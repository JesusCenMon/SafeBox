"""Persistencia, reconocimiento y salida de comandos de SAFEBOX AI."""

from __future__ import annotations

import json
import csv
import queue
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from control import AccessPinStore, LockController


MODEL_DIR = Path(__file__).resolve().parent / "models"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec_int8.onnx"
DEFAULT_SIMILARITY_THRESHOLD = 0.5
ENROLL_SAMPLES = 20


def load_models(model_dir: Path = MODEL_DIR):
    yunet_path, sface_path = model_dir / YUNET_FILE, model_dir / SFACE_FILE
    missing = [path.name for path in (yunet_path, sface_path)
               if not path.is_file() or path.stat().st_size < 100_000]
    if missing:
        raise FileNotFoundError(
            "Faltan modelos ONNX: " + ", ".join(missing) +
            ". Ejecuta python download_models.py"
        )
    try:
        detector = cv2.FaceDetectorYN.create(
            str(yunet_path), "", (320, 320), score_threshold=0.8,
            nms_threshold=0.3, top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
        recognizer = cv2.FaceRecognizerSF.create(
            str(sface_path), "", backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
    except cv2.error as exc:
        raise RuntimeError(f"No se pudieron cargar los modelos ONNX: {exc}") from exc
    return detector, recognizer


def detect_faces(detector, frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, 480 / width)
    sample = cv2.resize(frame, (round(width * scale), round(height * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1 else frame
    detector.setInputSize((sample.shape[1], sample.shape[0]))
    _, faces = detector.detect(sample)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    faces = faces.copy()
    faces[:, :14] /= scale
    return faces


def extract_feature(recognizer, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
    aligned = recognizer.alignCrop(frame, face)
    feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("No se pudo extraer un rasgo facial válido")
    return feature / norm


@dataclass(frozen=True)
class FaceObservation:
    box: tuple[int, int, int, int]
    person_id: str | None
    similarity: float | None
    feature: np.ndarray | None
    track_id: int = 0
    streak: int = 1


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    shared = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - shared
    return shared / union if union > 0 else 0.0


class FaceTracker:
    """Conserva una pista solo si posición, rasgos e identidad son inequívocos."""

    def __init__(self):
        self.previous: list[FaceObservation] = []
        self.next_id = 1

    def update(self, observations: list[FaceObservation]) -> list[FaceObservation]:
        matches: dict[int, list[int]] = {}
        for index, current in enumerate(observations):
            matches[index] = [old_index for old_index, old in enumerate(self.previous)
                              if current.person_id == old.person_id
                              and _overlap(current.box, old.box) >= 0.2
                              and current.feature is not None and old.feature is not None
                              and float(np.dot(current.feature, old.feature)) >= 0.5]
        updated = []
        for index, current in enumerate(observations):
            candidates = matches[index]
            old = self.previous[candidates[0]] if (len(candidates) == 1 and
                  sum(candidates[0] in found for found in matches.values()) == 1) else None
            track_id = old.track_id if old else self.next_id
            if old is None:
                self.next_id += 1
            updated.append(FaceObservation(current.box, current.person_id,
                                           current.similarity, current.feature, track_id,
                                           old.streak + 1 if old else 1))
        self.previous = updated
        return updated


class FaceStore:
    def __init__(self, root: Path):
        self.root = root
        self.faces_dir = root / "rostros"
        self.features_dir = root / "rasgos_sface"
        self.index_path = root / "usuarios.json"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.users = self._read_users()
        self.templates: dict[str, np.ndarray] = {}
        self._load_templates()

    def _read_users(self) -> dict[str, str]:
        if not self.index_path.exists():
            return {}
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(key, str) and key.isdecimal() and isinstance(value, str)
            for key, value in data.items()
        ):
            raise ValueError("datos/usuarios.json tiene un formato inválido")
        return data

    @staticmethod
    def _medoid(samples: np.ndarray) -> np.ndarray:
        if samples.ndim != 2 or samples.shape[1] != 128 or not np.isfinite(samples).all():
            raise ValueError("Rasgos SFace inválidos")
        norms = np.linalg.norm(samples, axis=1, keepdims=True)
        if not len(samples) or np.any(norms <= 0):
            raise ValueError("Rasgos SFace vacíos")
        normalized = samples / norms
        centrality = (normalized @ normalized.T).mean(axis=1)
        return normalized[int(np.argmax(centrality))].astype(np.float32)

    def _load_templates(self) -> None:
        for person_id in self.users:
            path = self.features_dir / f"{person_id}.npy"
            if path.is_file():
                samples = np.load(path, allow_pickle=False)
                self.templates[person_id] = self._medoid(samples)

    def add_person(self, name: str, samples: list[np.ndarray]) -> str:
        if len(samples) < ENROLL_SAMPLES:
            raise ValueError("Faltan muestras para registrar el rostro")
        existing_id = next((uid for uid, value in self.users.items()
                            if value.casefold() == name.casefold()), None)
        if existing_id in self.templates:
            raise ValueError("Ese nombre ya tiene un perfil SFace")
        person_id = existing_id or str(uuid.uuid4().int % 2_000_000_000)
        while person_id in self.users and existing_id is None:
            person_id = str(uuid.uuid4().int % 2_000_000_000)
        matrix = np.asarray(samples, dtype=np.float32)
        template = self._medoid(matrix)
        destination = self.features_dir / f"{person_id}.npy"
        temporary = self.features_dir / f"{person_id}.tmp"
        try:
            with temporary.open("wb") as file:
                np.save(file, matrix, allow_pickle=False)
            temporary.replace(destination)
            updated = {**self.users, person_id: name}
            if existing_id is None:
                temp = self.index_path.with_suffix(".tmp")
                temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
                temp.replace(self.index_path)
            self.users = updated
            self.templates[person_id] = template
            return person_id
        except Exception:
            temporary.unlink(missing_ok=True)
            if existing_id is None and person_id not in self.users:
                destination.unlink(missing_ok=True)
            raise

    def identify_id(self, feature: np.ndarray, threshold: float) -> tuple[str | None, float | None]:
        if not self.templates:
            return None, None
        probe = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(probe))
        if probe.size != 128 or norm <= 0 or not np.isfinite(norm):
            raise ValueError("Rasgo facial inválido")
        probe /= norm
        ranked = sorted(((uid, float(np.dot(probe, template)))
                         for uid, template in self.templates.items()),
                        key=lambda item: item[1], reverse=True)
        person_id, similarity = ranked[0]
        unambiguous = len(ranked) == 1 or similarity - ranked[1][1] >= 0.05
        return (person_id if similarity >= threshold and unambiguous else None, similarity)

    def identify(self, feature: np.ndarray, threshold: float) -> tuple[str | None, float | None]:
        person_id, similarity = self.identify_id(feature, threshold)
        return (self.users[person_id] if person_id is not None else None, similarity)

    def list_people(self) -> list[tuple[str, str, bool]]:
        return sorted(((uid, name, uid in self.templates) for uid, name in self.users.items()),
                      key=lambda item: item[1].casefold())

    def delete_person(self, person_id: str) -> str:
        if person_id not in self.users:
            raise ValueError("El usuario ya no existe")
        name = self.users[person_id]
        updated = {key: value for key, value in self.users.items() if key != person_id}
        temp = self.index_path.with_suffix(".tmp")
        temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.index_path)
        self.users = updated
        self.templates.pop(person_id, None)
        (self.features_dir / f"{person_id}.npy").unlink(missing_ok=True)
        shutil.rmtree(self.faces_dir / person_id, ignore_errors=True)
        return name


class AccessHistory:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.capture_dir = root / "capturas"
        self.capture_dir.mkdir(exist_ok=True)
        self.db = sqlite3.connect(root / "historial.db")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS accesos ("
            "id INTEGER PRIMARY KEY, fecha TEXT NOT NULL, usuario TEXT NOT NULL, "
            "resultado TEXT NOT NULL, confianza INTEGER, captura TEXT NOT NULL, "
            "distancia REAL, similitud REAL)"
        )
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(accesos)")}
        if "distancia" not in columns:
            self.db.execute("ALTER TABLE accesos ADD COLUMN distancia REAL")
        if "similitud" not in columns:
            self.db.execute("ALTER TABLE accesos ADD COLUMN similitud REAL")
        self.db.commit()

    def record(self, user: str, result: str, similarity: float | None, frame: np.ndarray) -> None:
        when = datetime.now().astimezone().isoformat(timespec="seconds")
        filename = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
        path = self.capture_dir / filename
        if not cv2.imwrite(str(path), frame):
            raise OSError("No se pudo guardar la captura del intento")
        try:
            with self.db:
                self.db.execute(
                    "INSERT INTO accesos (fecha, usuario, resultado, confianza, captura, distancia, similitud) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (when, user, result, None, filename, None, similarity),
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def recent(self, limit: int = 8) -> list[tuple]:
        return self.db.execute(
            "SELECT fecha, usuario, resultado, distancia, similitud FROM accesos "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def search(self, user: str = "", result: str = "Todos", limit: int = 31,
               offset: int = 0) -> list[tuple]:
        pattern = "%" + user.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        return self.db.execute(
            "SELECT fecha, usuario, resultado, distancia, similitud, captura FROM accesos "
            "WHERE usuario LIKE ? ESCAPE '\\' AND (? = 'Todos' OR resultado = ?) "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (pattern, result, result, limit, offset),
        ).fetchall()

    def stats(self) -> tuple[int, int, int, str | None]:
        total, allowed, denied, latest = self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(resultado='AUTORIZADO'),0), "
            "COALESCE(SUM(resultado='DENEGADO'),0), MAX(fecha) FROM accesos"
        ).fetchone()
        return int(total), int(allowed), int(denied), latest

    def export_csv(self, path: Path) -> int:
        rows = self.db.execute(
            "SELECT fecha, usuario, resultado, distancia, similitud FROM accesos ORDER BY id DESC"
        )
        count = 0
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("fecha", "usuario", "resultado", "distancia_lbph", "similitud_sface"))
            for row in rows:
                writer.writerow(row)
                count += 1
        return count

    def close(self) -> None:
        self.db.close()


class CommandBridge:
    """Comandos con ACK; ninguna acción de servo implícita al conectar o cerrar."""

    BOOT_WAIT_SECONDS = 2.0
    INITIAL_PING_TIMEOUT = 4.0
    INITIAL_PING_RETRIES = 2

    def __init__(self, mode: str, port: str | None = None, baudrate: int = 115200):
        self.mode = mode
        self.serial = None
        self.last_command = None
        self.sequence = 0
        self.boot_id = None
        self.servo_status: str | None = None
        if mode in {"hardware", "leds"}:
            if not port:
                raise ValueError("El modo hardware requiere --port COMx")
            try:
                import serial  # pyserial solo es necesario al conectar hardware
            except ImportError as exc:
                raise RuntimeError(
                    f"Falta pyserial en {sys.executable}; instálalo con ese Python: -m pip install pyserial"
                ) from exc
            self.serial = serial.Serial(port=port, baudrate=baudrate,
                                        timeout=0.1, write_timeout=0.5)
            try:
                time.sleep(self.BOOT_WAIT_SECONDS)
                for attempt in range(self.INITIAL_PING_RETRIES + 1):
                    try:
                        self.send("PING", timeout=self.INITIAL_PING_TIMEOUT,
                                  close_on_error=False)
                        break
                    except (OSError, TimeoutError, ConnectionError):
                        if attempt == self.INITIAL_PING_RETRIES:
                            raise
                if mode == "hardware":
                    self.servo_status = self.send("SERVO_STATUS")
            except Exception:
                self.close()
                raise

    def send(self, command: str, timeout: float = 1.5, *,
             close_on_error: bool = True, pulse_us: int | None = None) -> str | None:
        allowed = ({"PING", "AUTH_OK", "AUTH_FAIL", "LED_OFF"} if self.mode == "leds" else
                   {"OPEN", "LOCK", "DENIED", "IDLE", "PING", "AUTH_OK", "AUTH_FAIL",
                    "LED_OFF", "SERVO_STATUS", "TEST_SERVO", "SERVO_STOP"}
                   if self.mode == "hardware" else {"OPEN", "LOCK", "DENIED", "IDLE", "PING"})
        if command not in allowed:
            raise ValueError("Comando no permitido")
        if command == "TEST_SERVO":
            if not isinstance(pulse_us, int) or not 1300 <= pulse_us <= 1700:
                raise ValueError("Pulso de prueba: 1300 a 1700 microsegundos")
        elif pulse_us is not None:
            raise ValueError("Este comando no acepta un pulso")
        if self.mode in {"hardware", "leds"} and self.serial is None:
            raise ConnectionError("ESP32 sin conexión")
        if self.mode in {"hardware", "leds"} and command != "PING" and self.boot_id is None:
            raise ConnectionError("Falta PING inicial de la ESP32")
        if self.serial is not None:
            self.sequence += 1
            sequence = self.sequence
            status = None
            try:
                line = (f"PING {sequence}\n" if command == "PING" else
                        f"TEST_SERVO {sequence} {self.boot_id} {pulse_us}\n" if command == "TEST_SERVO" else
                        f"{command} {sequence} {self.boot_id}\n")
                self.serial.write(line.encode("ascii"))
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    line = self.serial.readline().decode("ascii", errors="replace").strip()
                    parts = line.split()
                    if (len(parts) == (5 if command == "SERVO_STATUS" else 4) and
                            parts[:3] == ["ACK", str(sequence), command] and
                            all(char in "0123456789ABCDEFabcdef" for char in parts[3])):
                        if self.boot_id is None:
                            self.boot_id = parts[3].upper()
                        elif self.boot_id != parts[3].upper():
                            raise ConnectionError("La ESP32 se reinició; confirma el cierre antes de abrir")
                        if command == "SERVO_STATUS":
                            if parts[4] not in {"LISTO", "NO_CONFIGURADO", "ERROR"}:
                                raise ConnectionError(f"Estado del servo inválido: {parts[4]}")
                            status = parts[4]
                        break
                    if parts and parts[0] in {"ACK", "ERR"}:
                        if parts[:3] == ["ERR", str(sequence), command]:
                            if command in {"TEST_SERVO", "SERVO_STOP"}:
                                raise ValueError(f"ESP32 rechazó {command}; revisa pulso, pasos y PWM")
                            raise ConnectionError(f"ESP32 rechazó {command}")
                        raise ConnectionError(f"Respuesta serial incorrecta para {command}: {line}")
                else:
                    raise TimeoutError(f"Sin ACK para {command}")
            except (OSError, TimeoutError, ConnectionError):
                if close_on_error:
                    self.close()
                raise
        if command != "PING":
            self.last_command = command
        return status if self.serial is not None else None

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None


class LedNotifier:
    """Sincroniza el último estado LED sin bloquear el motor ni tocar el servo."""

    def __init__(self, port: str, events: queue.Queue):
        self.port = port
        self.events = events
        self.desired = "LED_OFF"
        self.changed_at = 0.0
        self.generation = 0
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="safebox-leds")

    def start(self) -> None:
        self.thread.start()

    def notify(self, command: str) -> None:
        if command not in {"AUTH_OK", "AUTH_FAIL", "LED_OFF"}:
            raise ValueError("Aviso LED inválido")
        with self.lock:
            if command == self.desired:
                return
            self.desired = command
            self.changed_at = time.monotonic()
            self.generation += 1
        self.wake.set()

    def stop(self) -> None:
        self.stop_event.set()
        self.wake.set()

    def _run(self) -> None:
        bridge = None
        next_ping = 0.0
        applied_generation = -1
        auth_attempt_generation = -1
        announced = False
        try:
            while not self.stop_event.is_set():
                if bridge is None:
                    try:
                        bridge = CommandBridge("leds", self.port)
                        next_ping = time.monotonic() + 5.0
                    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
                        self.events.put(("led_device", False, f"LED ESP32 sin conexión: {exc}"))
                        self.stop_event.wait(5.0)
                        continue
                try:
                    with self.lock:
                        command, requested_at, generation = (
                            self.desired, self.changed_at, self.generation)
                    if generation != applied_generation:
                        # Un AUTH_OK ya enviado no debe reactivarse al reconectar.
                        if (command == "AUTH_OK" and
                                (generation == auth_attempt_generation or
                                 time.monotonic() - requested_at > 1.0)):
                            bridge.send("LED_OFF", timeout=0.75)
                        else:
                            if command == "AUTH_OK":
                                auth_attempt_generation = generation
                            bridge.send(command, timeout=0.75)
                        applied_generation = generation
                    if not announced:
                        self.events.put(("led_device", True, f"LED ESP32 conectado en {self.port}"))
                        announced = True
                    if time.monotonic() >= next_ping:
                        bridge.send("PING", timeout=0.75)
                        next_ping = time.monotonic() + 5.0
                except (OSError, TimeoutError, ConnectionError) as exc:
                    self.events.put(("led_device", False, f"LED ESP32 sin ACK: {exc}"))
                    bridge.close()
                    bridge = None
                    applied_generation = -1
                    announced = False
                    continue
                self.wake.wait(0.1)
                self.wake.clear()
        finally:
            if bridge is not None:
                bridge.close()


class SafeBackend:
    """La cámara y el motor son dueños de sus recursos; Tk nunca sale del hilo principal."""

    HEARTBEAT_INTERVAL = 5.0
    PING_TIMEOUT = 0.75

    def __init__(self, data_dir: Path, mode: str, port: str | None,
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 camera_index: int = 0, authorization_seconds: int = 8,
                 led_port: str | None = None):
        if not 0.5 <= similarity_threshold <= 0.9:
            raise ValueError("El umbral de similitud debe estar entre 0.5 y 0.9")
        if led_port and (mode != "simulacion" or not led_port.upper().startswith("COM")):
            raise ValueError("LED ESP32 requiere simulación y un puerto COM")
        self.data_dir = data_dir
        self.mode = mode
        self.port = port
        self.similarity_threshold = similarity_threshold
        self.camera_index = camera_index
        self.authorization_seconds = authorization_seconds
        self.admin_ready = False
        self.physical_unknown = mode == "hardware"
        self.led_port = led_port
        self.servo_status = "NO_CONFIGURADO"
        self.events: queue.Queue[tuple] = queue.Queue()
        self.led_notifier = LedNotifier(led_port, self.events) if led_port and mode == "simulacion" else None
        self.commands: queue.Queue[tuple] = queue.Queue()
        self.camera_commands: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_seq = 0
        self.camera_connected = False
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True, name="safebox-camera")
        self.engine_thread = threading.Thread(target=self._engine_loop, daemon=True, name="safebox-engine")

    def start(self) -> None:
        if self.led_notifier is not None:
            self.led_notifier.start()
        self.camera_thread.start()
        self.engine_thread.start()

    def latest_frame(self, after_seq: int) -> tuple[int, np.ndarray] | None:
        with self.frame_lock:
            if self.frame is None or self.frame_seq == after_seq:
                return None
            return self.frame_seq, self.frame

    def send(self, kind: str, value=None) -> None:
        self.commands.put((kind, value))

    def retry_camera(self) -> None:
        self.camera_commands.put("retry")

    def select_camera(self, index: int) -> None:
        self.camera_commands.put(("switch", index))

    def stop(self) -> None:
        self.stop_event.set()
        if self.led_notifier is not None:
            self.led_notifier.stop()
        self.commands.put(("stop", None))
        self.camera_commands.put("stop")

    def stopped(self) -> bool:
        return (not self.camera_thread.is_alive() and not self.engine_thread.is_alive() and
                (self.led_notifier is None or not self.led_notifier.thread.is_alive()))

    def _camera_loop(self) -> None:
        capture = None
        open_requested = True
        try:
            while not self.stop_event.is_set():
                if capture is None:
                    if not open_requested:
                        try:
                            camera_command = self.camera_commands.get(timeout=0.2)
                            if isinstance(camera_command, tuple) and camera_command[0] == "switch":
                                self.camera_index = camera_command[1]
                        except queue.Empty:
                            continue
                    open_requested = False
                    if self.stop_event.is_set():
                        break
                    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                    candidate = cv2.VideoCapture(self.camera_index, backend)
                    if not candidate.isOpened():
                        candidate.release()
                        self.camera_connected = False
                        self.events.put(("camera", False, f"Cámara {self.camera_index} no disponible"))
                        continue
                    capture = candidate
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.camera_connected = True
                    self.events.put(("camera", True, ""))
                try:
                    camera_command = self.camera_commands.get_nowait()
                except queue.Empty:
                    camera_command = None
                if isinstance(camera_command, tuple) and camera_command[0] == "switch":
                    self.camera_index = camera_command[1]
                    capture.release()
                    capture = None
                    self.camera_connected = False
                    self.events.put(("camera", False, "Cambiando cámara…"))
                    open_requested = True
                    continue
                read_started = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    capture.release()
                    capture = None
                    self.camera_connected = False
                    self.events.put(("camera", False, "No se pudo leer la cámara"))
                    continue
                frame = cv2.flip(frame, 1)
                with self.frame_lock:
                    self.frame = frame
                    self.frame_seq += 1
                self.stop_event.wait(max(0.0, 1 / 30 - (time.monotonic() - read_started)))
        except Exception as exc:
            self.camera_connected = False
            self.events.put(("camera", False, f"Error de cámara: {exc}"))
        finally:
            self.camera_connected = False
            if capture is not None:
                capture.release()

    def _engine_loop(self) -> None:
        history = None
        bridge = None
        controller = LockController(self.authorization_seconds, ready=self.mode == "simulacion")
        last_state: tuple | None = None
        last_led_command = "LED_OFF"
        last_seq = 0
        last_analyze = 0.0
        last_face_seen = 0.0
        candidate_name = None
        streak = 0
        logged_key = None
        enrolling_name = None
        samples: list[np.ndarray] = []
        last_sample = 0.0
        enrolling_pin = None
        pin_store = None
        pending_access = None
        prompted_id = None
        authorized_id = None
        authorized_track = None
        last_verified_id = None
        last_verified_at = 0.0
        last_verified_similarity = None
        last_verified_track = None
        tracker = FaceTracker()
        face_labels: list[tuple[str, str]] = []
        next_ping = time.monotonic() + self.HEARTBEAT_INTERVAL
        device_error_reported = False

        def state(name: str, detail: str, user: str = "—", similarity: float | None = None,
                  boxes: list[tuple[int, int, int, int]] | None = None) -> None:
            nonlocal last_state, last_led_command
            led_command = ("AUTH_FAIL" if name == "DENEGADO" else
                           "AUTH_OK" if name == "AUTORIZADO" else "LED_OFF")
            if led_command != last_led_command:
                if self.mode == "hardware" and bridge is not None and bridge.serial is not None:
                    try:
                        bridge.send(led_command, timeout=0.75)
                    except (OSError, TimeoutError, ConnectionError) as exc:
                        controller.fail(f"LED ESP32 sin respuesta: {exc}")
                        publish_lock()
                elif self.led_notifier is not None:
                    self.led_notifier.notify(led_command)
                last_led_command = led_command
            visible_boxes = (boxes if boxes is not None else
                             [item.box for item in tracker.previous]
                             if name in {"AUTORIZADO", "DENEGADO", "PIN REQUERIDO"} else [])
            labels = face_labels if visible_boxes else []
            payload = (name, detail, user, similarity, tuple(visible_boxes), tuple(labels))
            if payload != last_state:
                self.events.put(("state", name, detail, user, similarity,
                                 list(visible_boxes), list(labels)))
                last_state = payload

        def publish_lock() -> None:
            nonlocal device_error_reported
            self.events.put(("lock", controller.state, controller.reason))
            if (self.mode == "hardware" and controller.state == LockController.ERROR and
                    not (self.servo_status == "NO_CONFIGURADO" and bridge is not None and
                         bridge.serial is not None) and self.servo_status != "ERROR"):
                self.servo_status = "ERROR"
                self.events.put(("servo_status", "ERROR"))
            if (self.mode == "hardware" and controller.state == LockController.ERROR and
                    not (self.servo_status == "NO_CONFIGURADO" and bridge is not None and
                         bridge.serial is not None) and
                    not device_error_reported):
                self.events.put(("device_error", controller.reason))
                device_error_reported = True

        def record_attempt(user: str, result: str, similarity: float | None,
                           frame: np.ndarray) -> None:
            try:
                history.record(user, result, similarity, frame)
                self.events.put(("history", history.recent()))
                self.events.put(("stats", history.stats()))
            except (OSError, sqlite3.Error) as exc:
                self.events.put(("storage_error", str(exc)))

        def end_pending(reason: str) -> None:
            nonlocal pending_access
            if pending_access is None:
                return
            pending_access = None
            if self.frame is not None:
                record_attempt("Desconocido", "DENEGADO", last_verified_similarity, self.frame)
            self.events.put(("pin_expired", reason))

        try:
            store = FaceStore(self.data_dir)
            pin_store = AccessPinStore(self.data_dir)
            history = AccessHistory(self.data_dir)
            face_detector, face_recognizer = load_models()
            try:
                bridge = CommandBridge(self.mode, self.port)
                if self.mode == "hardware":
                    self.servo_status = bridge.servo_status or "ERROR"
                    if self.servo_status == "LISTO":
                        controller.state = LockController.WAITING_CLOSE
                        controller.reason = "Servo listo; confirma el cierre antes de bloquear"
                    else:
                        controller.fail("Servo sin calibrar" if self.servo_status == "NO_CONFIGURADO"
                                        else "No se pudo inicializar PWM del servo")
                self.events.put(("device", self.mode, self.port, self.physical_unknown))
            except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
                controller.fail(f"ESP32 sin conexión: {exc}")
                self.servo_status = "ERROR"
            self.events.put(("servo_status", self.servo_status if self.mode == "hardware" else "NO CONECTADO"))
            self.events.put(("ready", list(store.users.values()), history.recent(), store.list_people()))
            self.events.put(("pins", pin_store.ids()))
            self.events.put(("stats", history.stats()))
            publish_lock()
            while not self.stop_event.is_set():
                if (self.mode == "hardware" and bridge is not None and
                        bridge.serial is not None and time.monotonic() >= next_ping):
                    try:
                        bridge.send("PING", timeout=self.PING_TIMEOUT)
                    except (OSError, TimeoutError, ConnectionError) as exc:
                        controller.fail(f"Se perdió la conexión con la ESP32 ({exc})")
                        publish_lock()
                    next_ping = time.monotonic() + self.HEARTBEAT_INTERVAL
                if pending_access is not None and time.monotonic() > pending_access[1]:
                    end_pending("Tiempo agotado; vuelve a mostrar el rostro")
                if controller.tick():
                    controller.reason = "Tiempo cumplido; confirma el cierre antes de bloquear"
                    publish_lock()
                while True:
                    try:
                        kind, value = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "start_enroll":
                        name, access_pin = value if isinstance(value, tuple) else (value, None)
                        name = (name or "").strip()
                        if not self.camera_connected:
                            self.events.put(("enroll_error", "La cámara no está disponible"))
                        elif access_pin is None or not (6 <= len(access_pin) <= 12 and access_pin.isascii() and access_pin.isdecimal()):
                            self.events.put(("enroll_error", "PIN de acceso: 6 a 12 dígitos"))
                        elif any(uid in store.templates and existing.casefold() == name.casefold()
                                 for uid, existing in store.users.items()):
                            self.events.put(("enroll_error", "Ese nombre ya tiene un perfil SFace"))
                        else:
                            if pending_access is not None:
                                end_pending("Registro iniciado")
                            tracker.update([])
                            face_labels = []
                            enrolling_name = name
                            enrolling_pin = access_pin
                            samples = []
                            last_sample = 0.0
                            candidate_name, streak = None, 0
                            self.events.put(("enroll_started", name))
                    elif kind == "cancel_enroll":
                        enrolling_name = None
                        enrolling_pin = None
                        samples = []
                        tracker.update([])
                        face_labels = []
                        candidate_name, streak = None, 0
                        state("ESCANEANDO", "Esperando un rostro frente a la cámara")
                    elif kind == "list_users":
                        self.events.put(("users", store.list_people()))
                    elif kind == "delete_user":
                        try:
                            deleted = store.delete_person(value or "")
                            pin_store.remove(value or "")
                            self.events.put(("user_deleted", deleted, store.list_people()))
                            self.events.put(("pins", pin_store.ids()))
                            candidate_name, streak, logged_key = None, 0, None
                            state("ESCANEANDO", "Modelo facial actualizado")
                        except (OSError, ValueError) as exc:
                            self.events.put(("user_error", str(exc)))
                    elif kind == "set_access_pin":
                        person_id, access_pin = value
                        try:
                            if person_id not in store.users:
                                raise ValueError("El usuario ya no existe")
                            pin_store.set_pin(person_id, access_pin)
                            self.events.put(("pins", pin_store.ids()))
                            self.events.put(("pin_set", store.users[person_id]))
                        except (OSError, ValueError) as exc:
                            self.events.put(("user_error", str(exc)))
                    elif kind == "cancel_access":
                        end_pending("Verificación cancelada")
                    elif kind == "access_pin":
                        if not isinstance(value, tuple) or len(value) != 3:
                            self.events.put(("pin_failed", "Solicitud de PIN caducada"))
                            continue
                        person_id, access_pin, request_track = value
                        fresh = (self.camera_connected and enrolling_name is None and
                                 pending_access is not None and
                                 pending_access[0] == person_id and
                                 pending_access[2] == request_track and
                                 time.monotonic() <= pending_access[1] and
                                 last_verified_id == person_id and
                                 last_verified_track == pending_access[2] and
                                 time.monotonic() - last_verified_at <= 2.0 and
                                 self.frame_seq - last_seq <= 20 and
                                 len([face for face in tracker.previous if face.person_id]) == 1 and
                                 any(face.track_id == pending_access[2] and face.streak >= 4
                                     for face in tracker.previous))
                        if not fresh or controller.state != LockController.LOCKED or bridge is None:
                            end_pending("Rostro o cerradura no disponible; intenta de nuevo")
                            continue
                        try:
                            pin_store.verify(person_id, access_pin)
                        except (PermissionError, OSError) as exc:
                            self.events.put(("pin_failed", str(exc)))
                            state("DENEGADO", "PIN de acceso incorrecto")
                            if self.frame is not None:
                                record_attempt("Desconocido", "DENEGADO", last_verified_similarity, self.frame)
                            continue
                        pending_access = None
                        if controller.authorize(bridge, publish_lock):
                            authorized_id = person_id
                            authorized_track = last_verified_track
                            name = store.users[person_id]
                            face_labels = [(name, "AUTORIZADO") if item.track_id == authorized_track
                                           else label for item, label in zip(tracker.previous, face_labels)]
                            state("AUTORIZADO", "Rostro y PIN verificados", name, last_verified_similarity)
                            if self.frame is not None:
                                record_attempt(name, "AUTORIZADO", last_verified_similarity, self.frame)
                            self.events.put(("pin_success", name))
                            self.events.put(("welcome",))
                        else:
                            state("DENEGADO", "La cerradura no confirmó la apertura")
                            self.events.put(("pin_expired", "La cerradura no confirmó la apertura"))
                        publish_lock()
                    elif kind == "search_history":
                        try:
                            user, result, offset = value
                            self.events.put(("history_search", user, result, offset,
                                             history.search(user, result, offset=offset)))
                        except sqlite3.Error as exc:
                            self.events.put(("storage_error", str(exc)))
                    elif kind == "admin_ready":
                        self.admin_ready = bool(value)
                    elif kind == "update_settings":
                        self.similarity_threshold, controller.authorization_seconds = value
                    elif kind == "confirm_closed":
                        if bridge is None:
                            controller.fail("Sin comunicación con la ESP32")
                        else:
                            controller.confirm_closed(bridge)
                        authorized_id = None
                        publish_lock()
                    elif kind == "test_servo":
                        if self.mode != "hardware" or bridge is None or self.servo_status != "NO_CONFIGURADO":
                            self.events.put(("servo_test_result", False,
                                             "Prueba disponible solo en hardware sin calibrar"))
                        else:
                            try:
                                bridge.send("TEST_SERVO", pulse_us=value)
                                self.events.put(("servo_test_result", True,
                                                 f"PWM aceptado: {value} µs; verifica el movimiento real"))
                            except ValueError as exc:
                                self.events.put(("servo_test_result", False, str(exc)))
                            except (OSError, TimeoutError, ConnectionError) as exc:
                                controller.fail(f"Prueba del servo sin ACK: {exc}")
                                self.servo_status = "ERROR"
                                self.events.put(("servo_status", "ERROR"))
                                self.events.put(("servo_test_result", False, controller.reason))
                                publish_lock()
                    elif kind == "servo_stop":
                        if (self.mode == "hardware" and bridge is not None and
                                self.servo_status == "NO_CONFIGURADO"):
                            try:
                                bridge.send("SERVO_STOP")
                                self.events.put(("servo_test_result", True, "Señal PWM detenida"))
                            except (OSError, TimeoutError, ConnectionError, ValueError) as exc:
                                self.events.put(("servo_test_result", False, str(exc)))
                    elif kind == "connect_device":
                        new_mode, new_port = value
                        if controller.state in {LockController.UNLOCKED, LockController.PENDING}:
                            self.events.put(("configuration_error", "Confirma el cierre antes de cambiar de modo"))
                        else:
                            stopped_led = False
                            try:
                                if new_mode == "hardware" and self.led_notifier is not None:
                                    self.led_notifier.stop()
                                    self.led_notifier.thread.join(16)
                                    if self.led_notifier.thread.is_alive():
                                        raise RuntimeError("El puerto LED aún se está cerrando; intenta de nuevo")
                                    self.led_notifier = None
                                    stopped_led = True
                                candidate = CommandBridge(new_mode, new_port)
                                if bridge is not None:
                                    bridge.close()
                                bridge = candidate
                                self.mode, self.port = new_mode, new_port
                                self.physical_unknown = self.physical_unknown or new_mode == "hardware"
                                self.servo_status = (candidate.servo_status or "ERROR"
                                                     if new_mode == "hardware" else "NO_CONFIGURADO")
                                if new_mode == "simulacion":
                                    controller.state = LockController.LOCKED
                                    controller.reason = "Simulación activa; estado físico desconocido" if self.physical_unknown else "Simulación activa"
                                    if self.led_port and self.led_notifier is None:
                                        self.led_notifier = LedNotifier(self.led_port, self.events)
                                        self.led_notifier.start()
                                elif self.servo_status == "LISTO":
                                    controller.state = LockController.WAITING_CLOSE
                                    controller.reason = "Servo listo; confirma el cierre antes de bloquear"
                                else:
                                    controller.fail("Servo sin calibrar" if self.servo_status == "NO_CONFIGURADO"
                                                    else "No se pudo inicializar PWM del servo")
                                last_led_command = "LED_OFF"
                                next_ping = time.monotonic() + self.HEARTBEAT_INTERVAL
                                device_error_reported = False
                                self.events.put(("device", new_mode, new_port, self.physical_unknown))
                                self.events.put(("servo_status", self.servo_status if new_mode == "hardware"
                                                 else "NO CONECTADO"))
                                publish_lock()
                            except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
                                if stopped_led and self.led_port:
                                    self.led_notifier = LedNotifier(self.led_port, self.events)
                                    self.led_notifier.start()
                                self.events.put(("configuration_error", str(exc)))
                    elif kind == "stats":
                        self.events.put(("stats", history.stats()))
                    elif kind == "export_history":
                        try:
                            count = history.export_csv(Path(value))
                            self.events.put(("export_done", count, value))
                        except (OSError, sqlite3.Error) as exc:
                            self.events.put(("storage_error", str(exc)))
                    elif kind == "stop":
                        break
                if self.stop_event.is_set():
                    break
                if not self.camera_connected:
                    tracker.update([])
                    face_labels = []
                    candidate_name, streak = None, 0
                    authorized_id = None
                    last_verified_id = None
                    if pending_access is not None:
                        end_pending("Cámara desconectada")
                    state("CÁMARA DESCONECTADA", "Revisa conexión y permisos de Windows")
                    self.stop_event.wait(0.06)
                    continue
                now = time.monotonic()
                if now - last_analyze < 0.14:
                    self.stop_event.wait(0.025)
                    continue
                latest = self.latest_frame(last_seq)
                if latest is None:
                    self.stop_event.wait(0.025)
                    continue
                last_seq, frame = latest
                last_analyze = now
                found = detect_faces(face_detector, frame)
                boxes = [tuple(round(float(v)) for v in face[:4]) for face in found]
                count = len(boxes)
                if count == 0:
                    tracker.update([])
                    face_labels = []
                    candidate_name, streak = None, 0
                    authorized_id = None
                    authorized_track = None
                    last_verified_id = None
                    if now - last_face_seen > 1.0:
                        if pending_access is not None:
                            end_pending("Rostro fuera de cámara")
                        prompted_id = None
                        authorized_id = None
                        last_verified_id = None
                        logged_key = None
                    if enrolling_name:
                        state("REGISTRANDO", "Muestra un rostro para continuar", enrolling_name)
                    else:
                        state("ESCANEANDO", "Esperando un rostro frente a la cámara")
                    continue
                last_face_seen = now
                if enrolling_name and count > 1:
                    tracker.update([])
                    face_labels = []
                    state("REGISTRANDO", "Debe aparecer una sola persona", enrolling_name, boxes=boxes)
                    continue
                if enrolling_name:
                    try:
                        feature = extract_feature(face_recognizer, frame, found[0])
                    except (cv2.error, ValueError):
                        state("REGISTRANDO", "Ajusta el rostro frente a la cámara", boxes=boxes)
                        continue
                    tracker.update([])
                    face_labels = []
                    if now - last_sample >= 0.25:
                        samples.append(feature)
                        last_sample = now
                        self.events.put(("enroll_progress", len(samples)))
                    state("REGISTRANDO", f"Capturando · {len(samples)}/{ENROLL_SAMPLES}", enrolling_name, boxes=boxes)
                    if len(samples) >= ENROLL_SAMPLES:
                        name = enrolling_name
                        enrolling_name = None
                        state("VERIFICANDO", "Guardando perfil SFace…")
                        try:
                            person_id = store.add_person(name, samples)
                        except Exception as exc:
                            self.events.put(("enroll_error", str(exc)))
                        else:
                            pin_ok = True
                            try:
                                pin_store.set_pin(person_id, enrolling_pin)
                            except (OSError, ValueError) as exc:
                                pin_ok = False
                                self.events.put(("user_error", f"Rostro guardado, PIN no guardado: {exc}. Asigna el PIN en Usuarios."))
                            self.events.put(("enroll_done", name, list(store.users.values()), pin_ok))
                            self.events.put(("pins", pin_store.ids()))
                        samples = []
                        enrolling_pin = None
                        candidate_name, streak = None, 0
                    continue
                observations = []
                for face, box in zip(found, boxes):
                    try:
                        current_feature = extract_feature(face_recognizer, frame, face)
                        current_id, current_similarity = store.identify_id(current_feature, self.similarity_threshold)
                    except (cv2.error, ValueError):
                        current_feature, current_id, current_similarity = None, None, None
                    observations.append(FaceObservation(box, current_id, current_similarity,
                                                        current_feature))
                tracked = tracker.update(observations)
                face_labels = [
                    (store.users[item.person_id],
                     "AUTORIZADO" if authorized_id == item.person_id and
                     authorized_track == item.track_id else
                     "PIN PENDIENTE" if pending_access is not None and
                     pending_access[0] == item.person_id and
                     pending_access[2] == item.track_id else
                     "IDENTIDAD" if item.streak >= 4 else "VERIFICANDO")
                    if item.person_id else ("Desconocido", "DESCONOCIDO")
                    for item in tracked
                ]
                # Conserva la correspondencia con los recuadros incluso si falla un rasgo.
                boxes = [item.box for item in tracked]
                known = [item for item in tracked if item.person_id is not None]
                if pending_access is not None and (len(known) != 1 or
                        known[0].track_id != pending_access[2] or
                        known[0].person_id != pending_access[0] or known[0].streak < 4):
                    end_pending("Se perdió el seguimiento del rostro; vuelve a intentar")
                    last_verified_id = None
                    last_verified_track = None
                if len(known) > 1:
                    candidate_name, streak = None, 0
                    last_verified_id = None
                    last_verified_track = None
                    state("VERIFICANDO", "Varios usuarios registrados; deja solo uno para acceder",
                          boxes=boxes)
                    continue
                if not tracked:
                    state("VERIFICANDO", "Ajusta el rostro frente a la cámara", boxes=boxes)
                    continue
                selected = known[0] if known else tracked[0]
                person_id, similarity = selected.person_id, selected.similarity
                identity = person_id or "Desconocido"
                streak = selected.streak
                candidate_name = identity
                if prompted_id is not None and prompted_id != (person_id, selected.track_id):
                    prompted_id = None
                    authorized_id = None
                    authorized_track = None
                if streak < 4:
                    state("VERIFICANDO", f"Lectura {streak}/4", similarity=similarity, boxes=boxes)
                elif person_id is not None:
                    last_verified_id = person_id
                    last_verified_track = selected.track_id
                    last_verified_at = now
                    last_verified_similarity = similarity
                    if authorized_id == person_id and authorized_track == selected.track_id:
                        state("AUTORIZADO", "Rostro y PIN verificados", store.users[person_id], similarity, boxes)
                    elif not self.admin_ready or not pin_store.has_pin(person_id):
                        detail = ("Configura el PIN de administrador" if not self.admin_ready else
                                  "Esta identidad necesita un PIN de acceso")
                        state("DENEGADO", detail, "Desconocido", similarity, boxes)
                        key = f"no-pin:{person_id}"
                        if logged_key != key:
                            record_attempt("Desconocido", "DENEGADO", similarity, frame)
                            logged_key = key
                    elif controller.state != LockController.LOCKED or bridge is None:
                        state("VERIFICANDO", "Cerradura no disponible para una nueva apertura",
                              similarity=similarity, boxes=boxes)
                    else:
                        if prompted_id == (person_id, selected.track_id) and pending_access is None:
                            state("DENEGADO", "Retira el rostro y vuelve a intentar", boxes=boxes)
                        else:
                            state("PIN REQUERIDO", "Posible coincidencia; introduce el PIN de acceso",
                                  store.users[person_id], similarity, boxes)
                        if prompted_id != (person_id, selected.track_id):
                            pending_access = (person_id, now + 15.0, selected.track_id)
                            prompted_id = (person_id, selected.track_id)
                            self.events.put(("pin_required", person_id, store.users[person_id],
                                             selected.track_id))
                else:
                    detail = "Rostro no registrado" if similarity is not None else "No hay perfiles SFace activos"
                    state("DENEGADO", detail, "Desconocido", similarity, boxes)
                    if logged_key != "unknown":
                        record_attempt("Desconocido", "DENEGADO", similarity, frame)
                        logged_key = "unknown"
        except Exception as exc:
            self.events.put(("engine_error", str(exc)))
        finally:
            if bridge is not None:
                try:
                    bridge.close()
                except Exception:
                    pass
            if history is not None:
                history.close()
