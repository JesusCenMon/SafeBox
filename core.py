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

    def __init__(self, mode: str, port: str | None = None, baudrate: int = 115200):
        self.mode = mode
        self.serial = None
        self.last_command = None
        self.sequence = 0
        if mode == "hardware":
            if not port:
                raise ValueError("El modo hardware requiere --port COMx")
            try:
                import serial  # pyserial solo es necesario al conectar hardware
            except ImportError as exc:
                raise RuntimeError(
                    "Instala pyserial para usar hardware: python -m pip install pyserial"
                ) from exc
            self.serial = serial.Serial(port=port, baudrate=baudrate,
                                        timeout=0.1, write_timeout=0.5)
            try:
                self.send("PING", timeout=4.0)
            except Exception:
                self.serial.close()
                raise

    def send(self, command: str, timeout: float = 1.5) -> None:
        if command not in {"OPEN", "LOCK", "DENIED", "IDLE", "PING"}:
            raise ValueError("Comando no permitido")
        if self.mode == "hardware" and self.serial is None:
            raise ConnectionError("ESP32 sin conexión")
        if self.serial is not None:
            self.sequence += 1
            sequence = self.sequence
            try:
                self.serial.write(f"{command} {sequence}\n".encode("ascii"))
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    line = self.serial.readline().decode("ascii", errors="replace").strip()
                    if line == f"ACK {sequence} {command}":
                        break
                    if line == f"ERR {sequence} {command}":
                        raise ConnectionError(f"ESP32 rechazó {command}")
                else:
                    raise TimeoutError(f"Sin ACK para {command}")
            except (OSError, TimeoutError, ConnectionError):
                self.close()
                raise
        if command != "PING":
            self.last_command = command

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None


class SafeBackend:
    """La cámara y el motor son dueños de sus recursos; Tk nunca sale del hilo principal."""

    def __init__(self, data_dir: Path, mode: str, port: str | None,
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 camera_index: int = 0, authorization_seconds: int = 8):
        if not 0.5 <= similarity_threshold <= 0.9:
            raise ValueError("El umbral de similitud debe estar entre 0.5 y 0.9")
        self.data_dir = data_dir
        self.mode = mode
        self.port = port
        self.similarity_threshold = similarity_threshold
        self.camera_index = camera_index
        self.authorization_seconds = authorization_seconds
        self.admin_ready = False
        self.events: queue.Queue[tuple] = queue.Queue()
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
        self.commands.put(("stop", None))
        self.camera_commands.put("stop")

    def stopped(self) -> bool:
        return not self.camera_thread.is_alive() and not self.engine_thread.is_alive()

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
        last_verified_id = None
        last_verified_at = 0.0
        last_verified_similarity = None

        def state(name: str, detail: str, user: str = "—", similarity: float | None = None,
                  boxes: list[tuple[int, int, int, int]] | None = None) -> None:
            nonlocal last_state
            payload = (name, detail, user, similarity, tuple(boxes or ()))
            if payload != last_state:
                self.events.put(("state", name, detail, user, similarity, list(boxes or ())))
                last_state = payload

        def publish_lock() -> None:
            self.events.put(("lock", controller.state, controller.reason))

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
            if controller.state == LockController.LOCKED and bridge is not None:
                try:
                    bridge.send("DENIED")
                except (OSError, TimeoutError, ConnectionError) as exc:
                    controller.fail(f"ESP32 sin respuesta: {exc}")
                    publish_lock()
            self.events.put(("pin_expired", reason))

        try:
            store = FaceStore(self.data_dir)
            pin_store = AccessPinStore(self.data_dir)
            history = AccessHistory(self.data_dir)
            face_detector, face_recognizer = load_models()
            try:
                bridge = CommandBridge(self.mode, self.port)
                if self.mode == "hardware":
                    controller.state = LockController.WAITING_CLOSE
                    controller.reason = "Conexión verificada; confirma el cierre antes de bloquear"
                self.events.put(("device", self.mode, self.port))
            except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
                controller.fail(f"ESP32 sin conexión: {exc}")
                self.events.put(("device_error", controller.reason))
            self.events.put(("ready", list(store.users.values()), history.recent(), store.list_people()))
            self.events.put(("pins", pin_store.ids()))
            self.events.put(("stats", history.stats()))
            publish_lock()
            while not self.stop_event.is_set():
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
                        person_id, access_pin = value
                        fresh = (self.camera_connected and pending_access is not None and
                                 pending_access[0] == person_id and
                                 time.monotonic() <= pending_access[1] and
                                 last_verified_id == person_id and
                                 time.monotonic() - last_verified_at <= 2.0)
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
                            try:
                                bridge.send("DENIED")
                            except (OSError, TimeoutError, ConnectionError) as error:
                                controller.fail(f"ESP32 sin respuesta: {error}")
                                pending_access = None
                                self.events.put(("pin_expired", controller.reason))
                                publish_lock()
                            continue
                        pending_access = None
                        if controller.authorize(bridge, publish_lock):
                            authorized_id = person_id
                            name = store.users[person_id]
                            state("AUTORIZADO", "Rostro y PIN verificados", name, last_verified_similarity)
                            if self.frame is not None:
                                record_attempt(name, "AUTORIZADO", last_verified_similarity, self.frame)
                            self.events.put(("pin_success", name))
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
                    elif kind == "connect_device":
                        new_mode, new_port = value
                        if controller.state in {LockController.UNLOCKED, LockController.PENDING}:
                            self.events.put(("device_error", "Confirma el cierre antes de cambiar de modo"))
                        else:
                            try:
                                candidate = CommandBridge(new_mode, new_port)
                                if bridge is not None:
                                    bridge.close()
                                bridge = candidate
                                self.mode, self.port = new_mode, new_port
                                controller.state = (LockController.LOCKED if new_mode == "simulacion"
                                                    else LockController.WAITING_CLOSE)
                                controller.reason = ("Simulación activa" if new_mode == "simulacion" else
                                                     "ESP32 conectada; confirma el cierre")
                                self.events.put(("device", new_mode, new_port))
                                publish_lock()
                            except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
                                self.events.put(("device_error", str(exc)))
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
                    candidate_name, streak = None, 0
                    authorized_id = None
                    last_verified_id = None
                    if now - last_face_seen > 1.0:
                        if pending_access is not None:
                            end_pending("Rostro fuera de cámara")
                        prompted_id = None
                        authorized_id = None
                        last_verified_id = None
                        if logged_key is not None and controller.state == LockController.LOCKED and bridge is not None:
                            try:
                                bridge.send("IDLE")
                            except (OSError, TimeoutError, ConnectionError) as exc:
                                controller.fail(f"ESP32 sin respuesta: {exc}")
                                publish_lock()
                        logged_key = None
                    if enrolling_name:
                        state("REGISTRANDO", "Muestra un rostro para continuar", enrolling_name)
                    else:
                        state("ESCANEANDO", "Esperando un rostro frente a la cámara")
                    continue
                last_face_seen = now
                if count > 1:
                    candidate_name, streak = None, 0
                    authorized_id = None
                    if pending_access is not None:
                        end_pending("Hay varios rostros en cámara")
                    last_verified_id = None
                    if enrolling_name:
                        state("REGISTRANDO", "Debe aparecer una sola persona", enrolling_name, boxes=boxes)
                    else:
                        state("DENEGADO", "Hay varios rostros en cámara", "Varios rostros", boxes=boxes)
                        if logged_key != "multiple":
                            record_attempt("Varios rostros", "DENEGADO", None, frame)
                            if controller.state == LockController.LOCKED and bridge is not None:
                                try:
                                    bridge.send("DENIED")
                                except (OSError, TimeoutError, ConnectionError) as exc:
                                    controller.fail(f"ESP32 sin respuesta: {exc}")
                                    publish_lock()
                            logged_key = "multiple"
                    continue
                try:
                    feature = extract_feature(face_recognizer, frame, found[0])
                except (cv2.error, ValueError):
                    candidate_name, streak = None, 0
                    state("VERIFICANDO", "Ajusta el rostro frente a la cámara", boxes=boxes)
                    continue
                if enrolling_name:
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
                person_id, similarity = store.identify_id(feature, self.similarity_threshold)
                identity = person_id or "Desconocido"
                streak = streak + 1 if identity == candidate_name else 1
                candidate_name = identity
                if pending_access is not None and person_id != pending_access[0]:
                    end_pending("La identidad cambió; vuelve a intentar")
                    last_verified_id = None
                if person_id != prompted_id:
                    prompted_id = None
                    authorized_id = None
                if streak < 4:
                    state("VERIFICANDO", f"Lectura {streak}/4", similarity=similarity, boxes=boxes)
                elif person_id is not None:
                    last_verified_id = person_id
                    last_verified_at = now
                    last_verified_similarity = similarity
                    if authorized_id == person_id:
                        state("AUTORIZADO", "Rostro y PIN verificados", store.users[person_id], similarity, boxes)
                    elif not self.admin_ready or not pin_store.has_pin(person_id):
                        detail = ("Configura el PIN de administrador" if not self.admin_ready else
                                  "Esta identidad necesita un PIN de acceso")
                        state("DENEGADO", detail, "Desconocido", similarity, boxes)
                        key = f"no-pin:{person_id}"
                        if logged_key != key:
                            record_attempt("Desconocido", "DENEGADO", similarity, frame)
                            if controller.state == LockController.LOCKED and bridge is not None:
                                try:
                                    bridge.send("DENIED")
                                except (OSError, TimeoutError, ConnectionError) as exc:
                                    controller.fail(f"ESP32 sin respuesta: {exc}")
                                    publish_lock()
                            logged_key = key
                    elif controller.state != LockController.LOCKED or bridge is None:
                        state("VERIFICANDO", "Cerradura no disponible para una nueva apertura",
                              similarity=similarity, boxes=boxes)
                    else:
                        if prompted_id == person_id and pending_access is None:
                            state("DENEGADO", "Retira el rostro y vuelve a intentar", boxes=boxes)
                        else:
                            state("PIN REQUERIDO", "Posible coincidencia; introduce el PIN de acceso",
                                  store.users[person_id], similarity, boxes)
                        if prompted_id != person_id:
                            pending_access = (person_id, now + 15.0)
                            prompted_id = person_id
                            self.events.put(("pin_required", person_id, store.users[person_id]))
                else:
                    detail = "Rostro no registrado" if similarity is not None else "No hay perfiles SFace activos"
                    state("DENEGADO", detail, "Desconocido", similarity, boxes)
                    if logged_key != "unknown":
                        record_attempt("Desconocido", "DENEGADO", similarity, frame)
                        if controller.state == LockController.LOCKED and bridge is not None:
                            try:
                                bridge.send("DENIED")
                            except (OSError, TimeoutError, ConnectionError) as exc:
                                controller.fail(f"ESP32 sin respuesta: {exc}")
                                publish_lock()
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
