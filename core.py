"""Persistencia, reconocimiento y salida de comandos de SAFEBOX AI."""

from __future__ import annotations

import json
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


FACE_SIZE = (160, 160)
MAX_DISTANCE = 55.0


def detector() -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("No se pudo cargar el detector facial de OpenCV")
    return cascade


def extract_face(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    face = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(face)


class FaceStore:
    def __init__(self, root: Path):
        self.root = root
        self.faces_dir = root / "rostros"
        self.index_path = root / "usuarios.json"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.users = self._read_users()
        self.recognizer = None
        self.train()

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

    def train(self) -> None:
        images: list[np.ndarray] = []
        labels: list[int] = []
        for person_id in self.users:
            for image_path in sorted((self.faces_dir / person_id).glob("*.png")):
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is not None and image.shape == FACE_SIZE:
                    images.append(image)
                    labels.append(int(person_id))
        self.recognizer = None
        if images:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, np.asarray(labels, dtype=np.int32))
            self.recognizer = recognizer

    def add_person(self, name: str, samples: list[np.ndarray]) -> None:
        if len(samples) < 20:
            raise ValueError("Faltan muestras para registrar el rostro")
        if any(existing.casefold() == name.casefold() for existing in self.users.values()):
            raise ValueError("Ese nombre ya está registrado")
        person_id = str(uuid.uuid4().int % 2_000_000_000)
        while person_id in self.users:
            person_id = str(uuid.uuid4().int % 2_000_000_000)
        folder = self.faces_dir / person_id
        folder.mkdir()
        try:
            for number, image in enumerate(samples, 1):
                if image.shape != FACE_SIZE or not cv2.imwrite(
                    str(folder / f"{number:02d}.png"), image
                ):
                    raise OSError("No se pudo guardar una muestra facial")
            updated = {**self.users, person_id: name}
            temp = self.index_path.with_suffix(".tmp")
            temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.index_path)
            self.users = updated
            self.train()
        except Exception:
            if person_id not in self.users:
                shutil.rmtree(folder, ignore_errors=True)
            raise

    def identify(self, face: np.ndarray) -> tuple[str | None, float | None]:
        if self.recognizer is None:
            return None, None
        label, distance = self.recognizer.predict(face)
        name = self.users.get(str(label)) if distance <= MAX_DISTANCE else None
        return name, float(distance)

    def list_people(self) -> list[tuple[str, str]]:
        return sorted(self.users.items(), key=lambda item: item[1].casefold())

    def delete_person(self, person_id: str) -> str:
        if person_id not in self.users:
            raise ValueError("El usuario ya no existe")
        name = self.users[person_id]
        updated = {key: value for key, value in self.users.items() if key != person_id}
        temp = self.index_path.with_suffix(".tmp")
        temp.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.index_path)
        self.users = updated
        self.train()
        shutil.rmtree(self.faces_dir / person_id)
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
            "distancia REAL)"
        )
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(accesos)")}
        if "distancia" not in columns:
            self.db.execute("ALTER TABLE accesos ADD COLUMN distancia REAL")
        self.db.commit()

    def record(self, user: str, result: str, distance: float | None, frame: np.ndarray) -> None:
        when = datetime.now().astimezone().isoformat(timespec="seconds")
        filename = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
        path = self.capture_dir / filename
        if not cv2.imwrite(str(path), frame):
            raise OSError("No se pudo guardar la captura del intento")
        try:
            with self.db:
                self.db.execute(
                    "INSERT INTO accesos (fecha, usuario, resultado, confianza, captura, distancia) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (when, user, result, None, filename, distance),
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def recent(self, limit: int = 8) -> list[tuple[str, str, str, float | None]]:
        return self.db.execute(
            "SELECT fecha, usuario, resultado, distancia FROM accesos "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def search(self, user: str = "", result: str = "Todos", limit: int = 31,
               offset: int = 0) -> list[tuple]:
        pattern = "%" + user.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        return self.db.execute(
            "SELECT fecha, usuario, resultado, distancia, captura FROM accesos "
            "WHERE usuario LIKE ? ESCAPE '\\' AND (? = 'Todos' OR resultado = ?) "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (pattern, result, result, limit, offset),
        ).fetchall()

    def close(self) -> None:
        self.db.close()


class CommandBridge:
    """Simulación por defecto; serial opcional para la WeMos."""

    def __init__(self, mode: str, port: str | None = None, baudrate: int = 115200):
        self.mode = mode
        self.serial = None
        self.last_command = None
        if mode == "hardware":
            if not port:
                raise ValueError("El modo hardware requiere --port COMx")
            try:
                import serial  # pyserial solo es necesario al conectar hardware
            except ImportError as exc:
                raise RuntimeError(
                    "Instala pyserial para usar hardware: python -m pip install pyserial"
                ) from exc
            self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        self.send("LOCK")

    def send(self, command: str) -> None:
        if command not in {"OPEN", "LOCK", "DENIED"}:
            raise ValueError("Comando no permitido")
        if command == self.last_command:
            return
        if self.serial is not None:
            self.serial.write((command + "\n").encode("ascii"))
        self.last_command = command

    def close(self) -> None:
        try:
            self.send("LOCK")
        finally:
            if self.serial is not None:
                self.serial.close()


class SafeBackend:
    """La cámara y el motor son dueños de sus recursos; Tk nunca sale del hilo principal."""

    def __init__(self, data_dir: Path, mode: str, port: str | None):
        self.data_dir = data_dir
        self.mode = mode
        self.port = port
        self.events: queue.Queue[tuple] = queue.Queue()
        self.commands: queue.Queue[tuple] = queue.Queue()
        self.camera_commands: queue.Queue[str] = queue.Queue()
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

    def send(self, kind: str, value: str | None = None) -> None:
        self.commands.put((kind, value))

    def retry_camera(self) -> None:
        self.camera_commands.put("retry")

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
                            self.camera_commands.get(timeout=0.2)
                        except queue.Empty:
                            continue
                    open_requested = False
                    if self.stop_event.is_set():
                        break
                    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                    candidate = cv2.VideoCapture(0, backend)
                    if not candidate.isOpened():
                        candidate.release()
                        self.camera_connected = False
                        self.events.put(("camera", False, "Cámara no disponible"))
                        continue
                    capture = candidate
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.camera_connected = True
                    self.events.put(("camera", True, ""))
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

        def state(name: str, detail: str, user: str = "—", distance: float | None = None,
                  boxes: list[tuple[int, int, int, int]] | None = None) -> None:
            nonlocal last_state
            payload = (name, detail, user, distance, tuple(boxes or ()))
            if payload != last_state:
                self.events.put(("state", name, detail, user, distance, list(boxes or ())))
                last_state = payload
            command = "OPEN" if name == "AUTORIZADO" else "DENIED" if name == "DENEGADO" else "LOCK"
            bridge.send(command)

        def record_attempt(user: str, result: str, distance: float | None,
                           frame: np.ndarray) -> None:
            try:
                history.record(user, result, distance, frame)
                self.events.put(("history", history.recent()))
            except (OSError, sqlite3.Error) as exc:
                self.events.put(("storage_error", str(exc)))

        try:
            store = FaceStore(self.data_dir)
            history = AccessHistory(self.data_dir)
            face_detector = detector()
            bridge = CommandBridge(self.mode, self.port)
            self.events.put(("ready", list(store.users.values()), history.recent(), store.list_people()))
            while not self.stop_event.is_set():
                while True:
                    try:
                        kind, value = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "start_enroll":
                        name = (value or "").strip()
                        if not self.camera_connected:
                            self.events.put(("enroll_error", "La cámara no está disponible"))
                        elif any(existing.casefold() == name.casefold() for existing in store.users.values()):
                            self.events.put(("enroll_error", "Ese nombre ya está registrado"))
                        else:
                            enrolling_name = name
                            samples = []
                            last_sample = 0.0
                            candidate_name, streak = None, 0
                            self.events.put(("enroll_started", name))
                    elif kind == "cancel_enroll":
                        enrolling_name = None
                        samples = []
                        candidate_name, streak = None, 0
                        state("ESCANEANDO", "Esperando un rostro frente a la cámara")
                    elif kind == "list_users":
                        self.events.put(("users", store.list_people()))
                    elif kind == "delete_user":
                        try:
                            deleted = store.delete_person(value or "")
                            self.events.put(("user_deleted", deleted, store.list_people()))
                            candidate_name, streak, logged_key = None, 0, None
                            state("ESCANEANDO", "Modelo facial actualizado")
                        except (OSError, ValueError) as exc:
                            self.events.put(("user_error", str(exc)))
                    elif kind == "search_history":
                        try:
                            user, result, offset = value
                            self.events.put(("history_search", user, result, offset,
                                             history.search(user, result, offset=offset)))
                        except sqlite3.Error as exc:
                            self.events.put(("storage_error", str(exc)))
                    elif kind == "stop":
                        break
                if self.stop_event.is_set():
                    break
                if not self.camera_connected:
                    candidate_name, streak = None, 0
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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scale = min(1.0, 480 / gray.shape[1])
                small = cv2.resize(gray, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA) if scale < 1 else gray
                found = face_detector.detectMultiScale(
                    small, scaleFactor=1.1, minNeighbors=5,
                    minSize=(max(60, int(80 * scale)), max(60, int(80 * scale))),
                )
                boxes = [tuple(round(int(v) / scale) for v in box) for box in found]
                count = len(boxes)
                if count == 0:
                    candidate_name, streak = None, 0
                    if now - last_face_seen > 1.0:
                        logged_key = None
                    if enrolling_name:
                        state("REGISTRANDO", "Muestra un rostro para continuar", enrolling_name)
                    else:
                        state("ESCANEANDO", "Esperando un rostro frente a la cámara")
                    continue
                last_face_seen = now
                if count > 1:
                    candidate_name, streak = None, 0
                    if enrolling_name:
                        state("REGISTRANDO", "Debe aparecer una sola persona", enrolling_name, boxes=boxes)
                    else:
                        state("DENEGADO", "Hay varios rostros en cámara", "Varios rostros", boxes=boxes)
                        if logged_key != "multiple":
                            record_attempt("Varios rostros", "DENEGADO", None, frame)
                            logged_key = "multiple"
                    continue
                face = extract_face(frame, boxes[0])
                if enrolling_name:
                    if now - last_sample >= 0.25:
                        samples.append(face)
                        last_sample = now
                        self.events.put(("enroll_progress", len(samples)))
                    state("REGISTRANDO", f"Capturando · {len(samples)}/20", enrolling_name, boxes=boxes)
                    if len(samples) >= 20:
                        name = enrolling_name
                        enrolling_name = None
                        state("VERIFICANDO", "Guardando y entrenando el modelo…")
                        try:
                            store.add_person(name, samples)
                        except Exception as exc:
                            self.events.put(("enroll_error", str(exc)))
                        else:
                            self.events.put(("enroll_done", name, list(store.users.values())))
                        samples = []
                        candidate_name, streak = None, 0
                    continue
                name, distance = store.identify(face)
                identity = name or "Desconocido"
                streak = streak + 1 if identity == candidate_name else 1
                candidate_name = identity
                if streak < 3:
                    state("VERIFICANDO", f"Lectura {streak}/3", distance=distance, boxes=boxes)
                elif name is not None:
                    state("AUTORIZADO", "Coincidencia confirmada", name, distance, boxes)
                    key = f"allow:{name}"
                    if logged_key != key:
                        record_attempt(name, "AUTORIZADO", distance, frame)
                        logged_key = key
                else:
                    detail = "Rostro no registrado" if distance is not None else "No hay usuarios registrados"
                    state("DENEGADO", detail, "Desconocido", distance, boxes)
                    if logged_key != "unknown":
                        record_attempt("Desconocido", "DENEGADO", distance, frame)
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
