"""PIN local, preferencias y máquina de estados de la cerradura."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path


PIN_ITERATIONS = 600_000
MAX_PIN_FAILURES = 5
PIN_LOCK_SECONDS = 60


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class AdminManager:
    def __init__(self, root: Path):
        self.path = root / "admin.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None

    @property
    def configured(self) -> bool:
        return self.data is not None

    @staticmethod
    def _validate(pin: str) -> None:
        if not (6 <= len(pin) <= 12 and pin.isascii() and pin.isdecimal()):
            raise ValueError("El PIN debe tener de 6 a 12 dígitos")

    @staticmethod
    def _digest(pin: str, salt: bytes) -> str:
        return hashlib.pbkdf2_hmac("sha256", pin.encode("ascii"), salt, PIN_ITERATIONS).hex()

    def create(self, pin: str) -> None:
        if self.configured:
            raise ValueError("El PIN de administrador ya existe")
        self._validate(pin)
        salt = os.urandom(32)
        data = {"salt": salt.hex(), "hash": self._digest(pin, salt),
                "iterations": PIN_ITERATIONS, "failures": 0, "locked_until": 0.0}
        _atomic_json(self.path, data)
        self.data = data

    def verify(self, pin: str) -> None:
        if not self.configured:
            raise ValueError("Primero crea el PIN de administrador")
        now = time.time()
        if now < self.data["locked_until"]:
            raise PermissionError(f"PIN bloqueado temporalmente; espera {int(self.data['locked_until'] - now) + 1} s")
        salt = bytes.fromhex(self.data["salt"])
        valid = pin.isascii() and hmac.compare_digest(
            self._digest(pin, salt), self.data["hash"])
        if not valid:
            self.data["failures"] += 1
            if self.data["failures"] >= MAX_PIN_FAILURES:
                self.data["failures"] = 0
                self.data["locked_until"] = now + PIN_LOCK_SECONDS
            _atomic_json(self.path, self.data)
            raise PermissionError("PIN incorrecto o bloqueado temporalmente")
        self.data["failures"] = 0
        self.data["locked_until"] = 0.0
        _atomic_json(self.path, self.data)

    def change(self, old_pin: str, new_pin: str) -> None:
        self.verify(old_pin)
        self._validate(new_pin)
        salt = os.urandom(32)
        data = {**self.data, "salt": salt.hex(), "hash": self._digest(new_pin, salt),
                "iterations": PIN_ITERATIONS, "failures": 0, "locked_until": 0.0}
        _atomic_json(self.path, data)
        self.data = data


class SettingsStore:
    DEFAULTS = {"similarity_threshold": 0.363, "camera_index": 0,
                "authorization_seconds": 8, "mode": "simulacion", "port": ""}

    def __init__(self, root: Path):
        self.path = root / "config.json"
        saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.values = {**self.DEFAULTS, **saved}
        self.validate(self.values)

    @staticmethod
    def validate(values: dict) -> None:
        threshold = values["similarity_threshold"]
        duration = values["authorization_seconds"]
        camera = values["camera_index"]
        mode = values["mode"]
        port = values["port"]
        if not isinstance(threshold, (int, float)) or not 0.1 <= threshold <= 0.9:
            raise ValueError("Umbral: usa un valor entre 0.1 y 0.9")
        if not isinstance(duration, int) or not 3 <= duration <= 60:
            raise ValueError("Tiempo de autorización: 3 a 60 segundos")
        if not isinstance(camera, int) or not 0 <= camera <= 9:
            raise ValueError("Cámara: índice de 0 a 9")
        if mode not in {"simulacion", "hardware"}:
            raise ValueError("Modo inválido")
        if not isinstance(port, str) or len(port) > 32:
            raise ValueError("Puerto COM inválido")
        if mode == "hardware" and not port.upper().startswith("COM"):
            raise ValueError("El modo hardware requiere un puerto COM")

    def save(self, values: dict) -> None:
        merged = {**self.values, **values}
        self.validate(merged)
        _atomic_json(self.path, merged)
        self.values = merged


class LockController:
    LOCKED = "BLOQUEADA"
    PENDING = "AUTORIZACIÓN PENDIENTE"
    UNLOCKED = "DESBLOQUEADA"
    WAITING_CLOSE = "ESPERANDO CIERRE"
    ERROR = "ERROR"

    def __init__(self, authorization_seconds: int = 8, ready: bool = False):
        self.authorization_seconds = authorization_seconds
        self.state = self.LOCKED if ready else self.ERROR
        self.deadline = 0.0
        self.reason = "" if ready else "Conexión no verificada"

    def authorize(self, bridge, on_pending=None) -> bool:
        if self.state != self.LOCKED:
            return False
        self.state = self.PENDING
        self.reason = "Esperando ACK de apertura"
        if on_pending is not None:
            on_pending()
        try:
            bridge.send("OPEN")
        except (OSError, TimeoutError, ConnectionError) as exc:
            self.fail(f"Apertura no confirmada: {exc}")
            return False
        self.state = self.UNLOCKED
        self.reason = "Apertura ordenada; sin sensor físico"
        self.deadline = time.monotonic() + self.authorization_seconds
        return True

    def tick(self) -> bool:
        if self.state == self.UNLOCKED and time.monotonic() >= self.deadline:
            self.state = self.WAITING_CLOSE
            return True
        return False

    def confirm_closed(self, bridge) -> bool:
        if self.state not in {self.UNLOCKED, self.WAITING_CLOSE}:
            return False
        try:
            bridge.send("LOCK")
        except (OSError, TimeoutError, ConnectionError) as exc:
            self.fail(f"Bloqueo no confirmado: {exc}")
            return False
        self.state = self.LOCKED
        self.reason = "Cierre confirmado por el operador; sin sensor físico"
        self.deadline = 0.0
        return True

    def fail(self, reason: str) -> None:
        self.state = self.ERROR
        self.reason = reason
        self.deadline = 0.0
