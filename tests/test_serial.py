"""Protocolo serial y transiciones seguras sin ESP32 física."""

import queue
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import core
import numpy as np
from control import AccessPinStore


class FakeSerial:
    def __init__(self):
        self.writes = []
        self.closed = False
        self.reply = b""
        self.wrong_ack = False
        self.malformed_ack = False
        self.silent = False
        self.disconnect = False
        self.boot_id = "A1B2C3D4"

    def write(self, data):
        if self.disconnect:
            raise OSError("Puerto desconectado")
        self.writes.append(data)
        parts = data.decode().strip().split()
        command, seq = parts[:2]
        if command != "PING" and (len(parts) != 3 or parts[2] != self.boot_id):
            self.reply = f"ERR {seq} {command} {self.boot_id}\n".encode()
            return
        self.reply = (b"" if self.silent else
                      b"ACK\n" if self.malformed_ack else
                      f"ACK 999 {command} {self.boot_id}\n".encode() if self.wrong_ack else
                      f"ACK {seq} {command} {self.boot_id}\n".encode())

    def readline(self):
        response, self.reply = self.reply, b""
        return response

    def close(self):
        self.closed = True


class SerialTests(unittest.TestCase):
    def test_ack_errors_timeout_disconnect_and_reconnect_without_motion(self):
        serials = []

        def open_serial(**_kwargs):
            device = FakeSerial()
            serials.append(device)
            return device

        with patch.dict(sys.modules, {"serial": types.SimpleNamespace(Serial=open_serial)}):
            bridge = core.CommandBridge("hardware", "COM3")
            self.assertEqual(serials[0].writes, [b"PING 1\n"])
            bridge.send("PING")
            serials[0].wrong_ack = True
            with self.assertRaises(ConnectionError):
                bridge.send("OPEN")
            self.assertTrue(serials[0].closed)
            self.assertIsNone(bridge.serial)

            bridge = core.CommandBridge("hardware", "COM3")
            self.assertEqual(serials[1].writes, [b"PING 1\n"])
            serials[1].silent = True
            with self.assertRaises(TimeoutError):
                bridge.send("PING", timeout=0.02)
            self.assertTrue(serials[1].closed)

            bridge = core.CommandBridge("hardware", "COM3")
            serials[2].disconnect = True
            with self.assertRaises(OSError):
                bridge.send("PING")
            self.assertTrue(serials[2].closed)

            bridge = core.CommandBridge("hardware", "COM3")
            self.assertEqual(serials[3].writes, [b"PING 1\n"])
            serials[3].boot_id = "FEED1234"  # Reinicio aun con el puerto abierto.
            with self.assertRaises(ConnectionError):
                bridge.send("OPEN")
            self.assertTrue(serials[3].closed)
            self.assertEqual(serials[3].writes[-1], b"OPEN 2 A1B2C3D4\n")

            bridge = core.CommandBridge("hardware", "COM3")
            serials[4].malformed_ack = True
            with self.assertRaises(ConnectionError):
                bridge.send("PING")
            self.assertTrue(serials[4].closed)

            failed = FakeSerial()
            failed.wrong_ack = True
            with patch.dict(sys.modules, {"serial": types.SimpleNamespace(Serial=lambda **_kw: failed)}):
                with self.assertRaises(ConnectionError):
                    core.CommandBridge("hardware", "COM6")
            self.assertEqual(failed.writes, [b"PING 1\n"])
            self.assertTrue(failed.closed)

    def test_heartbeat_loss_mode_switch_and_manual_lock(self):
        instances = []

        class Bridge:
            def __init__(self, mode, _port):
                self.mode = mode
                self.serial = object() if mode == "hardware" else None
                self.commands = ["PING"] if mode == "hardware" else []
                self.fail_ping = False
                instances.append(self)

            def send(self, command, timeout=1.5):
                if command == "PING" and self.fail_ping:
                    self.serial = None
                    raise TimeoutError("Sin ACK")
                self.commands.append(command)

            def close(self):
                self.serial = None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = np.eye(1, 128, 0, dtype=np.float32).reshape(-1)
            person_id = core.FaceStore(root).add_person("Ana", [feature] * core.ENROLL_SAMPLES)
            AccessPinStore(root).set_pin(person_id, "456789")
            backend = core.SafeBackend(root, "hardware", "COM3")
            backend.HEARTBEAT_INTERVAL = 0.12
            backend.camera_connected = True

            def until(kind, predicate=lambda _event: True, feed=False):
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if feed:
                        backend.frame = np.zeros((32, 32, 3), np.uint8)
                        backend.frame_seq += 1
                    try:
                        event = backend.events.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if event[0] == "engine_error":
                        self.fail(event[1])
                    if event[0] == kind and predicate(event):
                        return event
                self.fail(f"No llegó {kind}")

            with (patch.object(core, "load_models", return_value=(object(), object())),
                  patch.object(core, "detect_faces", return_value=np.zeros((1, 15), np.float32)),
                  patch.object(core, "extract_feature", return_value=feature),
                  patch.object(core, "CommandBridge", Bridge)):
                backend.engine_thread.start()
                backend.send("admin_ready", True)
                self.assertEqual(until("device")[1], "hardware")
                self.assertEqual(until("lock")[1], "ESPERANDO CIERRE")
                instances[0].fail_ping = True
                self.assertIn("Estado físico desconocido", until("device_error")[1])
                self.assertNotIn("OPEN", instances[0].commands)

                backend.send("connect_device", ("hardware", "COM4"))
                self.assertEqual(until("device")[2], "COM4")
                self.assertEqual(until("lock")[1], "ESPERANDO CIERRE")
                self.assertTrue(all(command == "PING" for command in instances[1].commands))
                backend.send("connect_device", ("simulacion", ""))
                switched = until("device", lambda event: event[1] == "simulacion")
                self.assertTrue(switched[3])
                lock = until("lock")
                self.assertEqual(lock[1], "BLOQUEADA")
                self.assertIn("estado físico desconocido", lock[2])
                self.assertTrue(all(command == "PING" for command in instances[1].commands))

                backend.send("connect_device", ("hardware", "COM5"))
                self.assertEqual(until("device")[2], "COM5")
                self.assertEqual(until("lock")[1], "ESPERANDO CIERRE")
                self.assertTrue(all(command == "PING" for command in instances[3].commands))
                backend.send("confirm_closed")
                self.assertEqual(until("lock")[1], "BLOQUEADA")
                self.assertEqual([command for command in instances[3].commands if command != "PING"], ["LOCK"])
                self.assertEqual(until("pin_required", feed=True)[1], person_id)
                instances[3].fail_ping = True
                until("device_error")
                backend.send("access_pin", (person_id, "456789"))
                until("pin_expired")
                self.assertNotIn("OPEN", instances[3].commands)
                backend.stop()
                backend.engine_thread.join(2)
                self.assertFalse(backend.engine_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
