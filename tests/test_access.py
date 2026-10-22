"""Regresiones de acceso: una coincidencia facial nunca abre por sí sola."""

import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import core
from control import AccessPinStore, SettingsStore

FACE = np.array([[0, 0, 16, 16] + [0] * 11], dtype=np.float32)


class AccessTests(unittest.TestCase):
    def test_access_pin_hash_lockout_and_existing_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = AccessPinStore(root)
            pins.set_pin("123", "456789")
            pins.set_pin("124", "987654")
            self.assertNotIn("456789", (root / "access_pins.json").read_text())
            with self.assertRaises(PermissionError):
                pins.verify("123", "987654")
            pins.verify("124", "987654")
            for _ in range(5):
                with self.assertRaises(PermissionError):
                    pins.verify("123", "000000")
            with self.assertRaises(PermissionError):
                pins.verify("123", "456789")
            pins.data["123"]["locked_until"] = 0
            pins.verify("123", "456789")
            pins.remove("123")
            self.assertFalse(AccessPinStore(root).has_pin("123"))

    def test_threshold_upgrade_and_ambiguous_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"similarity_threshold": 0.363}')
            self.assertEqual(SettingsStore(root).values["similarity_threshold"], 0.5)
            store = core.FaceStore(root)
            a = np.eye(1, 128, 0, dtype=np.float32).reshape(-1)
            b = np.eye(1, 128, 1, dtype=np.float32).reshape(-1)
            store.add_person("Ana", [a] * core.ENROLL_SAMPLES)
            store.add_person("Beto", [b] * core.ENROLL_SAMPLES)
            probe = (a + b) / np.sqrt(2)
            self.assertIsNone(store.identify_id(probe, 0.5)[0])
            self.assertEqual(store.identify_id(a, 0.5)[0],
                             next(uid for uid, name in store.users.items() if name == "Ana"))

    def test_face_needs_pin_before_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = core.FaceStore(root)
            feature = np.eye(1, 128, 0, dtype=np.float32).reshape(-1)
            person_id = store.add_person("Ana", [feature] * core.ENROLL_SAMPLES)
            pins = AccessPinStore(root)
            pins.set_pin(person_id, "456789")
            sent = []

            class Bridge:
                def __init__(self, *_args):
                    pass

                def send(self, command):
                    sent.append(command)

                def close(self):
                    pass

            backend = core.SafeBackend(root, "simulacion", None, 0.5)
            backend.led_notifier = Mock()
            led_at = []
            backend.led_notifier.notify.side_effect = lambda command: led_at.append((command, tuple(sent)))
            backend.camera_connected = True
            with (patch.object(core, "load_models", return_value=(object(), object())),
                  patch.object(core, "detect_faces", return_value=FACE),
                  patch.object(core, "extract_feature", return_value=feature),
                  patch.object(core, "CommandBridge", Bridge)):
                backend.engine_thread.start()
                backend.send("admin_ready", True)
                def until(kind, timeout=4.0):
                    deadline = time.monotonic() + timeout
                    while time.monotonic() < deadline:
                        backend.frame = np.zeros((32, 32, 3), np.uint8)
                        backend.frame_seq += 1
                        try:
                            event = backend.events.get(timeout=0.08)
                        except queue.Empty:
                            continue
                        if event[0] == "engine_error":
                            self.fail(event[1])
                        if event[0] == kind:
                            return event
                    self.fail(f"No llegó {kind}")

                request = until("pin_required")
                self.assertEqual(request[1], person_id)
                self.assertNotIn("OPEN", sent)
                backend.send("access_pin", (person_id, "000000", request[3]))
                until("pin_failed")
                self.assertNotIn("OPEN", sent)
                self.assertNotIn("AUTH_OK", [call.args[0] for call in backend.led_notifier.notify.call_args_list])
                self.assertIn("AUTH_FAIL", [call.args[0] for call in backend.led_notifier.notify.call_args_list])
                backend.send("access_pin", (person_id, "456789", request[3]))
                until("pin_success")
                self.assertEqual(sent.count("OPEN"), 1)
                self.assertEqual([call.args[0] for call in backend.led_notifier.notify.call_args_list].count("AUTH_OK"), 1)
                self.assertTrue(any(command == "AUTH_OK" and commands.count("OPEN") == 1
                                    for command, commands in led_at))
                backend.send("confirm_closed")
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and "LOCK" not in sent:
                    time.sleep(0.02)
                self.assertEqual(sent.count("LOCK"), 1)
                self.assertEqual(sent.count("OPEN"), 1)
                backend.stop()
                backend.engine_thread.join(2)

    def test_unknown_face_never_opens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            known = np.eye(1, 128, 0, dtype=np.float32).reshape(-1)
            unknown = np.eye(1, 128, 1, dtype=np.float32).reshape(-1)
            uid = core.FaceStore(root).add_person("Ana", [known] * core.ENROLL_SAMPLES)
            AccessPinStore(root).set_pin(uid, "456789")
            sent = []

            class Bridge:
                def __init__(self, *_args):
                    pass

                def send(self, command):
                    sent.append(command)

                def close(self):
                    pass

            backend = core.SafeBackend(root, "simulacion", None, 0.5)
            backend.led_notifier = Mock()
            backend.camera_connected = True
            with (patch.object(core, "load_models", return_value=(object(), object())),
                  patch.object(core, "detect_faces", return_value=FACE),
                  patch.object(core, "extract_feature", return_value=unknown),
                  patch.object(core, "CommandBridge", Bridge)):
                backend.engine_thread.start()
                backend.send("admin_ready", True)
                deadline = time.monotonic() + 3
                denied = False
                while time.monotonic() < deadline and not denied:
                    backend.frame = np.zeros((32, 32, 3), np.uint8)
                    backend.frame_seq += 1
                    try:
                        event = backend.events.get(timeout=0.08)
                    except queue.Empty:
                        continue
                    if event[0] == "engine_error":
                        self.fail(event[1])
                    denied = event[0] == "state" and event[1] == "DENEGADO"
                self.assertTrue(denied)
                self.assertNotIn("OPEN", sent)
                backend.camera_connected = False
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        event = backend.events.get(timeout=0.08)
                    except queue.Empty:
                        continue
                    if event[0] == "state" and event[1] == "CÁMARA DESCONECTADA":
                        break
                else:
                    self.fail("No se apagaron los LEDs al salir de DENEGADO")
                commands = [call.args[0] for call in backend.led_notifier.notify.call_args_list]
                self.assertEqual(commands, ["AUTH_FAIL", "LED_OFF"])
                backend.stop()
                backend.engine_thread.join(2)
                self.assertFalse(backend.engine_thread.is_alive())

    def test_existing_user_without_pin_cannot_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = np.eye(1, 128, 0, dtype=np.float32).reshape(-1)
            core.FaceStore(root).add_person("Usuario anterior", [feature] * core.ENROLL_SAMPLES)
            sent = []

            class Bridge:
                def __init__(self, *_args):
                    pass

                def send(self, command):
                    sent.append(command)

                def close(self):
                    pass

            backend = core.SafeBackend(root, "simulacion", None, 0.5)
            backend.camera_connected = True
            with (patch.object(core, "load_models", return_value=(object(), object())),
                  patch.object(core, "detect_faces", return_value=FACE),
                  patch.object(core, "extract_feature", return_value=feature),
                  patch.object(core, "CommandBridge", Bridge)):
                backend.engine_thread.start()
                backend.send("admin_ready", True)
                denied = False
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not denied:
                    backend.frame = np.zeros((32, 32, 3), np.uint8)
                    backend.frame_seq += 1
                    try:
                        event = backend.events.get(timeout=0.08)
                    except queue.Empty:
                        continue
                    if event[0] == "engine_error":
                        self.fail(event[1])
                    denied = event[0] == "state" and event[1] == "DENEGADO"
                self.assertTrue(denied)
                self.assertNotIn("OPEN", sent)
                backend.stop()
                backend.engine_thread.join(2)


if __name__ == "__main__":
    unittest.main()
