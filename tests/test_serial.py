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
        expected_len = 4 if command == "TEST_SERVO" else 2 if command == "PING" else 3
        if len(parts) != expected_len or (command != "PING" and parts[2] != self.boot_id):
            self.reply = f"ERR {seq} {command} {self.boot_id}\n".encode()
            return
        self.reply = (b"" if self.silent else
                      b"ACK\n" if self.malformed_ack else
                      f"ACK 999 {command} {self.boot_id}\n".encode() if self.wrong_ack else
                      f"ACK {seq} {command} {self.boot_id} NO_CONFIGURADO\n".encode()
                      if command == "SERVO_STATUS" else
                      f"ACK {seq} {command} {self.boot_id}\n".encode())

    def readline(self):
        response, self.reply = self.reply, b""
        return response

    def close(self):
        self.closed = True


@patch.object(core.CommandBridge, "BOOT_WAIT_SECONDS", 0)
class SerialTests(unittest.TestCase):
    def test_missing_pyserial_names_active_interpreter(self):
        with patch.dict(sys.modules, {"serial": None}):
            with self.assertRaises(RuntimeError) as failure:
                core.CommandBridge("leds", "COM4")
        self.assertIn(sys.executable, str(failure.exception))
        self.assertIn("pyserial", str(failure.exception))

    def test_initial_ping_waits_for_boot_and_recovers_from_two_timeouts(self):
        class BootSerial(FakeSerial):
            def write(self, data):
                super().write(data)
                if data.startswith(b"PING") and len(self.writes) < 3:
                    self.reply = b""

        device = BootSerial()
        with (patch.dict(sys.modules, {"serial": types.SimpleNamespace(Serial=lambda **_kw: device)}),
              patch.object(core.CommandBridge, "BOOT_WAIT_SECONDS", 2.0),
              patch.object(core.CommandBridge, "INITIAL_PING_TIMEOUT", 0.01),
              patch.object(core.time, "sleep") as sleep):
            bridge = core.CommandBridge("leds", "COM4")
            sleep.assert_called_once_with(2.0)
            self.assertEqual(device.writes, [b"PING 1\n", b"PING 2\n", b"PING 3\n"])
            self.assertEqual(bridge.boot_id, device.boot_id)
            self.assertFalse(device.closed)
            bridge.close()

            silent = FakeSerial()
            silent.silent = True
            with patch.dict(sys.modules, {"serial": types.SimpleNamespace(Serial=lambda **_kw: silent)}):
                with self.assertRaises(TimeoutError):
                    core.CommandBridge("leds", "COM4")
            self.assertEqual(silent.writes, [b"PING 1\n", b"PING 2\n", b"PING 3\n"])
            self.assertTrue(silent.closed)

    def test_ack_errors_timeout_disconnect_and_reconnect_without_motion(self):
        serials = []

        def open_serial(**_kwargs):
            device = FakeSerial()
            serials.append(device)
            return device

        with patch.dict(sys.modules, {"serial": types.SimpleNamespace(Serial=open_serial)}):
            bridge = core.CommandBridge("hardware", "COM3")
            self.assertEqual(serials[0].writes, [b"PING 1\n", b"SERVO_STATUS 2 A1B2C3D4\n"])
            bridge.send("PING")
            serials[0].wrong_ack = True
            with self.assertRaises(ConnectionError):
                bridge.send("OPEN")
            self.assertTrue(serials[0].closed)
            self.assertIsNone(bridge.serial)

            bridge = core.CommandBridge("hardware", "COM3")
            self.assertEqual(serials[1].writes, [b"PING 1\n", b"SERVO_STATUS 2 A1B2C3D4\n"])
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
            self.assertEqual(serials[3].writes, [b"PING 1\n", b"SERVO_STATUS 2 A1B2C3D4\n"])
            serials[3].boot_id = "FEED1234"  # Reinicio aun con el puerto abierto.
            with self.assertRaises(ConnectionError):
                bridge.send("OPEN")
            self.assertTrue(serials[3].closed)
            self.assertEqual(serials[3].writes[-1], b"OPEN 3 A1B2C3D4\n")

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
            self.assertEqual(failed.writes, [b"PING 1\n", b"PING 2\n", b"PING 3\n"])
            self.assertTrue(failed.closed)

    def test_led_only_protocol_ack_and_no_servo_commands(self):
        device = FakeSerial()
        with patch.dict(sys.modules, {"serial": types.SimpleNamespace(Serial=lambda **_kw: device)}):
            bridge = core.CommandBridge("leds", "COM3")
            self.assertEqual(device.writes, [b"PING 1\n"])
            with self.assertRaises(ValueError):
                bridge.send("OPEN")
            bridge.send("AUTH_OK")
            bridge.send("AUTH_FAIL")
            bridge.send("LED_OFF")
            self.assertEqual(device.writes[1:],
                             [b"AUTH_OK 2 A1B2C3D4\n", b"AUTH_FAIL 3 A1B2C3D4\n",
                              b"LED_OFF 4 A1B2C3D4\n"])
            device.wrong_ack = True
            with self.assertRaises(ConnectionError):
                bridge.send("AUTH_OK")
            self.assertTrue(device.closed)

    def test_combined_servo_protocol_requires_status_and_explicit_pulse(self):
        device = FakeSerial()
        with patch.dict(sys.modules, {"serial": types.SimpleNamespace(Serial=lambda **_kw: device)}):
            bridge = core.CommandBridge("hardware", "COM4")
            self.assertEqual(bridge.servo_status, "NO_CONFIGURADO")
            bridge.send("AUTH_OK")
            bridge.send("AUTH_FAIL")
            with self.assertRaises(ValueError):
                bridge.send("TEST_SERVO", pulse_us=900)
            with self.assertRaises(ValueError):
                bridge.send("TEST_SERVO", pulse_us=2001)
            bridge.send("TEST_SERVO", pulse_us=1500)
            bridge.send("SERVO_STOP")
            bridge.send("OPEN")
            bridge.send("LOCK")
            self.assertEqual(device.writes, [
                b"PING 1\n", b"SERVO_STATUS 2 A1B2C3D4\n",
                b"AUTH_OK 3 A1B2C3D4\n", b"AUTH_FAIL 4 A1B2C3D4\n",
                b"TEST_SERVO 5 A1B2C3D4 1500\n", b"SERVO_STOP 6 A1B2C3D4\n",
                b"OPEN 7 A1B2C3D4\n", b"LOCK 8 A1B2C3D4\n",
            ])
            bridge.close()

    def test_led_notifier_runs_outside_engine_and_reports_bad_ack(self):
        device = FakeSerial()
        events = queue.Queue()
        with patch.dict(sys.modules, {"serial": types.SimpleNamespace(Serial=lambda **_kw: device)}):
            notifier = core.LedNotifier("COM3", events)
            notifier.start()
            try:
                self.assertEqual(events.get(timeout=2)[0:2], ("led_device", True))
                self.assertEqual(device.writes[:2], [b"PING 1\n", b"LED_OFF 2 A1B2C3D4\n"])
                started = time.monotonic()
                notifier.notify("AUTH_FAIL")
                self.assertLess(time.monotonic() - started, 0.05)
                deadline = time.monotonic() + 2
                while len(device.writes) < 3 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(device.writes[2], b"AUTH_FAIL 3 A1B2C3D4\n")
                notifier.notify("AUTH_FAIL")
                time.sleep(0.15)
                self.assertEqual(len(device.writes), 3)
                notifier.notify("LED_OFF")
                deadline = time.monotonic() + 2
                while len(device.writes) < 4 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(device.writes[3], b"LED_OFF 4 A1B2C3D4\n")
                notifier.notify("AUTH_OK")
                deadline = time.monotonic() + 2
                while len(device.writes) < 5 and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(device.writes[4], b"AUTH_OK 5 A1B2C3D4\n")
                device.wrong_ack = True
                notifier.notify("LED_OFF")
                failure = events.get(timeout=2)
                self.assertEqual(failure[:2], ("led_device", False))
                self.assertIn("Respuesta serial incorrecta", failure[2])
                self.assertTrue(device.closed)
            finally:
                notifier.stop()
                notifier.thread.join(2)
                self.assertFalse(notifier.thread.is_alive())

    def test_heartbeat_loss_mode_switch_and_manual_lock(self):
        instances = []

        class Bridge:
            def __init__(self, mode, _port):
                self.mode = mode
                self.serial = object() if mode == "hardware" else None
                self.servo_status = "LISTO" if mode == "hardware" else None
                self.commands = ["PING", "SERVO_STATUS"] if mode == "hardware" else []
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
                  patch.object(core, "detect_faces", return_value=np.array([[0, 0, 16, 16] + [0] * 11], np.float32)),
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
                self.assertTrue(all(command in {"PING", "SERVO_STATUS"} for command in instances[1].commands))
                backend.send("connect_device", ("simulacion", ""))
                switched = until("device", lambda event: event[1] == "simulacion")
                self.assertTrue(switched[3])
                lock = until("lock")
                self.assertEqual(lock[1], "BLOQUEADA")
                self.assertIn("estado físico desconocido", lock[2])
                self.assertTrue(all(command in {"PING", "SERVO_STATUS"} for command in instances[1].commands))

                backend.send("connect_device", ("hardware", "COM5"))
                self.assertEqual(until("device")[2], "COM5")
                self.assertEqual(until("lock")[1], "ESPERANDO CIERRE")
                self.assertTrue(all(command in {"PING", "SERVO_STATUS"} for command in instances[3].commands))
                backend.send("confirm_closed")
                self.assertEqual(until("lock")[1], "BLOQUEADA")
                self.assertEqual([command for command in instances[3].commands
                                  if command not in {"PING", "SERVO_STATUS"}], ["LOCK"])
                request = until("pin_required", feed=True)
                self.assertEqual(request[1], person_id)
                instances[3].fail_ping = True
                until("device_error")
                backend.send("access_pin", (person_id, "456789", request[3]))
                until("pin_expired")
                self.assertNotIn("OPEN", instances[3].commands)
                backend.stop()
                backend.engine_thread.join(2)
                self.assertFalse(backend.engine_thread.is_alive())

    def test_physical_access_needs_face_pin_and_manual_close(self):
        instances = []

        class Bridge:
            def __init__(self, mode, _port):
                self.mode = mode
                self.serial = object()
                self.servo_status = "LISTO"
                self.commands = ["PING", "SERVO_STATUS"]
                instances.append(self)

            def send(self, command, timeout=1.5):
                self.commands.append(command)

            def close(self):
                self.serial = None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = np.eye(1, 128, 0, dtype=np.float32).reshape(-1)
            person_id = core.FaceStore(root).add_person("Ana", [feature] * core.ENROLL_SAMPLES)
            AccessPinStore(root).set_pin(person_id, "456789")
            backend = core.SafeBackend(root, "hardware", "COM4")
            backend.camera_connected = True
            observed_feature = [np.eye(1, 128, 1, dtype=np.float32).reshape(-1)]

            def until(kind, feed=False, predicate=lambda _event: True):
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

            try:
                with (patch.object(core, "load_models", return_value=(object(), object())),
                      patch.object(core, "detect_faces", return_value=np.array([[0, 0, 16, 16] + [0] * 11], np.float32)),
                      patch.object(core, "extract_feature", side_effect=lambda *_args: observed_feature[0]),
                      patch.object(core, "CommandBridge", Bridge)):
                    backend.engine_thread.start()
                    backend.send("admin_ready", True)
                    until("device")
                    self.assertEqual(until("lock")[1], "ESPERANDO CIERRE")
                    self.assertNotIn("OPEN", instances[0].commands)
                    backend.send("confirm_closed")
                    self.assertEqual(until("lock", predicate=lambda e: e[1] == "BLOQUEADA")[1], "BLOQUEADA")
                    self.assertEqual(instances[0].commands.count("LOCK"), 1)
                    self.assertEqual(until("state", feed=True,
                                           predicate=lambda e: e[1] == "DENEGADO")[1], "DENEGADO")
                    self.assertNotIn("OPEN", instances[0].commands)
                    observed_feature[0] = feature
                    request = until("pin_required", feed=True)
                    backend.send("access_pin", (person_id, "000000", request[3]))
                    until("pin_failed", feed=True)
                    self.assertNotIn("OPEN", instances[0].commands)
                    self.assertIn("AUTH_FAIL", instances[0].commands)
                    backend.send("access_pin", (person_id, "456789", request[3]))
                    until("pin_success", feed=True)
                    self.assertEqual(instances[0].commands.count("OPEN"), 1)
                    self.assertIn("AUTH_OK", instances[0].commands)
                    backend.send("confirm_closed")
                    self.assertEqual(until("lock", predicate=lambda e: e[1] == "BLOQUEADA")[1], "BLOQUEADA")
                    self.assertEqual(instances[0].commands.count("LOCK"), 2)
            finally:
                backend.stop()
                backend.engine_thread.join(2)
                self.assertFalse(backend.engine_thread.is_alive())

    def test_uncalibrated_servo_never_opens_and_accepts_manual_test(self):
        instances = []

        class Bridge:
            def __init__(self, mode, _port):
                self.serial = object()
                self.servo_status = "NO_CONFIGURADO"
                self.commands = []
                instances.append(self)

            def send(self, command, timeout=1.5, *, pulse_us=None):
                self.commands.append((command, pulse_us))

            def close(self):
                self.serial = None

        with tempfile.TemporaryDirectory() as directory:
            backend = core.SafeBackend(Path(directory), "hardware", "COM4")
            try:
                with (patch.object(core, "load_models", return_value=(object(), object())),
                      patch.object(core, "CommandBridge", Bridge)):
                    backend.engine_thread.start()
                    deadline = time.monotonic() + 2
                    seen = []
                    while time.monotonic() < deadline and not any(e[0] == "lock" for e in seen):
                        try:
                            seen.append(backend.events.get(timeout=0.05))
                        except queue.Empty:
                            pass
                    self.assertIn(("servo_status", "NO_CONFIGURADO"), seen)
                    self.assertTrue(any(e[0] == "lock" and e[1] == "ERROR" for e in seen))
                    backend.send("test_servo", 1500)
                    while time.monotonic() < deadline:
                        try:
                            event = backend.events.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if event[0] == "servo_test_result":
                            self.assertTrue(event[1], event)
                            break
                    else:
                        self.fail("No llegó resultado de prueba")
                    self.assertEqual(instances[0].commands, [("TEST_SERVO", 1500)])
            finally:
                backend.stop()
                backend.engine_thread.join(2)


if __name__ == "__main__":
    unittest.main()
