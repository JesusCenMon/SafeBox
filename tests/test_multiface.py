"""Asociación individual de rostros, PIN y apertura con varios sujetos."""

import queue
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import core
import app
from control import AccessPinStore


FEATURES = np.eye(3, 128, dtype=np.float32)


def face(x, marker):
    row = np.zeros(15, np.float32)
    row[:4] = (x, 2, 20, 20)
    row[14] = marker
    return row


class MultiFaceTests(unittest.TestCase):
    def test_audio_failure_does_not_interrupt_access(self):
        broken_audio = SimpleNamespace(Beep=Mock(side_effect=OSError("Sin audio")))
        with patch.dict(sys.modules, {"winsound": broken_audio}):
            app.play_welcome_sound()
        broken_audio.Beep.assert_called_once()

    @contextmanager
    def running(self, two_users=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = core.FaceStore(root)
            ana = store.add_person("Ana", [FEATURES[0]] * core.ENROLL_SAMPLES)
            pins = AccessPinStore(root)
            pins.set_pin(ana, "456789")
            if two_users:
                beto = store.add_person("Beto", [FEATURES[1]] * core.ENROLL_SAMPLES)
                pins.set_pin(beto, "987654")
            current = [face(2, 0), face(32, 2 if not two_users else 1)]
            sent = []

            class Bridge:
                def __init__(self, *_args):
                    pass

                def send(self, command):
                    sent.append(command)

                def close(self):
                    pass

            backend = core.SafeBackend(root, "simulacion", None)
            self.seen = []
            backend.camera_connected = True
            with (patch.object(core, "load_models", return_value=(object(), object())),
                  patch.object(core, "detect_faces", side_effect=lambda *_: np.array(current)),
                  patch.object(core, "extract_feature", side_effect=lambda _r, _f, f: FEATURES[int(f[14])]),
                  patch.object(core, "CommandBridge", Bridge)):
                backend.engine_thread.start()
                backend.send("admin_ready", True)
                try:
                    yield backend, current, sent, ana
                finally:
                    backend.stop()
                    backend.engine_thread.join(3)
                    self.assertFalse(backend.engine_thread.is_alive())

    def next_event(self, backend, kind, timeout=4.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            backend.frame = np.zeros((32, 64, 3), np.uint8)
            backend.frame_seq += 1
            try:
                event = backend.events.get(timeout=0.06)
            except queue.Empty:
                continue
            if event[0] == "engine_error":
                self.fail(event[1])
            self.seen.append(event)
            if event[0] == kind:
                return event
        self.fail(f"No llegó {kind}: {self.seen[-8:]}")

    def pump_for(self, backend, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            backend.frame = np.zeros((32, 64, 3), np.uint8)
            backend.frame_seq += 1
            try:
                event = backend.events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event[0] == "engine_error":
                self.fail(event[1])
            self.seen.append(event)

    def test_known_and_unknown_pin_and_welcome_once(self):
        with self.running() as (backend, _faces, sent, ana):
            request = self.next_event(backend, "pin_required")
            self.assertEqual(request[1], ana)
            self.assertNotIn("OPEN", sent)
            self.pump_for(backend, 0.3)
            states = [event for event in self.seen if event[0] == "state"]
            self.assertTrue(any(("Ana", "PIN PENDIENTE") in event[6] and
                                ("Desconocido", "DESCONOCIDO") in event[6]
                                for event in states))
            backend.send("access_pin", (ana, "000000", request[3]))
            self.next_event(backend, "pin_failed")
            self.assertNotIn("OPEN", sent)
            self.assertFalse(any(event[0] == "welcome" for event in self.seen))
            backend.send("access_pin", (ana, "456789", request[3]))
            self.next_event(backend, "pin_success")
            self.next_event(backend, "welcome")
            self.assertEqual(sent.count("OPEN"), 1)
            self.pump_for(backend, 0.5)
            self.assertEqual(sent.count("OPEN"), 1)
            self.assertEqual(sum(event[0] == "welcome" for event in self.seen), 1)

    def test_two_registered_faces_block_selection_until_one_leaves(self):
        with self.running(two_users=True) as (backend, faces, sent, ana):
            self.pump_for(backend, 1.1)
            states = [event for event in self.seen if event[0] == "state"]
            self.assertTrue(any(event[5] == [(2, 2, 20, 20), (32, 2, 20, 20)] and
                                event[6] == [("Ana", "VERIFICANDO"), ("Beto", "VERIFICANDO")]
                                for event in states))
            self.assertTrue(any("Varios usuarios registrados" in event[2] and
                                event[6] == [("Ana", "IDENTIDAD"), ("Beto", "IDENTIDAD")]
                                for event in states))
            faces.reverse()
            self.seen.clear()
            self.pump_for(backend, 0.4)
            self.assertTrue(any(event[0] == "state" and
                                event[5] == [(32, 2, 20, 20), (2, 2, 20, 20)] and
                                [label[0] for label in event[6]] == ["Beto", "Ana"]
                                for event in self.seen))
            faces.reverse()
            self.assertNotIn("OPEN", sent)
            self.assertFalse(any(event[0] == "welcome" for event in self.seen))
            self.assertFalse(any(event[0] == "pin_required" for event in list(backend.events.queue)))
            faces.pop()
            request = self.next_event(backend, "pin_required")
            self.assertEqual(request[1], ana)
            self.assertNotIn("OPEN", sent)

    def test_position_swap_cancels_old_pin_request(self):
        with self.running() as (backend, faces, sent, ana):
            request = self.next_event(backend, "pin_required")
            faces[0], faces[1] = face(32, 0), face(2, 2)
            self.next_event(backend, "pin_expired")
            replacement = self.next_event(backend, "pin_required")
            self.assertNotEqual(replacement[3], request[3])
            backend.send("access_pin", (ana, "456789", request[3]))
            self.pump_for(backend, 0.4)
            self.assertNotIn("OPEN", sent)
            self.assertFalse(any(event[0] == "welcome" for event in self.seen))


if __name__ == "__main__":
    unittest.main()
