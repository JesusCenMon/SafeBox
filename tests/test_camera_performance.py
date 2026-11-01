"""Cámara y recursos con captura simulada; nunca abre un dispositivo real."""

import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import core
from icons import IconProvider


class CameraPerformanceTests(unittest.TestCase):
    def test_windows_falls_back_after_directshow_open_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = core.SafeBackend(Path(directory), "simulacion", None, camera_index=4)
            failed = Mock()
            failed.isOpened.return_value = False
            working = Mock()
            working.isOpened.return_value = True
            frame = np.full((12, 16, 3), 100, np.uint8)

            def read():
                backend.stop_event.set()
                return True, frame

            working.read.side_effect = read
            captures = []

            def open_capture(index, api):
                captures.append((index, api))
                return failed if api == core.cv2.CAP_DSHOW else working

            with (patch.object(core.sys, "platform", "win32"),
                  patch.object(core.cv2, "VideoCapture", side_effect=open_capture)):
                backend._camera_loop()
            self.assertEqual(captures, [(4, core.cv2.CAP_DSHOW), (4, core.cv2.CAP_MSMF)])
            self.assertEqual(failed.release.call_count, 1)
            self.assertEqual(working.release.call_count, 1)
            self.assertEqual(backend.latest_frame(0)[1].shape, frame.shape)
            self.assertIn(("camera", True, ""), list(backend.events.queue))

    def test_invalid_frame_retries_without_publishing_live_image(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = core.SafeBackend(Path(directory), "simulacion", None)
            broken = Mock()
            broken.isOpened.return_value = True
            broken.read.return_value = (True, None)
            backend.camera_commands = Mock()
            backend.camera_commands.get_nowait.side_effect = queue.Empty

            def finish(timeout):
                backend.stop_event.set()
                raise queue.Empty

            backend.camera_commands.get.side_effect = finish
            with (patch.object(core.sys, "platform", "linux"),
                  patch.object(core.cv2, "VideoCapture", return_value=broken)):
                backend._camera_loop()
            self.assertIsNone(backend.latest_frame(0))
            events = list(backend.events.queue)
            self.assertFalse(any(event[0] == "camera" and event[1] for event in events))
            self.assertTrue(any("fotograma válido" in event[2] for event in events))

    def test_pause_releases_camera_and_resumes_fresh_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = core.SafeBackend(Path(directory), "simulacion", None)
            captures = []

            def open_capture(*_args):
                capture = Mock()
                capture.isOpened.return_value = True
                capture.read.return_value = (True, np.full((12, 16, 3), len(captures) + 1, np.uint8))
                captures.append(capture)
                return capture

            with patch.object(core.cv2, "VideoCapture", side_effect=open_capture):
                backend.camera_thread.start()

                def until(kind, value=None):
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        try:
                            event = backend.events.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        if event[0] == kind and (value is None or event[1] == value):
                            return event
                    self.fail(f"No llegó {kind}={value}")

                try:
                    until("camera", True)
                    first = backend.latest_frame(0)
                    self.assertIsNotNone(first)
                    backend.set_capture_active(False)
                    until("camera_paused", True)
                    self.assertIsNone(backend.latest_frame(0))
                    self.assertEqual(captures[0].release.call_count, 1)
                    backend.set_capture_active(True)
                    until("camera_paused", False)
                    until("camera", True)
                    second = backend.latest_frame(first[0])
                    self.assertIsNotNone(second)
                    self.assertGreater(second[0], first[0])
                    self.assertEqual(len(captures), 2)
                finally:
                    backend.stop()
                    backend.camera_thread.join(2)
                self.assertFalse(backend.camera_thread.is_alive())

    def test_icon_provider_caches_images_and_falls_back(self):
        provider = IconProvider()
        loader = Mock()
        loader.load.side_effect = lambda *_args, **kwargs: Image.new("RGBA", (20, 20))
        outline = Mock()
        outline.HOME = object()
        provider._api = (outline, loader)
        first = provider.pillow("HOME", 20, "#FFFFFF")
        self.assertIs(first, provider.pillow("HOME", 20, "#FFFFFF"))
        self.assertEqual(loader.load.call_count, 1)
        picture = provider.ctk_image("HOME", 20, "#FFFFFF")
        self.assertIsNotNone(picture)
        self.assertIs(picture, provider.ctk_image("HOME", 20, "#FFFFFF"))
        provider.pillow("HOME", 20, "#1ED760")
        self.assertEqual(loader.load.call_count, 2)
        unavailable = IconProvider()
        unavailable._unavailable = True
        self.assertIsNone(unavailable.ctk_image("HOME"))


if __name__ == "__main__":
    unittest.main()
