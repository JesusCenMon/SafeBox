"""Comprobaciones locales de recuperación, historial y respaldo; sin hardware."""

import queue
import sqlite3
import tempfile
import time
import unittest
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import core
import app
from control import AccessPinStore


class UpgradeV3Tests(unittest.TestCase):
    def test_camera_retry_has_bounded_backoff_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = core.SafeBackend(Path(directory), "simulacion", None, camera_index=3)
            candidate = Mock()
            candidate.isOpened.return_value = False
            waits = []

            def waiting(timeout):
                waits.append(timeout)
                if len(waits) == 4:
                    backend.stop_event.set()
                raise queue.Empty

            commands = Mock()
            commands.get_nowait.side_effect = queue.Empty
            commands.get.side_effect = waiting
            backend.camera_commands = commands
            with patch.object(core.cv2, "VideoCapture", return_value=candidate) as opened:
                backend._camera_loop()
            self.assertEqual(waits, [0.5, 1.0, 2.0, 4.0])
            self.assertEqual(opened.call_count, 4)
            self.assertTrue(all(call.args[0] == 3 for call in opened.call_args_list))
            self.assertEqual(candidate.release.call_count, 4)

    def test_enrollment_quality_rejects_only_clear_defects(self):
        feature = np.eye(1, 128, dtype=np.float32).reshape(-1)
        face = np.array([0, 0, 80, 80], np.float32)
        dark = np.zeros((80, 80, 3), np.uint8)
        blurred = np.full((80, 80, 3), 120, np.uint8)
        rng = np.random.default_rng(7)
        clear = rng.integers(75, 175, (80, 80, 3), dtype=np.uint8)
        self.assertIn("iluminación", core.enrollment_quality(dark, face, feature, []))
        self.assertIn("enfocado", core.enrollment_quality(blurred, face, feature, []))
        self.assertIsNone(core.enrollment_quality(clear, face, feature, []))
        self.assertIn("Mueve", core.enrollment_quality(clear, face, feature, [feature]))
        self.assertEqual(core.ENROLL_SAMPLES, 20)

    def test_incomplete_user_pin_can_be_completed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = np.eye(1, 128, dtype=np.float32).reshape(-1)
            uid = core.FaceStore(root).add_person("Ana", [feature] * core.ENROLL_SAMPLES)
            backend = core.SafeBackend(root, "simulacion", None)
            with (patch.object(core, "load_models", return_value=(object(), object())),
                  patch.object(core, "CommandBridge", return_value=Mock())):
                backend.engine_thread.start()

                def until(kind):
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        try:
                            event = backend.events.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        if event[0] == "engine_error":
                            self.fail(event[1])
                        if event[0] == kind:
                            return event
                    self.fail(f"No llegó {kind}")

                try:
                    backend.send("complete_access_pin", (uid, "123456"))
                    until("pin_set")
                    original = (root / "access_pins.json").read_bytes()
                    backend.send("complete_access_pin", (uid, "654321"))
                    error = until("user_error")
                    self.assertIn("ya tiene PIN", error[1])
                    self.assertEqual(original, (root / "access_pins.json").read_bytes())
                    AccessPinStore(root).verify(uid, "123456")
                    self.assertEqual(len(core.FaceStore(root).users), 1)
                finally:
                    backend.stop()
                    backend.engine_thread.join(2)

    def test_history_date_filter_combines_user_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            history = core.AccessHistory(Path(directory))
            try:
                history.db.executemany(
                    "INSERT INTO accesos (fecha,usuario,resultado,captura) VALUES (?,?,?,?)",
                    [("2026-10-10T23:30:00-05:00", "Ana", "AUTORIZADO", ""),
                     ("2026-10-11T01:00:00+01:00", "Ana", "DENEGADO", ""),
                     ("2026-10-12T12:00:00-05:00", "Beto", "AUTORIZADO", "")])
                history.db.commit()
                rows = history.search("Ana", "AUTORIZADO", start=date(2026, 10, 10),
                                      end=date(2026, 10, 10))
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0][0].startswith("2026-10-10"))
                self.assertEqual(history.daily_activity(),
                                 [("2026-10-10", 1, 0), ("2026-10-11", 0, 1),
                                  ("2026-10-12", 1, 0)])
                with self.assertRaises(ValueError):
                    history.search(start=date(2026, 10, 12), end=date(2026, 10, 10))
            finally:
                history.close()

    def test_backup_contains_local_data_and_valid_sqlite_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "datos"
            history = core.AccessHistory(root)
            try:
                (root / "usuarios.json").write_text('{"1":"Ana"}', encoding="utf-8")
                (root / "access_pins.json").write_text('{"1":"hash"}', encoding="utf-8")
                (root / "rasgos_sface").mkdir()
                (root / "rasgos_sface" / "1.npy").write_bytes(b"profile")
                (root / "capturas" / "one.jpg").write_bytes(b"photo")
                history.db.execute(
                    "INSERT INTO accesos (fecha,usuario,resultado,captura) VALUES (?,?,?,?)",
                    ("2026-10-10T10:00:00-05:00", "Ana", "AUTORIZADO", "one.jpg"))
                history.db.commit()
                destination = Path(directory) / "backup.zip"
                path, count = core.create_local_backup(root, destination, history)
                self.assertEqual(path, destination)
                with zipfile.ZipFile(path) as archive:
                    self.assertEqual(archive.testzip(), None)
                    self.assertEqual(count, len(archive.namelist()))
                    self.assertTrue({"historial.db", "usuarios.json", "access_pins.json",
                                     "rasgos_sface/1.npy", "capturas/one.jpg"}.issubset(archive.namelist()))
                    self.assertFalse(any(name.endswith(".py") for name in archive.namelist()))
                    snapshot = Path(directory) / "snapshot.db"
                    snapshot.write_bytes(archive.read("historial.db"))
                self.assertEqual(core.verify_local_backup(path), (True, []))
                db = sqlite3.connect(snapshot)
                try:
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM accesos").fetchone()[0], 1)
                finally:
                    db.close()
                with self.assertRaises(FileExistsError):
                    core.create_local_backup(root, destination, history)
            finally:
                history.close()

    def test_filtered_export_uses_all_matching_pages_without_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = core.AccessHistory(root)
            try:
                history.db.executemany(
                    "INSERT INTO accesos (fecha,usuario,resultado,captura) VALUES (?,?,?,?)",
                    [("2026-10-10T10:00:00-05:00", "Ana", "AUTORIZADO", "secret.jpg") for _ in range(501)] +
                    [("2026-10-11T10:00:00-05:00", "Beto", "DENEGADO", "other.jpg")])
                history.db.commit()
                target = root / "filtered.csv"
                count = history.export_filtered_csv(target, "Ana", "AUTORIZADO",
                                                    date(2026, 10, 10), date(2026, 10, 10))
                self.assertEqual(count, 501)
                content = target.read_text(encoding="utf-8-sig")
                self.assertEqual(len(content.splitlines()), 502)
                self.assertNotIn("secret.jpg", content)
                self.assertNotIn("Beto", content)
                with self.assertRaises(FileExistsError):
                    history.export_filtered_csv(target)
            finally:
                history.close()

    def test_graph_png_uses_real_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "charts.png"
            app.save_graphs_png(path, (3, 2, 1), [("2026-10-10", 2, 1)],
                                [("2026-10-10", 8, 3)])
            from PIL import Image
            with Image.open(path) as picture:
                self.assertEqual(picture.format, "PNG")
                self.assertEqual(picture.size, (1050, 430))

    def test_integrity_and_storage_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "usuarios.json").write_text('{"1":"Ana"}', encoding="utf-8")
            (root / "access_pins.json").write_text('{}', encoding="utf-8")
            (root / "config.json").write_text('{}', encoding="utf-8")
            (root / "historial.db").write_bytes(b"damaged")
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            findings = core.check_data_integrity(root)
            self.assertTrue(any("rostro sin PIN" in finding for finding in findings))
            self.assertTrue(any("historial.db" in finding for finding in findings))
            self.assertTrue(any("config.json" in finding for finding in findings))
            sizes = core.storage_sizes(root)
            self.assertGreater(sizes["historial"], 0)
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})

    def test_backup_verifier_reports_corrupt_or_incomplete_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("notes.txt", "x")
            valid, findings = core.verify_local_backup(path)
            self.assertFalse(valid)
            self.assertTrue(any("historial.db" in finding for finding in findings))


if __name__ == "__main__":
    unittest.main()
