import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from PySide6.QtWidgets import QApplication
import meeting_transcriber_gui as gui


class FileGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        with patch.object(gui, "list_input_devices", side_effect=RuntimeError("no device")):
            self.window = gui.MainWindow()
        self.addCleanup(self.cleanup_window)

    def cleanup_window(self):
        self.window.file_worker = None
        self.window.engine = None
        self.window._stopping = False
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()

    def start_file(self):
        with patch.object(gui.QFileDialog, "getOpenFileName", return_value=("C:/meeting.mp4", "")), patch.object(
            gui.QFileDialog, "getSaveFileName", return_value=("C:/meeting.txt", "")
        ), patch.object(gui, "FileTranscriptionWorker") as worker:
            self.window._start_file_transcription()
            worker.return_value.thread.is_alive.return_value = False
            return worker.return_value

    def test_file_without_devices_preserves_live_output_and_blocks_recording(self):
        output = self.window.output_path.text()
        worker = self.start_file()
        worker.start.assert_called_once()
        self.assertEqual(self.window.output_path.text(), output)
        self.assertFalse(self.window.record_button.isEnabled())
        self.assertFalse(self.window.pause_button.isEnabled())
        self.assertFalse(self.window.file_button.isEnabled())
        with patch.object(gui, "MeetingTranscriberEngine") as engine:
            self.window._start_recording()
            engine.assert_not_called()
        self.window._stop_recording()
        worker.cancel.assert_called_once()
        self.window._handle_engine_event("file_finished", {"outcome": "cancelled", "output_path": "C:/meeting.txt"})
        self.assertIsNone(self.window.file_worker)
        self.assertTrue(self.window.file_button.isEnabled())
        self.assertTrue(self.window.record_button.isEnabled())

    def test_recording_blocks_file_picker(self):
        self.window.engine = MagicMock()
        with patch.object(gui.QFileDialog, "getOpenFileName") as picker:
            self.window._start_file_transcription()
            picker.assert_not_called()

    def test_failed_completion_and_progress(self):
        self.start_file()
        self.window._handle_engine_event("file_progress", {"processed": 5, "total": 10})
        self.assertIn("50%", self.window.file_progress_label.text())
        self.window._handle_engine_event("file_finished", {"outcome": "failed", "output_path": None})
        self.assertIn("failed", self.window.status_label.text())
        self.assertTrue(self.window.file_button.isEnabled())

    def test_close_requests_cancel_and_waits(self):
        worker = self.start_file()
        event = MagicMock()
        self.window.closeEvent(event)
        worker.cancel.assert_called_once()
        event.ignore.assert_called_once()
        self.assertTrue(self.window._close_after_file)


if __name__ == "__main__":
    unittest.main()
