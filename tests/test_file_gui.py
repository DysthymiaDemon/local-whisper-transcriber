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

    def test_home_page_shows_three_actions_on_start(self):
        self.assertEqual(self.window.pages.currentIndex(), gui.HOME_PAGE)
        self.assertEqual(self.window.windowTitle(), "Meeting Scribe")
        self.assertEqual(
            [
                self.window.home_record_button.text(),
                self.window.home_file_button.text(),
                self.window.home_close_button.text(),
            ],
            ["Record and Transcribe This Meeting", "Transcribe Existing Meeting", "Close This App"],
        )

    def test_record_action_opens_session_without_starting(self):
        with patch.object(gui, "MeetingTranscriberEngine") as engine:
            self.window.home_record_button.click()
            engine.assert_not_called()
        self.assertEqual(self.window.pages.currentIndex(), gui.SESSION_PAGE)
        self.window.home_button.click()
        self.assertEqual(self.window.pages.currentIndex(), gui.HOME_PAGE)

    def test_existing_meeting_stays_home_when_picker_cancelled(self):
        with patch.object(gui.QFileDialog, "getOpenFileName", return_value=("", "")) as picker:
            self.window.home_file_button.click()
        self.assertIn("*.m4a", picker.call_args.args[3])
        self.assertEqual(self.window.pages.currentIndex(), gui.HOME_PAGE)

    def test_existing_meeting_opens_session_and_locks_home(self):
        with patch.object(gui.QFileDialog, "getOpenFileName", return_value=("C:/meeting.m4a", "")), patch.object(
            gui.QFileDialog, "getSaveFileName", return_value=("C:/meeting.txt", "")
        ), patch.object(gui, "FileTranscriptionWorker"):
            self.window.home_file_button.click()
        self.assertEqual(self.window.pages.currentIndex(), gui.SESSION_PAGE)
        self.assertFalse(self.window.home_button.isEnabled())
        self.window._show_home()
        self.assertEqual(self.window.pages.currentIndex(), gui.SESSION_PAGE)

    def test_close_action_closes_window(self):
        with patch.object(self.window, "close") as close:
            self.window.home_close_button.click()
        close.assert_called_once()

    def test_model_and_output_paths_are_under_collapsed_advanced(self):
        self.assertTrue(self.window.advanced_panel.isHidden())
        self.window.advanced_toggle.setChecked(True)
        self.assertFalse(self.window.advanced_panel.isHidden())

    def test_saved_microphone_is_restored(self):
        devices = [{"index": 3, "name": "Headset", "default_samplerate": 48000, "hostapi": "WASAPI"}]
        with patch.object(gui, "INPUT_DEVICE", 3), patch.object(gui, "list_input_devices", return_value=devices), patch.object(
            gui, "list_output_devices", return_value=[]
        ):
            self.window._load_devices()
        self.assertEqual(self.window.device_combo.currentData(), 3)

    def test_each_recording_gets_its_own_transcript_file(self):
        from datetime import datetime

        path = gui.session_output_path("C:/t/meeting_transcript.txt", datetime(2026, 10, 12, 9, 5, 7))
        self.assertEqual(Path(path), Path("C:/t/meeting_transcript_2026-10-12_090507.txt"))

    def test_stop_failure_still_leaves_stopping_state(self):
        engine = MagicMock()
        engine.stop.side_effect = RuntimeError("device vanished")
        self.window.engine = engine
        self.window._set_running(True)
        self.window._stop_recording()
        for _ in range(100):
            self.application.processEvents()
            if self.window.engine is None:
                break
            import time

            time.sleep(0.01)
        self.assertIsNone(self.window.engine)
        self.assertFalse(self.window._stopping)
        self.assertIn("device vanished", self.window.log_box.toPlainText())

    def test_fatal_engine_event_stops_session(self):
        self.window.engine = MagicMock()
        with patch.object(self.window, "_stop_recording") as stop:
            self.window._handle_engine_event("fatal", {"message": "Whisper model load failed"})
        stop.assert_called_once()

    def test_transcript_rows_are_joined_with_single_spaces(self):
        row = lambda text: gui.TranscriptRow(id="1", chunk_index=0, start=0, end=1, text=text)
        for text in ("Hello team.", "Next item.", ""):
            self.window._append_transcript(row(text))
        self.assertEqual(self.window.transcript_text.toPlainText(), "Hello team. Next item.")

    def test_close_requests_cancel_and_waits(self):
        worker = self.start_file()
        event = MagicMock()
        self.window.closeEvent(event)
        worker.cancel.assert_called_once()
        event.ignore.assert_called_once()
        self.assertTrue(self.window._close_after_file)


if __name__ == "__main__":
    unittest.main()
