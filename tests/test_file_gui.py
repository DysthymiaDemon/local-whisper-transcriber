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
        self.preload = self.start_patch(patch.object(gui, "preload_whisper_model"))
        self.unload = self.start_patch(patch.object(gui, "unload_whisper_model"))
        with patch.object(gui, "list_input_devices", side_effect=RuntimeError("no device")):
            self.window = gui.MainWindow()
        self.addCleanup(self.cleanup_window)

    def start_patch(self, patcher):
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def wait_for_model_state(self, state):
        import time

        for _ in range(200):
            self.application.processEvents()
            if self.window.model_button.state == state:
                return
            time.sleep(0.01)
        self.fail(f"model state stayed {self.window.model_button.state}, expected {state}")

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
        with patch.object(gui, "MeetingTranscriberEngine") as engine:
            self.window._start_recording()
            engine.assert_not_called()
        self.window._stop_recording()
        worker.cancel.assert_called_once()
        self.window._handle_engine_event("file_finished", {"outcome": "cancelled", "output_path": "C:/meeting.txt"})
        self.assertIsNone(self.window.file_worker)
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
        self.assertTrue(self.window.record_button.isEnabled())

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
        self.window._set_model_state(gui.MODEL_STARTING)
        with patch.object(self.window, "_stop_recording") as stop:
            self.window._handle_engine_event("fatal", {"message": "Whisper model load failed"})
        stop.assert_called_once()
        self.assertEqual(self.window.model_button.state, gui.MODEL_STOPPED)

    def test_transcript_rows_are_joined_with_single_spaces(self):
        row = lambda text: gui.TranscriptRow(id="1", chunk_index=0, start=0, end=1, text=text)
        for text in ("Hello team.", "Next item.", ""):
            self.window._append_transcript(row(text))
        self.assertEqual(self.window.transcript_text.toPlainText(), "Hello team. Next item.")

    def test_model_pill_colour_follows_traffic_light(self):
        from ui_theme import MODEL_PILL_COLOURS, MODEL_STATE_LABELS

        self.assertEqual(set(MODEL_PILL_COLOURS), set(MODEL_STATE_LABELS))
        fill = {state: colours[0] for state, colours in MODEL_PILL_COLOURS.items()}
        self.assertEqual(fill[gui.MODEL_STARTING], fill[gui.MODEL_STOPPING])
        self.assertEqual(len({fill[gui.MODEL_STOPPED], fill[gui.MODEL_STARTING], fill[gui.MODEL_READY]}), 3)

    def test_model_is_not_loaded_at_launch(self):
        self.preload.assert_not_called()
        self.assertEqual(self.window.model_button.state, gui.MODEL_STOPPED)
        self.assertEqual(self.window.model_button.text(), "Start Model")

    def test_unverified_model_is_flagged_from_the_engine_check(self):
        for repo_id, flagged in (("distil-whisper/distil-large-v3.5-ct2", False), (None, True)):
            with self.subTest(repo_id=repo_id), patch.object(gui.EngineConfig, "validate", return_value=[]), patch.object(
                gui, "verified_model_repo", return_value=repo_id
            ):
                self.window.log_box.clear()
                self.window.model_button.click()
                self.wait_for_model_state(gui.MODEL_READY)
                self.assertEqual("Custom model folder" in self.window.log_box.toPlainText(), flagged)
                self.window.model_button.click()
                self.wait_for_model_state(gui.MODEL_STOPPED)

    def test_home_record_starts_model_red_yellow_green(self):
        with patch.object(gui.EngineConfig, "validate", return_value=[]):
            self.window.home_record_button.click()
            self.assertIn(self.window.model_button.state, (gui.MODEL_STARTING, gui.MODEL_READY))
            self.wait_for_model_state(gui.MODEL_READY)
        self.preload.assert_called_once()
        self.assertEqual(self.window.model_button.text(), "Ready to Transcribe")
        self.assertIn("Speech model ready", self.window.model_status_label.text())

    def test_clicking_green_stops_model_yellow_red(self):
        with patch.object(gui.EngineConfig, "validate", return_value=[]):
            self.window.model_button.click()
            self.wait_for_model_state(gui.MODEL_READY)
            self.window.model_button.click()
            self.wait_for_model_state(gui.MODEL_STOPPED)
        self.unload.assert_called_once()
        self.assertEqual(self.window.model_button.text(), "Start Model")
        self.assertIn("stopped", self.window.model_status_label.text())

    def test_model_button_is_locked_while_recording(self):
        self.window.engine = MagicMock()
        self.window._set_running(True)
        self.assertFalse(self.window.model_button.isEnabled())
        self.window._toggle_model()
        self.preload.assert_not_called()
        self.window.engine = None
        self.window._set_running(False)
        self.assertTrue(self.window.model_button.isEnabled())

    def test_model_load_failure_returns_to_red(self):
        self.preload.side_effect = RuntimeError("bad model")
        with patch.object(gui.EngineConfig, "validate", return_value=[]):
            self.window.model_button.click()
            self.wait_for_model_state(gui.MODEL_STOPPED)
        self.assertIn("failed to load", self.window.model_status_label.text())

    def test_changed_model_files_keep_the_light_red_with_a_setup_hint(self):
        from model_integrity import ModelIntegrityError

        self.preload.side_effect = ModelIntegrityError(
            "Speech model files changed since setup (changed: model.bin). Run setup to restore the approved model."
        )
        with patch.object(gui.EngineConfig, "validate", return_value=[]):
            self.window.model_button.click()
            self.wait_for_model_state(gui.MODEL_STOPPED)
        self.assertIn("changed since setup", self.window.model_status_label.text())
        self.assertIn("Run setup", self.window.log_box.toPlainText())

    def test_missing_model_folder_keeps_red(self):
        with patch.object(gui.EngineConfig, "validate", return_value=["Whisper model path does not exist: C:/missing"]):
            self.window.model_button.click()
        self.preload.assert_not_called()
        self.assertEqual(self.window.model_button.state, gui.MODEL_STOPPED)
        self.assertIn("not found", self.window.model_status_label.text())

    def test_record_with_model_stopped_starts_model_and_records(self):
        with patch.object(gui.EngineConfig, "validate", return_value=[]), patch.object(
            gui, "MeetingTranscriberEngine"
        ) as engine, patch.object(gui, "save_portable_config"):
            self.window._start_recording()
            self.assertIn(self.window.model_button.state, (gui.MODEL_STARTING, gui.MODEL_READY))
            engine.return_value.start.assert_called_once()
            self.assertFalse(self.window.model_button.isEnabled())
            self.window.engine = None
            self.window._set_running(False)
            self.wait_for_model_state(gui.MODEL_READY)

    def test_record_is_blocked_while_model_unloads(self):
        self.window._set_model_state(gui.MODEL_STOPPING)
        self.assertFalse(self.window.record_button.isEnabled())
        self.assertFalse(self.window.model_button.isEnabled())

    def test_record_pause_stop_have_their_own_colours(self):
        self.assertEqual(
            [self.window.record_button.objectName(), self.window.pause_button.objectName(), self.window.stop_button.objectName()],
            ["recordButton", "pauseButton", "stopButton"],
        )
        from ui_theme import APP_STYLESHEET

        for name in ("recordButton", "pauseButton", "stopButton"):
            self.assertIn(f"QPushButton#{name} {{", APP_STYLESHEET)

    def test_transcribe_file_button_is_gone_but_home_still_opens_files(self):
        self.assertFalse(hasattr(self.window, "file_button"))
        self.assertEqual(self.window.home_file_button.text(), "Transcribe Existing Meeting")

    def test_model_button_is_standard_size_and_session_buttons_are_large(self):
        from ui_theme import SESSION_BUTTON_HEIGHT, STANDARD_BUTTON_HEIGHT

        self.window.resize(1040, 720)
        self.window.pages.setCurrentIndex(gui.SESSION_PAGE)
        self.window.show()
        self.addCleanup(self.window.hide)
        self.application.processEvents()
        row = [self.window.record_button, self.window.pause_button, self.window.stop_button, self.window.copy_button]

        self.assertEqual(self.window.model_button.height(), STANDARD_BUTTON_HEIGHT)
        self.assertEqual([button.height() for button in row], [SESSION_BUTTON_HEIGHT] * 4)
        widths = [button.width() for button in row]
        self.assertLessEqual(max(widths) - min(widths), 1)
        spanned = row[-1].geometry().right() - row[0].geometry().left() + 1
        self.assertEqual(spanned, self.window.model_button.width())
        self.assertEqual(self.window.copy_button.objectName(), "copyButton")

    def test_record_flashes_when_model_turns_ready(self):
        from ui_theme import RECORD_FLASH_PULSES

        self.window._set_model_state(gui.MODEL_STARTING)
        self.window._handle_engine_event("model_status", {"state": gui.MODEL_READY, "message": "Speech model ready"})

        self.assertTrue(self.window._flash_timer.isActive())
        self.assertTrue(self.window.record_button.property("flash"))
        seen = []
        for _ in range(2 * RECORD_FLASH_PULSES + 1):
            self.window._advance_record_flash()
            seen.append(bool(self.window.record_button.property("flash")))
        self.assertIn(False, seen[:-1])
        self.assertFalse(seen[-1])
        self.assertFalse(self.window._flash_timer.isActive())

    def test_pressing_record_stops_the_flash(self):
        self.window._set_model_state(gui.MODEL_STARTING)
        self.window._handle_engine_event("model_status", {"state": gui.MODEL_READY, "message": "Speech model ready"})
        with patch.object(gui.EngineConfig, "validate", return_value=["stop here"]), patch.object(
            gui.QMessageBox, "critical"
        ):
            self.window._start_recording()

        self.assertFalse(self.window._flash_timer.isActive())
        self.assertFalse(self.window.record_button.property("flash"))

    def test_no_flash_when_model_becomes_ready_during_recording(self):
        self.window._set_model_state(gui.MODEL_STARTING)
        self.window.engine = MagicMock()
        self.window._set_running(True)
        self.window._handle_engine_event("model_status", {"state": gui.MODEL_READY, "message": "Speech model ready"})

        self.assertFalse(self.window._flash_timer.isActive())
        self.window.engine = None
        self.window._set_running(False)

    def test_close_requests_cancel_and_waits(self):
        worker = self.start_file()
        event = MagicMock()
        self.window.closeEvent(event)
        worker.cancel.assert_called_once()
        event.ignore.assert_called_once()
        self.assertTrue(self.window._close_after_file)


if __name__ == "__main__":
    unittest.main()
