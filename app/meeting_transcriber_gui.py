from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any
from file_transcriber import FileTranscriptionWorker

from app_config import (
    append_error_log,
    application_root,
    load_portable_config,
    save_portable_config,
)
from transcriber_engine import (
    EngineConfig,
    MeetingTranscriberEngine,
    TranscriptRow,
    CAPTURE_MODE_LOOPBACK,
    CAPTURE_MODE_MIC,
    CAPTURE_MODE_MIXED,
    list_input_devices,
    list_output_devices,
    rms_to_meter_percent,
)
from ui_helpers import meter_bar_geometry


PORTABLE_DEFAULTS = load_portable_config()
WHISPER_MODEL_DIR = PORTABLE_DEFAULTS.whisper_model_dir
OUTPUT_FILE = PORTABLE_DEFAULTS.output_file
SAMPLE_RATE = PORTABLE_DEFAULTS.sample_rate
CHUNK_SECONDS = PORTABLE_DEFAULTS.chunk_seconds
OVERLAP_SECONDS = PORTABLE_DEFAULTS.overlap_seconds
CAPTURE_MODE = PORTABLE_DEFAULTS.capture_mode
SYSTEM_DEVICE = PORTABLE_DEFAULTS.system_device_index
ALLOW_MIXED_FALLBACK = PORTABLE_DEFAULTS.allow_mixed_fallback
COMPUTE_TYPE = PORTABLE_DEFAULTS.compute_type
DEVICE = PORTABLE_DEFAULTS.device
CPU_THREADS = PORTABLE_DEFAULTS.cpu_threads
NUM_WORKERS = PORTABLE_DEFAULTS.num_workers
LANGUAGE = PORTABLE_DEFAULTS.language


try:
    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QCheckBox,
        QSizePolicy,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - depends on local GUI deps
    print(
        "PySide6 is not installed. Install offline wheels from docs/offline_setup.md, "
        "then run this file again.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


class EngineSignalBridge(QObject):
    event = Signal(str, object)


class MicLevelMeter(QWidget):
    def __init__(self, bar_width: int = 7, gap: int = 4, min_segments: int = 18) -> None:
        super().__init__()
        self.bar_width = bar_width
        self.gap = gap
        self.min_segments = min_segments
        self.level = 0
        self.setMinimumHeight(28)

    def set_level(self, value: int) -> None:
        self.level = max(0, min(100, value))
        self.update()

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual widget
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        width = self.width()
        height = self.height()
        segments, bar_width, gap = meter_bar_geometry(
            width,
            bar_width=self.bar_width,
            gap=self.gap,
            min_segments=self.min_segments,
        )
        bar_height = max(14, min(20, height - 6))
        top = int((height - bar_height) / 2)
        active_segments = round((self.level / 100) * segments)

        for index in range(segments):
            left = index * (bar_width + gap)
            if left + bar_width > width:
                break
            active = index < active_segments
            ratio = (index + 1) / segments
            if not active:
                color = QColor("#d8d8dc")
            elif ratio <= 0.60:
                color = QColor("#4ac11f")
            elif ratio <= 0.82:
                color = QColor("#f1c232")
            else:
                color = QColor("#d93025")
            painter.fillRect(left, top, bar_width, bar_height, color)
            painter.setPen(QPen(QColor("#8b8b8b"), 1))
            painter.drawRect(left, top, bar_width, bar_height)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Offline Meeting Transcriber")
        self.resize(1040, 720)
        self.engine: MeetingTranscriberEngine | None = None
        self.file_worker: FileTranscriptionWorker | None = None
        self._close_after_file = False
        self._stopping = False
        self.bridge = EngineSignalBridge()
        self.bridge.event.connect(self._handle_engine_event)
        self._output_devices_by_index: dict[int, dict[str, Any]] = {}
        self._behind_seconds = 0.0
        self._spinner_index = 0
        self._spinner_frames = ["|", "/", "-", "\\"]
        self._copy_feedback_tokens: dict[int, int] = {}
        self._copy_button_originals: dict[int, tuple[str, QIcon, str]] = {}
        self.spinner_timer = QTimer(self)
        self.spinner_timer.setInterval(160)
        self.spinner_timer.timeout.connect(self._tick_spinner)

        self._build_ui()
        self._load_devices()
        self._set_running(False)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(main_splitter, 1)

        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setAcceptRichText(False)
        self.transcript_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.transcript_text.setPlaceholderText("Transcription appears here...")
        self.transcript_text.setStyleSheet(
            "QTextEdit {"
            "background: #2b2b2b;"
            "color: #f2f2f2;"
            "border: 1px solid #444;"
            "padding: 14px;"
            "font: 14px 'Segoe UI';"
            "selection-background-color: #174ea6;"
            "}"
        )
        main_splitter.addWidget(self.transcript_text)

        right_panel = QWidget()
        right_panel.setMinimumWidth(260)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([680, 340])

        controls = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls)
        button_row = QHBoxLayout()
        self.record_button = QPushButton("Record")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.copy_button = QPushButton("Copy")
        self.record_button.clicked.connect(self._start_recording)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop_recording)
        self.copy_button.clicked.connect(lambda: self._copy_transcript(self.copy_button))
        button_row.addWidget(self.record_button)
        button_row.addWidget(self.pause_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.copy_button)
        controls_layout.addLayout(button_row)
        self.file_button = QPushButton("Transcribe File")
        self.file_button.clicked.connect(self._start_file_transcription)
        controls_layout.addWidget(self.file_button)
        self.file_progress_label = QLabel("")
        controls_layout.addWidget(self.file_progress_label)

        status_row = QFormLayout()
        self.status_label = QLabel("Idle")
        self.level_meter = MicLevelMeter()
        status_row.addRow("Status", self.status_label)
        status_row.addRow("Mic level", self.level_meter)
        controls_layout.addLayout(status_row)
        right_layout.addWidget(controls)

        settings = QGroupBox("Settings")
        settings_layout = QFormLayout(settings)
        self.device_combo = QComboBox()
        self.system_device_combo = QComboBox()
        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.addItem("Microphone", CAPTURE_MODE_MIC)
        self.capture_mode_combo.addItem("System audio (loopback)", CAPTURE_MODE_LOOPBACK)
        self.capture_mode_combo.addItem("Mixed (mic + system)", CAPTURE_MODE_MIXED)
        self.capture_mode_combo.currentIndexChanged.connect(self._update_capture_controls)
        self.allow_fallback_checkbox = QCheckBox("Allow mixed fallback to single source")
        self.whisper_path = QLineEdit(WHISPER_MODEL_DIR)
        self.output_path = QLineEdit(OUTPUT_FILE)
        for widget in (
            self.device_combo,
            self.system_device_combo,
            self.capture_mode_combo,
            self.whisper_path,
            self.output_path,
            self.allow_fallback_checkbox,
        ):
            self._allow_field_to_shrink(widget)

        self.capture_mode_combo.setCurrentText("Microphone")
        settings_layout.addRow("Capture mode", self.capture_mode_combo)
        settings_layout.addRow("Microphone", self.device_combo)
        settings_layout.addRow("System output", self.system_device_combo)
        settings_layout.addRow("Mixed fallback", self.allow_fallback_checkbox)
        settings_layout.addRow("Whisper", self._path_row(self.whisper_path, folder=True))
        settings_layout.addRow("Output", self._path_row(self.output_path, folder=False))
        right_layout.addWidget(settings)
        right_layout.addStretch(1)

        footer_row = QHBoxLayout()
        self.transcription_state_label = QLabel("Idle")
        self.copy_footer_button = QPushButton("Copy to Clipboard")
        self.copy_footer_button.setIcon(QIcon.fromTheme("edit-copy"))
        self.copy_footer_button.clicked.connect(lambda: self._copy_transcript(self.copy_footer_button))
        footer_row.addWidget(self.transcription_state_label, 1)
        footer_row.addWidget(self.copy_footer_button)
        root_layout.addLayout(footer_row)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        root_layout.addWidget(self.log_box)

    @staticmethod
    def _allow_field_to_shrink(widget: QWidget) -> None:
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def _path_row(self, line_edit: QLineEdit, folder: bool) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse_path(line_edit, folder))
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse)
        return widget

    def _browse_path(self, line_edit: QLineEdit, folder: bool) -> None:
        if folder:
            selected = QFileDialog.getExistingDirectory(self, "Select folder", line_edit.text())
        else:
            selected, _ = QFileDialog.getSaveFileName(
                self,
                "Select transcript file",
                line_edit.text(),
                "Text files (*.txt);;All files (*.*)",
            )
        if selected:
            line_edit.setText(selected)

    def _load_devices(self) -> None:
        self.device_combo.clear()
        self._output_devices_by_index = {}
        try:
            devices = list_input_devices()
        except Exception as exc:
            devices = []
            self._append_log(f"Audio device discovery failed; file transcription remains available: {exc}")
        if not devices:
            self.device_combo.addItem("Default input", None)
            self.system_device_combo.clear()
            self.system_device_combo.addItem("Default output", None)
            self._set_capture_default_values()
            self._update_capture_controls()
            return

        self.device_combo.addItem("Default input", None)
        for device in devices:
            rate = device.get("default_samplerate")
            hostapi = device.get("hostapi")
            suffix = f" - {hostapi}" if hostapi else ""
            rate_text = f" ({int(rate)} Hz)" if rate else ""
            self.device_combo.addItem(f"{device['index']}: {device['name']}{suffix}{rate_text}", device["index"])
        for item in (None, *[device["index"] for device in devices]):
            if item is not None and not any(self.device_combo.itemData(i) == item for i in range(self.device_combo.count())):
                pass

        self.system_device_combo.clear()
        try:
            output_devices = list_output_devices()
        except Exception as exc:
            output_devices = []
            self._append_log(f"Output device discovery failed: {exc}")
        if not output_devices:
            self.system_device_combo.addItem("Default output", None)
            self._update_capture_controls()
            self._set_capture_default_values()
            return
        self.system_device_combo.addItem("Default output", None)
        for device in output_devices:
            rate = device.get("default_samplerate")
            hostapi = device.get("hostapi")
            suffix = f" - {hostapi}" if hostapi else ""
            rate_text = f" ({int(rate)} Hz)" if rate else ""
            index = int(device["index"])
            self._output_devices_by_index[index] = device
            wasapi_suffix = " (WASAPI)" if device.get("wasapi") else ""
            self.system_device_combo.addItem(
                f"{index}: {device['name']}{suffix}{rate_text}{wasapi_suffix}",
                index,
            )
        self._set_capture_default_values()
        self._update_capture_controls()

    def _set_capture_default_values(self) -> None:
        self._set_combo_current_data(self.device_combo, DEVICE)
        self._set_combo_current_data(self.system_device_combo, SYSTEM_DEVICE)
        self._set_combo_current_data(self.capture_mode_combo, CAPTURE_MODE)
        self.allow_fallback_checkbox.setChecked(bool(ALLOW_MIXED_FALLBACK))

    @staticmethod
    def _set_combo_current_data(combo: QComboBox, value: object) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return
        combo.setCurrentIndex(0)

    def _update_capture_controls(self) -> None:
        mode = str(self.capture_mode_combo.currentData() or CAPTURE_MODE_MIC).strip().lower()
        loopback_supported = self._is_loopback_supported()
        loopback_mode_enabled = loopback_supported
        self.system_device_combo.setEnabled(mode != CAPTURE_MODE_MIC)
        self.allow_fallback_checkbox.setEnabled(mode == CAPTURE_MODE_MIXED)

        if not loopback_mode_enabled and mode != CAPTURE_MODE_MIC:
            self.capture_mode_combo.blockSignals(True)
            self._set_combo_current_data(self.capture_mode_combo, CAPTURE_MODE_MIC)
            self.capture_mode_combo.blockSignals(False)
            self.system_device_combo.setEnabled(False)
            self.allow_fallback_checkbox.setEnabled(False)
            self._append_log("Loopback mode is unavailable because no WASAPI output device is detected.")

        selected_system_is_wasapi = self._is_selected_system_device_wasapi()
        if mode in (CAPTURE_MODE_LOOPBACK, CAPTURE_MODE_MIXED) and not selected_system_is_wasapi:
            self.system_device_combo.setToolTip("Loopback and mixed modes require a WASAPI output device.")
        else:
            self.system_device_combo.setToolTip("")

        if mode == CAPTURE_MODE_MIXED and self.allow_fallback_checkbox.isChecked():
            if not self.system_device_combo.itemData(self.system_device_combo.currentIndex()):
                self.allow_fallback_checkbox.setEnabled(False)
            else:
                self.allow_fallback_checkbox.setEnabled(True)
        else:
            self.allow_fallback_checkbox.setEnabled(mode == CAPTURE_MODE_MIXED)

    def _selected_system_device(self) -> int | None:
        selected = self.system_device_combo.currentData()
        return int(selected) if isinstance(selected, int) else None

    def _is_loopback_supported(self) -> bool:
        return any(bool(device.get("wasapi")) for device in self._output_devices_by_index.values())

    def _is_selected_system_device_wasapi(self) -> bool:
        device_index = self._selected_system_device()
        if device_index is None:
            return False
        return bool(self._output_devices_by_index.get(device_index, {}).get("wasapi", False))

    def _read_config(self) -> EngineConfig:
        selected_mode = str(self.capture_mode_combo.currentData() or CAPTURE_MODE_MIC)
        return EngineConfig(
            whisper_model_dir=self.whisper_path.text().strip(),
            output_file=self.output_path.text().strip(),
            sample_rate=SAMPLE_RATE,
            chunk_seconds=float(CHUNK_SECONDS),
            overlap_seconds=float(OVERLAP_SECONDS),
            compute_type=COMPUTE_TYPE,
            device=DEVICE,
            cpu_threads=int(CPU_THREADS),
            num_workers=int(NUM_WORKERS),
            input_device=self.device_combo.currentData(),
            capture_mode=selected_mode,
            system_device_index=self.system_device_combo.currentData(),
            allow_mixed_fallback=self.allow_fallback_checkbox.isChecked(),
            language=LANGUAGE,
        )

    def _start_recording(self) -> None:
        if self._stopping or self.engine or self.file_worker:
            return
        config = self._read_config()
        save_portable_config(config, root=application_root())
        errors = config.validate()
        if errors:
            message = "\n".join(errors)
            log_path = self._write_error_log("Configuration error", message, {"errors": errors})
            QMessageBox.critical(self, "Configuration error", f"{message}\n\nError details saved to {log_path}")
            return

        self.transcript_text.clear()
        self.engine = MeetingTranscriberEngine(config)
        self.engine.on_event(lambda event_type, payload: self.bridge.event.emit(event_type, payload))
        try:
            self.engine.start()
        except Exception as exc:
            message = str(exc)
            log_path = self._write_error_log("Start failed", message)
            self._append_log(message)
            QMessageBox.critical(self, "Start failed", f"{message}\n\nError details saved to {log_path}")
            self.engine = None
            return
        self._stopping = False
        self._set_running(True)

    def _start_file_transcription(self) -> None:
        if self.engine or self.file_worker or self._stopping:
            return
        source, _ = QFileDialog.getOpenFileName(
            self, "Select meeting recording", str(Path.home() / "Downloads"), "MP4 recordings (*.mp4)"
        )
        if not source:
            return
        output, _ = QFileDialog.getSaveFileName(
            self, "Save transcript", str(Path(source).with_suffix(".txt")), "Text files (*.txt)"
        )
        if not output:
            return
        if Path(source).resolve() == Path(output).resolve():
            QMessageBox.critical(self, "Invalid output", "Output must differ from the source recording.")
            return
        self.transcript_text.clear()
        self.file_progress_label.setText("Preparing recording")
        self.file_worker = FileTranscriptionWorker(
            self._read_config(), source, output,
            lambda event, payload: self.bridge.event.emit(event, payload),
        )
        self._set_running(True)
        self.pause_button.setEnabled(False)
        self.status_label.setText("Transcribing file")
        try:
            self.file_worker.start()
        except Exception as exc:
            self.file_worker = None
            self._set_running(False)
            self.status_label.setText("File transcription failed")
            self._append_log(str(exc))

    def _toggle_pause(self) -> None:
        if not self.engine:
            return
        if self.pause_button.text() == "Pause":
            self.engine.pause()
            self.pause_button.setText("Resume")
        else:
            self.engine.resume()
            self.pause_button.setText("Pause")

    def _stop_recording(self) -> None:
        if self.file_worker:
            self.file_worker.cancel()
            self._stopping = True
            self.stop_button.setEnabled(False)
            self.status_label.setText("Finishing current file chunk")
            return
        if not self.engine or self._stopping:
            return
        self._stopping = True
        self.status_label.setText("Finishing transcription")
        self.record_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.level_meter.set_level(0)
        self._update_transcription_state(running=True)
        if not self.spinner_timer.isActive():
            self.spinner_timer.start()
        for widget in (
            self.device_combo,
            self.system_device_combo,
            self.capture_mode_combo,
            self.allow_fallback_checkbox,
            self.whisper_path,
            self.output_path,
        ):
            widget.setEnabled(False)
        engine = self.engine

        def stop_worker() -> None:
            engine.stop()
            self.bridge.event.emit("engine_stopped", {})

        threading.Thread(target=stop_worker, name="gui-stop-engine", daemon=True).start()

    def _copy_transcript(self, feedback_button: QPushButton | None = None) -> None:
        clipboard = QApplication.clipboard()
        clipboard.setText(self.transcript_text.toPlainText())
        if feedback_button is not None:
            self._show_copy_feedback(feedback_button)

    def _show_copy_feedback(self, button: QPushButton) -> None:
        button_id = id(button)
        if button_id not in self._copy_button_originals:
            self._copy_button_originals[button_id] = (button.text(), button.icon(), button.styleSheet())
        token = self._copy_feedback_tokens.get(button_id, 0) + 1
        self._copy_feedback_tokens[button_id] = token
        button.setText("✓ Copied!")
        button.setStyleSheet("QPushButton { color: #2e7d32; }")

        def restore() -> None:
            if self._copy_feedback_tokens.get(button_id) != token:
                return
            original_text, original_icon, original_style = self._copy_button_originals.pop(button_id)
            self._copy_feedback_tokens.pop(button_id, None)
            button.setText(original_text)
            button.setIcon(original_icon)
            button.setStyleSheet(original_style)

        QTimer.singleShot(1500, restore)

    def _handle_engine_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "file_progress":
            processed, total = payload["processed"], payload["total"]
            text = f"{processed:.1f}s processed"
            if total:
                text = f"{min(processed, total):.1f}s / {total:.1f}s ({min(100, processed / total * 100):.0f}%)"
            self.file_progress_label.setText(text)
        elif event_type == "file_finished":
            self._finish_file_when_stopped(payload)
        elif event_type == "status":
            self.status_label.setText(str(payload.get("message", "")))
        elif event_type == "level":
            rms = float(payload.get("rms", 0.0))
            percent = payload.get("percent")
            self.level_meter.set_level(int(percent) if percent is not None else rms_to_meter_percent(rms))
        elif event_type == "lag":
            self._behind_seconds = float(payload.get("behind_seconds", 0.0))
            self._update_transcription_state(running=True)
        elif event_type == "transcript":
            self._append_transcript(payload["row"])
        elif event_type == "error":
            message = str(payload.get("message", ""))
            if message:
                log_path = self._write_error_log("Runtime error", message)
                self._append_log(f"{message}\nError details saved to {log_path}")
        elif event_type == "log":
            self._append_log(str(payload.get("message", "")))
        elif event_type == "engine_stopped":
            self._finish_engine_stop()

    def _append_transcript(self, row: TranscriptRow) -> None:
        text = row.text.strip()
        if not text:
            return
        cursor = self.transcript_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        existing = self.transcript_text.toPlainText()
        separator = "" if not existing or existing.endswith((" ", "\n")) else " "
        cursor.insertText(separator + text)
        self.transcript_text.setTextCursor(cursor)
        self.transcript_text.ensureCursorVisible()

    def _finish_file_when_stopped(self, payload: dict[str, Any]) -> None:
        if self.file_worker and self.file_worker.thread and self.file_worker.thread.is_alive():
            QTimer.singleShot(25, lambda: self._finish_file_when_stopped(payload))
            return
        self.file_worker = None
        self._stopping = False
        self._set_running(False)
        self.status_label.setText(f"File transcription {payload['outcome']}")
        if payload.get("output_path"):
            self._append_log(f"Transcript saved: {payload['output_path']}")
        if self._close_after_file:
            self.close()

    def _append_log(self, message: str) -> None:
        if message:
            self.log_box.append(message)

    def _tick_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        if self.engine or self._stopping:
            self._update_transcription_state(running=True)

    def _update_transcription_state(self, running: bool) -> None:
        if self.file_worker:
            self.transcription_state_label.setText("Transcribing saved recording locally")
            return
        if not running:
            self.transcription_state_label.setText("Idle, 0s behind")
            return
        frame = self._spinner_frames[self._spinner_index]
        self.transcription_state_label.setText(f"Transcribing, {int(round(self._behind_seconds))}s behind {frame}")

    def _write_error_log(self, context: str, message: str, details: Any | None = None) -> str:
        try:
            return str(append_error_log(application_root(), context, message, details))
        except Exception as exc:  # pragma: no cover - logging must not block UI
            return f"error_log.txt (failed to write: {exc})"

    def _set_running(self, running: bool) -> None:
        self.file_button.setEnabled(not running)
        self.record_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        self.pause_button.setText("Pause")
        if running:
            self._behind_seconds = 0.0
            self._update_transcription_state(running=True)
            if not self.spinner_timer.isActive():
                self.spinner_timer.start()
        else:
            self.spinner_timer.stop()
            self._update_transcription_state(running=False)
        if not running:
            self.level_meter.set_level(0)
        for widget in (
            self.device_combo,
            self.system_device_combo,
            self.capture_mode_combo,
            self.allow_fallback_checkbox,
            self.whisper_path,
            self.output_path,
        ):
            widget.setEnabled(not running)

    def _finish_engine_stop(self) -> None:
        self._behind_seconds = 0.0
        self._stopping = False
        self.engine = None
        self._set_running(False)
        self.status_label.setText("Stopped")

    def closeEvent(self, event: Any) -> None:  # pragma: no cover - GUI lifecycle
        if self.file_worker:
            self._close_after_file = True
            self._stop_recording()
            event.ignore()
            return
        if self.engine or self._stopping:
            self.status_label.setText("Finishing transcription")
            self._append_log("Finishing transcription before closing. Wait for Stopped.")
            if self.engine and not self._stopping:
                self._stop_recording()
            event.ignore()
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
