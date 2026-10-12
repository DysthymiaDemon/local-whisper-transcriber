from __future__ import annotations

import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from file_transcriber import FileTranscriptionWorker

from app_config import (
    append_error_log,
    application_root,
    load_portable_config_safely,
    save_portable_config,
)
from offline_guard import enforce_offline
from transcriber_engine import (
    EngineConfig,
    MeetingTranscriberEngine,
    TranscriptRow,
    CAPTURE_MODE_LOOPBACK,
    CAPTURE_MODE_MIC,
    CAPTURE_MODE_MIXED,
    list_input_devices,
    list_output_devices,
    preload_whisper_model,
    rms_to_meter_percent,
    unload_whisper_model,
)
from ui_helpers import meter_bar_geometry


PORTABLE_DEFAULTS, CONFIG_LOAD_PROBLEM = load_portable_config_safely()
WHISPER_MODEL_DIR = PORTABLE_DEFAULTS.whisper_model_dir
OUTPUT_FILE = PORTABLE_DEFAULTS.output_file
SAMPLE_RATE = PORTABLE_DEFAULTS.sample_rate
CHUNK_SECONDS = PORTABLE_DEFAULTS.chunk_seconds
OVERLAP_SECONDS = PORTABLE_DEFAULTS.overlap_seconds
CAPTURE_MODE = PORTABLE_DEFAULTS.capture_mode
INPUT_DEVICE = PORTABLE_DEFAULTS.input_device
SYSTEM_DEVICE = PORTABLE_DEFAULTS.system_device_index
ALLOW_MIXED_FALLBACK = PORTABLE_DEFAULTS.allow_mixed_fallback
COMPUTE_TYPE = PORTABLE_DEFAULTS.compute_type
DEVICE = PORTABLE_DEFAULTS.device
CPU_THREADS = PORTABLE_DEFAULTS.cpu_threads
NUM_WORKERS = PORTABLE_DEFAULTS.num_workers
LANGUAGE = PORTABLE_DEFAULTS.language

MEETING_FILE_FILTER = (
    "Meeting recordings (*.mp4 *.m4a *.mp3 *.wav *.webm *.mkv *.mov *.aac *.wma);;All files (*.*)"
)
HOME_PAGE = 0
SESSION_PAGE = 1


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
        QStackedWidget,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
    from ui_theme import (
        APP_NAME,
        APP_STYLESHEET,
        GradientBackground,
        HoverIconButton,
        add_soft_shadow,
        draw_icon,
        TITLE_BLUE,
        MODEL_READY,
        MODEL_STARTING,
        MODEL_STOPPED,
        MODEL_STOPPING,
        ModelLightButton,
    )
except ImportError as exc:  # pragma: no cover - depends on local GUI deps
    print(
        "PySide6 is not installed. Install offline wheels from docs/offline_setup.md, "
        "then run this file again.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def session_output_path(configured: str, now: datetime | None = None) -> str:
    """Give each recording its own transcript file so earlier meetings are never overwritten."""
    path = Path(configured)
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    suffix = path.suffix or ".txt"
    return str(path.with_name(f"{path.stem}_{stamp}{suffix}"))


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
                color = QColor("#d8e3ef")
            elif ratio <= 0.60:
                color = QColor("#4ac11f")
            elif ratio <= 0.82:
                color = QColor("#f1c232")
            else:
                color = QColor("#d93025")
            painter.fillRect(left, top, bar_width, bar_height, color)
            painter.setPen(QPen(QColor("#9fb4cc"), 1))
            painter.drawRect(left, top, bar_width, bar_height)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(draw_icon("mic", TITLE_BLUE, 64))
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
        self._loaded_model_config: EngineConfig | None = None
        self.spinner_timer = QTimer(self)
        self.spinner_timer.setInterval(160)
        self.spinner_timer.timeout.connect(self._tick_spinner)

        self.pages = QStackedWidget()
        self.setCentralWidget(self.pages)
        self.pages.addWidget(self._build_home_page())
        self.pages.addWidget(self._build_session_page())
        self.pages.setCurrentIndex(HOME_PAGE)
        QTimer.singleShot(0, self._home_page.setFocus)

        self._load_devices()
        self._set_running(False)
        if CONFIG_LOAD_PROBLEM:
            log_path = self._write_error_log("Configuration file ignored", CONFIG_LOAD_PROBLEM)
            self._append_log(f"{CONFIG_LOAD_PROBLEM}. Using default settings. Details saved to {log_path}")

    def _build_home_page(self) -> QWidget:
        page = GradientBackground()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 22, 28, 28)

        header = QHBoxLayout()
        header.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(draw_icon("mic", TITLE_BLUE, 32).pixmap(32, 32))
        title = QLabel(APP_NAME)
        title.setObjectName("appTitle")
        header.addWidget(logo)
        header.addWidget(title)
        header.addStretch(1)
        layout.addLayout(header)
        layout.addStretch(2)

        self.home_record_button = HoverIconButton("Record and Transcribe This Meeting", "record")
        self.home_file_button = HoverIconButton("Transcribe Existing Meeting", "folder_mic")
        self.home_close_button = HoverIconButton("Close This App", "close")
        self.home_record_button.clicked.connect(self._open_recording_session)
        self.home_file_button.clicked.connect(self._open_existing_meeting)
        self.home_close_button.clicked.connect(self._close_app)

        column = QVBoxLayout()
        column.setSpacing(20)
        for button in (self.home_record_button, self.home_file_button, self.home_close_button):
            button.setObjectName("homeAction")
            button.setMinimumHeight(64)
            button.setMinimumWidth(460)
            button.setMaximumWidth(640)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            add_soft_shadow(button)
            column.addWidget(button)

        centered = QHBoxLayout()
        centered.addStretch(1)
        centered.addLayout(column, 6)
        centered.addStretch(1)
        layout.addLayout(centered)
        self.model_status_label = QLabel("")
        self.model_status_label.setObjectName("modelStatus")
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(18)
        layout.addWidget(self.model_status_label)
        layout.addStretch(3)
        # Keep keyboard access to the buttons without drawing a focus ring on one of them at start-up.
        page.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._home_page = page
        return page

    def _build_session_page(self) -> QWidget:
        root = QWidget()
        root.setObjectName("sessionPage")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        top_bar = QHBoxLayout()
        self.home_button = QPushButton("Home")
        self.home_button.setIcon(draw_icon("mic", TITLE_BLUE, 18))
        self.home_button.clicked.connect(self._show_home)
        session_title = QLabel(APP_NAME)
        session_title.setStyleSheet(f"color: {TITLE_BLUE}; font-size: 13pt; font-weight: 600;")
        top_bar.addWidget(self.home_button)
        top_bar.addSpacing(8)
        top_bar.addWidget(session_title)
        top_bar.addStretch(1)
        root_layout.addLayout(top_bar)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(main_splitter, 1)

        self.transcript_text = QTextEdit()
        self.transcript_text.setObjectName("transcript")
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setAcceptRichText(False)
        self.transcript_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.transcript_text.setPlaceholderText("Transcription appears here...")
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
        self.model_button = ModelLightButton()
        self.model_button.clicked.connect(self._toggle_model)
        controls_layout.addWidget(self.model_button)
        button_row = QHBoxLayout()
        self.record_button = QPushButton("Record")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.record_button.setObjectName("recordButton")
        self.pause_button.setObjectName("pauseButton")
        self.stop_button.setObjectName("stopButton")
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
        self.system_device_combo.currentIndexChanged.connect(self._update_capture_controls)
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

        settings_layout.addRow("Capture mode", self.capture_mode_combo)
        settings_layout.addRow("Microphone", self.device_combo)
        settings_layout.addRow("System output", self.system_device_combo)
        settings_layout.addRow("Mixed fallback", self.allow_fallback_checkbox)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("advancedToggle")
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.advanced_panel = QWidget()
        advanced_layout = QFormLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.addRow("Model folder", self._path_row(self.whisper_path, folder=True))
        advanced_layout.addRow("Transcript file", self._path_row(self.output_path, folder=False))
        self.advanced_panel.setVisible(False)
        settings_layout.addRow(self.advanced_toggle)
        settings_layout.addRow(self.advanced_panel)
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
        self.log_box.setObjectName("log")
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        root_layout.addWidget(self.log_box)
        return root

    def _toggle_model(self) -> None:
        if self.model_button.state == MODEL_STOPPED:
            self._start_model()
        elif self.model_button.state == MODEL_READY:
            self._stop_model()

    def _start_model(self) -> bool:
        """Load and warm up the speech model in the background (red -> yellow -> green)."""
        if self.model_button.state != MODEL_STOPPED:
            return self.model_button.state in (MODEL_STARTING, MODEL_READY)
        config = self._read_config()
        model_errors = [error for error in config.validate() if "model" in error.lower()]
        if model_errors:
            self._set_model_status("Speech model not found. Run setup or set Advanced > Model folder.")
            return False
        self._loaded_model_config = config
        self._set_model_state(MODEL_STARTING)
        self._set_model_status("Starting speech model...")

        def preload() -> None:
            started = time.monotonic()
            try:
                model = preload_whisper_model(config)
            except Exception as exc:
                self.bridge.event.emit(
                    "model_status", {"state": "failed", "message": f"Speech model failed to load: {exc}"}
                )
                return
            seconds = time.monotonic() - started
            where = "on the Intel GPU" if config.device.lower().startswith("openvino") else "on the CPU"
            fallback = getattr(model, "fallback_reason", "")
            if isinstance(fallback, str) and fallback:
                where = "on the CPU (Intel GPU unavailable)"
                self.bridge.event.emit("log", {"message": fallback})
            self.bridge.event.emit(
                "model_status",
                {"state": MODEL_READY, "message": f"Speech model ready {where} ({seconds:.0f}s to load)"},
            )

        threading.Thread(target=preload, name="model-start", daemon=True).start()
        return True

    def _stop_model(self) -> None:
        """Unload the speech model to free memory (green -> yellow -> red)."""
        if self.model_button.state != MODEL_READY or self._busy():
            return
        config = self._loaded_model_config or self._read_config()
        self._set_model_state(MODEL_STOPPING)
        self._set_model_status("Stopping speech model...")

        def unload() -> None:
            try:
                unload_whisper_model(config)
            finally:
                self.bridge.event.emit("model_status", {"state": MODEL_STOPPED, "message": "Speech model stopped"})

        threading.Thread(target=unload, name="model-stop", daemon=True).start()

    def _busy(self) -> bool:
        return bool(self.engine or self.file_worker or self._stopping)

    def _set_model_state(self, state: str) -> None:
        self.model_button.set_state(state)
        self._refresh_model_controls()

    def _refresh_model_controls(self) -> None:
        busy = self._busy()
        changing = self.model_button.state in (MODEL_STARTING, MODEL_STOPPING)
        self.model_button.setEnabled(not busy and not changing)
        # Record is allowed while the model starts (audio is buffered) but not while it is being unloaded.
        unloading = self.model_button.state == MODEL_STOPPING
        self.record_button.setEnabled(not busy and not unloading)
        self.file_button.setEnabled(not busy and not unloading)

    def _set_model_status(self, message: str) -> None:
        self.model_status_label.setText(message)
        self._append_log(message)

    def _toggle_advanced(self, expanded: bool) -> None:
        self.advanced_panel.setVisible(expanded)
        self.advanced_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def _open_recording_session(self) -> None:
        self.pages.setCurrentIndex(SESSION_PAGE)
        self._start_model()

    def _open_existing_meeting(self) -> None:
        if self._start_file_transcription():
            self.pages.setCurrentIndex(SESSION_PAGE)

    def _close_app(self) -> None:
        self.close()

    def _show_home(self) -> None:
        if self.engine or self.file_worker or self._stopping:
            return
        self.pages.setCurrentIndex(HOME_PAGE)
        self._home_page.setFocus()

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

    @staticmethod
    def _device_label(device: dict[str, Any]) -> str:
        rate = device.get("default_samplerate")
        hostapi = device.get("hostapi")
        suffix = f" - {hostapi}" if hostapi else ""
        rate_text = f" ({int(rate)} Hz)" if rate else ""
        return f"{device['index']}: {device['name']}{suffix}{rate_text}"

    def _load_devices(self) -> None:
        self.device_combo.clear()
        self.system_device_combo.clear()
        self._output_devices_by_index = {}
        try:
            devices = list_input_devices()
        except Exception as exc:
            devices = []
            self._append_log(f"Audio device discovery failed; file transcription remains available: {exc}")
        self.device_combo.addItem("Default input", None)
        for device in devices:
            self.device_combo.addItem(self._device_label(device), device["index"])

        try:
            output_devices = list_output_devices()
        except Exception as exc:
            output_devices = []
            self._append_log(f"Output device discovery failed: {exc}")
        self.system_device_combo.addItem("Default output", None)
        for device in output_devices:
            index = int(device["index"])
            self._output_devices_by_index[index] = device
            wasapi_suffix = " (WASAPI)" if device.get("wasapi") else ""
            self.system_device_combo.addItem(f"{self._device_label(device)}{wasapi_suffix}", index)
        self._set_capture_default_values()
        self._update_capture_controls()

    def _set_capture_default_values(self) -> None:
        self._set_combo_current_data(self.device_combo, INPUT_DEVICE)
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
        if not self._is_loopback_supported() and mode != CAPTURE_MODE_MIC:
            self.capture_mode_combo.blockSignals(True)
            self._set_combo_current_data(self.capture_mode_combo, CAPTURE_MODE_MIC)
            self.capture_mode_combo.blockSignals(False)
            mode = CAPTURE_MODE_MIC
            self._append_log("Loopback mode is unavailable because no WASAPI output device is detected.")

        self.system_device_combo.setEnabled(mode != CAPTURE_MODE_MIC)
        if mode in (CAPTURE_MODE_LOOPBACK, CAPTURE_MODE_MIXED) and not self._is_selected_system_device_wasapi():
            self.system_device_combo.setToolTip("Loopback and mixed modes require a WASAPI output device.")
        else:
            self.system_device_combo.setToolTip("")

        fallback_possible = mode == CAPTURE_MODE_MIXED and self._selected_system_device() is not None
        self.allow_fallback_checkbox.setEnabled(fallback_possible)

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

    def _save_settings(self, config: EngineConfig) -> None:
        try:
            save_portable_config(config, root=application_root())
        except Exception as exc:
            self._append_log(f"Settings could not be saved; recording continues: {exc}")

    def _start_recording(self) -> None:
        if self._stopping or self.engine or self.file_worker:
            return
        config = self._read_config()
        errors = config.validate()
        if errors:
            message = "\n".join(errors)
            log_path = self._write_error_log("Configuration error", message, {"errors": errors})
            QMessageBox.critical(self, "Configuration error", f"{message}\n\nError details saved to {log_path}")
            return
        self._save_settings(config)
        if self.model_button.state == MODEL_STOPPING:
            return
        # Starting the model here does not block: the engine buffers audio until the model is ready.
        self._start_model()

        session_config = replace(config, output_file=session_output_path(config.output_file))
        self.transcript_text.clear()
        self.engine = MeetingTranscriberEngine(session_config)
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
        self._append_log(f"Saving transcript to {session_config.output_file}")
        self._stopping = False
        self._set_running(True)

    def _start_file_transcription(self) -> bool:
        if self.engine or self.file_worker or self._stopping:
            return False
        source, _ = QFileDialog.getOpenFileName(
            self, "Select meeting recording", str(Path.home() / "Downloads"), MEETING_FILE_FILTER
        )
        if not source:
            return False
        output, _ = QFileDialog.getSaveFileName(
            self, "Save transcript", str(Path(source).with_suffix(".txt")), "Text files (*.txt)"
        )
        if not output:
            return False
        if Path(source).resolve() == Path(output).resolve():
            QMessageBox.critical(self, "Invalid output", "Output must differ from the source recording.")
            return False
        self._start_model()
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
        return True

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
        for widget in self._settings_widgets():
            widget.setEnabled(False)
        engine = self.engine

        def stop_worker() -> None:
            try:
                engine.stop()
            except Exception as exc:
                self.bridge.event.emit("error", {"message": f"Stopping transcription failed: {exc}"})
            finally:
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
        elif event_type == "model_status":
            message = str(payload.get("message", ""))
            state = payload.get("state")
            self._set_model_status(message)
            if state == MODEL_READY:
                self._set_model_state(MODEL_READY)
            else:
                self._set_model_state(MODEL_STOPPED)
                if state == "failed":
                    self._write_error_log("Speech model start", message)
        elif event_type == "fatal":
            self._set_model_state(MODEL_STOPPED)
            if self.engine and not self._stopping:
                self._stop_recording()
        elif event_type == "engine_stopped":
            self._finish_engine_stop()

    def _append_transcript(self, row: TranscriptRow) -> None:
        text = row.text.strip()
        if not text:
            return
        document = self.transcript_text.document()
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.End)
        separator = ""
        if not document.isEmpty():
            # QTextDocument stores line breaks as U+2029 paragraph separators.
            last_char = document.characterAt(document.characterCount() - 2)
            separator = "" if last_char in (" ", "\n", "\u2029") else " "
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

    def _settings_widgets(self) -> tuple[QWidget, ...]:
        return (
            self.device_combo,
            self.system_device_combo,
            self.capture_mode_combo,
            self.allow_fallback_checkbox,
            self.whisper_path,
            self.output_path,
        )

    def _set_running(self, running: bool) -> None:
        self.file_button.setEnabled(not running)
        self.record_button.setEnabled(not running)
        self.home_button.setEnabled(not running)
        self._refresh_model_controls()
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
            self.level_meter.set_level(0)
        for widget in self._settings_widgets():
            widget.setEnabled(not running)
        if not running:
            self._update_capture_controls()

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
            self.pages.setCurrentIndex(SESSION_PAGE)
            self.status_label.setText("Finishing transcription")
            self._append_log("Finishing transcription before closing. Wait for Stopped.")
            if self.engine and not self._stopping:
                self._stop_recording()
            event.ignore()
            return
        event.accept()


def main() -> int:
    enforce_offline()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
