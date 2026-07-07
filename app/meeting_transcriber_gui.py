from __future__ import annotations

import sys
import threading
from typing import Any

from app_config import append_error_log, application_root, load_portable_config
from transcriber_engine import (
    DIARIZATION_BACKEND_LOCAL_ECAPA,
    DIARIZATION_BACKEND_PYANNOTE,
    EngineConfig,
    MeetingTranscriberEngine,
    TranscriptRow,
    format_timestamp,
    list_input_devices,
)


PORTABLE_DEFAULTS = load_portable_config()
WHISPER_MODEL_DIR = PORTABLE_DEFAULTS.whisper_model_dir
DIARIZATION_BACKEND = PORTABLE_DEFAULTS.diarization_backend
SPEAKER_EMBEDDING_MODEL_DIR = PORTABLE_DEFAULTS.speaker_embedding_model_dir
PYANNOTE_PIPELINE_DIR = PORTABLE_DEFAULTS.pyannote_pipeline_dir
PYANNOTE_EMBEDDING_MODEL_DIR = PORTABLE_DEFAULTS.pyannote_embedding_model_dir
OUTPUT_FILE = PORTABLE_DEFAULTS.output_file
SAMPLE_RATE = PORTABLE_DEFAULTS.sample_rate
CHUNK_SECONDS = PORTABLE_DEFAULTS.chunk_seconds
OVERLAP_SECONDS = PORTABLE_DEFAULTS.overlap_seconds
COMPUTE_TYPE = PORTABLE_DEFAULTS.compute_type


try:
    from PySide6.QtCore import QObject, Qt, Signal
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
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
    def __init__(self, segments: int = 30) -> None:
        super().__init__()
        self.segments = segments
        self.level = 0
        self.setMinimumHeight(24)

    def set_level(self, value: int) -> None:
        self.level = max(0, min(100, value))
        self.update()

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - visual widget
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        width = self.width()
        height = self.height()
        gap = 5
        bar_width = max(3, int((width - gap * (self.segments - 1)) / self.segments))
        bar_height = max(10, min(18, height - 6))
        top = int((height - bar_height) / 2)
        active_segments = round((self.level / 100) * self.segments)

        for index in range(self.segments):
            left = index * (bar_width + gap)
            if left + bar_width > width:
                break
            active = index < active_segments
            ratio = (index + 1) / self.segments
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
        self.bridge = EngineSignalBridge()
        self.bridge.event.connect(self._handle_engine_event)
        self.row_indexes: dict[str, int] = {}
        self.speaker_keys_by_item: dict[int, str] = {}

        self._build_ui()
        self._load_devices()
        self._set_running(False)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        top = QGridLayout()
        top.setColumnStretch(0, 2)
        top.setColumnStretch(1, 1)
        root_layout.addLayout(top)

        self.transcript_table = QTableWidget(0, 4)
        self.transcript_table.setHorizontalHeaderLabels(["Time", "Speaker", "Text", "State"])
        self.transcript_table.verticalHeader().setVisible(False)
        self.transcript_table.setAlternatingRowColors(True)
        self.transcript_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.transcript_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.transcript_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.transcript_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.transcript_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.transcript_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        top.addWidget(self.transcript_table, 0, 0, 3, 1)

        controls = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls)
        button_row = QHBoxLayout()
        self.record_button = QPushButton("Record")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.record_button.clicked.connect(self._start_recording)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop_recording)
        button_row.addWidget(self.record_button)
        button_row.addWidget(self.pause_button)
        button_row.addWidget(self.stop_button)
        controls_layout.addLayout(button_row)

        status_row = QFormLayout()
        self.status_label = QLabel("Idle")
        self.lag_label = QLabel("Diarization: 0 chunks behind")
        self.level_meter = MicLevelMeter()
        status_row.addRow("Status", self.status_label)
        status_row.addRow("Speaker sync", self.lag_label)
        status_row.addRow("Mic level", self.level_meter)
        controls_layout.addLayout(status_row)
        top.addWidget(controls, 0, 1)

        settings = QGroupBox("Settings")
        settings_layout = QFormLayout(settings)
        self.device_combo = QComboBox()
        self.whisper_path = QLineEdit(WHISPER_MODEL_DIR)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Local ECAPA (one-touch default)", DIARIZATION_BACKEND_LOCAL_ECAPA)
        self.backend_combo.addItem("Pyannote advanced", DIARIZATION_BACKEND_PYANNOTE)
        backend_index = self.backend_combo.findData(DIARIZATION_BACKEND)
        self.backend_combo.setCurrentIndex(max(0, backend_index))
        self.speaker_model_path = QLineEdit(SPEAKER_EMBEDDING_MODEL_DIR)
        self.pyannote_path = QLineEdit(PYANNOTE_PIPELINE_DIR)
        self.embedding_path = QLineEdit(PYANNOTE_EMBEDDING_MODEL_DIR)
        self.output_path = QLineEdit(OUTPUT_FILE)
        self.chunk_seconds = QSpinBox()
        self.chunk_seconds.setRange(3, 60)
        self.chunk_seconds.setValue(int(CHUNK_SECONDS))
        self.overlap_seconds = QSpinBox()
        self.overlap_seconds.setRange(0, 20)
        self.overlap_seconds.setValue(int(OVERLAP_SECONDS))

        settings_layout.addRow("Microphone", self.device_combo)
        settings_layout.addRow("Whisper", self._path_row(self.whisper_path, folder=True))
        settings_layout.addRow("Diarization", self.backend_combo)
        settings_layout.addRow("Speaker model", self._path_row(self.speaker_model_path, folder=True))
        self.pyannote_label = QLabel("Pyannote")
        self.pyannote_row = self._path_row(self.pyannote_path, folder=True)
        self.pyannote_embedding_label = QLabel("Pyannote embedding")
        self.pyannote_embedding_row = self._path_row(self.embedding_path, folder=True)
        settings_layout.addRow(self.pyannote_label, self.pyannote_row)
        settings_layout.addRow(self.pyannote_embedding_label, self.pyannote_embedding_row)
        settings_layout.addRow("Output", self._path_row(self.output_path, folder=False))
        settings_layout.addRow("Chunk seconds", self.chunk_seconds)
        settings_layout.addRow("Overlap seconds", self.overlap_seconds)
        self.backend_combo.currentIndexChanged.connect(self._toggle_backend_fields)
        self._toggle_backend_fields()
        top.addWidget(settings, 1, 1)

        speakers = QGroupBox("Speakers")
        speaker_layout = QVBoxLayout(speakers)
        self.speaker_list = QListWidget()
        rename_row = QHBoxLayout()
        self.speaker_name = QLineEdit()
        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self._rename_selected_speaker)
        rename_row.addWidget(self.speaker_name)
        rename_row.addWidget(self.rename_button)
        speaker_layout.addWidget(self.speaker_list)
        speaker_layout.addLayout(rename_row)
        top.addWidget(speakers, 2, 1)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(120)
        root_layout.addWidget(self.log_box)

    def _path_row(self, line_edit: QLineEdit, folder: bool) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse_path(line_edit, folder))
        layout.addWidget(line_edit)
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
        devices = list_input_devices()
        if not devices:
            self.device_combo.addItem("Default input", None)
            return
        self.device_combo.addItem("Default input", None)
        for device in devices:
            self.device_combo.addItem(f"{device['index']}: {device['name']}", device["index"])

    def _read_config(self) -> EngineConfig:
        return EngineConfig(
            whisper_model_dir=self.whisper_path.text().strip(),
            diarization_backend=str(self.backend_combo.currentData() or DIARIZATION_BACKEND_LOCAL_ECAPA),
            speaker_embedding_model_dir=self.speaker_model_path.text().strip(),
            speaker_cluster_distance_threshold=PORTABLE_DEFAULTS.speaker_cluster_distance_threshold,
            pyannote_pipeline_dir=self.pyannote_path.text().strip(),
            pyannote_embedding_model_dir=self.embedding_path.text().strip(),
            output_file=self.output_path.text().strip(),
            sample_rate=SAMPLE_RATE,
            chunk_seconds=float(self.chunk_seconds.value()),
            overlap_seconds=float(self.overlap_seconds.value()),
            compute_type=COMPUTE_TYPE,
            input_device=self.device_combo.currentData(),
        )

    def _toggle_backend_fields(self) -> None:
        pyannote_selected = self.backend_combo.currentData() == DIARIZATION_BACKEND_PYANNOTE
        for widget in (
            self.pyannote_label,
            self.pyannote_row,
            self.pyannote_embedding_label,
            self.pyannote_embedding_row,
        ):
            widget.setVisible(pyannote_selected)

    def _start_recording(self) -> None:
        config = self._read_config()
        errors = config.validate()
        if errors:
            message = "\n".join(errors)
            log_path = self._write_error_log("Configuration error", message, {"errors": errors})
            QMessageBox.critical(self, "Configuration error", f"{message}\n\nError details saved to {log_path}")
            return

        self.transcript_table.setRowCount(0)
        self.row_indexes.clear()
        self.speaker_list.clear()
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
        self._set_running(True)

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
        if not self.engine:
            return
        self._set_running(False)
        engine = self.engine
        self.engine = None
        threading.Thread(target=engine.stop, name="gui-stop-engine", daemon=True).start()

    def _rename_selected_speaker(self) -> None:
        if not self.engine:
            return
        item = self.speaker_list.currentItem()
        if item is None:
            return
        speaker_key = item.data(Qt.UserRole)
        self.engine.rename_speaker(speaker_key, self.speaker_name.text())

    def _handle_engine_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "status":
            self.status_label.setText(str(payload.get("message", "")))
        elif event_type == "level":
            rms = float(payload.get("rms", 0.0))
            self.level_meter.set_level(min(100, int(rms * 800)))
        elif event_type == "lag":
            self.lag_label.setText(str(payload.get("message", "")))
        elif event_type == "transcript":
            self._upsert_row(payload["row"])
        elif event_type == "speaker_update":
            self._upsert_row(payload["row"])
        elif event_type == "speakers":
            self._update_speakers(payload.get("speakers", []))
        elif event_type == "error":
            message = str(payload.get("message", ""))
            if message:
                log_path = self._write_error_log("Runtime error", message)
                self._append_log(f"{message}\nError details saved to {log_path}")
        elif event_type == "log":
            self._append_log(str(payload.get("message", "")))

    def _upsert_row(self, row: TranscriptRow) -> None:
        if row.id in self.row_indexes:
            index = self.row_indexes[row.id]
        else:
            index = self.transcript_table.rowCount()
            self.transcript_table.insertRow(index)
            self.row_indexes[row.id] = index
        values = [
            format_timestamp(row.start),
            row.speaker_label,
            row.text,
            "Speaker pending" if row.speaker_label == "Speaker ?" else "Speaker set",
        ]
        for column, value in enumerate(values):
            self.transcript_table.setItem(index, column, QTableWidgetItem(value))
        self.transcript_table.scrollToBottom()

    def _update_speakers(self, speakers: list[Any]) -> None:
        current_key = None
        current_item = self.speaker_list.currentItem()
        if current_item is not None:
            current_key = current_item.data(Qt.UserRole)
        self.speaker_list.clear()
        restore_row = -1
        for index, speaker in enumerate(speakers):
            item = QListWidgetItem(speaker.display_name_value)
            item.setData(Qt.UserRole, speaker.key)
            self.speaker_list.addItem(item)
            if speaker.key == current_key:
                restore_row = index
        if restore_row >= 0:
            self.speaker_list.setCurrentRow(restore_row)

    def _append_log(self, message: str) -> None:
        if message:
            self.log_box.append(message)

    def _write_error_log(self, context: str, message: str, details: Any | None = None) -> str:
        try:
            return str(append_error_log(application_root(), context, message, details))
        except Exception as exc:  # pragma: no cover - logging must not block UI
            return f"error_log.txt (failed to write: {exc})"

    def _set_running(self, running: bool) -> None:
        self.record_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        self.pause_button.setText("Pause")
        for widget in (
            self.device_combo,
            self.whisper_path,
            self.backend_combo,
            self.speaker_model_path,
            self.pyannote_path,
            self.embedding_path,
            self.output_path,
            self.chunk_seconds,
            self.overlap_seconds,
        ):
            widget.setEnabled(not running)

    def closeEvent(self, event: Any) -> None:  # pragma: no cover - GUI lifecycle
        if self.engine:
            self.status_label.setText("Stopping")
            engine = self.engine
            self.engine = None
            engine.stop()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
