from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

# pythonw.exe starts .pyw files without console streams; tqdm expects writable streams.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from app_config import append_error_log, application_root, error_log_root, source_root


REQUIRED_IMPORTS: dict[str, str] = {
    "PySide6": "PySide6",
    "sounddevice": "sounddevice",
    "faster-whisper": "faster_whisper",
    "speechbrain": "speechbrain",
    "scikit-learn": "sklearn",
    "huggingface_hub": "huggingface_hub",
    "truststore": "truststore",
    "torch": "torch",
}

DEFAULT_MODEL_DIRS = ("faster-whisper", "speechbrain-ecapa")
OPTIONAL_MODEL_DIRS = ("pyannote-pipeline", "pyannote-embedding")
MODEL_DIRS = DEFAULT_MODEL_DIRS + OPTIONAL_MODEL_DIRS
MODEL_GUIDES: dict[str, str] = {
    "faster-whisper": (
        "Default setup downloads Systran/faster-whisper-small here.\n\n"
        "Required file:\n"
        "- model.bin\n\n"
        "Example source model: Systran/faster-whisper-small\n"
    ),
    "speechbrain-ecapa": (
        "Default setup downloads the non-gated SpeechBrain ECAPA speaker model here.\n\n"
        "Required files:\n"
        "- hyperparams.yaml\n"
        "- embedding_model.ckpt or model.ckpt\n\n"
        "Example source model: speechbrain/spkrec-ecapa-voxceleb\n"
    ),
    "pyannote-pipeline": (
        "Optional advanced backend only. Copy the local pyannote diarization pipeline here.\n\n"
        "Required file:\n"
        "- config.yaml\n\n"
        "The config.yaml must reference local model paths only.\n"
    ),
    "pyannote-embedding": (
        "Optional advanced backend only. Copy the local pyannote embedding model here.\n\n"
        "Required files:\n"
        "- config.yaml\n"
        "- pytorch_model.bin or model.safetensors\n"
    ),
}
SETUP_MARKER = ".setup_complete"
IMPORTANT_MESSAGE_SECONDS = 5
RUNTIME_DIR = ".runtime"
PACKAGE_DIR = "site-packages"
RUNTIME_ENV = "LOCAL_WHISPER_RUNTIME_ROOT"
RUNTIME_APP_FOLDER_NAME = "OfflineMeetingTranscriberRuntime"
APP_PUBLISHER = "Ameen Khan"
APP_VERSION = "local"
HF_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
HF_PROGRESS_ENV_VARS = ("HF_HUB_DISABLE_PROGRESS_BARS",)
CA_BUNDLE_ENV = "LOCAL_WHISPER_CA_BUNDLE"
TLS_CERT_ENV_VARS = ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE")
CA_BUNDLE_FILENAMES = ("company-ca.pem", "corporate-ca.pem", "ca-bundle.pem")
WINDOWS_CA_BUNDLE_NAME = "windows-ca-bundle.pem"
INSTALL_LOG_FILE_NAME = "install_log.txt"
SETUP_PROGRESS_STYLE = "Setup.Horizontal.TProgressbar"
DETAIL_PROGRESS_STYLE = "Detail.Horizontal.TProgressbar"
COMPLETE_PROGRESS_STYLE = "Complete.Horizontal.TProgressbar"
STEP_WEIGHTS = {
    "Checking Python runtime": 2,
    "Preparing app folders": 2,
    "Checking Python packages": 3,
    "Installing Python packages": 40,
    "Downloading AI models": 50,
    "Checking local model folders": 1,
    "Finishing setup": 1,
    "Launching transcriber": 1,
}
MODEL_DOWNLOAD_POLL_SECONDS = 0.25

BOOTSTRAP_STEPS = [
    "Checking Python runtime",
    "Preparing app folders",
    "Checking Python packages",
    "Installing Python packages",
    "Downloading AI models",
    "Checking local model folders",
    "Finishing setup",
    "Launching transcriber",
]


class BootstrapStatus(StrEnum):
    READY = "ready"
    MISSING = "missing"


class StepState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class PipProgressEvent:
    phase: str
    detail: str
    progress_percent: int | None = None
    raw_line: str = ""


@dataclass
class StepDiagnostic:
    name: str
    state: StepState = StepState.PENDING
    detail: str = ""
    command: str = ""
    last_output: str = ""
    error: str = ""
    start_time: float | None = None
    end_time: float | None = None


@dataclass(frozen=True)
class PipInstallResult:
    code: int
    command: str
    recent_output: list[str]


@dataclass(frozen=True)
class ModelDownloadSpec:
    name: str
    repo_id: str
    target_subdir: str


DEFAULT_MODEL_DOWNLOADS = (
    ModelDownloadSpec(
        name="faster-whisper",
        repo_id="Systran/faster-whisper-small",
        target_subdir="models/faster-whisper",
    ),
    ModelDownloadSpec(
        name="speechbrain-ecapa",
        repo_id="speechbrain/spkrec-ecapa-voxceleb",
        target_subdir="models/speechbrain-ecapa",
    ),
)


def setup_install_summary(package_root: Path | None = None) -> str:
    package_lines = "\n".join(f"- {package}" for package in setup_package_list(package_root))
    model_lines = "\n".join(f"- {model}" for model in setup_model_list())
    return (
        "Install Offline Meeting Transcriber?\n"
        "Local Windows App\n"
        f"Publisher: {APP_PUBLISHER}\n"
        f"Version: {APP_VERSION}\n\n"
        "Models to install locally:\n"
        f"{model_lines}\n\n"
        "Python packages to install locally:\n"
        f"{package_lines}\n\n"
        "Install location:\n"
        "- App files, config, models, transcripts: folder beside this launcher\n"
        "- Python dependencies: app runtime folder"
    )


def setup_model_list() -> list[str]:
    return [f"{spec.repo_id} -> {spec.target_subdir.replace('/', os.sep)}" for spec in DEFAULT_MODEL_DOWNLOADS]


def setup_package_list(package_root: Path | None = None) -> list[str]:
    package_root = package_root or installer_source_root()
    requirements_path = resource_root(package_root) / "requirements.txt"
    if not requirements_path.exists():
        return ["Python package requirements"]
    packages: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            packages.append(line)
    return packages or ["Python package requirements"]


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "0s"
    seconds = max(0, int(seconds))
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {remaining}s"
    if minutes:
        return f"{minutes}m {remaining}s"
    return f"{remaining}s"


def format_clock_duration(seconds: float | None) -> str:
    seconds = max(0, int(seconds or 0))
    minutes, remaining = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining:02d}"
    return f"{minutes}:{remaining:02d}"


def format_bytes(value: int | float | None) -> str:
    size = max(0.0, float(value or 0))
    units = ("B", "KB", "MB", "GB", "TB")
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    if index == 0:
        return f"{int(size)} B"
    if size >= 100 or size.is_integer():
        return f"{int(size)} {units[index]}"
    return f"{size:.1f} {units[index]}"


def format_model_download_status(
    filename: str,
    downloaded_bytes: int,
    total_bytes: int | None,
    elapsed_seconds: float,
) -> str:
    current = format_bytes(downloaded_bytes)
    elapsed = format_clock_duration(elapsed_seconds)
    if total_bytes and total_bytes > 0:
        return f"{filename} \u2022 {current} / {format_bytes(total_bytes)} \u2022 {elapsed} elapsed"
    return f"{filename} \u2022 {current} downloaded \u2022 {elapsed} elapsed"


def folder_size_bytes(folder: Path) -> int:
    if not folder.exists():
        return 0
    total = 0
    for path in folder.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def largest_file_name(folder: Path, fallback: str) -> str:
    largest_name = fallback
    largest_size = -1
    if not folder.exists():
        return largest_name
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > largest_size:
            largest_size = size
            largest_name = path.name
    return largest_name


def _metadata_size(value: Any) -> int | None:
    if isinstance(value, Mapping):
        size = value.get("size")
    else:
        size = getattr(value, "size", None)
    return int(size) if isinstance(size, (int, float)) and size > 0 else None


def model_info_total_size(info: Any) -> int | None:
    total = 0
    found = False
    for sibling in getattr(info, "siblings", []) or []:
        size = getattr(sibling, "size", None)
        if not isinstance(size, (int, float)) or size <= 0:
            size = _metadata_size(getattr(sibling, "lfs", None))
        if isinstance(size, (int, float)) and size > 0:
            total += int(size)
            found = True
    return total if found else None


def fetch_model_repo_size(repo_id: str, root: Path | None = None) -> int | None:
    try:
        configure_download_tls(root)
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception:
        return None
    return model_info_total_size(info)


def estimate_model_download_sizes(
    specs: tuple[ModelDownloadSpec, ...] = DEFAULT_MODEL_DOWNLOADS,
    root: Path | None = None,
) -> dict[str, int | None]:
    return {spec.name: fetch_model_repo_size(spec.repo_id, root) for spec in specs}


def total_known_size(sizes: Mapping[str, int | None]) -> int | None:
    values = [size for size in sizes.values() if isinstance(size, int) and size > 0]
    return sum(values) if values and len(values) == len(sizes) else None


def weighted_overall_progress(states: Mapping[str, StepState], progress: Mapping[str, int]) -> int:
    total_weight = sum(STEP_WEIGHTS.values())
    weighted = 0.0
    for step in BOOTSTRAP_STEPS:
        weight = STEP_WEIGHTS[step]
        state = states.get(step, StepState.PENDING)
        if state in (StepState.DONE, StepState.WARNING):
            step_progress = 100
        elif state == StepState.RUNNING:
            step_progress = max(0, min(100, int(progress.get(step, 0))))
        elif int(progress.get(step, 0)) > 0:
            step_progress = max(0, min(100, int(progress.get(step, 0))))
        else:
            step_progress = 0
        weighted += weight * (step_progress / 100)
    return max(0, min(100, int(weighted / total_weight * 100)))


def install_log_path(root: Path | str | None = None) -> Path:
    app_root = Path(root).expanduser().resolve() if root else application_root()
    log_root = error_log_root(app_root)
    log_root.mkdir(parents=True, exist_ok=True)
    return log_root / INSTALL_LOG_FILE_NAME


def _append_log_details(lines: list[str], details: Mapping[str, Any] | Iterable[str] | str | None) -> None:
    if not details:
        return
    lines.append("details:")
    if isinstance(details, Mapping):
        for key, value in details.items():
            if isinstance(value, (list, tuple)):
                lines.append(f"  {key}:")
                for item in value:
                    lines.append(f"    {item}")
            else:
                lines.append(f"  {key}: {value}")
    elif isinstance(details, str):
        lines.append(details)
    else:
        for item in details:
            lines.append(f"  {item}")


def reset_install_log(root: Path | str | None = None, message: str = "Setup started") -> Path:
    app_root = Path(root).expanduser().resolve() if root else application_root()
    log_path = install_log_path(app_root)
    lines = [
        "=" * 72,
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"app_root: {app_root}",
        f"message: {message}",
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def append_install_log(
    root: Path | str | None,
    context: str,
    message: str,
    details: Mapping[str, Any] | Iterable[str] | str | None = None,
) -> Path:
    app_root = Path(root).expanduser().resolve() if root else application_root()
    log_path = install_log_path(app_root)
    lines = [
        "-" * 72,
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"context: {context}",
        f"message: {message}",
    ]
    _append_log_details(lines, details)
    lines.append("")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return log_path


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SIZE_UNIT_RE = re.compile(
    r"(?P<current>\d+(?:\.\d+)?)\s*(?P<current_unit>[KMGTPE]?i?B|[KMGTPE]?B|[KMGTPE])?"
    r"\s*/\s*"
    r"(?P<total>\d+(?:\.\d+)?)\s*(?P<total_unit>[KMGTPE]?i?B|[KMGTPE]?B|[KMGTPE])",
    re.IGNORECASE,
)
_UNIT_FACTORS = {
    "": 1.0,
    "B": 1.0,
    "K": 1024.0,
    "KB": 1024.0,
    "KIB": 1024.0,
    "M": 1024.0**2,
    "MB": 1024.0**2,
    "MIB": 1024.0**2,
    "G": 1024.0**3,
    "GB": 1024.0**3,
    "GIB": 1024.0**3,
    "T": 1024.0**4,
    "TB": 1024.0**4,
    "TIB": 1024.0**4,
    "P": 1024.0**5,
    "PB": 1024.0**5,
    "PIB": 1024.0**5,
    "E": 1024.0**6,
    "EB": 1024.0**6,
    "EIB": 1024.0**6,
}


def _clean_cli_progress_line(line: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", line).strip()


def _unit_factor(unit: str | None, fallback: str | None = None) -> float:
    normalized = (unit or fallback or "").upper()
    return _UNIT_FACTORS.get(normalized, 1.0)


def parse_cli_download_percent(line: str) -> int | None:
    clean = _clean_cli_progress_line(line)
    percent = re.search(r"(?<![\d.])(\d{1,3})\s*%", clean)
    if percent:
        return max(0, min(100, int(percent.group(1))))

    size = _SIZE_UNIT_RE.search(clean)
    if not size:
        return None
    total_unit = size.group("total_unit")
    current_unit = size.group("current_unit") or total_unit
    current = float(size.group("current")) * _unit_factor(current_unit, total_unit)
    total = float(size.group("total")) * _unit_factor(total_unit)
    if total <= 0:
        return None
    return max(0, min(100, int((current / total) * 100)))


def cli_download_progress_event(line: str, phase: str, detail_prefix: str = "") -> PipProgressEvent | None:
    clean = _clean_cli_progress_line(line)
    if not clean:
        return None
    progress = parse_cli_download_percent(clean)
    if progress is None:
        return None
    detail = clean
    if detail_prefix and detail_prefix not in detail:
        detail = f"{detail_prefix}: {detail}"
    return PipProgressEvent(phase, detail, progress, line)


def iter_cli_progress_output(stream) -> Iterable[str]:
    if not hasattr(stream, "read"):
        for line in stream:
            yield line.rstrip("\r\n")
        return

    buffer: list[str] = []
    while True:
        char = stream.read(1)
        if char == "":
            break
        if char in ("\r", "\n"):
            if buffer:
                yield "".join(buffer).rstrip()
                buffer = []
        else:
            buffer.append(char)
    if buffer:
        yield "".join(buffer).rstrip()


def build_step_tooltip(diagnostic: StepDiagnostic, now: float | None = None) -> str:
    current = time.time() if now is None else now
    elapsed_end = diagnostic.end_time if diagnostic.end_time is not None else current
    started = diagnostic.start_time if diagnostic.start_time is not None else elapsed_end
    lines = [
        diagnostic.name,
        f"Status: {diagnostic.state.value}",
        f"Elapsed: {format_duration(elapsed_end - started)}",
    ]
    if diagnostic.detail:
        lines.append(f"Detail: {diagnostic.detail}")
    if diagnostic.command:
        lines.append(f"Command: {diagnostic.command}")
    if diagnostic.last_output:
        lines.append(f"Last output: {diagnostic.last_output}")
    if diagnostic.error:
        lines.append(f"Error: {diagnostic.error}")
    return "\n".join(lines)


class PipInstallProgressTracker:
    def __init__(self, start_time: float | None = None) -> None:
        self.start_time = time.time() if start_time is None else start_time
        self.event_count = 0
        self.progress = 0
        self.installing_started = False
        self.installing_message = ""

    def record(self, event: PipProgressEvent, now: float | None = None) -> tuple[int, str]:
        self.event_count += 1
        current = time.time() if now is None else now
        if event.progress_percent == 100:
            self.progress = 100
        elif event.progress_percent == 0:
            self.installing_started = True
            self.installing_message = event.raw_line.strip() or f"Installing collected packages: {event.detail}"
            self.progress = max(self.progress, 80)
        elif event.progress_percent is not None:
            self.progress = max(self.progress, event.progress_percent)
        else:
            cap = 92 if self.installing_started else 95
            self.progress = min(cap, max(self.progress + 3, 8 + self.event_count * 4))

        elapsed = current - self.start_time
        if self.progress >= 100:
            eta = "ETA complete"
        elif self.installing_started:
            eta = self.installing_message or "Installing collected packages..."
        elif self.event_count < 3 or self.progress < 10:
            eta = "ETA estimating..."
        else:
            remaining = elapsed * ((100 - self.progress) / self.progress)
            eta = f"ETA ~{format_duration(remaining)}"
        return self.progress, f"Elapsed {format_duration(elapsed)} | {eta}"


class PipProgressParser:
    def __init__(self) -> None:
        self.install_total = 0

    def parse(self, line: str) -> PipProgressEvent:
        stripped = line.strip()
        if not stripped:
            return PipProgressEvent("Running pip", "", None, line)

        download_progress = cli_download_progress_event(stripped, "Downloading package")
        if download_progress is not None:
            return download_progress

        already = re.match(r"Requirement already satisfied:\s+(.+?)(?:\s+in\s+.+)?$", stripped)
        if already:
            return PipProgressEvent("Already installed", already.group(1), 100, line)

        collecting = re.match(r"Collecting\s+(.+)$", stripped)
        if collecting:
            return PipProgressEvent("Resolving package", collecting.group(1), None, line)

        downloading = re.match(r"Downloading\s+(.+)$", stripped)
        if downloading:
            return PipProgressEvent("Downloading package", downloading.group(1), 0, line)

        installing = re.match(r"Installing collected packages:\s+(.+)$", stripped)
        if installing:
            packages = [item.strip() for item in installing.group(1).split(",") if item.strip()]
            self.install_total = len(packages)
            return PipProgressEvent("Installing packages", ", ".join(packages[:5]), 0, line)

        installed = re.match(r"Successfully installed\s+(.+)$", stripped)
        if installed:
            packages = [item.strip() for item in installed.group(1).split() if item.strip()]
            detail = f"{len(packages)} packages installed" if packages else "Packages installed"
            return PipProgressEvent("Install complete", detail, 100, line)

        if re.search(r"\d+(\.\d+)?/\d+(\.\d+)?\s+[kMG]?B", stripped):
            return PipProgressEvent("Downloading package", stripped, None, line)

        return PipProgressEvent("Running pip", stripped, None, line)


class ToolTip:
    def __init__(self, widget, text_callback: Callable[[], str]) -> None:
        self.widget = widget
        self.text_callback = text_callback
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
        widget.bind("<Motion>", self.move)

    def show(self, event=None) -> None:
        del event
        if self.tip is not None:
            return
        text = self.text_callback()
        if not text:
            return
        import tkinter as tk

        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        label = tk.Label(
            self.tip,
            text=text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=6,
            pady=4,
        )
        label.pack()
        self.move()

    def move(self, event=None) -> None:
        del event
        if self.tip is None:
            return
        x = self.widget.winfo_pointerx() + 14
        y = self.widget.winfo_pointery() + 12
        self.tip.wm_geometry(f"+{x}+{y}")

    def hide(self, event=None) -> None:
        del event
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


def runtime_root() -> Path:
    return application_root()


def installer_source_root() -> Path:
    return source_root()


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        resolved_parent = parent.expanduser().resolve()
        resolved_child = child.expanduser().resolve()
    except OSError:
        return False
    return resolved_child == resolved_parent or resolved_parent in resolved_child.parents


def is_onedrive_path(path: Path) -> bool:
    for env_name in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        env_value = os.environ.get(env_name)
        if env_value and _path_contains(Path(env_value), path):
            return True
    return any("onedrive" in part.lower() for part in path.expanduser().parts)


def local_runtime_dir(root: Path) -> Path:
    configured = os.environ.get(RUNTIME_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    if is_onedrive_path(root):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return (Path(local_app_data) / RUNTIME_APP_FOLDER_NAME).resolve()
        return (Path.home() / "AppData" / "Local" / RUNTIME_APP_FOLDER_NAME).resolve()
    return root / RUNTIME_DIR


def local_package_dir(root: Path) -> Path:
    return local_runtime_dir(root) / PACKAGE_DIR


def migrate_legacy_onedrive_runtime(root: Path) -> Path | None:
    legacy_runtime = root / RUNTIME_DIR
    target_runtime = local_runtime_dir(root)
    if not is_onedrive_path(root) or target_runtime == legacy_runtime or not legacy_runtime.exists():
        return None
    if not target_runtime.exists():
        target_runtime.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy_runtime), str(target_runtime))
        return target_runtime

    backup = target_runtime / "legacy-runtime"
    suffix = 1
    while backup.exists():
        suffix += 1
        backup = target_runtime / f"legacy-runtime-{suffix}"
    shutil.move(str(legacy_runtime), str(backup))
    return backup


def app_source_dir(root: Path | None = None) -> Path:
    base = (root or installer_source_root()).resolve()
    candidate = base / "app"
    return candidate if candidate.is_dir() else base


def resource_root(root: Path | None = None) -> Path:
    base = (root or installer_source_root()).resolve()
    candidate = base / "resources"
    return candidate if candidate.is_dir() else base


def missing_imports(required: Mapping[str, str] = REQUIRED_IMPORTS) -> list[str]:
    missing: list[str] = []
    for label, module_name in required.items():
        try:
            if importlib.util.find_spec(module_name) is None:
                missing.append(label)
        except ModuleNotFoundError:
            missing.append(label)
    return missing


def missing_runtime_imports(root: Path, required: Mapping[str, str] = REQUIRED_IMPORTS) -> list[str]:
    if getattr(sys, "frozen", False):
        return missing_imports(required)

    package_path = local_package_dir(root)
    if not package_path.exists():
        return list(required.keys())

    probe = (
        "import importlib.util, json, sys; "
        f"required = {json.dumps(dict(required))}; "
        "missing = [label for label, module in required.items() if importlib.util.find_spec(module) is None]; "
        "print(json.dumps(missing)); "
        "sys.exit(1 if missing else 0)"
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(package_path) if not existing_pythonpath else str(package_path) + os.pathsep + existing_pythonpath
    result = subprocess.run(
        [sys.executable, "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    try:
        parsed = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return list(required.keys())
    return [item for item in parsed if isinstance(item, str)]


def ensure_portable_layout(root: Path, template_root: Path | None = None) -> Path:
    migrate_legacy_onedrive_runtime(root)
    local_runtime_dir(root).mkdir(parents=True, exist_ok=True)
    models = root / "models"
    for name in MODEL_DIRS:
        model_dir = models / name
        model_dir.mkdir(parents=True, exist_ok=True)
        guide = model_dir / "README_MODEL_FILES.txt"
        if not guide.exists():
            guide.write_text(MODEL_GUIDES[name], encoding="utf-8")
    (root / "transcripts").mkdir(parents=True, exist_ok=True)

    config_path = root / "config.json"
    if not config_path.exists():
        template_path = resource_root(template_root or root) / "config.template.json"
        if template_path.exists():
            config_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            config_path.write_text(
                json.dumps(
                    {
                        "whisper_model_dir": "models/faster-whisper",
                        "diarization_backend": "local-ecapa",
                        "speaker_embedding_model_dir": "models/speechbrain-ecapa",
                        "speaker_cluster_distance_threshold": 0.55,
                        "pyannote_pipeline_dir": "models/pyannote-pipeline",
                        "pyannote_embedding_model_dir": "models/pyannote-embedding",
                        "output_file": "transcripts/meeting_transcript.txt",
                        "sample_rate": 16000,
                        "chunk_seconds": 8.0,
                        "overlap_seconds": 2.0,
                        "compute_type": "int8",
                        "speaker_match_threshold": 0.7,
                        "min_speaker_confidence": 0.5,
                        "min_overlap_ratio": 0.35,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return config_path


def _model_folder_ready(root: Path, name: str) -> bool:
    return _model_folder_ready_at(root / "models" / name, name)


def _model_folder_ready_at(folder: Path, name: str) -> bool:
    if name == "faster-whisper":
        return (folder / "model.bin").is_file()
    if name == "speechbrain-ecapa":
        return (folder / "hyperparams.yaml").is_file() and any(
            (folder / filename).is_file() for filename in ("embedding_model.ckpt", "model.ckpt")
        )
    if name == "pyannote-pipeline":
        return (folder / "config.yaml").is_file()
    if name == "pyannote-embedding":
        return (folder / "config.yaml").is_file() and any(
            (folder / filename).is_file() for filename in ("pytorch_model.bin", "model.safetensors")
        )
    return False


def model_folder_status(root: Path, include_optional: bool = False) -> dict[str, BootstrapStatus]:
    status: dict[str, BootstrapStatus] = {}
    model_names = MODEL_DIRS if include_optional else DEFAULT_MODEL_DIRS
    for name in model_names:
        ready = _model_folder_ready(root, name)
        status[name] = BootstrapStatus.READY if ready else BootstrapStatus.MISSING
    return status


def missing_model_setup_message(missing_models: list[str]) -> str:
    joined = "\n".join(missing_models)
    return (
        "Setup stopped because required local AI model files are missing after download.\n\n"
        "Missing model folders:\n"
        f"{joined}\n\n"
        "Recording cannot start until these model folders contain the expected files."
    )


def setup_error_card_message(message: str) -> str:
    if "CERTIFICATE_VERIFY_FAILED" in message.upper():
        return (
            "Your corporate network is intercepting HTTPS, so the model download certificate cannot be verified.\n\n"
            "Fix: export your proxy/root CA certificate, save it as company-ca.pem beside the launcher, "
            "then click Retry this step."
        )
    for line in message.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return "Setup failed. See details below."


def find_local_ca_bundle(root: Path | None = None) -> Path | None:
    configured = os.environ.get(CA_BUNDLE_ENV)
    if configured:
        candidate = Path(configured).expanduser()
        return candidate.resolve() if candidate.is_file() else None

    roots: list[Path] = []
    if root is not None:
        roots.extend([error_log_root(root), root, root.parent])
    else:
        roots.append(error_log_root())

    seen: set[Path] = set()
    for base in roots:
        resolved_base = base.expanduser().resolve()
        if resolved_base in seen:
            continue
        seen.add(resolved_base)
        for filename in CA_BUNDLE_FILENAMES:
            candidate = resolved_base / filename
            if candidate.is_file():
                return candidate.resolve()
    return None


def read_certifi_bundle_pem() -> str:
    try:
        import certifi

        return Path(certifi.where()).read_text(encoding="utf-8")
    except Exception:
        return ""


def build_windows_ca_bundle_pem(
    base_pem: str,
    enum_certificates: Callable[[str], Iterable[tuple[bytes, str, Any]]] | None = None,
    der_to_pem: Callable[[bytes], str] | None = None,
) -> str:
    enum_certificates = enum_certificates or getattr(ssl, "enum_certificates", None)
    der_to_pem = der_to_pem or ssl.DER_cert_to_PEM_cert
    if enum_certificates is None:
        return base_pem

    parts: list[str] = [base_pem.rstrip(), ""]
    seen: set[str] = set()
    for store_name in ("ROOT", "CA"):
        try:
            certificates = enum_certificates(store_name)
        except Exception:
            continue
        for cert_bytes, encoding, _trust in certificates:
            if encoding != "x509_asn":
                continue
            try:
                pem = der_to_pem(cert_bytes).strip()
            except Exception:
                continue
            if pem and pem not in seen:
                seen.add(pem)
                parts.append(pem)
    return "\n".join(part for part in parts if part) + "\n"


def ensure_windows_ca_bundle(root: Path | None = None) -> Path | None:
    if root is None:
        return None
    base_pem = read_certifi_bundle_pem()
    combined_pem = build_windows_ca_bundle_pem(base_pem)
    if not combined_pem.strip():
        return None
    bundle = local_runtime_dir(root) / WINDOWS_CA_BUNDLE_NAME
    bundle.parent.mkdir(parents=True, exist_ok=True)
    current = bundle.read_text(encoding="utf-8") if bundle.exists() else None
    if current != combined_pem:
        bundle.write_text(combined_pem, encoding="utf-8")
    return bundle


def configure_download_tls(root: Path | None = None) -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass

    bundle = find_local_ca_bundle(root)
    if bundle is None:
        bundle = ensure_windows_ca_bundle(root)
    if bundle is None:
        return
    for name in TLS_CERT_ENV_VARS:
        os.environ.setdefault(name, str(bundle))


@contextmanager
def online_huggingface_download_env(root: Path | None = None):
    previous = {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS + HF_PROGRESS_ENV_VARS + TLS_CERT_ENV_VARS}
    try:
        for name in HF_OFFLINE_ENV_VARS:
            os.environ.pop(name, None)
        for name in HF_PROGRESS_ENV_VARS:
            os.environ[name] = "1"
        configure_download_tls(root)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class CliDownloadProgressStream:
    def __init__(
        self,
        on_event: Callable[[PipProgressEvent], None],
        phase: str,
        detail_prefix: str = "",
        wrapped=None,
        progress_mapper: Callable[[PipProgressEvent], PipProgressEvent] | None = None,
    ) -> None:
        self.on_event = on_event
        self.phase = phase
        self.detail_prefix = detail_prefix
        self.wrapped = wrapped
        self.progress_mapper = progress_mapper
        self.buffer: list[str] = []
        self.encoding = getattr(wrapped, "encoding", "utf-8")
        self.errors = getattr(wrapped, "errors", "replace")

    def write(self, value: str) -> int:
        if self.wrapped is not None:
            self.wrapped.write(value)
        for char in value:
            if char in ("\r", "\n"):
                self.emit_buffer()
            else:
                self.buffer.append(char)
        return len(value)

    def flush(self) -> None:
        self.emit_buffer()
        if self.wrapped is not None:
            self.wrapped.flush()

    def isatty(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def emit_buffer(self) -> None:
        if not self.buffer:
            return
        line = "".join(self.buffer).rstrip()
        self.buffer = []
        event = cli_download_progress_event(line, self.phase, self.detail_prefix)
        if event is None:
            return
        if self.progress_mapper is not None:
            event = self.progress_mapper(event)
        self.on_event(event)


@contextmanager
def redirect_cli_download_progress(
    on_event: Callable[[PipProgressEvent], None],
    phase: str,
    detail_prefix: str = "",
    progress_mapper: Callable[[PipProgressEvent], PipProgressEvent] | None = None,
):
    previous_stdout = sys.stdout
    previous_stderr = sys.stderr
    stream = CliDownloadProgressStream(on_event, phase, detail_prefix, previous_stderr, progress_mapper)
    try:
        sys.stdout = CliDownloadProgressStream(on_event, phase, detail_prefix, previous_stdout, progress_mapper)
        sys.stderr = stream
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = previous_stdout
        sys.stderr = previous_stderr


def _model_folder_preview(folder: Path, limit: int = 12) -> str:
    if not folder.exists():
        return "<folder does not exist>"
    entries = []
    for item in sorted(folder.iterdir(), key=lambda path: path.name.lower()):
        suffix = "/" if item.is_dir() else ""
        entries.append(f"{item.name}{suffix}")
        if len(entries) >= limit:
            break
    return ", ".join(entries) if entries else "<empty>"


def model_download_staging_dir(root: Path, spec: ModelDownloadSpec) -> Path:
    runtime_root = local_runtime_dir(root).resolve()
    staging = (runtime_root / "model-downloads" / spec.name).resolve()
    if not staging.is_relative_to(runtime_root):
        raise RuntimeError(f"Unsafe model download staging path: {staging}")
    return staging


def prepare_model_download_staging_dir(root: Path, spec: ModelDownloadSpec) -> Path:
    staging = model_download_staging_dir(root, spec)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def copy_staged_model_to_target(staging: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staging, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))


def verify_downloaded_model(
    root: Path,
    spec: ModelDownloadSpec,
    staging: Path | None = None,
    returned_path: str | list[Any] | None = None,
) -> None:
    folder = staging or root / spec.target_subdir
    if _model_folder_ready_at(folder, spec.name):
        return
    target = root / spec.target_subdir
    raise RuntimeError(
        "Model download did not produce expected files for "
        f"{spec.repo_id}.\n"
        f"Staging folder: {folder}\n"
        f"Final target folder: {target}\n"
        f"snapshot_download returned: {returned_path}\n"
        f"Staging top-level files: {_model_folder_preview(folder)}\n"
        f"Final target top-level files: {_model_folder_preview(target)}"
    )


def download_snapshot_with_progress(
    spec: ModelDownloadSpec,
    staging: Path,
    total_bytes: int | None,
    base_progress: int,
    end_progress: int,
    on_event,
    downloader: Callable[..., str],
) -> str:
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["path"] = downloader(
                repo_id=spec.repo_id,
                local_dir=str(staging),
                local_files_only=False,
                force_download=True,
            )
        except BaseException as exc:  # pragma: no cover - propagated below
            result["error"] = exc

    thread = threading.Thread(target=worker, name=f"download-{spec.name}", daemon=True)
    started = time.time()
    thread.start()

    while thread.is_alive():
        downloaded = folder_size_bytes(staging)
        elapsed = time.time() - started
        filename = largest_file_name(staging, spec.repo_id)
        progress = None
        if total_bytes and total_bytes > 0:
            local_percent = max(0, min(100, int((min(downloaded, total_bytes) / total_bytes) * 100)))
            progress = base_progress + int(((end_progress - base_progress) * local_percent) / 100)
        detail = format_model_download_status(filename, downloaded, total_bytes, elapsed)
        on_event(PipProgressEvent("Downloading model", detail, progress, detail))
        thread.join(MODEL_DOWNLOAD_POLL_SECONDS)

    thread.join()
    if "error" in result:
        raise result["error"]

    downloaded = folder_size_bytes(staging)
    elapsed = time.time() - started
    filename = largest_file_name(staging, spec.repo_id)
    detail = format_model_download_status(filename, downloaded, total_bytes, elapsed)
    on_event(PipProgressEvent("Downloading model", detail, end_progress, detail))
    return str(result.get("path", ""))


def download_default_models(root: Path, on_event, downloader: Callable[..., str] | None = None) -> list[str]:
    package_path = local_package_dir(root)
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))
    if downloader is None:
        with online_huggingface_download_env(root):
            from huggingface_hub import snapshot_download

        downloader = snapshot_download

    downloaded: list[str] = []
    total = len(DEFAULT_MODEL_DOWNLOADS)
    for index, spec in enumerate(DEFAULT_MODEL_DOWNLOADS, start=1):
        target = root / spec.target_subdir
        target.mkdir(parents=True, exist_ok=True)
        base_progress = int(((index - 1) / total) * 100)
        if _model_folder_ready(root, spec.name):
            on_event(
                PipProgressEvent(
                    "Model ready",
                    f"{spec.repo_id} already exists in {target}",
                    int((index / total) * 100),
                    f"skip {spec.repo_id}",
                )
            )
            continue

        detail = f"{spec.repo_id} -> {target}"
        on_event(PipProgressEvent("Downloading model", detail, base_progress, detail))
        staging = prepare_model_download_staging_dir(root, spec)
        end_progress = int((index / total) * 100)
        total_bytes = fetch_model_repo_size(spec.repo_id, root)
        if total_bytes:
            size_detail = f"{spec.repo_id} download size: {format_bytes(total_bytes)}"
            on_event(PipProgressEvent("Model size", size_detail, base_progress, size_detail))
        else:
            size_detail = f"{spec.repo_id} download size unavailable; showing elapsed time"
            on_event(PipProgressEvent("Model size unavailable", size_detail, None, size_detail))

        with online_huggingface_download_env(root):
            returned_path = download_snapshot_with_progress(
                spec,
                staging,
                total_bytes,
                base_progress,
                end_progress,
                on_event,
                downloader,
            )
        verify_downloaded_model(root, spec, staging, returned_path)
        copy_staged_model_to_target(staging, target)
        verify_downloaded_model(root, spec, target, returned_path)
        downloaded.append(spec.name)
        on_event(PipProgressEvent("Model downloaded", detail, end_progress, detail))
    return downloaded


def create_startup_splash():
    import tkinter as tk
    from tkinter import ttk

    splash = tk.Tk()
    splash.title("Starting Offline Meeting Transcriber")
    splash.geometry("420x170")
    splash.resizable(False, False)
    splash.configure(bg="#f7f7f7")

    width = 420
    height = 170
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    left = int((screen_width - width) / 2)
    top = int((screen_height - height) / 2)
    splash.geometry(f"{width}x{height}+{left}+{top}")

    title = tk.Label(
        splash,
        text="Offline Meeting Transcriber",
        font=("Segoe UI", 14, "bold"),
        bg="#f7f7f7",
        fg="#1f1f1f",
    )
    title.pack(fill="x", padx=24, pady=(22, 8))

    status = tk.StringVar(value="Starting...")
    status_label = tk.Label(
        splash,
        textvariable=status,
        font=("Segoe UI", 10),
        bg="#f7f7f7",
        fg="#333333",
        anchor="w",
    )
    status_label.pack(fill="x", padx=24, pady=(0, 12))

    progress = ttk.Progressbar(splash, mode="indeterminate")
    progress.pack(fill="x", padx=24, pady=(0, 18))
    progress.start(12)

    splash.update_idletasks()
    splash.update()
    return splash, status


def update_startup_splash(splash, status, message: str) -> None:
    if splash is None:
        return
    status.set(message)
    splash.update_idletasks()
    splash.update()


def close_startup_splash(splash) -> None:
    if splash is not None:
        splash.destroy()


def launch_gui(root: Path, splash=None, splash_status=None) -> int:
    os.chdir(root)
    os.environ["LOCAL_WHISPER_APP_ROOT"] = str(root)
    os.environ["LOCAL_WHISPER_SOURCE_ROOT"] = str(installer_source_root())
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    if splash is None:
        try:
            splash, splash_status = create_startup_splash()
        except Exception:
            splash = None
            splash_status = None
    update_startup_splash(splash, splash_status, "Checking local app files...")

    app_dir = str(app_source_dir())
    package_dir = str(local_package_dir(root))
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    try:
        update_startup_splash(splash, splash_status, "Loading audio and AI components...")
        from meeting_transcriber_gui import main

        update_startup_splash(splash, splash_status, "Opening transcriber window...")
    finally:
        close_startup_splash(splash)
    return main()


def run_pip_install(root: Path, on_event, package_root: Path | None = None) -> PipInstallResult:
    package_root = package_root or installer_source_root()
    requirements_path = resource_root(package_root) / "requirements.txt"
    package_dir = local_package_dir(root)
    package_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--progress-bar",
        "on",
        "--upgrade",
        "--target",
        str(package_dir),
        "-r",
        str(requirements_path),
    ]
    command = subprocess.list2cmdline(cmd)
    on_event(PipProgressEvent("Running command", command, 0, command))
    process = subprocess.Popen(
        cmd,
        cwd=str(package_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    parser = PipProgressParser()
    recent_output: deque[str] = deque(maxlen=25)
    for line in iter_cli_progress_output(process.stdout):
        stripped = line.rstrip()
        recent_output.append(stripped)
        on_event(parser.parse(stripped))
    return PipInstallResult(process.wait(), command, list(recent_output))


def write_setup_error_log(
    root: Path,
    context: str,
    message: str,
    command: str = "",
    recent_output: list[str] | None = None,
) -> Path:
    return append_error_log(
        root,
        context,
        message,
        {
            "command": command,
            "recent_output": recent_output or [],
        },
    )


def needs_setup(root: Path, required: Mapping[str, str] = REQUIRED_IMPORTS) -> bool:
    ensure_portable_layout(root, installer_source_root())
    models_missing = any(value == BootstrapStatus.MISSING for value in model_folder_status(root).values())
    return bool(missing_runtime_imports(root, required)) or models_missing or not (root / SETUP_MARKER).exists()


def run_bootstrap() -> int:
    root = runtime_root()
    package_root = installer_source_root()
    configure_download_tls(root)
    splash = None
    splash_status = None
    try:
        splash, splash_status = create_startup_splash()
        update_startup_splash(splash, splash_status, "Preparing local app folder...")
    except Exception:
        splash = None
        splash_status = None

    ensure_portable_layout(root, package_root)
    update_startup_splash(splash, splash_status, "Checking Python packages...")
    missing_packages = missing_runtime_imports(root)
    update_startup_splash(splash, splash_status, "Checking default AI models...")
    models_missing = any(value == BootstrapStatus.MISSING for value in model_folder_status(root).values())
    update_startup_splash(splash, splash_status, "Checking first-time setup status...")
    setup_missing = not (root / SETUP_MARKER).exists()
    if not missing_packages and not models_missing and not setup_missing:
        update_startup_splash(splash, splash_status, "Dependencies ready.")
        return launch_gui(root, splash, splash_status)

    if missing_packages:
        update_startup_splash(splash, splash_status, "Dependencies missing. Opening first-time setup...")
    elif models_missing:
        update_startup_splash(splash, splash_status, "Default AI models missing. Opening setup...")
    else:
        update_startup_splash(splash, splash_status, "First-time setup required. Opening installer...")
    time.sleep(0.8)
    close_startup_splash(splash)

    install_log_file = reset_install_log(root, setup_install_summary(package_root))
    append_install_log(root, "Setup", "Installer opened", {"install_log": str(install_log_file)})

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    window = tk.Tk()
    window.title("Offline Meeting Transcriber Setup")
    window.geometry("860x680")
    window.minsize(820, 650)
    window.resizable(False, False)
    window.configure(bg="#f7f7f7")

    style = ttk.Style(window)
    style.theme_use("clam")
    style.configure(SETUP_PROGRESS_STYLE, troughcolor="#e1e1e1", background="#2d6cdf")
    style.configure(DETAIL_PROGRESS_STYLE, troughcolor="#e1e1e1", background="#4f9cff")
    style.configure(COMPLETE_PROGRESS_STYLE, troughcolor="#e1e1e1", background="#2e7d32")

    step_states = {step: StepState.PENDING for step in BOOTSTRAP_STEPS}
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_index = 0
    current_step = BOOTSTRAP_STEPS[0]
    detail_progress_value = tk.IntVar(value=0)
    overall_progress_value = tk.IntVar(value=0)
    package_detail = tk.StringVar(value=setup_install_summary(package_root))
    command_detail = tk.StringVar(value="")
    eta_detail = tk.StringVar(value="")
    install_path_detail = tk.StringVar(value=f"Install path: {root}")
    model_size_summary = tk.StringVar(value="Estimated model download: estimating...")
    final_message_active = tk.BooleanVar(value=False)
    ui_thread = threading.current_thread()
    step_diagnostics = {step: StepDiagnostic(step) for step in BOOTSTRAP_STEPS}
    step_progress_values = {step: 0 for step in BOOTSTRAP_STEPS}
    tooltip_refs = []
    pip_tracker: dict[str, PipInstallProgressTracker | None] = {"value": None}
    install_button_ref: dict[str, Any] = {"value": None}

    def log_install(context: str, message: str, details: Mapping[str, Any] | Iterable[str] | str | None = None) -> None:
        try:
            append_install_log(root, context, message, details)
        except Exception:
            pass

    title = tk.Label(
        window,
        text="Offline Meeting Transcriber",
        font=("Segoe UI", 18, "bold"),
        anchor="w",
        bg="#f7f7f7",
        fg="#1f1f1f",
    )
    title.pack(fill="x", padx=24, pady=(22, 4))

    summary = tk.Label(
        window,
        text=(
            "First-time setup checks local prerequisites. Future launches open "
            "the transcriber immediately."
        ),
        wraplength=800,
        justify="left",
        anchor="w",
        bg="#f7f7f7",
        fg="#333333",
    )
    summary.pack(fill="x", padx=24, pady=(0, 18))

    overall_bar = ttk.Progressbar(
        window,
        maximum=100,
        variable=overall_progress_value,
        style=SETUP_PROGRESS_STYLE,
    )
    overall_bar.pack(fill="x", padx=24, pady=(0, 20))

    body = tk.Frame(window, bg="#f7f7f7")
    body.pack(fill="both", expand=True, padx=24, pady=(0, 16))

    task_frame = tk.Frame(body, bg="#f7f7f7")
    task_frame.pack(side="left", fill="both", expand=False, padx=(0, 24))

    detail_frame = tk.Frame(body, bg="#ffffff", highlightbackground="#dddddd", highlightthickness=1)
    detail_frame.pack(side="right", fill="both", expand=True)

    task_labels: dict[str, tk.Label] = {}
    for step in BOOTSTRAP_STEPS:
        row = tk.Label(
            task_frame,
            text=f"○ {step}",
            font=("Segoe UI", 10),
            anchor="w",
            width=34,
            bg="#f7f7f7",
            fg="#222222",
        )
        row.pack(fill="x", pady=3)
        task_labels[step] = row
        tooltip_refs.append(ToolTip(row, lambda name=step: build_step_tooltip(step_diagnostics[name])))

    install_frame = tk.Frame(detail_frame, bg="#ffffff")
    install_frame.pack(fill="both", expand=True)

    install_button_row = tk.Frame(install_frame, bg="#ffffff")
    install_button_row.pack(side="bottom", fill="x", padx=20, pady=(8, 16))

    install_content = tk.Frame(install_frame, bg="#ffffff")
    install_content.pack(side="top", fill="both", expand=True)

    install_title = tk.Label(
        install_content,
        text="Install Offline Meeting Transcriber?",
        font=("Segoe UI", 16, "bold"),
        anchor="w",
        bg="#ffffff",
        fg="#1f1f1f",
    )
    install_title.pack(fill="x", padx=20, pady=(14, 4))

    app_type = tk.Label(
        install_content,
        text="Local Windows App",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#0969da",
    )
    app_type.pack(fill="x", padx=20)

    publisher = tk.Label(
        install_content,
        text=f"Publisher: {APP_PUBLISHER}",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#333333",
    )
    publisher.pack(fill="x", padx=20)

    version = tk.Label(
        install_content,
        text=f"Version: {APP_VERSION}",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#333333",
    )
    version.pack(fill="x", padx=20, pady=(0, 10))

    install_path_label = tk.Label(
        install_content,
        textvariable=install_path_detail,
        font=("Segoe UI", 9),
        anchor="w",
        bg="#ffffff",
        fg="#333333",
        wraplength=500,
        justify="left",
    )
    install_path_label.pack(fill="x", padx=20, pady=(0, 8))

    models_title = tk.Label(
        install_content,
        text="Models to install locally:",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#222222",
    )
    models_title.pack(fill="x", padx=20, pady=(0, 4))

    model_size_label = tk.Label(
        install_content,
        textvariable=model_size_summary,
        font=("Segoe UI", 9),
        anchor="w",
        bg="#ffffff",
        fg="#666666",
    )
    model_size_label.pack(fill="x", padx=20, pady=(0, 4))

    for item in setup_model_list():
        tk.Label(
            install_content,
            text=f"- {item}",
            font=("Segoe UI", 10),
            anchor="w",
            bg="#ffffff",
            fg="#666666",
        ).pack(fill="x", padx=28)

    packages_title = tk.Label(
        install_content,
        text="Python packages to install locally:",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#222222",
    )
    packages_title.pack(fill="x", padx=20, pady=(10, 4))

    packages_box = tk.Frame(install_content, bg="#ffffff")
    packages_box.pack(fill="x", padx=20)
    package_list = tk.Text(
        packages_box,
        height=3,
        wrap="none",
        font=("Cascadia Mono", 9),
        bg="#ffffff",
        fg="#555555",
        relief="solid",
        borderwidth=1,
        highlightthickness=0,
        padx=6,
        pady=4,
    )
    package_scroll = ttk.Scrollbar(packages_box, orient="vertical", command=package_list.yview)
    package_list.configure(yscrollcommand=package_scroll.set)
    package_scroll.pack(side="right", fill="y")
    package_list.pack(side="left", fill="both", expand=True)
    package_list.insert("1.0", "\n".join(f"- {package}" for package in setup_package_list(package_root)))
    package_list.configure(state="disabled", cursor="arrow", takefocus=False)

    install_note = tk.Label(
        install_content,
        text="Internet required for first setup downloads. Runtime stays local/offline.",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#333333",
    )
    install_note.pack(fill="x", padx=20, pady=(8, 8))

    onedrive_warning_frame = tk.Frame(
        install_content,
        bg="#fff4ce",
        highlightbackground="#d29922",
        highlightthickness=1,
    )
    onedrive_warning = tk.Label(
        onedrive_warning_frame,
        text=(
            "This install path is under OneDrive. Models and runtime files can be large; "
            "a local folder such as C:\\Dev avoids sync delays."
        ),
        font=("Segoe UI", 9),
        anchor="w",
        justify="left",
        bg="#fff4ce",
        fg="#5f3b00",
        wraplength=480,
    )
    onedrive_warning.pack(fill="x", padx=10, pady=(8, 6))
    onedrive_actions = tk.Frame(onedrive_warning_frame, bg="#fff4ce")
    onedrive_actions.pack(fill="x", padx=10, pady=(0, 8))

    progress_frame = tk.Frame(detail_frame, bg="#ffffff")

    detail_title = tk.Label(
        progress_frame,
        text="Preparing setup",
        font=("Segoe UI", 14, "bold"),
        anchor="w",
        bg="#ffffff",
        fg="#1f1f1f",
    )
    detail_title.pack(fill="x", padx=18, pady=(18, 6))

    def readonly_text(parent, height: int, font: tuple[str, int], foreground: str, wrap: str = "word") -> tk.Text:
        widget = tk.Text(
            parent,
            height=height,
            wrap=wrap,
            font=font,
            bg="#ffffff",
            fg=foreground,
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=0,
            pady=0,
        )
        widget.configure(state="disabled", cursor="arrow", takefocus=False)
        return widget

    command_text = readonly_text(
        progress_frame,
        height=4,
        font=("Cascadia Mono", 9),
        foreground="#505050",
        wrap="char",
    )
    command_text.pack(fill="x", padx=18, pady=(0, 10))

    detail_bar = ttk.Progressbar(
        progress_frame,
        maximum=100,
        variable=detail_progress_value,
        style=DETAIL_PROGRESS_STYLE,
    )
    detail_bar.pack(fill="x", padx=18, pady=(0, 12))
    detail_bar_mode = {"indeterminate": False}

    eta_label = tk.Label(
        progress_frame,
        textvariable=eta_detail,
        font=("Segoe UI", 9),
        anchor="w",
        justify="left",
        bg="#ffffff",
        fg="#666666",
    )
    eta_label.pack(fill="x", padx=18, pady=(0, 8))

    package_text = readonly_text(
        progress_frame,
        height=7,
        font=("Segoe UI", 10),
        foreground="#222222",
        wrap="word",
    )
    package_text.pack(fill="x", expand=False, padx=18, pady=(0, 14))

    button_row = tk.Frame(progress_frame, bg="#ffffff")
    button_row.pack(fill="x", padx=18, pady=(0, 6))

    model_button_row = tk.Frame(progress_frame, bg="#ffffff")
    model_button_row.pack(fill="x", padx=18, pady=(0, 18))

    details_button = tk.Button(button_row, text="Show install log", width=16)
    details_button.pack(side="left")

    launch_now_button = tk.Button(button_row, text="Launch now", width=14, state="disabled")
    launch_now_button.pack(side="right")

    details_frame = tk.Frame(progress_frame, bg="#ffffff")
    details_text = tk.Text(
        details_frame,
        height=6,
        wrap="word",
        font=("Cascadia Mono", 9),
        bg="#ffffff",
        fg="#333333",
        relief="solid",
        borderwidth=1,
        highlightthickness=0,
        padx=6,
        pady=4,
    )
    details_scroll = ttk.Scrollbar(details_frame, orient="vertical", command=details_text.yview)
    details_text.configure(yscrollcommand=details_scroll.set, state="disabled", cursor="arrow")
    details_scroll.pack(side="right", fill="y")
    details_text.pack(side="left", fill="both", expand=True)

    error_frame = tk.Frame(detail_frame, bg="#ffffff")
    error_title = tk.Label(
        error_frame,
        text="Setup needs attention",
        font=("Segoe UI", 14, "bold"),
        anchor="w",
        bg="#ffffff",
        fg="#b00020",
    )
    error_title.pack(fill="x", padx=18, pady=(18, 8))
    error_message = tk.Label(
        error_frame,
        text="",
        font=("Segoe UI", 10),
        anchor="nw",
        justify="left",
        bg="#ffffff",
        fg="#222222",
        wraplength=500,
    )
    error_message.pack(fill="x", padx=18, pady=(0, 12))
    error_detail_text = tk.Text(
        error_frame,
        height=9,
        wrap="word",
        font=("Cascadia Mono", 9),
        bg="#ffffff",
        fg="#333333",
        relief="solid",
        borderwidth=1,
        highlightthickness=0,
        padx=6,
        pady=4,
    )
    error_detail_text.configure(state="disabled", cursor="arrow", takefocus=False)
    error_detail_text.pack(fill="both", expand=True, padx=18, pady=(0, 12))
    error_button_row = tk.Frame(error_frame, bg="#ffffff")
    error_button_row.pack(fill="x", padx=18, pady=(0, 18))
    retry_button = tk.Button(error_button_row, text="Retry this step", width=16)
    retry_button.pack(side="left")
    copy_details_button = tk.Button(error_button_row, text="Copy details", width=14)
    copy_details_button.pack(side="left", padx=(8, 0))
    open_error_log_button = tk.Button(error_button_row, text="Open error_log.txt", width=16)
    open_error_log_button.pack(side="right")
    error_state: dict[str, Any] = {"step": "", "details": "", "log_path": ""}

    def set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def append_details_log(line: str) -> None:
        if not line:
            return
        details_text.configure(state="normal")
        details_text.insert("end", line.rstrip() + "\n")
        details_text.see("end")
        details_text.configure(state="disabled")

    def toggle_details_log() -> None:
        if details_frame.winfo_ismapped():
            details_frame.pack_forget()
            details_button.config(text="Show install log")
        else:
            details_frame.pack(fill="both", expand=True, padx=18, pady=(0, 12), before=button_row)
            details_button.config(text="Hide install log")

    details_button.config(command=toggle_details_log)

    def set_detail_bar_indeterminate(active: bool) -> None:
        if active and not detail_bar_mode["indeterminate"]:
            detail_bar.stop()
            detail_bar.configure(mode="indeterminate")
            detail_bar.start(12)
            detail_bar_mode["indeterminate"] = True
        elif not active and detail_bar_mode["indeterminate"]:
            detail_bar.stop()
            detail_bar.configure(mode="determinate")
            detail_bar_mode["indeterminate"] = False

    def set_command_text(value: str) -> None:
        command_detail.set(value)
        set_text(command_text, value)
        step_diagnostics[current_step].last_output = value

    def run_on_ui(callback):
        if threading.current_thread() is ui_thread:
            return callback()
        done = threading.Event()
        result = {}

        def wrapped() -> None:
            try:
                result["value"] = callback()
            except Exception as exc:  # pragma: no cover - defensive UI bridge
                result["error"] = exc
            finally:
                done.set()

        window.after(0, wrapped)
        done.wait()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def set_install_button_enabled(enabled: bool) -> None:
        button = install_button_ref.get("value")
        if button is not None:
            button.config(state="normal" if enabled else "disabled")

    def refresh_onedrive_warning() -> None:
        install_path_detail.set(f"Install path: {root}")
        if is_onedrive_path(root):
            if not onedrive_warning_frame.winfo_ismapped():
                onedrive_warning_frame.pack(fill="x", padx=20, pady=(0, 10), before=models_title)
            set_install_button_enabled(False)
        else:
            onedrive_warning_frame.pack_forget()
            set_install_button_enabled(True)

    def continue_onedrive_install() -> None:
        onedrive_warning_frame.pack_forget()
        set_install_button_enabled(True)

    def choose_local_install_folder() -> None:
        nonlocal root
        selected = filedialog.askdirectory(title="Choose local install folder")
        if not selected:
            return
        root = Path(selected).expanduser().resolve()
        install_path_detail.set(f"Install path: {root}")
        refresh_onedrive_warning()

    tk.Button(onedrive_actions, text="Continue anyway", width=16, command=continue_onedrive_install).pack(side="left")
    tk.Button(onedrive_actions, text="Choose local folder", width=18, command=choose_local_install_folder).pack(
        side="left", padx=(8, 0)
    )

    def refresh_model_size_estimate() -> None:
        with online_huggingface_download_env(root):
            sizes = estimate_model_download_sizes(DEFAULT_MODEL_DOWNLOADS, root)
        total = total_known_size(sizes)
        if total:
            message = f"Estimated model download: {format_bytes(total)}"
        else:
            known = [size for size in sizes.values() if isinstance(size, int) and size > 0]
            message = (
                f"Estimated model download: at least {format_bytes(sum(known))}"
                if known
                else "Estimated model download: unavailable"
            )
        run_on_ui(lambda: model_size_summary.set(message))

    def start_model_size_estimate() -> None:
        threading.Thread(target=refresh_model_size_estimate, name="model-size-estimate", daemon=True).start()

    sensitive_install_phase = {"active": False}

    def set_sensitive_install_phase(active: bool) -> None:
        sensitive_install_phase["active"] = active

    def short_error_cause(message: str) -> str:
        return setup_error_card_message(message)

    def copy_error_details() -> None:
        window.clipboard_clear()
        window.clipboard_append(error_state.get("details", ""))

    def open_error_log_file() -> None:
        log_path = str(error_state.get("log_path", ""))
        if log_path and Path(log_path).is_file():
            os.startfile(log_path)
        else:
            messagebox.showwarning("Error log not found", "error_log.txt was not found for this failure.")

    def show_progress_pane() -> None:
        error_frame.pack_forget()
        progress_frame.pack(fill="both", expand=True)

    def show_error_card(step: str, message: str, log_path: Path | str = "", details: str = "") -> None:
        detail_text = details or message
        if log_path:
            detail_text = f"{detail_text}\n\nError details saved to {log_path}"
        error_state.update({"step": step, "details": detail_text, "log_path": str(log_path)})

        def apply() -> None:
            progress_frame.pack_forget()
            error_message.config(text=short_error_cause(message))
            set_text(error_detail_text, detail_text)
            error_frame.pack(fill="both", expand=True)
            retry_button.config(state="normal")
            copy_details_button.config(state="normal")
            open_error_log_button.config(state="normal" if log_path else "disabled")
            window.update_idletasks()

        run_on_ui(apply)

    def retry_failed_step() -> None:
        step = str(error_state.get("step") or current_step)

        def reset_failed_steps() -> None:
            try:
                start_index = BOOTSTRAP_STEPS.index(step)
            except ValueError:
                start_index = 0
            for reset_step in BOOTSTRAP_STEPS[start_index:]:
                step_states[reset_step] = StepState.PENDING
                step_progress_values[reset_step] = 0
            show_progress_pane()
            set_detail_bar_indeterminate(False)
            detail_progress_value.set(0)
            eta_detail.set("")
            render_steps()

        run_on_ui(reset_failed_steps)
        threading.Thread(target=lambda: run_setup_flow(step), name="setup-retry", daemon=True).start()

    def on_close_setup() -> None:
        if sensitive_install_phase["active"]:
            if not messagebox.askyesno(
                "Setup still running",
                "Package or model install is in progress. Closing now can leave a half-installed setup.\n\nClose anyway?",
            ):
                return
        window.destroy()

    retry_button.config(command=retry_failed_step)
    copy_details_button.config(command=copy_error_details)
    open_error_log_button.config(command=open_error_log_file)
    window.protocol("WM_DELETE_WINDOW", on_close_setup)

    def render_steps() -> None:
        overall_progress_value.set(weighted_overall_progress(step_states, step_progress_values))
        for step, label in task_labels.items():
            state = step_states[step]
            if state == StepState.DONE:
                label.config(text=f"✓ {step}", fg="#2e7d32")
            elif state == StepState.WARNING:
                label.config(text=f"! {step}", fg="#a66a00")
            elif state == StepState.ERROR:
                label.config(text=f"× {step}", fg="#b00020")
            elif state == StepState.RUNNING:
                label.config(text=f"{spinner_frames[spinner_index % len(spinner_frames)]} {step}", fg="#1f6feb")
            else:
                label.config(text=f"○ {step}", fg="#555555")

    def tick_spinner() -> None:
        nonlocal spinner_index
        spinner_index += 1
        render_steps()
        if not final_message_active.get():
            window.after(180, tick_spinner)

    def set_step(step: str, state: StepState, detail: str = "", progress: int | None = None) -> None:
        nonlocal current_step
        log_details = {"state": state.value}
        if progress is not None:
            log_details["progress_percent"] = progress
        log_install(step, detail or step, log_details)

        def apply() -> None:
            nonlocal current_step
            current_step = step
            step_states[step] = state
            diagnostic = step_diagnostics[step]
            diagnostic.state = state
            if diagnostic.start_time is None and state == StepState.RUNNING:
                diagnostic.start_time = time.time()
            if state in (StepState.DONE, StepState.WARNING, StepState.ERROR):
                diagnostic.end_time = time.time()
            if detail:
                diagnostic.detail = detail
            if state == StepState.ERROR:
                diagnostic.error = detail
            detail_title.config(text=step)
            append_details_log(f"{step}: {state.value}" + (f" - {detail}" if detail else ""))
            if detail:
                package_detail.set(detail)
                set_text(package_text, detail)
            if progress is not None:
                set_detail_bar_indeterminate(False)
                detail_progress_value.set(progress)
                step_progress_values[step] = progress
            elif state == StepState.DONE:
                step_progress_values[step] = 100
            render_steps()
            window.update_idletasks()

        run_on_ui(apply)

    def important_message(step: str, message: str) -> None:
        set_step(step, StepState.RUNNING, message, None)
        def apply_message() -> None:
            set_command_text(message)

        run_on_ui(apply_message)
        time.sleep(IMPORTANT_MESSAGE_SECONDS)

    def choose_model_folder(model_name: str) -> None:
        selected = filedialog.askdirectory(title=f"Select folder for {model_name}")
        if not selected:
            return
        target = root / "models" / model_name
        target.mkdir(parents=True, exist_ok=True)
        marker = target / "README_SELECTED_MODEL_PATH.txt"
        marker.write_text(
            f"Selected source folder:\n{selected}\n\nCopy model files here or update config.json manually.\n",
            encoding="utf-8",
        )
        detail = f"Recorded selected folder for {model_name}:\n{selected}"
        package_detail.set(detail)
        set_text(package_text, detail)
        render_steps()

    def open_model_folder() -> None:
        model_root = root / "models"
        os.startfile(model_root)
        detail = f"Opened model folder:\n{model_root}"
        package_detail.set(detail)
        set_text(package_text, detail)

    def update_from_pip(event: PipProgressEvent) -> None:
        log_install(
            BOOTSTRAP_STEPS[3],
            event.phase,
            {
                "detail": event.detail,
                "progress_percent": event.progress_percent,
                "raw_line": event.raw_line,
            },
        )

        def apply_event() -> None:
            set_command_text(event.phase)
            append_details_log(event.raw_line or event.detail or event.phase)
            if event.detail:
                package_detail.set(event.detail)
                set_text(package_text, event.detail)
                step_diagnostics[BOOTSTRAP_STEPS[3]].detail = event.detail
            step_diagnostics[BOOTSTRAP_STEPS[3]].last_output = event.raw_line or event.detail or event.phase
            tracker = pip_tracker["value"]
            progress = None
            if tracker is not None:
                progress, eta = tracker.record(event)
                eta_detail.set(eta)
                if tracker.installing_started and event.progress_percent != 100:
                    set_detail_bar_indeterminate(True)
                else:
                    set_detail_bar_indeterminate(False)
                    detail_progress_value.set(progress)
                    step_progress_values[BOOTSTRAP_STEPS[3]] = progress
            if event.progress_percent is not None:
                if event.progress_percent == 100:
                    set_detail_bar_indeterminate(False)
                    detail_progress_value.set(100)
                elif not (tracker is not None and tracker.installing_started):
                    current_progress = progress if progress is not None else detail_progress_value.get()
                    detail_progress_value.set(max(current_progress, event.progress_percent))
                step_progress_values[BOOTSTRAP_STEPS[3]] = detail_progress_value.get()
            render_steps()

        window.after(0, apply_event)

    def update_from_model_download(event: PipProgressEvent) -> None:
        log_install(
            BOOTSTRAP_STEPS[4],
            event.phase,
            {
                "detail": event.detail,
                "progress_percent": event.progress_percent,
                "raw_line": event.raw_line,
            },
        )

        def apply_event() -> None:
            set_command_text(event.phase)
            append_details_log(event.raw_line or event.detail or event.phase)
            if event.detail:
                package_detail.set(event.detail)
                set_text(package_text, event.detail)
                step_diagnostics[BOOTSTRAP_STEPS[4]].detail = event.detail
            step_diagnostics[BOOTSTRAP_STEPS[4]].last_output = event.raw_line or event.detail or event.phase
            if event.progress_percent is not None:
                set_detail_bar_indeterminate(False)
                detail_progress_value.set(event.progress_percent)
                step_progress_values[BOOTSTRAP_STEPS[4]] = event.progress_percent
            else:
                set_detail_bar_indeterminate(True)
            render_steps()

        window.after(0, apply_event)

    def finish_with_launch_countdown() -> None:
        final_message_active.set(True)
        step_states[BOOTSTRAP_STEPS[-1]] = StepState.RUNNING
        log_install(BOOTSTRAP_STEPS[-1], "Launch countdown started", {"delay_seconds": IMPORTANT_MESSAGE_SECONDS})
        launch_now_button.config(state="normal")
        render_steps()

        remaining = {"seconds": IMPORTANT_MESSAGE_SECONDS}
        launched = {"value": False}

        def launch_now() -> None:
            if launched["value"]:
                return
            launched["value"] = True
            step_states[BOOTSTRAP_STEPS[-1]] = StepState.DONE
            (root / SETUP_MARKER).write_text("complete\n", encoding="utf-8")
            window.destroy()
            raise SystemExit(launch_gui(root))

        launch_now_button.config(command=launch_now)

        def countdown() -> None:
            if launched["value"]:
                return
            seconds = remaining["seconds"]
            detail_title.config(text="Prerequisites setup done")
            command_detail.set("")
            set_text(command_text, "")
            eta_detail.set("")
            set_detail_bar_indeterminate(False)
            detail = f"Launching in {seconds}..."
            package_detail.set(detail)
            set_text(package_text, detail)
            overall_bar.configure(style=COMPLETE_PROGRESS_STYLE)
            detail_bar.configure(style=COMPLETE_PROGRESS_STYLE)
            overall_progress_value.set(100)
            detail_progress_value.set(100)
            if seconds <= 0:
                launch_now()
                return
            remaining["seconds"] -= 1
            window.after(1000, countdown)

        countdown()

    def launch_if_ready() -> None:
        missing = missing_runtime_imports(root)
        if missing:
            messagebox.showerror("Missing packages", "Install packages first:\n" + "\n".join(missing))
            return
        model_status = model_folder_status(root)
        if any(value == BootstrapStatus.MISSING for value in model_status.values()):
            messagebox.showwarning(
                "Models missing",
                "GUI can open, but recording cannot start until model folders contain local model files.",
            )
        finish_with_launch_countdown()

    def run_setup_flow(start_step: str = BOOTSTRAP_STEPS[0]) -> None:
        try:
            try:
                start_index = BOOTSTRAP_STEPS.index(start_step)
            except ValueError:
                start_index = 0

            if start_index <= 0:
                important_message(BOOTSTRAP_STEPS[0], f"Using Python: {sys.executable}")
                set_step(BOOTSTRAP_STEPS[0], StepState.DONE, "Python runtime ready", 100)

            if start_index <= 1:
                important_message(BOOTSTRAP_STEPS[1], f"Preparing folders under:\n{root}")
                ensure_portable_layout(root, package_root)
                set_step(BOOTSTRAP_STEPS[1], StepState.DONE, "App folders ready", 100)

            if start_index <= 3:
                if start_index <= 2:
                    important_message(BOOTSTRAP_STEPS[2], "Checking required Python packages...")
                missing = missing_runtime_imports(root)
                if missing:
                    set_step(BOOTSTRAP_STEPS[2], StepState.WARNING, "Missing: " + ", ".join(missing), 100)
                    set_step(BOOTSTRAP_STEPS[3], StepState.RUNNING, "Installing missing packages", 0)
                    command = subprocess.list2cmdline(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--no-cache-dir",
                            "--progress-bar",
                            "on",
                            "--upgrade",
                            "--target",
                            str(local_package_dir(root)),
                            "-r",
                            str(resource_root(package_root) / "requirements.txt"),
                        ]
                    )

                    def show_command() -> None:
                        set_command_text("Running command:\n" + command)
                        step_diagnostics[BOOTSTRAP_STEPS[3]].command = command

                    log_install(BOOTSTRAP_STEPS[3], "Running pip command", {"command": command})
                    run_on_ui(show_command)
                    time.sleep(IMPORTANT_MESSAGE_SECONDS)
                    pip_tracker["value"] = PipInstallProgressTracker()
                    set_sensitive_install_phase(True)
                    try:
                        result = run_pip_install(root, update_from_pip, package_root)
                    finally:
                        set_sensitive_install_phase(False)
                    if result.code != 0:
                        log_path = write_setup_error_log(
                            root,
                            BOOTSTRAP_STEPS[3],
                            f"pip failed with exit code {result.code}",
                            result.command,
                            result.recent_output,
                        )
                        step_diagnostics[BOOTSTRAP_STEPS[3]].command = result.command
                        recent = "\n".join(result.recent_output[-8:])
                        step_diagnostics[BOOTSTRAP_STEPS[3]].last_output = recent
                        message = f"pip failed with exit code {result.code}"
                        set_step(
                            BOOTSTRAP_STEPS[3],
                            StepState.ERROR,
                            f"{message}\nError details saved to {log_path}",
                            100,
                        )
                        show_error_card(BOOTSTRAP_STEPS[3], message, log_path, recent)
                        return
                    set_step(BOOTSTRAP_STEPS[3], StepState.DONE, "Python packages installed", 100)
                else:
                    set_step(BOOTSTRAP_STEPS[2], StepState.DONE, "Python packages ready", 100)
                    set_step(BOOTSTRAP_STEPS[3], StepState.DONE, "No install needed", 100)

            if start_index <= 4:
                important_message(BOOTSTRAP_STEPS[4], "Downloading default local AI models...")
                set_step(BOOTSTRAP_STEPS[4], StepState.RUNNING, "Downloading default models", 0)
                try:
                    set_sensitive_install_phase(True)
                    downloaded = download_default_models(root, update_from_model_download)
                except Exception as exc:
                    log_path = write_setup_error_log(
                        root,
                        BOOTSTRAP_STEPS[4],
                        f"Model download failed: {exc}",
                        "",
                        [str(exc)],
                    )
                    message = f"Model download failed:\n{exc}"
                    set_step(
                        BOOTSTRAP_STEPS[4],
                        StepState.ERROR,
                        f"{message}\nError details saved to {log_path}",
                        100,
                    )
                    show_error_card(BOOTSTRAP_STEPS[4], message, log_path, str(exc))
                    return
                finally:
                    set_sensitive_install_phase(False)
                model_detail = "Downloaded: " + ", ".join(downloaded) if downloaded else "Default models already present"
                set_step(BOOTSTRAP_STEPS[4], StepState.DONE, model_detail, 100)

            if start_index <= 5:
                important_message(BOOTSTRAP_STEPS[5], "Checking local AI model folders...")
                models = model_folder_status(root)
                missing_models = [name for name, value in models.items() if value == BootstrapStatus.MISSING]
                if missing_models:
                    message = missing_model_setup_message(missing_models)
                    log_path = write_setup_error_log(
                        root,
                        BOOTSTRAP_STEPS[5],
                        "Required model files missing after download",
                        "",
                        missing_models,
                    )
                    set_step(
                        BOOTSTRAP_STEPS[5],
                        StepState.ERROR,
                        f"{message}\n\nError details saved to {log_path}",
                        100,
                    )
                    show_error_card(BOOTSTRAP_STEPS[5], message, log_path, "\n".join(missing_models))
                    return
                set_step(BOOTSTRAP_STEPS[5], StepState.DONE, "Model folders ready", 100)

            if start_index <= 6:
                important_message(BOOTSTRAP_STEPS[6], "Saving setup state and preparing launch...")
                set_step(BOOTSTRAP_STEPS[6], StepState.DONE, "Setup complete", 100)
            window.after(0, finish_with_launch_countdown)
        except Exception as exc:
            log_path = write_setup_error_log(root, current_step, f"Setup failed: {exc}")
            message = f"Setup failed:\n{exc}"
            window.after(0, lambda: set_step(current_step, StepState.ERROR, f"{message}\nError details saved to {log_path}", 100))
            show_error_card(current_step, message, log_path, str(exc))

    def start_flow() -> None:
        for child in install_button_row.winfo_children():
            child.config(state="disabled")
        install_frame.pack_forget()
        progress_frame.pack(fill="both", expand=True)
        details_frame.pack_forget()
        details_button.config(text="Show install log")
        set_text(command_text, "")
        set_text(package_text, "")
        overall_bar.configure(style=SETUP_PROGRESS_STYLE)
        detail_bar.configure(style=DETAIL_PROGRESS_STYLE)
        detail_progress_value.set(0)
        eta_detail.set("")
        set_detail_bar_indeterminate(False)
        render_steps()
        for child in button_row.winfo_children():
            if child is details_button:
                child.config(state="normal")
            else:
                child.config(state="disabled")
        for child in model_button_row.winfo_children():
            child.config(state="disabled")
        threading.Thread(target=run_setup_flow, name="setup-flow", daemon=True).start()

    cancel_button = tk.Button(install_button_row, text="Cancel", width=14, command=on_close_setup)
    cancel_button.pack(side="right", padx=(0, 20))
    install_button = tk.Button(install_button_row, text="Install", width=14, command=start_flow)
    install_button.pack(side="right", padx=(0, 8))
    install_button_ref["value"] = install_button
    refresh_onedrive_warning()

    set_text(command_text, command_detail.get())
    set_text(package_text, "")

    render_steps()
    tick_spinner()
    window.after(100, start_model_size_estimate)
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_bootstrap())
