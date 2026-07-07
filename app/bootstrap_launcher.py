from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping

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
CA_BUNDLE_ENV = "LOCAL_WHISPER_CA_BUNDLE"
TLS_CERT_ENV_VARS = ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")
CA_BUNDLE_FILENAMES = ("company-ca.pem", "corporate-ca.pem", "ca-bundle.pem")

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

        already = re.match(r"Requirement already satisfied:\s+(.+?)(?:\s+in\s+.+)?$", stripped)
        if already:
            return PipProgressEvent("Already installed", already.group(1), 100, line)

        collecting = re.match(r"Collecting\s+(.+)$", stripped)
        if collecting:
            return PipProgressEvent("Resolving package", collecting.group(1), None, line)

        downloading = re.match(r"Downloading\s+(.+)$", stripped)
        if downloading:
            return PipProgressEvent("Downloading package", downloading.group(1), None, line)

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


def configure_download_tls(root: Path | None = None) -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass

    bundle = find_local_ca_bundle(root)
    if bundle is None:
        return
    for name in TLS_CERT_ENV_VARS:
        os.environ.setdefault(name, str(bundle))


@contextmanager
def online_huggingface_download_env(root: Path | None = None):
    previous = {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS + TLS_CERT_ENV_VARS}
    try:
        for name in HF_OFFLINE_ENV_VARS:
            os.environ.pop(name, None)
        configure_download_tls(root)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
        with online_huggingface_download_env(root):
            returned_path = downloader(
                repo_id=spec.repo_id,
                local_dir=str(staging),
                local_files_only=False,
                force_download=True,
            )
        verify_downloaded_model(root, spec, staging, returned_path)
        copy_staged_model_to_target(staging, target)
        verify_downloaded_model(root, spec, target, returned_path)
        downloaded.append(spec.name)
        on_event(PipProgressEvent("Model downloaded", detail, int((index / total) * 100), detail))
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
    for line in process.stdout:
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

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    window = tk.Tk()
    window.title("Offline Meeting Transcriber Setup")
    window.geometry("860x560")
    window.minsize(820, 540)
    window.resizable(False, False)
    window.configure(bg="#f7f7f7")

    style = ttk.Style(window)
    style.theme_use("clam")
    style.configure("Setup.Horizontal.TProgressbar", troughcolor="#e1e1e1", background="#2d6cdf")
    style.configure("Detail.Horizontal.TProgressbar", troughcolor="#e1e1e1", background="#4f9cff")

    step_states = {step: StepState.PENDING for step in BOOTSTRAP_STEPS}
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_index = 0
    current_step = BOOTSTRAP_STEPS[0]
    detail_progress_value = tk.IntVar(value=0)
    overall_progress_value = tk.IntVar(value=0)
    package_detail = tk.StringVar(value=setup_install_summary(package_root))
    command_detail = tk.StringVar(value="")
    eta_detail = tk.StringVar(value="")
    final_message_active = tk.BooleanVar(value=False)
    ui_thread = threading.current_thread()
    step_diagnostics = {step: StepDiagnostic(step) for step in BOOTSTRAP_STEPS}
    tooltip_refs = []
    pip_tracker: dict[str, PipInstallProgressTracker | None] = {"value": None}

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
        style="Setup.Horizontal.TProgressbar",
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

    install_title = tk.Label(
        install_frame,
        text="Install Offline Meeting Transcriber?",
        font=("Segoe UI", 16, "bold"),
        anchor="w",
        bg="#ffffff",
        fg="#1f1f1f",
    )
    install_title.pack(fill="x", padx=20, pady=(14, 4))

    app_type = tk.Label(
        install_frame,
        text="Local Windows App",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#0969da",
    )
    app_type.pack(fill="x", padx=20)

    publisher = tk.Label(
        install_frame,
        text=f"Publisher: {APP_PUBLISHER}",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#333333",
    )
    publisher.pack(fill="x", padx=20)

    version = tk.Label(
        install_frame,
        text=f"Version: {APP_VERSION}",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#333333",
    )
    version.pack(fill="x", padx=20, pady=(0, 10))

    models_title = tk.Label(
        install_frame,
        text="Models to install locally:",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#222222",
    )
    models_title.pack(fill="x", padx=20, pady=(0, 4))

    for item in setup_model_list():
        tk.Label(
            install_frame,
            text=f"- {item}",
            font=("Segoe UI", 10),
            anchor="w",
            bg="#ffffff",
            fg="#666666",
        ).pack(fill="x", padx=28)

    packages_title = tk.Label(
        install_frame,
        text="Python packages to install locally:",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#222222",
    )
    packages_title.pack(fill="x", padx=20, pady=(10, 4))

    packages_box = tk.Frame(install_frame, bg="#ffffff")
    packages_box.pack(fill="x", padx=20)
    package_list = tk.Text(
        packages_box,
        height=4,
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
        install_frame,
        text="Internet required for first setup downloads. Runtime stays local/offline.",
        font=("Segoe UI", 10),
        anchor="w",
        bg="#ffffff",
        fg="#333333",
    )
    install_note.pack(fill="x", padx=20, pady=(8, 52))

    install_button_row = tk.Frame(install_frame, bg="#ffffff")
    install_button_row.place(relx=0, rely=1, relwidth=1, anchor="sw", y=-16)

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
        style="Detail.Horizontal.TProgressbar",
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
    package_text.pack(fill="both", expand=True, padx=18, pady=(0, 14))

    button_row = tk.Frame(progress_frame, bg="#ffffff")
    button_row.pack(fill="x", padx=18, pady=(0, 6))

    model_button_row = tk.Frame(progress_frame, bg="#ffffff")
    model_button_row.pack(fill="x", padx=18, pady=(0, 18))

    def set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

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

    def render_steps() -> None:
        done_count = sum(1 for state in step_states.values() if state in (StepState.DONE, StepState.WARNING))
        overall_progress_value.set(int(done_count / len(BOOTSTRAP_STEPS) * 100))
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
            if detail:
                package_detail.set(detail)
                set_text(package_text, detail)
            if progress is not None:
                set_detail_bar_indeterminate(False)
                detail_progress_value.set(progress)
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
        def apply_event() -> None:
            set_command_text(event.phase)
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
            if event.progress_percent is not None:
                if event.progress_percent == 100:
                    set_detail_bar_indeterminate(False)
                    detail_progress_value.set(100)
                elif not (tracker is not None and tracker.installing_started):
                    current_progress = progress if progress is not None else detail_progress_value.get()
                    detail_progress_value.set(max(current_progress, event.progress_percent))
            render_steps()

        window.after(0, apply_event)

    def update_from_model_download(event: PipProgressEvent) -> None:
        def apply_event() -> None:
            set_command_text(event.phase)
            if event.detail:
                package_detail.set(event.detail)
                set_text(package_text, event.detail)
                step_diagnostics[BOOTSTRAP_STEPS[4]].detail = event.detail
            step_diagnostics[BOOTSTRAP_STEPS[4]].last_output = event.raw_line or event.detail or event.phase
            if event.progress_percent is not None:
                set_detail_bar_indeterminate(False)
                detail_progress_value.set(event.progress_percent)
            render_steps()

        window.after(0, apply_event)

    def finish_with_launch_countdown() -> None:
        final_message_active.set(True)
        step_states[BOOTSTRAP_STEPS[-1]] = StepState.RUNNING
        render_steps()

        remaining = {"seconds": IMPORTANT_MESSAGE_SECONDS}

        def countdown() -> None:
            seconds = remaining["seconds"]
            detail_title.config(text="Prerequisites setup done")
            command_detail.set("")
            set_text(command_text, "")
            eta_detail.set("")
            set_detail_bar_indeterminate(False)
            detail = f"Launching transcriber in {seconds} seconds..."
            package_detail.set(detail)
            set_text(package_text, detail)
            detail_progress_value.set(100)
            if seconds <= 0:
                step_states[BOOTSTRAP_STEPS[-1]] = StepState.DONE
                (root / SETUP_MARKER).write_text("complete\n", encoding="utf-8")
                window.destroy()
                raise SystemExit(launch_gui(root))
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

    def run_setup_flow() -> None:
        try:
            important_message(BOOTSTRAP_STEPS[0], f"Using Python: {sys.executable}")
            set_step(BOOTSTRAP_STEPS[0], StepState.DONE, "Python runtime ready", 100)

            important_message(BOOTSTRAP_STEPS[1], f"Preparing folders under:\n{root}")
            ensure_portable_layout(root, package_root)
            set_step(BOOTSTRAP_STEPS[1], StepState.DONE, "App folders ready", 100)

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

                run_on_ui(show_command)
                time.sleep(IMPORTANT_MESSAGE_SECONDS)
                pip_tracker["value"] = PipInstallProgressTracker()
                result = run_pip_install(root, update_from_pip, package_root)
                if result.code != 0:
                    log_path = write_setup_error_log(
                        root,
                        BOOTSTRAP_STEPS[3],
                        f"pip failed with exit code {result.code}",
                        result.command,
                        result.recent_output,
                    )
                    step_diagnostics[BOOTSTRAP_STEPS[3]].command = result.command
                    step_diagnostics[BOOTSTRAP_STEPS[3]].last_output = "\n".join(result.recent_output[-3:])
                    set_step(
                        BOOTSTRAP_STEPS[3],
                        StepState.ERROR,
                        f"pip failed with exit code {result.code}\nError details saved to {log_path}",
                        100,
                    )
                    return
                set_step(BOOTSTRAP_STEPS[3], StepState.DONE, "Python packages installed", 100)
            else:
                set_step(BOOTSTRAP_STEPS[2], StepState.DONE, "Python packages ready", 100)
                set_step(BOOTSTRAP_STEPS[3], StepState.DONE, "No install needed", 100)

            important_message(BOOTSTRAP_STEPS[4], "Downloading default local AI models...")
            set_step(BOOTSTRAP_STEPS[4], StepState.RUNNING, "Downloading default models", 0)
            try:
                downloaded = download_default_models(root, update_from_model_download)
            except Exception as exc:
                log_path = write_setup_error_log(
                    root,
                    BOOTSTRAP_STEPS[4],
                    f"Model download failed: {exc}",
                    "",
                    [str(exc)],
                )
                set_step(
                    BOOTSTRAP_STEPS[4],
                    StepState.ERROR,
                    f"Model download failed:\n{exc}\nError details saved to {log_path}",
                    100,
                )
                return
            model_detail = "Downloaded: " + ", ".join(downloaded) if downloaded else "Default models already present"
            set_step(BOOTSTRAP_STEPS[4], StepState.DONE, model_detail, 100)

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
                return
            else:
                set_step(BOOTSTRAP_STEPS[5], StepState.DONE, "Model folders ready", 100)

            important_message(BOOTSTRAP_STEPS[6], "Saving setup state and preparing launch...")
            set_step(BOOTSTRAP_STEPS[6], StepState.DONE, "Setup complete", 100)
            window.after(0, finish_with_launch_countdown)
        except Exception as exc:
            log_path = write_setup_error_log(root, current_step, f"Setup failed: {exc}")
            window.after(
                0,
                lambda: set_step(current_step, StepState.ERROR, f"Setup failed:\n{exc}\nError details saved to {log_path}", 100),
            )

    def start_flow() -> None:
        for child in install_button_row.winfo_children():
            child.config(state="disabled")
        install_frame.pack_forget()
        progress_frame.pack(fill="both", expand=True)
        set_text(command_text, "")
        set_text(package_text, "")
        detail_progress_value.set(0)
        eta_detail.set("")
        set_detail_bar_indeterminate(False)
        render_steps()
        for child in button_row.winfo_children():
            child.config(state="disabled")
        for child in model_button_row.winfo_children():
            child.config(state="disabled")
        threading.Thread(target=run_setup_flow, daemon=True).start()

    cancel_button = tk.Button(install_button_row, text="Cancel", width=14, command=window.destroy)
    cancel_button.pack(side="right", padx=(0, 20))
    install_button = tk.Button(install_button_row, text="Install", width=14, command=start_flow)
    install_button.pack(side="right", padx=(0, 8))

    set_text(command_text, command_detail.get())
    set_text(package_text, "")

    render_steps()
    tick_spinner()
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_bootstrap())
