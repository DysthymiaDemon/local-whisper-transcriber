from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping

from app_config import append_error_log, application_root, source_root


REQUIRED_IMPORTS: dict[str, str] = {
    "PySide6": "PySide6",
    "sounddevice": "sounddevice",
    "faster-whisper": "faster_whisper",
    "pyannote.audio": "pyannote.audio",
    "torch": "torch",
}

MODEL_DIRS = ("faster-whisper", "pyannote-pipeline", "pyannote-embedding")
MODEL_GUIDES: dict[str, str] = {
    "faster-whisper": (
        "Copy a CTranslate2 faster-whisper model here.\n\n"
        "Required file:\n"
        "- model.bin\n\n"
        "Example source model: Systran/faster-whisper-small\n"
    ),
    "pyannote-pipeline": (
        "Copy the local pyannote diarization pipeline here.\n\n"
        "Required file:\n"
        "- config.yaml\n\n"
        "The config.yaml must reference local model paths only.\n"
    ),
    "pyannote-embedding": (
        "Copy the local pyannote embedding model here.\n\n"
        "Required files:\n"
        "- config.yaml\n"
        "- pytorch_model.bin or model.safetensors\n"
    ),
}
SETUP_MARKER = ".setup_complete"
IMPORTANT_MESSAGE_SECONDS = 5
RUNTIME_DIR = ".runtime"
VENV_DIR = "venv"

BOOTSTRAP_STEPS = [
    "Checking Python runtime",
    "Preparing app folders",
    "Checking Python packages",
    "Installing Python packages",
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

    def record(self, event: PipProgressEvent, now: float | None = None) -> tuple[int, str]:
        self.event_count += 1
        current = time.time() if now is None else now
        if event.progress_percent == 100:
            self.progress = 100
        elif event.progress_percent == 0:
            self.progress = max(self.progress, 20)
        else:
            self.progress = min(95, max(self.progress + 3, 8 + self.event_count * 4))

        elapsed = current - self.start_time
        if self.progress >= 100:
            eta = "ETA complete"
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


def local_runtime_dir(root: Path) -> Path:
    return root / RUNTIME_DIR


def local_venv_dir(root: Path) -> Path:
    return local_runtime_dir(root) / VENV_DIR


def local_venv_python(root: Path, prefer_windowed: bool = False) -> Path:
    scripts = local_venv_dir(root) / "Scripts"
    preferred = scripts / ("pythonw.exe" if prefer_windowed else "python.exe")
    if preferred.exists():
        return preferred
    return scripts / "python.exe"


def running_in_local_venv(root: Path) -> bool:
    if getattr(sys, "frozen", False):
        return True
    try:
        executable = Path(sys.executable).resolve()
        venv = local_venv_dir(root).resolve()
        return executable == local_venv_python(root).resolve() or venv in executable.parents
    except OSError:
        return False


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

    python_path = local_venv_python(root)
    if not python_path.exists():
        return list(required.keys())

    probe = (
        "import importlib.util, json, sys; "
        f"required = {json.dumps(dict(required))}; "
        "missing = [label for label, module in required.items() if importlib.util.find_spec(module) is None]; "
        "print(json.dumps(missing)); "
        "sys.exit(1 if missing else 0)"
    )
    result = subprocess.run(
        [str(python_path), "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    try:
        parsed = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return list(required.keys())
    return [item for item in parsed if isinstance(item, str)]


def ensure_portable_layout(root: Path, template_root: Path | None = None) -> Path:
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


def model_folder_status(root: Path) -> dict[str, BootstrapStatus]:
    status: dict[str, BootstrapStatus] = {}
    for name in MODEL_DIRS:
        folder = root / "models" / name
        if name == "faster-whisper":
            ready = (folder / "model.bin").is_file()
        elif name == "pyannote-pipeline":
            ready = (folder / "config.yaml").is_file()
        else:
            ready = (folder / "config.yaml").is_file() and any(
                (folder / filename).is_file() for filename in ("pytorch_model.bin", "model.safetensors")
            )
        status[name] = BootstrapStatus.READY if ready else BootstrapStatus.MISSING
    return status


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

    if not running_in_local_venv(root):
        python_path = local_venv_python(root, prefer_windowed=True)
        if python_path.exists():
            update_startup_splash(splash, splash_status, "Switching to local app Python...")
            close_startup_splash(splash)
            script_path = app_source_dir(root) / "bootstrap_launcher.py"
            env = os.environ.copy()
            env["LOCAL_WHISPER_APP_ROOT"] = str(root)
            env["LOCAL_WHISPER_SOURCE_ROOT"] = str(root)
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")
            env.setdefault("HF_DATASETS_OFFLINE", "1")
            return subprocess.call([str(python_path), str(script_path)], cwd=str(root), env=env)

    if splash is None:
        try:
            splash, splash_status = create_startup_splash()
        except Exception:
            splash = None
            splash_status = None
    update_startup_splash(splash, splash_status, "Checking local app files...")

    app_dir = str(app_source_dir())
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
    venv_path = local_venv_dir(root)
    if not local_venv_python(root).exists():
        venv_cmd = [sys.executable, "-m", "venv", str(venv_path)]
        venv_command = subprocess.list2cmdline(venv_cmd)
        on_event(PipProgressEvent("Creating local Python environment", venv_command, 0, venv_command))
        venv_result = subprocess.run(
            venv_cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if venv_result.returncode != 0:
            recent = [line for line in venv_result.stdout.splitlines() if line.strip()][-25:]
            return PipInstallResult(venv_result.returncode, venv_command, recent)

    python_path = local_venv_python(root)
    cmd = [str(python_path), "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements_path)]
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
    return bool(missing_runtime_imports(root, required)) or not (root / SETUP_MARKER).exists()


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
    update_startup_splash(splash, splash_status, "Checking first-time setup status...")
    setup_missing = not (root / SETUP_MARKER).exists()
    if not missing_packages and not setup_missing:
        update_startup_splash(splash, splash_status, "Dependencies ready.")
        return launch_gui(root, splash, splash_status)

    if missing_packages:
        update_startup_splash(splash, splash_status, "Dependencies missing. Opening first-time setup...")
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
    spinner_frames = ["|", "/", "-", "\\"]
    spinner_index = 0
    current_step = BOOTSTRAP_STEPS[0]
    detail_progress_value = tk.IntVar(value=0)
    overall_progress_value = tk.IntVar(value=0)
    package_detail = tk.StringVar(value="Waiting to start")
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

    detail_title = tk.Label(
        detail_frame,
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
        detail_frame,
        height=4,
        font=("Cascadia Mono", 9),
        foreground="#505050",
        wrap="char",
    )
    command_text.pack(fill="x", padx=18, pady=(0, 10))

    detail_bar = ttk.Progressbar(
        detail_frame,
        maximum=100,
        variable=detail_progress_value,
        style="Detail.Horizontal.TProgressbar",
    )
    detail_bar.pack(fill="x", padx=18, pady=(0, 12))

    eta_label = tk.Label(
        detail_frame,
        textvariable=eta_detail,
        font=("Segoe UI", 9),
        anchor="w",
        justify="left",
        bg="#ffffff",
        fg="#666666",
    )
    eta_label.pack(fill="x", padx=18, pady=(0, 8))

    package_text = readonly_text(
        detail_frame,
        height=7,
        font=("Segoe UI", 10),
        foreground="#222222",
        wrap="word",
    )
    package_text.pack(fill="both", expand=True, padx=18, pady=(0, 14))

    button_row = tk.Frame(detail_frame, bg="#ffffff")
    button_row.pack(fill="x", padx=18, pady=(0, 6))

    model_button_row = tk.Frame(detail_frame, bg="#ffffff")
    model_button_row.pack(fill="x", padx=18, pady=(0, 18))

    def set_text(widget: tk.Text, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

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
                label.config(text=f"{spinner_frames[spinner_index % len(spinner_frames)]} {step}", fg="#1a5fb4")
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
                detail_progress_value.set(progress)
            if event.progress_percent is not None:
                current_progress = progress if progress is not None else detail_progress_value.get()
                detail_progress_value.set(max(current_progress, event.progress_percent))
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
                        str(local_venv_python(root)),
                        "-m",
                        "pip",
                        "install",
                        "--no-cache-dir",
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

            important_message(BOOTSTRAP_STEPS[4], "Checking local AI model folders...")
            models = model_folder_status(root)
            missing_models = [name for name, value in models.items() if value == BootstrapStatus.MISSING]
            if missing_models:
                set_step(
                    BOOTSTRAP_STEPS[4],
                    StepState.WARNING,
                    "Model folders still need files:\n" + "\n".join(missing_models),
                    100,
                )
            else:
                set_step(BOOTSTRAP_STEPS[4], StepState.DONE, "Model folders ready", 100)

            important_message(BOOTSTRAP_STEPS[5], "Saving setup state and preparing launch...")
            set_step(BOOTSTRAP_STEPS[5], StepState.DONE, "Setup complete", 100)
            window.after(0, finish_with_launch_countdown)
        except Exception as exc:
            log_path = write_setup_error_log(root, current_step, f"Setup failed: {exc}")
            window.after(
                0,
                lambda: set_step(current_step, StepState.ERROR, f"Setup failed:\n{exc}\nError details saved to {log_path}", 100),
            )

    def start_flow() -> None:
        for child in button_row.winfo_children():
            child.config(state="disabled")
        for child in model_button_row.winfo_children():
            child.config(state="disabled")
        threading.Thread(target=run_setup_flow, daemon=True).start()

    install_button = tk.Button(button_row, text="Start setup", command=start_flow)
    install_button.pack(side="left", padx=(0, 8))
    tk.Button(button_row, text="Open model folder", command=open_model_folder).pack(side="left", padx=(0, 8))
    tk.Button(button_row, text="Launch GUI", command=launch_if_ready).pack(side="right")
    for model in MODEL_DIRS:
        tk.Button(model_button_row, text=f"Set {model}", command=lambda name=model: choose_model_folder(name)).pack(
            side="left", padx=(0, 8)
        )

    set_text(command_text, command_detail.get())
    set_text(package_text, package_detail.get())

    render_steps()
    tick_spinner()
    window.after(400, start_flow)
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_bootstrap())
