from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from app_config import application_root, source_root


REQUIRED_IMPORTS: dict[str, str] = {
    "PySide6": "PySide6",
    "sounddevice": "sounddevice",
    "faster-whisper": "faster_whisper",
    "pyannote.audio": "pyannote.audio",
    "torch": "torch",
}

MODEL_DIRS = ("faster-whisper", "pyannote-pipeline", "pyannote-embedding")
SETUP_MARKER = ".setup_complete"
IMPORTANT_MESSAGE_SECONDS = 5

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


def runtime_root() -> Path:
    return application_root()


def installer_source_root() -> Path:
    return source_root()


def missing_imports(required: Mapping[str, str] = REQUIRED_IMPORTS) -> list[str]:
    missing: list[str] = []
    for label, module_name in required.items():
        try:
            if importlib.util.find_spec(module_name) is None:
                missing.append(label)
        except ModuleNotFoundError:
            missing.append(label)
    return missing


def ensure_portable_layout(root: Path, template_root: Path | None = None) -> Path:
    models = root / "models"
    for name in MODEL_DIRS:
        (models / name).mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)

    config_path = root / "config.json"
    if not config_path.exists():
        template_path = (template_root or root) / "config.template.json"
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
        ready = folder.is_dir() and any(folder.iterdir())
        status[name] = BootstrapStatus.READY if ready else BootstrapStatus.MISSING
    return status


def launch_gui(root: Path) -> int:
    os.chdir(root)
    os.environ["LOCAL_WHISPER_APP_ROOT"] = str(root)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from meeting_transcriber_gui import main

    return main()


def run_pip_install(root: Path, on_event, package_root: Path | None = None) -> int:
    package_root = package_root or installer_source_root()
    requirements_path = package_root / "requirements.txt"
    cmd = [sys.executable, "-m", "pip", "install", "--user", "-r", str(requirements_path)]
    on_event(PipProgressEvent("Running command", " ".join(cmd), 0, " ".join(cmd)))
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
    for line in process.stdout:
        on_event(parser.parse(line.rstrip()))
    return process.wait()


def needs_setup(root: Path, required: Mapping[str, str] = REQUIRED_IMPORTS) -> bool:
    ensure_portable_layout(root, installer_source_root())
    return bool(missing_imports(required)) or not (root / SETUP_MARKER).exists()


def run_bootstrap() -> int:
    root = runtime_root()
    package_root = installer_source_root()
    ensure_portable_layout(root, package_root)
    if not needs_setup(root):
        return launch_gui(root)

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    window = tk.Tk()
    window.title("Offline Meeting Transcriber Setup")
    window.geometry("780x520")
    window.minsize(720, 500)
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
    final_message_active = tk.BooleanVar(value=False)
    ui_thread = threading.current_thread()

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
        wraplength=720,
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

    detail_title = tk.Label(
        detail_frame,
        text="Preparing setup",
        font=("Segoe UI", 14, "bold"),
        anchor="w",
        bg="#ffffff",
        fg="#1f1f1f",
    )
    detail_title.pack(fill="x", padx=18, pady=(18, 6))

    command_label = tk.Label(
        detail_frame,
        textvariable=command_detail,
        font=("Cascadia Mono", 9),
        anchor="w",
        justify="left",
        wraplength=350,
        bg="#ffffff",
        fg="#505050",
    )
    command_label.pack(fill="x", padx=18, pady=(0, 10))

    detail_bar = ttk.Progressbar(
        detail_frame,
        maximum=100,
        variable=detail_progress_value,
        style="Detail.Horizontal.TProgressbar",
    )
    detail_bar.pack(fill="x", padx=18, pady=(0, 12))

    package_label = tk.Label(
        detail_frame,
        textvariable=package_detail,
        font=("Segoe UI", 10),
        anchor="nw",
        justify="left",
        wraplength=360,
        bg="#ffffff",
        fg="#222222",
    )
    package_label.pack(fill="both", expand=True, padx=18, pady=(0, 14))

    button_row = tk.Frame(detail_frame, bg="#ffffff")
    button_row.pack(fill="x", padx=18, pady=(0, 18))

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
            detail_title.config(text=step)
            if detail:
                package_detail.set(detail)
            if progress is not None:
                detail_progress_value.set(progress)
            render_steps()
            window.update_idletasks()

        run_on_ui(apply)

    def important_message(step: str, message: str) -> None:
        set_step(step, StepState.RUNNING, message, None)
        run_on_ui(lambda: command_detail.set(message))
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
        package_detail.set(f"Recorded selected folder for {model_name}: {selected}")
        render_steps()

    def open_model_folder() -> None:
        model_root = root / "models"
        os.startfile(model_root)
        package_detail.set(f"Opened model folder:\n{model_root}")

    def update_from_pip(event: PipProgressEvent) -> None:
        def apply_event() -> None:
            command_detail.set(event.phase)
            if event.detail:
                package_detail.set(event.detail)
            if event.progress_percent is not None:
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
            package_detail.set(f"Launching transcriber in {seconds} seconds...")
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
        missing = missing_imports()
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
            missing = missing_imports()
            if missing:
                set_step(BOOTSTRAP_STEPS[2], StepState.WARNING, "Missing: " + ", ".join(missing), 100)
                set_step(BOOTSTRAP_STEPS[3], StepState.RUNNING, "Installing missing packages", 0)
                command = f"{sys.executable} -m pip install --user -r {package_root / 'requirements.txt'}"
                command_detail.set("Running command:\n" + command)
                time.sleep(IMPORTANT_MESSAGE_SECONDS)
                code = run_pip_install(root, update_from_pip, package_root)
                if code != 0:
                    set_step(BOOTSTRAP_STEPS[3], StepState.ERROR, f"pip failed with exit code {code}", 100)
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
            window.after(
                0,
                lambda: set_step(current_step, StepState.ERROR, f"Setup failed:\n{exc}", 100),
            )

    def start_flow() -> None:
        for child in button_row.winfo_children():
            child.config(state="disabled")
        threading.Thread(target=run_setup_flow, daemon=True).start()

    install_button = tk.Button(button_row, text="Start setup", command=start_flow)
    install_button.pack(side="left", padx=(0, 8))
    tk.Button(button_row, text="Open model folder", command=open_model_folder).pack(side="left", padx=(0, 8))
    for model in MODEL_DIRS:
        tk.Button(button_row, text=f"Set {model}", command=lambda name=model: choose_model_folder(name)).pack(
            side="left", padx=(0, 8)
        )
    tk.Button(button_row, text="Launch GUI", command=launch_if_ready).pack(side="right")

    render_steps()
    tick_spinner()
    window.after(400, start_flow)
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_bootstrap())
