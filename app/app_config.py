from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from transcriber_engine import EngineConfig


CONFIG_FILE_NAME = "config.json"
ERROR_LOG_FILE_NAME = "error_log.txt"
APP_FOLDER_NAME = "OfflineMeetingTranscriber"
ENV_APP_ROOT = "LOCAL_WHISPER_APP_ROOT"
ENV_SOURCE_ROOT = "LOCAL_WHISPER_SOURCE_ROOT"
ENV_LOG_ROOT = "LOCAL_WHISPER_LOG_ROOT"


def source_root() -> Path:
    configured = os.environ.get(ENV_SOURCE_ROOT)
    if configured:
        return Path(configured).expanduser().resolve()

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    module_dir = Path(__file__).resolve().parent
    if module_dir.name == "app":
        return module_dir.parent
    return module_dir


def default_local_app_root() -> Path:
    configured = os.environ.get(ENV_APP_ROOT)
    if configured:
        return Path(configured).expanduser().resolve()

    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        return (Path(user_profile) / "Apps" / APP_FOLDER_NAME).resolve()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return (Path(local_app_data) / APP_FOLDER_NAME).resolve()

    return (Path.home() / "Apps" / APP_FOLDER_NAME).resolve()


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return default_local_app_root()


def error_log_root(app_root: Path | None = None) -> Path:
    configured = os.environ.get(ENV_LOG_ROOT)
    if configured:
        return Path(configured).expanduser().resolve()
    if app_root is not None and app_root.name.lower() == APP_FOLDER_NAME.lower():
        return app_root.parent.resolve()
    return application_root()


def default_portable_config(root: Path | None = None) -> EngineConfig:
    app_root = (root or application_root()).resolve()
    models_root = app_root / "models"
    return EngineConfig(
        whisper_model_dir=str(models_root / "faster-whisper"),
        output_file=str(app_root / "transcripts" / "meeting_transcript.txt"),
    )


def _uses_generated_latency_defaults(raw: Mapping[str, Any]) -> bool:
    try:
        chunk_seconds = float(raw.get("chunk_seconds"))
        overlap_seconds = float(raw.get("overlap_seconds"))
    except (TypeError, ValueError):
        return False
    return (chunk_seconds, overlap_seconds) in {
        (8.0, 2.0),
        (4.0, 0.5),
        (3.0, 0.25),
        (5.0, 0.5),
    }


def load_portable_config(root: Path | None = None) -> EngineConfig:
    app_root = (root or application_root()).resolve()
    config = default_portable_config(app_root)
    config_path = app_root / CONFIG_FILE_NAME
    if not config_path.exists():
        return config

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")

    allowed = set(EngineConfig.__dataclass_fields__.keys())
    overrides: dict[str, Any] = {key: value for key, value in raw.items() if key in allowed}
    if _uses_generated_latency_defaults(raw):
        overrides["chunk_seconds"] = config.chunk_seconds
        overrides["overlap_seconds"] = config.overlap_seconds
    for key in (
        "whisper_model_dir",
        "output_file",
    ):
        value = overrides.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            overrides[key] = str(app_root / value)
    return replace(config, **overrides)


def load_portable_config_safely(root: Path | None = None) -> tuple[EngineConfig, str | None]:
    """Load config.json, falling back to defaults when the file is unreadable or has wrong types."""
    app_root = (root or application_root()).resolve()
    try:
        config = load_portable_config(app_root)
        _check_config_types(config)
        return config, None
    except Exception as exc:
        return default_portable_config(app_root), f"{app_root / CONFIG_FILE_NAME} could not be used: {exc}"


def _check_config_types(config: EngineConfig) -> None:
    numeric = ("sample_rate", "chunk_seconds", "overlap_seconds", "cpu_threads", "num_workers")
    for name in numeric:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, got {value!r}")
    for name in ("whisper_model_dir", "output_file", "compute_type", "device", "capture_mode"):
        if not isinstance(getattr(config, name), str):
            raise ValueError(f"{name} must be text, got {getattr(config, name)!r}")
    for name in ("input_device", "system_device_index"):
        value = getattr(config, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name} must be a device number or null, got {value!r}")


def save_portable_config(config: EngineConfig, root: Path | None = None) -> Path:
    app_root = (root or application_root()).resolve()
    config_path = app_root / CONFIG_FILE_NAME
    raw = asdict(config)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{CONFIG_FILE_NAME}.", suffix=".tmp", dir=str(app_root))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(raw, indent=2) + "\n")
        os.replace(tmp_name, config_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return config_path


def append_error_log(
    root: Path | str | None,
    context: str,
    message: str,
    details: Mapping[str, Any] | Iterable[str] | str | None = None,
) -> Path:
    app_root = Path(root).expanduser().resolve() if root else application_root()
    log_root = error_log_root(app_root)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / ERROR_LOG_FILE_NAME

    lines = [
        "=" * 72,
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"app_root: {app_root}",
        f"context: {context}",
        f"message: {message}",
    ]
    if details:
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
    lines.append("")

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return log_path
