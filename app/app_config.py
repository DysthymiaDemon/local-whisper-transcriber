from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from transcriber_engine import EngineConfig


CONFIG_FILE_NAME = "config.json"
ERROR_LOG_FILE_NAME = "error_log.txt"
APP_FOLDER_NAME = "OfflineMeetingTranscriber"
ENV_APP_ROOT = "LOCAL_WHISPER_APP_ROOT"
ENV_SOURCE_ROOT = "LOCAL_WHISPER_SOURCE_ROOT"


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


def default_portable_config(root: Path | None = None) -> EngineConfig:
    app_root = (root or application_root()).resolve()
    models_root = app_root / "models"
    return EngineConfig(
        whisper_model_dir=str(models_root / "faster-whisper"),
        pyannote_pipeline_dir=str(models_root / "pyannote-pipeline"),
        pyannote_embedding_model_dir=str(models_root / "pyannote-embedding"),
        output_file=str(app_root / "transcripts" / "meeting_transcript.txt"),
    )


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
    for key in (
        "whisper_model_dir",
        "pyannote_pipeline_dir",
        "pyannote_embedding_model_dir",
        "output_file",
    ):
        value = overrides.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            overrides[key] = str(app_root / value)
    return replace(config, **overrides)


def append_error_log(
    root: Path | str | None,
    context: str,
    message: str,
    details: Mapping[str, Any] | Iterable[str] | str | None = None,
) -> Path:
    app_root = Path(root).expanduser().resolve() if root else application_root()
    app_root.mkdir(parents=True, exist_ok=True)
    log_path = app_root / ERROR_LOG_FILE_NAME

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
