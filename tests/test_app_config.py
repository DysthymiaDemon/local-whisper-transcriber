import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from app_config import (
    ENV_APP_ROOT,
    ENV_LOG_ROOT,
    append_error_log,
    default_local_app_root,
    default_portable_config,
    load_portable_config,
)


class PortableConfigTests(unittest.TestCase):
    def test_append_error_log_creates_debug_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = append_error_log(
                tmp,
                "Configuration error",
                "Missing model files",
                {"errors": ["model.bin missing"], "command": "python -m pip"},
            )
            content = log_path.read_text(encoding="utf-8")

        self.assertEqual(log_path.name, "error_log.txt")
        self.assertIn("context: Configuration error", content)
        self.assertIn("message: Missing model files", content)
        self.assertIn("model.bin missing", content)
        self.assertIn("python -m pip", content)

    def test_append_error_log_uses_launcher_folder_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher_root = Path(tmp)
            app_root = launcher_root / "OfflineMeetingTranscriber"
            app_root.mkdir()
            previous = os.environ.get(ENV_LOG_ROOT)
            os.environ[ENV_LOG_ROOT] = str(launcher_root)
            try:
                log_path = append_error_log(app_root, "Setup failed", "pip failed")
            finally:
                if previous is None:
                    os.environ.pop(ENV_LOG_ROOT, None)
                else:
                    os.environ[ENV_LOG_ROOT] = previous
            content = log_path.read_text(encoding="utf-8")

        self.assertEqual(log_path, launcher_root / "error_log.txt")
        self.assertIn(f"app_root: {app_root.resolve()}", content)
        self.assertFalse((app_root / "error_log.txt").exists())

    def test_append_error_log_uses_parent_for_portable_app_root_without_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher_root = Path(tmp)
            app_root = launcher_root / "OfflineMeetingTranscriber"
            app_root.mkdir()
            previous = os.environ.get(ENV_LOG_ROOT)
            os.environ.pop(ENV_LOG_ROOT, None)
            try:
                log_path = append_error_log(app_root, "Configuration error", "Missing models")
            finally:
                if previous is not None:
                    os.environ[ENV_LOG_ROOT] = previous
            content = log_path.read_text(encoding="utf-8")

        self.assertEqual(log_path, launcher_root / "error_log.txt")
        self.assertIn("context: Configuration error", content)
        self.assertFalse((app_root / "error_log.txt").exists())

    def test_default_local_app_root_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get(ENV_APP_ROOT)
            os.environ[ENV_APP_ROOT] = tmp
            try:
                root = default_local_app_root()
            finally:
                if previous is None:
                    os.environ.pop(ENV_APP_ROOT, None)
                else:
                    os.environ[ENV_APP_ROOT] = previous

        self.assertEqual(root, Path(tmp).resolve())

    def test_defaults_point_to_folders_inside_app_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            config = default_portable_config(root)

        self.assertEqual(config.whisper_model_dir, os.path.join(tmp, "models", "faster-whisper"))
        self.assertEqual(config.output_file, os.path.join(tmp, "transcripts", "meeting_transcript.txt"))
        self.assertEqual(config.language, "en")

    def test_config_file_overrides_portable_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_file = root / "config.json"
            config_file.write_text(
                json.dumps(
                    {
                        "whisper_model_dir": "D:/models/whisper",
                        "chunk_seconds": 12,
                        "overlap_seconds": 3,
                    }
                ),
                encoding="utf-8",
            )

            config = load_portable_config(root)

        self.assertEqual(config.whisper_model_dir, "D:/models/whisper")
        self.assertEqual(config.chunk_seconds, 12)
        self.assertEqual(config.overlap_seconds, 3)

    def test_relative_paths_in_config_resolve_against_app_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "whisper_model_dir": "custom/whisper",
                        "output_file": "custom-output/transcript.txt",
                    }
                ),
                encoding="utf-8",
            )

            config = load_portable_config(root)

        self.assertEqual(config.whisper_model_dir, os.path.join(tmp, "custom", "whisper"))
        self.assertEqual(config.output_file, os.path.join(tmp, "custom-output", "transcript.txt"))

    def test_legacy_diarization_config_uses_new_transcription_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "diarization_backend": "local-ecapa",
                        "speaker_embedding_model_dir": "models/speechbrain-ecapa",
                        "chunk_seconds": 8.0,
                        "overlap_seconds": 2.0,
                    }
                ),
                encoding="utf-8",
            )

            config = load_portable_config(root)

        self.assertEqual(config.chunk_seconds, 4.0)
        self.assertEqual(config.overlap_seconds, 0.5)
        self.assertFalse(hasattr(config, "diarization_backend"))


if __name__ == "__main__":
    unittest.main()
