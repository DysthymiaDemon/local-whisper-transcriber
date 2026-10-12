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
    load_portable_config_safely,
    save_portable_config,
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
        self.assertEqual(config.chunk_seconds, 10.0)
        self.assertEqual(config.overlap_seconds, 1.0)

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

        self.assertEqual(config.chunk_seconds, 10.0)
        self.assertEqual(config.overlap_seconds, 1.0)
        self.assertFalse(hasattr(config, "diarization_backend"))

    def test_previous_transcription_defaults_migrate_to_current_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "chunk_seconds": 4.0,
                        "overlap_seconds": 0.5,
                    }
                ),
                encoding="utf-8",
            )

            config = load_portable_config(root)

        self.assertEqual(config.chunk_seconds, 10.0)
        self.assertEqual(config.overlap_seconds, 1.0)

    def test_previous_short_latency_defaults_migrate_to_current_accuracy_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "chunk_seconds": 3.0,
                        "overlap_seconds": 0.25,
                    }
                ),
                encoding="utf-8",
            )

            config = load_portable_config(root)

        self.assertEqual(config.chunk_seconds, 10.0)
        self.assertEqual(config.overlap_seconds, 1.0)


    def test_generated_five_second_defaults_migrate_to_ten_second_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"chunk_seconds": 5.0, "overlap_seconds": 0.5}), encoding="utf-8"
            )

            config = load_portable_config(root)

        self.assertEqual((config.chunk_seconds, config.overlap_seconds), (10.0, 1.0))

    def test_broken_config_falls_back_to_defaults_with_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{not json", encoding="utf-8")
            config, problem = load_portable_config_safely(root)
            (root / "config.json").write_text(json.dumps({"chunk_seconds": "5"}), encoding="utf-8")
            typed_config, typed_problem = load_portable_config_safely(root)

        self.assertEqual(config, default_portable_config(root))
        self.assertIn("config.json", problem)
        self.assertEqual(typed_config.chunk_seconds, 10.0)
        self.assertIn("chunk_seconds", typed_problem)

    def test_save_writes_complete_json_without_leftover_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = save_portable_config(default_portable_config(root), root)

            saved = json.loads(path.read_text(encoding="utf-8"))
            leftovers = [item.name for item in root.iterdir() if item.suffix == ".tmp"]

        self.assertEqual(saved["chunk_seconds"], 10.0)
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
