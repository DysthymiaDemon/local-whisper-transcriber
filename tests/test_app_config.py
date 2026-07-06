import json
import os
import tempfile
import unittest
from pathlib import Path

from app_config import default_portable_config, load_portable_config


class PortableConfigTests(unittest.TestCase):
    def test_defaults_point_to_folders_inside_app_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            config = default_portable_config(root)

        self.assertEqual(config.whisper_model_dir, os.path.join(tmp, "models", "faster-whisper"))
        self.assertEqual(config.pyannote_pipeline_dir, os.path.join(tmp, "models", "pyannote-pipeline"))
        self.assertEqual(config.pyannote_embedding_model_dir, os.path.join(tmp, "models", "pyannote-embedding"))
        self.assertEqual(config.output_file, os.path.join(tmp, "transcripts", "meeting_transcript.txt"))

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
                        "pyannote_pipeline_dir": "custom/pipeline",
                        "pyannote_embedding_model_dir": "custom/embedding",
                        "output_file": "custom-output/transcript.txt",
                    }
                ),
                encoding="utf-8",
            )

            config = load_portable_config(root)

        self.assertEqual(config.whisper_model_dir, os.path.join(tmp, "custom", "whisper"))
        self.assertEqual(config.pyannote_pipeline_dir, os.path.join(tmp, "custom", "pipeline"))
        self.assertEqual(config.pyannote_embedding_model_dir, os.path.join(tmp, "custom", "embedding"))
        self.assertEqual(config.output_file, os.path.join(tmp, "custom-output", "transcript.txt"))


if __name__ == "__main__":
    unittest.main()
