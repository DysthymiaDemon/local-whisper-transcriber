import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = PROJECT_ROOT / "packaging" / "build_standalone_pyw.py"
LAUNCHER_PATH = PROJECT_ROOT / "Open Offline Meeting Transcriber.pyw"


def load_generator():
    spec = importlib.util.spec_from_file_location("build_standalone_pyw", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandaloneLauncherTests(unittest.TestCase):
    def test_payload_contains_required_app_and_resource_files(self):
        generator = load_generator()

        payload = generator.build_payload(PROJECT_ROOT)
        files = payload["files"]

        self.assertIn("app/bootstrap_launcher.py", files)
        self.assertIn("app/meeting_transcriber_gui.py", files)
        self.assertIn("app/transcriber_engine.py", files)
        self.assertIn("resources/requirements.txt", files)
        self.assertIn("resources/config.template.json", files)

    def test_committed_launcher_payload_is_current(self):
        generator = load_generator()

        expected = generator.generate(PROJECT_ROOT)
        actual = LAUNCHER_PATH.read_text(encoding="utf-8")

        self.assertEqual(actual, expected)

    def test_init_only_extracts_beside_copied_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            copied_launcher = folder / LAUNCHER_PATH.name
            shutil.copy2(LAUNCHER_PATH, copied_launcher)

            result = subprocess.run(
                [sys.executable, str(copied_launcher), "--init-only"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            app_root = folder / "OfflineMeetingTranscriber"
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), app_root.resolve())
            self.assertTrue((app_root / "app" / "bootstrap_launcher.py").is_file())
            self.assertTrue((app_root / "resources" / "requirements.txt").is_file())
            self.assertTrue((app_root / "config.json").is_file())
            self.assertTrue((app_root / "models" / "faster-whisper").is_dir())
            self.assertTrue((app_root / "transcripts").is_dir())

    def test_init_only_uses_current_folder_when_named_like_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_root = Path(tmp) / "OfflineMeetingTranscriber"
            app_root.mkdir()
            copied_launcher = app_root / LAUNCHER_PATH.name
            shutil.copy2(LAUNCHER_PATH, copied_launcher)

            result = subprocess.run(
                [sys.executable, str(copied_launcher), "--init-only"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), app_root.resolve())
            self.assertTrue((app_root / "app" / "bootstrap_launcher.py").is_file())

    def test_init_only_honors_install_root_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            copied_launcher = folder / LAUNCHER_PATH.name
            target_root = folder / "custom-install"
            shutil.copy2(LAUNCHER_PATH, copied_launcher)

            result = subprocess.run(
                [sys.executable, str(copied_launcher), "--init-only", "--install-root", str(target_root)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), target_root.resolve())
            self.assertTrue((target_root / "resources" / "config.template.json").is_file())


if __name__ == "__main__":
    unittest.main()
