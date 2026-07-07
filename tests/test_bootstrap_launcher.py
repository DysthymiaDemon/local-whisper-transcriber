import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from bootstrap_launcher import (
    APP_PUBLISHER,
    BOOTSTRAP_STEPS,
    CA_BUNDLE_ENV,
    DEFAULT_MODEL_DOWNLOADS,
    HF_OFFLINE_ENV_VARS,
    PipProgressParser,
    PipInstallProgressTracker,
    PipProgressEvent,
    REQUIRED_IMPORTS,
    SETUP_MARKER,
    BootstrapStatus,
    StepDiagnostic,
    StepState,
    build_step_tooltip,
    download_default_models,
    ensure_portable_layout,
    is_onedrive_path,
    local_package_dir,
    local_runtime_dir,
    migrate_legacy_onedrive_runtime,
    missing_imports,
    missing_model_setup_message,
    missing_runtime_imports,
    model_folder_status,
    needs_setup,
    online_huggingface_download_env,
    setup_install_summary,
    setup_package_list,
    verify_downloaded_model,
    write_setup_error_log,
)


class BootstrapLauncherTests(unittest.TestCase):
    def test_missing_imports_reports_unavailable_modules(self):
        result = missing_imports(
            {
                "ok": "json",
                "missing": "definitely_missing_local_transcriber_module",
            }
        )

        self.assertEqual(result, ["missing"])

    def test_required_imports_cover_runtime_stacks(self):
        self.assertIn("PySide6", REQUIRED_IMPORTS)
        self.assertIn("sounddevice", REQUIRED_IMPORTS)
        self.assertIn("faster-whisper", REQUIRED_IMPORTS)
        self.assertIn("speechbrain", REQUIRED_IMPORTS)
        self.assertIn("scikit-learn", REQUIRED_IMPORTS)
        self.assertIn("huggingface_hub", REQUIRED_IMPORTS)
        self.assertIn("truststore", REQUIRED_IMPORTS)
        self.assertNotIn("pyannote.audio", REQUIRED_IMPORTS)
        self.assertIn("torch", REQUIRED_IMPORTS)

    def test_setup_install_summary_lists_publisher_models_and_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = root / "resources"
            resources.mkdir()
            (resources / "requirements.txt").write_text(
                "# ignored\nalpha==1.0\n\nbeta>=2.0\n",
                encoding="utf-8",
            )

            summary = setup_install_summary(root)
            packages = setup_package_list(root)

        self.assertIn(f"Publisher: {APP_PUBLISHER}", summary)
        self.assertIn("Systran/faster-whisper-small -> models\\faster-whisper", summary)
        self.assertIn("speechbrain/spkrec-ecapa-voxceleb -> models\\speechbrain-ecapa", summary)
        self.assertEqual(packages, ["alpha==1.0", "beta>=2.0"])
        self.assertIn("- alpha==1.0", summary)
        self.assertIn("- beta>=2.0", summary)

    def test_setup_initial_buttons_do_not_include_manual_launch_or_model_shortcuts(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertNotIn('text="Launch GUI"', source)
        self.assertNotIn('text="Open model folder"', source)
        self.assertNotIn("text=f\"Set {model}\"", source)

    def test_bootstrap_steps_cover_setup_flow(self):
        self.assertEqual(BOOTSTRAP_STEPS[0], "Checking Python runtime")
        self.assertIn("Installing Python packages", BOOTSTRAP_STEPS)
        self.assertEqual(BOOTSTRAP_STEPS[-1], "Launching transcriber")

    def test_ensure_portable_layout_creates_config_and_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            config_path = ensure_portable_layout(root)

            self.assertTrue(config_path.exists())
            self.assertTrue((root / ".runtime").is_dir())
            self.assertTrue((root / "models" / "faster-whisper").is_dir())
            self.assertTrue((root / "models" / "speechbrain-ecapa").is_dir())
            self.assertTrue((root / "models" / "pyannote-pipeline").is_dir())
            self.assertTrue((root / "models" / "pyannote-embedding").is_dir())
            self.assertTrue((root / "transcripts").is_dir())
            self.assertTrue((root / "models" / "faster-whisper" / "README_MODEL_FILES.txt").is_file())
            config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["whisper_model_dir"], "models/faster-whisper")
        self.assertEqual(config["diarization_backend"], "local-ecapa")
        self.assertEqual(config["speaker_embedding_model_dir"], "models/speechbrain-ecapa")
        self.assertEqual(config["output_file"], "transcripts/meeting_transcript.txt")

    def test_local_runtime_paths_stay_under_install_root_without_venv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self.assertEqual(local_runtime_dir(root), root / ".runtime")
            self.assertEqual(local_package_dir(root), root / ".runtime" / "site-packages")

    def test_onedrive_install_uses_local_appdata_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            onedrive = base / "OneDrive - Company"
            root = onedrive / "OfflineMeetingTranscriber"
            local_app_data = base / "LocalAppData"
            root.mkdir(parents=True)
            local_app_data.mkdir()

            with patch.dict(
                os.environ,
                {"OneDriveCommercial": str(onedrive), "LOCALAPPDATA": str(local_app_data)},
                clear=False,
            ):
                runtime = local_runtime_dir(root)
                packages = local_package_dir(root)

            self.assertTrue(is_onedrive_path(root))
            self.assertEqual(runtime, local_app_data / "OfflineMeetingTranscriberRuntime")
            self.assertEqual(packages, runtime / "site-packages")

    def test_onedrive_install_migrates_existing_runtime_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            onedrive = base / "OneDrive - Company"
            root = onedrive / "OfflineMeetingTranscriber"
            local_app_data = base / "LocalAppData"
            legacy_runtime = root / ".runtime"
            legacy_runtime.mkdir(parents=True)
            (legacy_runtime / "marker.txt").write_text("keep", encoding="utf-8")
            local_app_data.mkdir()

            with patch.dict(
                os.environ,
                {"OneDriveCommercial": str(onedrive), "LOCALAPPDATA": str(local_app_data)},
                clear=False,
            ):
                migrated = migrate_legacy_onedrive_runtime(root)

            expected = local_app_data / "OfflineMeetingTranscriberRuntime"
            self.assertEqual(migrated, expected)
            self.assertFalse(legacy_runtime.exists())
            self.assertEqual((expected / "marker.txt").read_text(encoding="utf-8"), "keep")

    def test_onedrive_install_moves_legacy_runtime_as_backup_when_target_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            onedrive = base / "OneDrive - Company"
            root = onedrive / "OfflineMeetingTranscriber"
            local_app_data = base / "LocalAppData"
            legacy_runtime = root / ".runtime"
            target_runtime = local_app_data / "OfflineMeetingTranscriberRuntime"
            legacy_runtime.mkdir(parents=True)
            target_runtime.mkdir(parents=True)
            (legacy_runtime / "legacy.txt").write_text("keep legacy", encoding="utf-8")
            (target_runtime / "current.txt").write_text("keep current", encoding="utf-8")

            with patch.dict(
                os.environ,
                {"OneDriveCommercial": str(onedrive), "LOCALAPPDATA": str(local_app_data)},
                clear=False,
            ):
                migrated = migrate_legacy_onedrive_runtime(root)

            expected_backup = target_runtime / "legacy-runtime"
            self.assertEqual(migrated, expected_backup)
            self.assertFalse(legacy_runtime.exists())
            self.assertEqual((target_runtime / "current.txt").read_text(encoding="utf-8"), "keep current")
            self.assertEqual((expected_backup / "legacy.txt").read_text(encoding="utf-8"), "keep legacy")

    def test_missing_runtime_imports_requires_local_package_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            self.assertEqual(missing_runtime_imports(root, {"json": "json"}), ["json"])

    def test_run_pip_install_uses_target_package_dir_without_venv(self):
        from bootstrap_launcher import run_pip_install

        with tempfile.TemporaryDirectory() as runtime_tmp, tempfile.TemporaryDirectory() as package_tmp:
            root = Path(runtime_tmp)
            package_root = Path(package_tmp)
            resources = package_root / "resources"
            resources.mkdir()
            (resources / "requirements.txt").write_text("example-package==1.0\n", encoding="utf-8")
            captured = {}

            class FakeProcess:
                def __init__(self, cmd, **kwargs):
                    captured["cmd"] = cmd
                    captured["kwargs"] = kwargs
                    self.stdout = iter(["Successfully installed example-package-1.0\n"])

                def wait(self):
                    return 0

            with patch("bootstrap_launcher.subprocess.Popen", FakeProcess):
                result = run_pip_install(root, lambda event: None, package_root)

            self.assertEqual(result.code, 0)
            self.assertIn("--target", captured["cmd"])
            self.assertIn(str(root / ".runtime" / "site-packages"), captured["cmd"])
            self.assertNotIn("venv", captured["cmd"])

    def test_ensure_portable_layout_uses_external_template_root(self):
        with tempfile.TemporaryDirectory() as runtime_tmp, tempfile.TemporaryDirectory() as source_tmp:
            runtime_root = Path(runtime_tmp)
            source_root = Path(source_tmp)
            (source_root / "config.template.json").write_text(
                json.dumps({"chunk_seconds": 10, "output_file": "transcripts/custom.txt"}),
                encoding="utf-8",
            )

            config_path = ensure_portable_layout(runtime_root, source_root)
            config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["chunk_seconds"], 10)
        self.assertEqual(config["output_file"], "transcripts/custom.txt")

    def test_model_folder_status_requires_nonempty_model_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            (root / "models" / "faster-whisper" / "model.bin").write_bytes(b"model")
            (root / "models" / "speechbrain-ecapa" / "hyperparams.yaml").write_text("speaker", encoding="utf-8")

            status = model_folder_status(root)

        self.assertEqual(status["faster-whisper"], BootstrapStatus.READY)
        self.assertEqual(status["speechbrain-ecapa"], BootstrapStatus.MISSING)
        self.assertNotIn("pyannote-pipeline", status)

    def test_model_folder_status_can_include_optional_pyannote_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            (root / "models" / "pyannote-pipeline" / "config.yaml").write_text("pipeline", encoding="utf-8")

            status = model_folder_status(root, include_optional=True)

        self.assertEqual(status["pyannote-pipeline"], BootstrapStatus.READY)
        self.assertEqual(status["pyannote-embedding"], BootstrapStatus.MISSING)

    def test_missing_model_setup_message_blocks_launch(self):
        message = missing_model_setup_message(["faster-whisper", "speechbrain-ecapa"])

        self.assertIn("Setup stopped", message)
        self.assertIn("faster-whisper", message)
        self.assertIn("speechbrain-ecapa", message)
        self.assertIn("Recording cannot start", message)

    def test_model_downloader_skips_complete_default_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            (root / "models" / "faster-whisper" / "model.bin").write_bytes(b"model")
            speaker = root / "models" / "speechbrain-ecapa"
            (speaker / "hyperparams.yaml").write_text("speaker", encoding="utf-8")
            (speaker / "embedding_model.ckpt").write_bytes(b"speaker")
            calls = []

            downloaded = download_default_models(
                root,
                lambda event: None,
                downloader=lambda **kwargs: calls.append(kwargs) or kwargs["local_dir"],
            )

        self.assertEqual(downloaded, [])
        self.assertEqual(calls, [])

    def test_model_downloader_calls_snapshot_for_missing_default_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                target = Path(kwargs["local_dir"])
                if kwargs["repo_id"] == "Systran/faster-whisper-small":
                    (target / "model.bin").write_bytes(b"model")
                else:
                    (target / "hyperparams.yaml").write_text("speaker", encoding="utf-8")
                    (target / "embedding_model.ckpt").write_bytes(b"speaker")
                return str(target)

            downloaded = download_default_models(root, lambda event: None, downloader=fake_download)

            self.assertEqual(downloaded, ["faster-whisper", "speechbrain-ecapa"])
            self.assertEqual([call["repo_id"] for call in calls], [spec.repo_id for spec in DEFAULT_MODEL_DOWNLOADS])
            self.assertTrue(all(call["local_files_only"] is False for call in calls))
            self.assertTrue(all(call["force_download"] is True for call in calls))
            self.assertTrue(all(Path(call["local_dir"]).is_relative_to(local_runtime_dir(root)) for call in calls))
            self.assertTrue(all("model-downloads" in Path(call["local_dir"]).parts for call in calls))
            self.assertTrue((root / "models" / "faster-whisper" / "model.bin").is_file())
            self.assertTrue((root / "models" / "speechbrain-ecapa" / "hyperparams.yaml").is_file())
            self.assertTrue((root / "models" / "speechbrain-ecapa" / "embedding_model.ckpt").is_file())

    def test_model_downloader_does_not_treat_placeholder_as_downloaded_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            self.assertTrue((root / "models" / "faster-whisper" / "README_MODEL_FILES.txt").is_file())
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                target = Path(kwargs["local_dir"])
                if kwargs["repo_id"] == "Systran/faster-whisper-small":
                    (target / "model.bin").write_bytes(b"model")
                else:
                    (target / "hyperparams.yaml").write_text("speaker", encoding="utf-8")
                    (target / "model.ckpt").write_bytes(b"speaker")
                return str(target)

            downloaded = download_default_models(root, lambda event: None, downloader=fake_download)

            self.assertEqual(downloaded, ["faster-whisper", "speechbrain-ecapa"])
            self.assertEqual(len(calls), 2)
            self.assertTrue((root / "models" / "faster-whisper" / "README_MODEL_FILES.txt").is_file())
            self.assertTrue((root / "models" / "faster-whisper" / "model.bin").is_file())
            self.assertTrue((root / "models" / "speechbrain-ecapa" / "model.ckpt").is_file())

    def test_model_downloader_clears_stale_staging_before_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            staging = local_runtime_dir(root) / "model-downloads" / "faster-whisper"
            staging.mkdir(parents=True)
            (staging / "stale.txt").write_text("old", encoding="utf-8")

            def fake_download(**kwargs):
                target = Path(kwargs["local_dir"])
                self.assertFalse((target / "stale.txt").exists())
                if kwargs["repo_id"] == "Systran/faster-whisper-small":
                    (target / "model.bin").write_bytes(b"model")
                else:
                    (target / "hyperparams.yaml").write_text("speaker", encoding="utf-8")
                    (target / "embedding_model.ckpt").write_bytes(b"speaker")
                return str(target)

            download_default_models(root, lambda event: None, downloader=fake_download)

            self.assertFalse((staging / "stale.txt").exists())

    def test_model_downloader_temporarily_clears_offline_env_and_restores_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            seen_env = []

            def fake_download(**kwargs):
                seen_env.append({name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS})
                target = Path(kwargs["local_dir"])
                if kwargs["repo_id"] == "Systran/faster-whisper-small":
                    (target / "model.bin").write_bytes(b"model")
                else:
                    (target / "hyperparams.yaml").write_text("speaker", encoding="utf-8")
                    (target / "embedding_model.ckpt").write_bytes(b"speaker")
                return str(target)

            with patch.dict(
                os.environ,
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                },
                clear=False,
            ):
                download_default_models(root, lambda event: None, downloader=fake_download)
                restored = {name: os.environ.get(name) for name in HF_OFFLINE_ENV_VARS}

        self.assertTrue(all(all(value is None for value in item.values()) for item in seen_env))
        self.assertEqual(restored, {name: "1" for name in HF_OFFLINE_ENV_VARS})

    def test_online_download_env_uses_ca_bundle_beside_launcher_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher_root = Path(tmp)
            app_root = launcher_root / "OfflineMeetingTranscriber"
            app_root.mkdir()
            bundle = launcher_root / "company-ca.pem"
            bundle.write_text("certificate", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "LOCAL_WHISPER_LOG_ROOT": str(launcher_root),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                },
                clear=False,
            ):
                os.environ.pop("REQUESTS_CA_BUNDLE", None)
                os.environ.pop("SSL_CERT_FILE", None)
                with online_huggingface_download_env(app_root):
                    self.assertIsNone(os.environ.get("HF_HUB_OFFLINE"))
                    self.assertEqual(os.environ["REQUESTS_CA_BUNDLE"], str(bundle.resolve()))
                    self.assertEqual(os.environ["SSL_CERT_FILE"], str(bundle.resolve()))
                self.assertNotIn("REQUESTS_CA_BUNDLE", os.environ)
                self.assertNotIn("SSL_CERT_FILE", os.environ)

    def test_online_download_env_preserves_existing_ca_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "existing.pem"
            existing.write_text("existing", encoding="utf-8")
            app_root = Path(tmp) / "OfflineMeetingTranscriber"
            app_root.mkdir()
            (Path(tmp) / "company-ca.pem").write_text("ignored", encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "LOCAL_WHISPER_LOG_ROOT": str(tmp),
                    "REQUESTS_CA_BUNDLE": str(existing),
                    "SSL_CERT_FILE": str(existing),
                },
                clear=False,
            ):
                with online_huggingface_download_env(app_root):
                    self.assertEqual(os.environ["REQUESTS_CA_BUNDLE"], str(existing))
                    self.assertEqual(os.environ["SSL_CERT_FILE"], str(existing))

    def test_online_download_env_injects_windows_truststore_when_available(self):
        fake_truststore = types.ModuleType("truststore")
        calls = []

        def fake_inject_into_ssl():
            calls.append("called")

        fake_truststore.inject_into_ssl = fake_inject_into_ssl

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"truststore": fake_truststore}):
            with online_huggingface_download_env(Path(tmp)):
                pass

        self.assertEqual(calls, ["called"])

    def test_verify_downloaded_model_raises_clear_error_when_expected_files_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            spec = DEFAULT_MODEL_DOWNLOADS[0]
            staging = local_runtime_dir(root) / "model-downloads" / spec.name
            staging.mkdir(parents=True)
            (staging / "README_MODEL_FILES.txt").write_text("placeholder", encoding="utf-8")

            pattern = (
                r"(?s)Model download did not produce expected files.*"
                r"Staging folder.*Final target folder.*README_MODEL_FILES.txt"
            )
            with self.assertRaisesRegex(RuntimeError, pattern):
                verify_downloaded_model(root, spec, staging, "returned-path")

    def test_needs_setup_uses_marker_after_first_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)

            self.assertTrue(needs_setup(root, {"json": "json"}))
            (root / SETUP_MARKER).write_text("complete\n", encoding="utf-8")
            local_package_dir(root).mkdir(parents=True)
            (root / "models" / "faster-whisper" / "model.bin").write_bytes(b"model")
            speaker = root / "models" / "speechbrain-ecapa"
            (speaker / "hyperparams.yaml").write_text("speaker", encoding="utf-8")
            (speaker / "embedding_model.ckpt").write_bytes(b"speaker")

            self.assertFalse(needs_setup(root, {"json": "json"}))

    def test_pip_progress_parser_reports_download_and_install_progress(self):
        parser = PipProgressParser()

        collect = parser.parse("Collecting PySide6==6.11.1")
        download = parser.parse("Downloading pyside6-6.11.1-cp310-abi3-win_amd64.whl (578 kB)")
        installing = parser.parse("Installing collected packages: PySide6, sounddevice, torch")
        installed_one = parser.parse("Successfully installed PySide6-6.11.1 sounddevice-0.5.5 torch-2.11.0")

        self.assertEqual(collect.phase, "Resolving package")
        self.assertEqual(collect.detail, "PySide6==6.11.1")
        self.assertEqual(download.phase, "Downloading package")
        self.assertIn("pyside6", download.detail)
        self.assertEqual(installing.phase, "Installing packages")
        self.assertEqual(installing.progress_percent, 0)
        self.assertEqual(installed_one.phase, "Install complete")
        self.assertEqual(installed_one.progress_percent, 100)

    def test_pip_progress_parser_counts_already_satisfied(self):
        parser = PipProgressParser()

        event = parser.parse("Requirement already satisfied: numpy==2.4.2 in c:\\site-packages")

        self.assertEqual(event.phase, "Already installed")
        self.assertEqual(event.detail, "numpy==2.4.2")
        self.assertEqual(event.progress_percent, 100)

    def test_step_tooltip_includes_error_command_and_last_output(self):
        diagnostic = StepDiagnostic(
            name="Installing Python packages",
            state=StepState.ERROR,
            detail="pip failed",
            command="python -m pip install",
            last_output="ERROR: no matching distribution",
            error="exit code 1",
            start_time=100.0,
            end_time=145.0,
        )

        tooltip = build_step_tooltip(diagnostic, now=150.0)

        self.assertIn("Installing Python packages", tooltip)
        self.assertIn("Status: error", tooltip)
        self.assertIn("Elapsed: 45s", tooltip)
        self.assertIn("Command: python -m pip install", tooltip)
        self.assertIn("Last output: ERROR: no matching distribution", tooltip)
        self.assertIn("Error: exit code 1", tooltip)

    def test_pip_progress_tracker_reports_elapsed_and_eta(self):
        tracker = PipInstallProgressTracker(start_time=100.0)

        tracker.record(PipProgressEvent("Resolving package", "PySide6"), now=110.0)
        tracker.record(PipProgressEvent("Downloading package", "PySide6 wheel"), now=120.0)
        progress, detail = tracker.record(PipProgressEvent("Running pip", "Installing"), now=130.0)

        self.assertGreater(progress, 0)
        self.assertIn("Elapsed 30s", detail)
        self.assertIn("ETA", detail)

    def test_pip_progress_tracker_does_not_show_fake_short_eta_during_install(self):
        tracker = PipInstallProgressTracker(start_time=100.0)

        tracker.record(PipProgressEvent("Resolving package", "PySide6"), now=110.0)
        tracker.record(PipProgressEvent("Downloading package", "PySide6 wheel"), now=120.0)
        tracker.record(
            PipProgressEvent(
                "Installing packages",
                "torch, torchaudio",
                0,
                "Installing collected packages: torch, torchaudio",
            ),
            now=130.0,
        )
        progress, detail = tracker.record(PipProgressEvent("Running pip", "Building wheels"), now=180.0)

        self.assertLess(progress, 100)
        self.assertIn("Installing collected packages: torch, torchaudio", detail)
        self.assertNotIn("ETA ~1s", detail)

    def test_write_setup_error_log_records_command_and_recent_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = write_setup_error_log(
                Path(tmp),
                "Installing Python packages",
                "pip failed with exit code 1",
                "python -m pip install -r requirements.txt",
                ["line one", "line two"],
            )
            content = log_path.read_text(encoding="utf-8")

        self.assertIn("context: Installing Python packages", content)
        self.assertIn("pip failed with exit code 1", content)
        self.assertIn("python -m pip install -r requirements.txt", content)
        self.assertIn("line two", content)


if __name__ == "__main__":
    unittest.main()
