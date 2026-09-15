import io
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import bootstrap_launcher as bootstrap_launcher_module
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
    GpuVendor,
    StepDiagnostic,
    StepState,
    build_step_tooltip,
    choose_gpu_trial_backend,
    detect_gpu_vendors_from_names,
    download_default_models,
    ensure_portable_layout,
    gpu_trial_required_imports,
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
    parse_cli_download_status,
    setup_install_summary,
    setup_package_list,
    verify_downloaded_model,
    write_setup_error_log,
)


def write_complete_openvino_ir(folder: Path, content: str = "model") -> None:
    for stem in ("encoder_model", "decoder_model", "tokenizer", "detokenizer"):
        (folder / f"openvino_{stem}.xml").write_text(f"{content} xml", encoding="utf-8")
        (folder / f"openvino_{stem}.bin").write_bytes(f"{content} weights".encode())


class BootstrapLauncherTests(unittest.TestCase):
    def test_import_replaces_missing_pythonw_streams(self):
        script = (
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd() / 'app'))\n"
            "sys.stdout = None\n"
            "sys.stderr = None\n"
            "import bootstrap_launcher\n"
            "if sys.stdout is None or sys.stderr is None:\n"
            "    sys.__stdout__.write('streams still None')\n"
            "    raise SystemExit(2)\n"
            "if not callable(getattr(sys.stdout, 'write', None)):\n"
            "    sys.__stdout__.write('stdout cannot write')\n"
            "    raise SystemExit(3)\n"
            "if not callable(getattr(sys.stderr, 'write', None)):\n"
            "    sys.__stdout__.write('stderr cannot write')\n"
            "    raise SystemExit(4)\n"
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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
        self.assertIn("huggingface_hub", REQUIRED_IMPORTS)
        self.assertIn("truststore", REQUIRED_IMPORTS)
        self.assertNotIn("speechbrain", REQUIRED_IMPORTS)
        self.assertNotIn("scikit-learn", REQUIRED_IMPORTS)
        self.assertNotIn("pyannote.audio", REQUIRED_IMPORTS)
        self.assertNotIn("torch", REQUIRED_IMPORTS)

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
        self.assertIn("Disk space: ~1.8 GB for packages and models.", summary)
        self.assertIn("Systran/faster-whisper-small.en -> models\\faster-whisper", summary)
        self.assertNotIn("speechbrain", summary)
        self.assertEqual(packages, ["alpha==1.0", "beta>=2.0"])
        self.assertIn("- alpha==1.0", summary)
        self.assertIn("- beta>=2.0", summary)

    def test_intel_setup_summary_documents_model_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resources = root / "resources"
            resources.mkdir()
            (resources / "requirements.txt").write_text("alpha==1.0\n", encoding="utf-8")

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    summary = setup_install_summary(root)

        self.assertIn("828 MB", summary)
        self.assertIn("OpenVINO 2026.1", summary)
        self.assertIn("in place", summary)
        self.assertIn("models/.openvino-whisper.rollback", summary)

    def test_setup_initial_buttons_do_not_include_manual_launch_or_model_shortcuts(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertNotIn('text="Launch GUI"', source)
        self.assertNotIn('text="Open model folder"', source)
        self.assertNotIn("text=f\"Set {model}\"", source)

    def test_setup_spinner_uses_npm_dots_frames_in_task_and_detail_title(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertIn('NPM_DOTS5_SPINNER_FRAMES = ("⠋", "⠙", "⠚", "⠞", "⠖", "⠦", "⠴", "⠲", "⠳", "⠓")', source)
        self.assertIn("spinner_frames = list(NPM_DOTS5_SPINNER_FRAMES)", source)
        self.assertIn('detail_title.config(text=f"{spinner_frames[spinner_index % len(spinner_frames)]} {current_step}")', source)

    def test_setup_confirmation_mentions_disk_space(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertIn('INSTALL_DISK_SPACE_ESTIMATE = "~1.8 GB"', source)
        self.assertIn("this install uses {INSTALL_DISK_SPACE_ESTIMATE} for packages and models", source)

    def test_detect_gpu_vendors_from_video_controller_names(self):
        vendors = detect_gpu_vendors_from_names(
            [
                "Intel(R) Iris(R) Xe Graphics",
                "AMD Radeon 780M Graphics",
                "NVIDIA RTX 4060 Laptop GPU",
                "Microsoft Basic Display Adapter",
            ]
        )

        self.assertEqual(vendors, [GpuVendor.INTEL, GpuVendor.AMD, GpuVendor.NVIDIA])

    def test_choose_gpu_trial_backend_prefers_nvidia_then_intel_then_amd(self):
        nvidia = choose_gpu_trial_backend([GpuVendor.INTEL, GpuVendor.NVIDIA])
        intel = choose_gpu_trial_backend([GpuVendor.INTEL])
        amd = choose_gpu_trial_backend([GpuVendor.AMD])

        self.assertEqual(nvidia.key, "nvidia-cuda")
        self.assertEqual(nvidia.config_overrides["device"], "cuda")
        self.assertEqual(nvidia.config_overrides["compute_type"], "float16")
        self.assertNotIn("chunk_seconds", nvidia.config_overrides)
        self.assertNotIn("overlap_seconds", nvidia.config_overrides)
        self.assertEqual(intel.key, "intel-openvino")
        self.assertTrue(any(requirement.startswith("openvino-genai") for requirement in intel.requirements))
        self.assertEqual(amd.key, "amd-directml")
        self.assertTrue(any(requirement.startswith("onnxruntime-directml") for requirement in amd.requirements))

    def test_intel_gpu_trial_uses_large_v3_turbo_int8_contract(self):
        intel = choose_gpu_trial_backend([GpuVendor.INTEL])

        self.assertEqual(len(intel.model_downloads), 1)
        self.assertEqual(
            intel.model_downloads[0].repo_id,
            "OpenVINO/whisper-large-v3-turbo-int8-ov",
        )
        self.assertEqual(intel.model_downloads[0].target_subdir, "models/openvino-whisper")
        self.assertEqual(intel.config_overrides["device"], "openvino:GPU")
        self.assertEqual(intel.config_overrides["compute_type"], "int8")
        self.assertEqual(intel.config_overrides["chunk_seconds"], 10.0)
        self.assertEqual(intel.config_overrides["overlap_seconds"], 1.0)
        self.assertIn("openvino>=2026.1.0", intel.requirements)
        self.assertIn("openvino-tokenizers>=2026.1.0", intel.requirements)
        self.assertIn("openvino-genai>=2026.1.0", intel.requirements)

    def test_gpu_trial_required_imports_are_selected_by_vendor(self):
        imports = gpu_trial_required_imports([GpuVendor.INTEL])

        self.assertIn("openvino", imports)
        self.assertIn("openvino-genai", imports)
        self.assertIn("openvino-tokenizers", imports)
        self.assertNotIn("onnxruntime-directml", imports)

    def test_successful_package_install_marks_package_check_done(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertGreaterEqual(
            source.count('set_step(BOOTSTRAP_STEPS[2], StepState.DONE, "Python packages ready", 100)'),
            2,
        )

    def test_missing_packages_are_normal_install_work_not_warning(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertIn('package_word = "package" if len(missing) == 1 else "packages"', source)
        self.assertIn(
            'set_step(BOOTSTRAP_STEPS[2], StepState.DONE, f"{len(missing)} {package_word} to install", 100)',
            source,
        )
        self.assertNotIn('set_step(BOOTSTRAP_STEPS[2], StepState.WARNING, "Missing: " + ", ".join(missing), 100)', source)

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
            self.assertFalse((root / "models" / "speechbrain-ecapa").exists())
            self.assertFalse((root / "models" / "pyannote-pipeline").exists())
            self.assertFalse((root / "models" / "pyannote-embedding").exists())
            self.assertTrue((root / "transcripts").is_dir())
            self.assertTrue((root / "models" / "faster-whisper" / "README_MODEL_FILES.txt").is_file())
            config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["whisper_model_dir"], "models/faster-whisper")
        self.assertEqual(config["output_file"], "transcripts/meeting_transcript.txt")
        self.assertEqual(config["language"], "en")

    def test_gpu_trial_layout_uses_openvino_model_and_config_for_intel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    config_path = ensure_portable_layout(root)

            config = json.loads(config_path.read_text(encoding="utf-8"))
            requirements = local_runtime_dir(root) / "gpu-trial-requirements.txt"
            report = root / "gpu_trial_report.txt"
            self.assertTrue(requirements.is_file())
            self.assertTrue(report.is_file())
            requirements_content = requirements.read_text(encoding="utf-8")
            report_content = report.read_text(encoding="utf-8")
            guide_content = (
                root / "models" / "openvino-whisper" / "README_MODEL_FILES.txt"
            ).read_text(encoding="utf-8")

            self.assertEqual(config["whisper_model_dir"], "models/openvino-whisper")
            self.assertEqual(config["device"], "openvino:GPU")
            self.assertEqual(config["compute_type"], "int8")
            self.assertEqual(config["chunk_seconds"], 10.0)
            self.assertEqual(config["overlap_seconds"], 1.0)
            self.assertIn("openvino-genai>=2026.1.0", requirements_content)
            self.assertIn("selected_backend: intel-openvino", report_content)
            for content in (report_content, guide_content):
                self.assertIn("828 MB", content)
                self.assertIn("OpenVINO 2026.1", content)
                self.assertIn("in place", content)
                self.assertIn("models/.openvino-whisper.rollback", content)

    def test_gpu_trial_layout_migrates_untouched_intel_chunk_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "whisper_model_dir": "models/openvino-whisper",
                        "device": "openvino:GPU",
                        "compute_type": "fp16",
                        "chunk_seconds": 5.0,
                        "overlap_seconds": 0.5,
                        "language": "en",
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    ensure_portable_layout(root)

            config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["compute_type"], "int8")
        self.assertEqual(config["chunk_seconds"], 10.0)
        self.assertEqual(config["overlap_seconds"], 1.0)
        self.assertEqual(config["language"], "en")

    def test_gpu_trial_layout_preserves_custom_path_and_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "whisper_model_dir": "D:/models/custom-openvino",
                        "device": "openvino:GPU",
                        "compute_type": "fp16",
                        "chunk_seconds": 12.0,
                        "overlap_seconds": 3.0,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    ensure_portable_layout(root)

            config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["whisper_model_dir"], "D:/models/custom-openvino")
        self.assertEqual(config["compute_type"], "fp16")
        self.assertEqual(config["chunk_seconds"], 12.0)
        self.assertEqual(config["overlap_seconds"], 3.0)

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

    def test_missing_runtime_imports_reports_outdated_intel_openvino_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = local_package_dir(root)
            (package_dir / "openvino").mkdir(parents=True)
            (package_dir / "openvino" / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "openvino_genai.py").write_text("", encoding="utf-8")
            (package_dir / "openvino_tokenizers.py").write_text("", encoding="utf-8")
            for distribution in ("openvino", "openvino-genai", "openvino-tokenizers"):
                metadata_dir = distribution.replace("-", "_")
                metadata = package_dir / f"{metadata_dir}-2025.4.0.dist-info" / "METADATA"
                metadata.parent.mkdir()
                metadata.write_text(
                    f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 2025.4.0\n",
                    encoding="utf-8",
                )

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    missing = missing_runtime_imports(
                        root,
                        {
                            "openvino": "openvino",
                            "openvino-genai": "openvino_genai",
                            "openvino-tokenizers": "openvino_tokenizers",
                        },
                    )

        self.assertEqual(
            missing,
            ["openvino", "openvino-genai", "openvino-tokenizers"],
        )

    def test_missing_runtime_imports_accepts_compatible_intel_openvino_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_dir = local_package_dir(root)
            (package_dir / "openvino").mkdir(parents=True)
            (package_dir / "openvino" / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "openvino_genai.py").write_text("", encoding="utf-8")
            (package_dir / "openvino_tokenizers.py").write_text("", encoding="utf-8")
            versions = {
                "openvino": "2026.1.0",
                "openvino-genai": "2026.2.0",
                "openvino-tokenizers": "2026.1.1",
            }
            for distribution, version in versions.items():
                metadata_dir = distribution.replace("-", "_")
                metadata = package_dir / f"{metadata_dir}-{version}.dist-info" / "METADATA"
                metadata.parent.mkdir()
                metadata.write_text(
                    f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
                    encoding="utf-8",
                )

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    missing = missing_runtime_imports(
                        root,
                        {
                            "openvino": "openvino",
                            "openvino-genai": "openvino_genai",
                            "openvino-tokenizers": "openvino_tokenizers",
                        },
                    )

        self.assertEqual(missing, [])

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

    def test_run_pip_install_adds_gpu_trial_requirements_for_intel(self):
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
                    self.stdout = iter(["Successfully installed example-package-1.0\n"])

                def wait(self):
                    return 0

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    with patch("bootstrap_launcher.subprocess.Popen", FakeProcess):
                        result = run_pip_install(root, lambda event: None, package_root)

            requirement_args = [
                captured["cmd"][index + 1]
                for index, item in enumerate(captured["cmd"])
                if item == "-r"
            ]

        self.assertEqual(result.code, 0)
        self.assertEqual(len(requirement_args), 2)
        self.assertTrue(requirement_args[-1].endswith("gpu-trial-requirements.txt"))

    def test_windows_cpu_limit_uses_job_object_hard_cap_at_25_percent(self):
        calls = []

        class FakeKernel32:
            def CreateJobObjectW(self, security_attributes, name):
                calls.append(("CreateJobObjectW", security_attributes, name))
                return 1234

            def SetInformationJobObject(self, job, info_class, info_ptr, info_size):
                info = ctypes.cast(
                    info_ptr,
                    ctypes.POINTER(bootstrap_launcher_module.JOBOBJECT_CPU_RATE_CONTROL_INFORMATION),
                ).contents
                calls.append(("SetInformationJobObject", job, info_class, info.ControlFlags, info.CpuRate, info_size))
                return 1

            def AssignProcessToJobObject(self, job, process_handle):
                calls.append(("AssignProcessToJobObject", job, process_handle))
                return 1

            def CloseHandle(self, handle):
                calls.append(("CloseHandle", handle))
                return 1

        class FakeProcess:
            _handle = 5678

        applied = bootstrap_launcher_module.apply_windows_cpu_limit(
            FakeProcess(),
            percent=25,
            platform_name="win32",
            kernel32=FakeKernel32(),
        )

        self.assertTrue(applied)
        self.assertIn(("CreateJobObjectW", None, None), calls)
        self.assertIn(
            (
                "SetInformationJobObject",
                1234,
                bootstrap_launcher_module.JobObjectCpuRateControlInformation,
                bootstrap_launcher_module.JOB_OBJECT_CPU_RATE_CONTROL_ENABLE
                | bootstrap_launcher_module.JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
                2500,
                ctypes.sizeof(bootstrap_launcher_module.JOBOBJECT_CPU_RATE_CONTROL_INFORMATION),
            ),
            calls,
        )
        self.assertIn(("AssignProcessToJobObject", 1234, 5678), calls)
        self.assertNotIn(("CloseHandle", 1234), calls)

    def test_pip_install_applies_and_closes_cpu_limit_job(self):
        from bootstrap_launcher import run_pip_install

        with tempfile.TemporaryDirectory() as runtime_tmp, tempfile.TemporaryDirectory() as package_tmp:
            root = Path(runtime_tmp)
            package_root = Path(package_tmp)
            resources = package_root / "resources"
            resources.mkdir()
            (resources / "requirements.txt").write_text("example-package==1.0\n", encoding="utf-8")
            calls = []

            class FakeProcess:
                def __init__(self, cmd, **kwargs):
                    self.stdout = iter(["Successfully installed example-package-1.0\n"])
                    self._cpu_limit_job_handle = 1234

                def wait(self):
                    return 0

            with patch("bootstrap_launcher.subprocess.Popen", FakeProcess):
                with patch("bootstrap_launcher.apply_windows_cpu_limit", side_effect=lambda process: calls.append("apply") or True):
                    with patch("bootstrap_launcher.close_windows_cpu_limit", side_effect=lambda process: calls.append("close")):
                        result = run_pip_install(root, lambda event: None, package_root)

            self.assertEqual(result.code, 0)
            self.assertEqual(calls, ["apply", "close"])

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

            status = model_folder_status(root)

        self.assertEqual(status["faster-whisper"], BootstrapStatus.READY)
        self.assertEqual(set(status), {"faster-whisper"})

    def test_model_folder_status_has_no_optional_diarization_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)

            status = model_folder_status(root, include_optional=True)

        self.assertEqual(set(status), {"faster-whisper"})

    def test_intel_model_status_rejects_previous_small_fp16_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "openvino-whisper"
            model_dir.mkdir(parents=True)
            write_complete_openvino_ir(model_dir)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "d_model": 768,
                        "encoder_layers": 12,
                        "decoder_layers": 12,
                        "num_mel_bins": 80,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    status = model_folder_status(root)

        self.assertEqual(status["openvino-whisper"], BootstrapStatus.MISSING)

    def test_intel_model_status_accepts_large_v3_turbo_int8_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "openvino-whisper"
            model_dir.mkdir(parents=True)
            write_complete_openvino_ir(model_dir)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "d_model": 1280,
                        "encoder_layers": 32,
                        "decoder_layers": 4,
                        "num_mel_bins": 128,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    status = model_folder_status(root)

        self.assertEqual(status["openvino-whisper"], BootstrapStatus.READY)

    def test_intel_model_status_rejects_signature_with_only_encoder_ir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "openvino-whisper"
            model_dir.mkdir(parents=True)
            (model_dir / "openvino_encoder_model.xml").write_text("xml", encoding="utf-8")
            (model_dir / "openvino_encoder_model.bin").write_bytes(b"weights")
            (model_dir / "config.json").write_text(
                json.dumps(
                    {
                        "d_model": 1280,
                        "encoder_layers": 32,
                        "decoder_layers": 4,
                        "num_mel_bins": 128,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    status = model_folder_status(root)

        self.assertEqual(status["openvino-whisper"], BootstrapStatus.MISSING)

    def test_intel_model_status_rejects_partial_or_malformed_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "openvino-whisper"
            model_dir.mkdir(parents=True)
            (model_dir / "openvino_encoder_model.xml").write_text("xml", encoding="utf-8")
            (model_dir / "config.json").write_text("not json", encoding="utf-8")

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    status = model_folder_status(root)

        self.assertEqual(status["openvino-whisper"], BootstrapStatus.MISSING)

    def test_missing_model_setup_message_blocks_launch(self):
        message = missing_model_setup_message(["faster-whisper"])

        self.assertIn("Setup stopped", message)
        self.assertIn("faster-whisper", message)
        self.assertNotIn("speechbrain", message)
        self.assertIn("Recording cannot start", message)

    def test_model_downloader_skips_complete_default_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            (root / "models" / "faster-whisper" / "model.bin").write_bytes(b"model")
            calls = []

            downloaded = download_default_models(
                root,
                lambda event: None,
                downloader=lambda **kwargs: calls.append(kwargs) or kwargs["local_dir"],
            )

        self.assertEqual(downloaded, [])
        self.assertEqual(calls, [])

    def test_intel_model_downloader_replaces_small_model_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "models" / "openvino-whisper"
            target.mkdir(parents=True)
            write_complete_openvino_ir(target, "old")
            (target / "config.json").write_text(
                json.dumps(
                    {
                        "d_model": 768,
                        "encoder_layers": 12,
                        "decoder_layers": 12,
                        "num_mel_bins": 80,
                    }
                ),
                encoding="utf-8",
            )
            (target / "small-only.txt").write_text("legacy", encoding="utf-8")
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                staging = Path(kwargs["local_dir"])
                write_complete_openvino_ir(staging, "turbo")
                (staging / "config.json").write_text(
                    json.dumps(
                        {
                            "d_model": 1280,
                            "encoder_layers": 32,
                            "decoder_layers": 4,
                            "num_mel_bins": 128,
                        }
                    ),
                    encoding="utf-8",
                )
                return str(staging)

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ), patch.object(
                    bootstrap_launcher_module,
                    "fetch_model_repo_size",
                    return_value=None,
                ):
                    downloaded = download_default_models(root, lambda event: None, downloader=fake_download)

            installed = json.loads((target / "config.json").read_text(encoding="utf-8"))
            rollback = target.parent / ".openvino-whisper.rollback"
            small_only_exists = (target / "small-only.txt").exists()
            rollback_exists = rollback.exists()

        self.assertEqual(downloaded, ["openvino-whisper"])
        self.assertEqual(calls[0]["repo_id"], "OpenVINO/whisper-large-v3-turbo-int8-ov")
        self.assertEqual(installed["decoder_layers"], 4)
        self.assertFalse(small_only_exists)
        self.assertFalse(rollback_exists)

    def test_intel_model_downloader_skips_manually_copied_turbo_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "models" / "openvino-whisper"
            target.mkdir(parents=True)
            write_complete_openvino_ir(target)
            (target / "config.json").write_text(
                json.dumps(
                    {
                        "d_model": 1280,
                        "encoder_layers": 32,
                        "decoder_layers": 4,
                        "num_mel_bins": 128,
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ):
                    downloaded = download_default_models(
                        root,
                        lambda event: None,
                        downloader=lambda **kwargs: calls.append(kwargs),
                    )

        self.assertEqual(downloaded, [])
        self.assertEqual(calls, [])

    def test_intel_model_replacement_restores_small_model_when_copy_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "models" / "openvino-whisper"
            target.mkdir(parents=True)
            old_config = {
                "d_model": 768,
                "encoder_layers": 12,
                "decoder_layers": 12,
                "num_mel_bins": 80,
            }
            write_complete_openvino_ir(target, "old")
            (target / "config.json").write_text(json.dumps(old_config), encoding="utf-8")

            def fake_download(**kwargs):
                staging = Path(kwargs["local_dir"])
                write_complete_openvino_ir(staging, "turbo")
                (staging / "config.json").write_text(
                    json.dumps(
                        {
                            "d_model": 1280,
                            "encoder_layers": 32,
                            "decoder_layers": 4,
                            "num_mel_bins": 128,
                        }
                    ),
                    encoding="utf-8",
                )
                return str(staging)

            def failing_copytree(source, destination, **kwargs):
                del source, kwargs
                destination = Path(destination)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "config.json").write_text('{"decoder_layers": 4}', encoding="utf-8")
                raise OSError("simulated copy failure")

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ), patch.object(
                    bootstrap_launcher_module,
                    "fetch_model_repo_size",
                    return_value=None,
                ), patch.object(
                    bootstrap_launcher_module.shutil,
                    "copytree",
                    side_effect=failing_copytree,
                ):
                    with self.assertRaisesRegex(OSError, "simulated copy failure"):
                        download_default_models(root, lambda event: None, downloader=fake_download)

            restored = json.loads((target / "config.json").read_text(encoding="utf-8"))
            rollback = target.parent / ".openvino-whisper.rollback"
            rollback_exists = rollback.exists()

        self.assertEqual(restored, old_config)
        self.assertFalse(rollback_exists)

    def test_intel_model_replacement_restores_small_model_when_verification_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "models" / "openvino-whisper"
            target.mkdir(parents=True)
            old_config = {
                "d_model": 768,
                "encoder_layers": 12,
                "decoder_layers": 12,
                "num_mel_bins": 80,
            }
            write_complete_openvino_ir(target, "old")
            (target / "config.json").write_text(json.dumps(old_config), encoding="utf-8")

            def fake_download(**kwargs):
                staging = Path(kwargs["local_dir"])
                write_complete_openvino_ir(staging, "turbo")
                (staging / "config.json").write_text(
                    json.dumps(
                        {
                            "d_model": 1280,
                            "encoder_layers": 32,
                            "decoder_layers": 4,
                            "num_mel_bins": 128,
                        }
                    ),
                    encoding="utf-8",
                )
                return str(staging)

            def corrupting_copytree(source, destination, **kwargs):
                del kwargs
                source = Path(source)
                destination = Path(destination)
                destination.mkdir(parents=True)
                for source_file in source.iterdir():
                    if source_file.name != "openvino_detokenizer.bin":
                        (destination / source_file.name).write_bytes(source_file.read_bytes())
                return destination

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ), patch.object(
                    bootstrap_launcher_module,
                    "fetch_model_repo_size",
                    return_value=None,
                ), patch.object(
                    bootstrap_launcher_module.shutil,
                    "copytree",
                    side_effect=corrupting_copytree,
                ):
                    with self.assertRaisesRegex(RuntimeError, "failed validation"):
                        download_default_models(root, lambda event: None, downloader=fake_download)

            restored = json.loads((target / "config.json").read_text(encoding="utf-8"))
            rollback_exists = (target.parent / ".openvino-whisper.rollback").exists()

        self.assertEqual(restored, old_config)
        self.assertFalse(rollback_exists)

    def test_intel_model_replacement_aborts_when_rollback_folder_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "models" / "openvino-whisper"
            target.mkdir(parents=True)
            write_complete_openvino_ir(target, "old")
            (target / "config.json").write_text(
                json.dumps(
                    {
                        "d_model": 768,
                        "encoder_layers": 12,
                        "decoder_layers": 12,
                        "num_mel_bins": 80,
                    }
                ),
                encoding="utf-8",
            )
            rollback = target.parent / ".openvino-whisper.rollback"
            rollback.mkdir()
            (rollback / "keep.txt").write_text("recovery", encoding="utf-8")
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                staging = Path(kwargs["local_dir"])
                write_complete_openvino_ir(staging, "turbo")
                (staging / "config.json").write_text(
                    json.dumps(
                        {
                            "d_model": 1280,
                            "encoder_layers": 32,
                            "decoder_layers": 4,
                            "num_mel_bins": 128,
                        }
                    ),
                    encoding="utf-8",
                )
                return str(staging)

            with patch.dict(os.environ, {"LOCAL_WHISPER_GPU_TRIAL": "1"}, clear=False):
                with patch.object(
                    bootstrap_launcher_module,
                    "detect_windows_gpu_names",
                    return_value=["Intel(R) Iris(R) Xe Graphics"],
                ), patch.object(
                    bootstrap_launcher_module,
                    "fetch_model_repo_size",
                    return_value=None,
                ):
                    with self.assertRaisesRegex(RuntimeError, "rollback folder already exists"):
                        download_default_models(root, lambda event: None, downloader=fake_download)

            old_model_remains = json.loads((target / "config.json").read_text(encoding="utf-8"))
            rollback_content = (rollback / "keep.txt").read_text(encoding="utf-8")

        self.assertEqual(old_model_remains["decoder_layers"], 12)
        self.assertEqual(rollback_content, "recovery")
        self.assertEqual(calls, [])

    def test_model_downloader_calls_snapshot_for_missing_default_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                target = Path(kwargs["local_dir"])
                self.assertEqual(kwargs["repo_id"], "Systran/faster-whisper-small.en")
                (target / "model.bin").write_bytes(b"model")
                return str(target)

            downloaded = download_default_models(root, lambda event: None, downloader=fake_download)

            self.assertEqual(downloaded, ["faster-whisper"])
            self.assertEqual([call["repo_id"] for call in calls], [spec.repo_id for spec in DEFAULT_MODEL_DOWNLOADS])
            self.assertTrue(all(call["local_files_only"] is False for call in calls))
            self.assertTrue(all(call["force_download"] is True for call in calls))
            self.assertTrue(all(Path(call["local_dir"]).is_relative_to(local_runtime_dir(root)) for call in calls))
            self.assertTrue(all("model-downloads" in Path(call["local_dir"]).parts for call in calls))
            self.assertTrue((root / "models" / "faster-whisper" / "model.bin").is_file())

    def test_model_downloader_does_not_treat_placeholder_as_downloaded_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            self.assertTrue((root / "models" / "faster-whisper" / "README_MODEL_FILES.txt").is_file())
            calls = []

            def fake_download(**kwargs):
                calls.append(kwargs)
                target = Path(kwargs["local_dir"])
                self.assertEqual(kwargs["repo_id"], "Systran/faster-whisper-small.en")
                (target / "model.bin").write_bytes(b"model")
                return str(target)

            downloaded = download_default_models(root, lambda event: None, downloader=fake_download)

            self.assertEqual(downloaded, ["faster-whisper"])
            self.assertEqual(len(calls), 1)
            self.assertTrue((root / "models" / "faster-whisper" / "README_MODEL_FILES.txt").is_file())
            self.assertTrue((root / "models" / "faster-whisper" / "model.bin").is_file())

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
                self.assertEqual(kwargs["repo_id"], "Systran/faster-whisper-small.en")
                (target / "model.bin").write_bytes(b"model")
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
                self.assertEqual(kwargs["repo_id"], "Systran/faster-whisper-small.en")
                (target / "model.bin").write_bytes(b"model")
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

    def test_online_download_env_disables_huggingface_progress_bars_and_restores_previous_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

                with online_huggingface_download_env(Path(tmp)):
                    self.assertEqual(os.environ["HF_HUB_DISABLE_PROGRESS_BARS"], "1")

                self.assertNotIn("HF_HUB_DISABLE_PROGRESS_BARS", os.environ)

            with patch.dict(os.environ, {"HF_HUB_DISABLE_PROGRESS_BARS": "0"}, clear=False):
                with online_huggingface_download_env(Path(tmp)):
                    self.assertEqual(os.environ["HF_HUB_DISABLE_PROGRESS_BARS"], "1")

                self.assertEqual(os.environ["HF_HUB_DISABLE_PROGRESS_BARS"], "0")

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
                os.environ.pop("CURL_CA_BUNDLE", None)
                with online_huggingface_download_env(app_root):
                    self.assertIsNone(os.environ.get("HF_HUB_OFFLINE"))
                    self.assertEqual(os.environ["REQUESTS_CA_BUNDLE"], str(bundle.resolve()))
                    self.assertEqual(os.environ["SSL_CERT_FILE"], str(bundle.resolve()))
                    self.assertEqual(os.environ["CURL_CA_BUNDLE"], str(bundle.resolve()))
                self.assertNotIn("REQUESTS_CA_BUNDLE", os.environ)
                self.assertNotIn("SSL_CERT_FILE", os.environ)
                self.assertNotIn("CURL_CA_BUNDLE", os.environ)

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

    def test_online_download_env_prefers_local_whisper_ca_bundle_over_generated_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = Path(tmp) / "explicit.pem"
            explicit.write_text("explicit", encoding="utf-8")
            root = Path(tmp) / "OfflineMeetingTranscriber"
            root.mkdir()

            with patch.dict(os.environ, {CA_BUNDLE_ENV: str(explicit)}, clear=False):
                for name in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
                    os.environ.pop(name, None)
                with patch.object(bootstrap_launcher_module, "ensure_windows_ca_bundle") as generated:
                    with online_huggingface_download_env(root):
                        self.assertEqual(os.environ["REQUESTS_CA_BUNDLE"], str(explicit.resolve()))
                        self.assertEqual(os.environ["SSL_CERT_FILE"], str(explicit.resolve()))
                        self.assertEqual(os.environ["CURL_CA_BUNDLE"], str(explicit.resolve()))
                    generated.assert_not_called()

    def test_build_windows_ca_bundle_appends_windows_certs_to_certifi_pem(self):
        der_one = b"root-cert"
        der_two = b"ca-cert"
        calls = []

        def fake_enum_certificates(store_name):
            calls.append(store_name)
            if store_name == "ROOT":
                return [(der_one, "x509_asn", True), (b"ignored", "pkcs_7_asn", True)]
            if store_name == "CA":
                return [(der_two, "x509_asn", True), (der_one, "x509_asn", True)]
            return []

        def fake_der_to_pem(cert_bytes):
            return f"-----BEGIN CERTIFICATE-----\n{cert_bytes.decode('ascii')}\n-----END CERTIFICATE-----\n"

        pem = bootstrap_launcher_module.build_windows_ca_bundle_pem(
            "CERTIFI\n",
            enum_certificates=fake_enum_certificates,
            der_to_pem=fake_der_to_pem,
        )

        self.assertEqual(calls, ["ROOT", "CA"])
        self.assertTrue(pem.startswith("CERTIFI\n"))
        self.assertIn("root-cert", pem)
        self.assertIn("ca-cert", pem)
        self.assertEqual(pem.count("root-cert"), 1)
        self.assertNotIn("ignored", pem)

    def test_online_download_env_generates_windows_bundle_when_no_user_bundle_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            certifi_bundle = root / "certifi.pem"
            certifi_bundle.write_text("CERTIFI\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=False):
                for name in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE", "LOCAL_WHISPER_CA_BUNDLE"):
                    os.environ.pop(name, None)

                with patch.object(bootstrap_launcher_module, "read_certifi_bundle_pem", return_value="CERTIFI\n"):
                    with patch.object(
                        bootstrap_launcher_module,
                        "build_windows_ca_bundle_pem",
                        return_value="CERTIFI\nWINDOWS\n",
                    ):
                        with online_huggingface_download_env(root):
                            bundle = local_runtime_dir(root) / "windows-ca-bundle.pem"
                            self.assertEqual(os.environ["REQUESTS_CA_BUNDLE"], str(bundle))
                            self.assertEqual(os.environ["SSL_CERT_FILE"], str(bundle))
                            self.assertEqual(os.environ["CURL_CA_BUNDLE"], str(bundle))
                            self.assertIn("WINDOWS", bundle.read_text(encoding="utf-8"))

                self.assertNotIn("REQUESTS_CA_BUNDLE", os.environ)
                self.assertNotIn("SSL_CERT_FILE", os.environ)
                self.assertNotIn("CURL_CA_BUNDLE", os.environ)

    def test_fetch_model_repo_size_configures_tls_before_huggingface_import(self):
        calls = []
        fake_hf_module = types.ModuleType("huggingface_hub")

        class FakeSibling:
            size = 123

        class FakeInfo:
            siblings = [FakeSibling()]

        class FakeHfApi:
            def model_info(self, repo_id, files_metadata=False):
                calls.append(("model_info", repo_id, files_metadata))
                return FakeInfo()

        fake_hf_module.HfApi = FakeHfApi

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf_module}):
                with patch.object(bootstrap_launcher_module, "configure_download_tls") as configure_tls:
                    result = bootstrap_launcher_module.fetch_model_repo_size("org/model", Path(tmp))

        self.assertEqual(result, 123)
        configure_tls.assert_called_once_with(Path(tmp))
        self.assertEqual(calls, [("model_info", "org/model", True)])

    def test_certificate_verify_failure_gets_corporate_https_error_message(self):
        message = bootstrap_launcher_module.setup_error_card_message(
            "HTTPSConnectionPool failed: CERTIFICATE_VERIFY_FAILED unable to get local issuer certificate"
        )

        self.assertIn("corporate network is intercepting HTTPS", message)
        self.assertIn("company-ca.pem", message)
        self.assertIn("Retry", message)

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

            self.assertFalse(needs_setup(root, {"json": "json"}))

    def test_pip_progress_parser_reports_download_and_install_progress(self):
        parser = PipProgressParser()

        collect = parser.parse("Collecting PySide6==6.11.1")
        download = parser.parse("Downloading pyside6-6.11.1-cp310-abi3-win_amd64.whl (578 kB)")
        installing = parser.parse("Installing collected packages: PySide6, sounddevice")
        installed_one = parser.parse("Successfully installed PySide6-6.11.1 sounddevice-0.5.5")

        self.assertEqual(collect.phase, "Resolving package")
        self.assertEqual(collect.detail, "PySide6==6.11.1")
        self.assertEqual(download.phase, "Downloading package")
        self.assertIn("pyside6", download.detail)
        self.assertEqual(installing.phase, "Installing packages")
        self.assertEqual(installing.progress_percent, 0)
        self.assertEqual(installed_one.phase, "Install complete")
        self.assertEqual(installed_one.progress_percent, 100)

    def test_pip_progress_parser_reads_cli_download_bar_percent(self):
        parser = PipProgressParser()

        event = parser.parse("  42%|####2     | 42.0/100.0 MB [00:02<00:03, 19.3MB/s]")

        self.assertEqual(event.phase, "Downloading package")
        self.assertEqual(event.progress_percent, 42)
        self.assertIn("42.0/100.0 MB", event.detail)

    def test_pip_progress_parser_formats_pip_download_status(self):
        parser = PipProgressParser()

        start = parser.parse("Downloading av-18.0.0-cp311-abi3-win_amd64.whl (27.6 MB)")
        event = parser.parse("---------------------------------------- 13.8/27.6 MB 4.1 MB/s eta 0:00:03")

        self.assertEqual(start.progress_percent, 0)
        self.assertEqual(event.phase, "Downloading package")
        self.assertEqual(event.progress_percent, 50)
        self.assertIn("av-18.0.0-cp311-abi3-win_amd64.whl", event.detail)
        self.assertIn("13.8 MB / 27.6 MB", event.detail)
        self.assertIn("4.1 MB/s", event.detail)
        self.assertIn("ETA 0:00:03", event.detail)

    def test_cli_download_status_parses_speed_eta_and_sizes(self):
        status = parse_cli_download_status("---- 328.7/328.7 kB 4.1 MB/s eta 0:00:00")

        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.progress_percent, 100)
        self.assertEqual(status.current_text, "328.7 KB")
        self.assertEqual(status.total_text, "328.7 KB")
        self.assertEqual(status.speed_text, "4.1 MB/s")
        self.assertEqual(status.eta_text, "0:00:00")

    def test_run_pip_install_emits_carriage_return_download_progress(self):
        from bootstrap_launcher import run_pip_install

        with tempfile.TemporaryDirectory() as runtime_tmp, tempfile.TemporaryDirectory() as package_tmp:
            root = Path(runtime_tmp)
            package_root = Path(package_tmp)
            resources = package_root / "resources"
            resources.mkdir()
            (resources / "requirements.txt").write_text("example-package==1.0\n", encoding="utf-8")
            events = []

            class FakeProcess:
                def __init__(self, cmd, **kwargs):
                    self.stdout = io.StringIO(
                        "Downloading example-package-1.0.whl (100 MB)\n"
                        "  25%|##5       | 25.0/100.0 MB [00:01<00:03, 25.0MB/s]\r"
                        " 100%|##########| 100.0/100.0 MB [00:04<00:00, 25.0MB/s]\n"
                        "Successfully installed example-package-1.0\n"
                    )

                def wait(self):
                    return 0

            with patch("bootstrap_launcher.subprocess.Popen", FakeProcess):
                result = run_pip_install(root, events.append, package_root)

            self.assertEqual(result.code, 0)
            self.assertIn(25, [event.progress_percent for event in events])
            self.assertIn(100, [event.progress_percent for event in events])

    def test_model_downloader_reports_polled_staging_byte_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ensure_portable_layout(root)
            events = []

            def fake_download(**kwargs):
                target = Path(kwargs["local_dir"])
                self.assertEqual(kwargs["repo_id"], "Systran/faster-whisper-small.en")
                for size in (25, 50, 100):
                    (target / "model.bin").write_bytes(b"x" * size)
                    time.sleep(0.35)
                return str(target)

            with patch.object(bootstrap_launcher_module, "fetch_model_repo_size", return_value=100, create=True):
                download_default_models(root, events.append, downloader=fake_download)

            progress_values = [
                event.progress_percent
                for event in events
                if event.phase == "Downloading model" and event.progress_percent not in (None, 0, 50, 100)
            ]
            self.assertTrue(progress_values)
            self.assertTrue(any(" / " in event.detail and "elapsed" in event.detail for event in events))

    def test_model_download_formatting_and_manifest_size_helpers(self):
        sibling = types.SimpleNamespace(rfilename="model.bin", size=463 * 1024 * 1024)
        lfs_sibling = types.SimpleNamespace(rfilename="weights.bin", lfs={"size": 37 * 1024 * 1024})
        info = types.SimpleNamespace(siblings=[sibling, lfs_sibling])

        total = bootstrap_launcher_module.model_info_total_size(info)
        detail = bootstrap_launcher_module.format_model_download_status(
            "model.bin",
            downloaded_bytes=89 * 1024 * 1024,
            total_bytes=463 * 1024 * 1024,
            elapsed_seconds=47,
        )

        self.assertEqual(total, 500 * 1024 * 1024)
        self.assertEqual(bootstrap_launcher_module.format_bytes(463 * 1024 * 1024), "463 MB")
        self.assertEqual(detail, "model.bin \u2022 89 MB / 463 MB \u2022 0:47 elapsed")

    def test_folder_size_bytes_sums_nested_files_and_ignores_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            (root / "a.bin").write_bytes(b"a" * 7)
            (nested / "b.bin").write_bytes(b"b" * 11)

            self.assertEqual(bootstrap_launcher_module.folder_size_bytes(root), 18)
            self.assertEqual(bootstrap_launcher_module.folder_size_bytes(root / "missing"), 0)

    def test_weighted_overall_progress_uses_expected_step_durations(self):
        states = {step: StepState.PENDING for step in BOOTSTRAP_STEPS}
        for step in BOOTSTRAP_STEPS[:3]:
            states[step] = StepState.DONE

        progress = {step: 0 for step in BOOTSTRAP_STEPS}
        progress[BOOTSTRAP_STEPS[3]] = 50

        self.assertEqual(bootstrap_launcher_module.weighted_overall_progress(states, progress), 27)

        states[BOOTSTRAP_STEPS[3]] = StepState.DONE
        progress[BOOTSTRAP_STEPS[4]] = 50
        self.assertEqual(bootstrap_launcher_module.weighted_overall_progress(states, progress), 72)

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
                "faster-whisper, PySide6",
                0,
                "Installing collected packages: faster-whisper, PySide6",
            ),
            now=130.0,
        )
        progress, detail = tracker.record(PipProgressEvent("Running pip", "Building wheels"), now=180.0)

        self.assertLess(progress, 100)
        self.assertIn("Installing collected packages: faster-whisper, PySide6", detail)
        self.assertNotIn("ETA ~1s", detail)

    def test_install_log_records_setup_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OfflineMeetingTranscriber"
            root.mkdir()
            log_root = Path(tmp) / "logs"
            with patch.dict(os.environ, {"LOCAL_WHISPER_LOG_ROOT": str(log_root)}, clear=False):
                self.assertTrue(hasattr(bootstrap_launcher_module, "reset_install_log"))
                self.assertTrue(hasattr(bootstrap_launcher_module, "append_install_log"))

                log_path = bootstrap_launcher_module.reset_install_log(root, "Installing test build")
                bootstrap_launcher_module.append_install_log(
                    root,
                    "Downloading AI models",
                    "model.bin: 50%",
                    {"progress_percent": 50, "raw_line": "model.bin:  50%"},
                )
                content = log_path.read_text(encoding="utf-8")

        self.assertEqual(log_path, log_root / "install_log.txt")
        self.assertIn("Installing test build", content)
        self.assertIn("context: Downloading AI models", content)
        self.assertIn("progress_percent: 50", content)
        self.assertIn("model.bin:  50%", content)

    def test_setup_switches_progressbars_to_green_during_launch_countdown(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertIn("COMPLETE_PROGRESS_STYLE", source)
        self.assertIn('style.configure(COMPLETE_PROGRESS_STYLE', source)
        self.assertIn("overall_bar.configure(style=COMPLETE_PROGRESS_STYLE)", source)
        self.assertIn("detail_bar.configure(style=COMPLETE_PROGRESS_STYLE)", source)
        self.assertIn("overall_progress_value.set(100)", source)

    def test_setup_source_contains_error_card_details_expander_and_close_guard(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertIn('text="Retry this step"', source)
        self.assertIn('text="Copy details"', source)
        self.assertIn('text="Open error_log.txt"', source)
        self.assertIn('text="Show install log"', source)
        self.assertIn('text="Hide install log"', source)
        self.assertIn('text="Launch now"', source)
        self.assertIn("messagebox.askyesno", source)
        self.assertIn("details_frame.pack_forget()", source)
        self.assertIn("before=button_row", source)
        self.assertIn("replace_last=", source)
        self.assertIn("is_live_download_progress", source)

    def test_launch_now_is_hidden_until_launch_countdown(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertIn('launch_now_button = tk.Button(button_row, text="Launch now", width=14, state="disabled")', source)
        self.assertIn("def hide_launch_now_button() -> None:", source)
        self.assertIn("launch_now_button.pack_forget()", source)
        self.assertIn("hide_launch_now_button()", source)
        self.assertIn('launch_now_button.pack(side="right")', source)

    def test_confirm_screen_uses_fixed_footer_not_overlapping_content(self):
        source = (Path(__file__).resolve().parents[1] / "app" / "bootstrap_launcher.py").read_text(encoding="utf-8")

        self.assertIn('window.geometry("860x680")', source)
        self.assertIn("window.minsize(820, 650)", source)
        self.assertIn('install_button_row.pack(side="bottom"', source)
        self.assertNotIn("install_button_row.place(", source)
        self.assertNotIn("install_canvas = tk.Canvas", source)
        self.assertNotIn("install_content_window", source)
        self.assertIn("before=models_title", source)

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
