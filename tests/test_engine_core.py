import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from transcriber_engine import (
    AudioChunk,
    AudioPreprocessor,
    AtomicTranscriptWriter,
    DuplicateSuppressor,
    CAPTURE_MODE_LOOPBACK,
    EngineConfig,
    InputBlockBuffer,
    MeetingTranscriberEngine,
    TranscriptStore,
    audio_chunk_has_activity,
    audio_chunk_duration_seconds,
    default_cpu_thread_count,
    effective_cpu_threads,
    has_wasapi_output_device,
    microphone_health_message,
    preferred_input_sample_rate,
    resample_audio,
    rms_to_meter_percent,
    transcription_segment_is_usable,
)
from ui_helpers import meter_bar_geometry


class DuplicateSuppressorTests(unittest.TestCase):
    def test_duplicate_text_inside_overlap_window_is_suppressed(self):
        suppressor = DuplicateSuppressor(window_seconds=2.0)

        self.assertFalse(suppressor.is_duplicate(10.0, "Confirm the deadline."))
        self.assertTrue(suppressor.is_duplicate(11.5, " confirm  the deadline "))
        self.assertFalse(suppressor.is_duplicate(13.1, "Confirm the deadline."))


class TranscriptStoreTests(unittest.TestCase):
    def test_add_transcript_keeps_plain_text_rows(self):
        store = TranscriptStore()

        row = store.add_transcript(chunk_index=1, start=0.0, end=2.0, text=" Hello team. ")

        self.assertEqual(row.text, "Hello team.")
        self.assertEqual(store.snapshot()[0].text, "Hello team.")


class AtomicTranscriptWriterTests(unittest.TestCase):
    def test_refresh_writes_plain_transcript_atomically(self):
        store = TranscriptStore()
        store.add_transcript(chunk_index=1, start=0.0, end=2.0, text="Hello team.")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "transcript.txt")
            writer = AtomicTranscriptWriter(path)
            writer.refresh(store.rows)

            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()

        self.assertIn("[00:00:00] Hello team.", content)
        self.assertNotIn("Speaker", content)

    def test_recovery_write_preserves_transcript_when_primary_path_is_locked(self):
        store = TranscriptStore()
        store.add_transcript(chunk_index=1, start=0.0, end=2.0, text="Hello team.")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "meeting_transcript.txt")
            writer = AtomicTranscriptWriter(path)

            with patch.object(writer, "refresh", side_effect=PermissionError("locked")):
                recovery_path = writer.refresh_with_recovery(store.rows)

            with open(recovery_path, "r", encoding="utf-8") as handle:
                content = handle.read()

        self.assertTrue(str(recovery_path).endswith(".recovery.txt"))
        self.assertIn("[00:00:00] Hello team.", content)


class InputBlockBufferTests(unittest.TestCase):
    def test_callback_block_is_copied_before_queueing(self):
        buffer = InputBlockBuffer(max_blocks=2)
        block = [[0.1], [0.2], [0.3]]

        buffer.push_from_callback(block)
        block[0][0] = 9.9

        queued = buffer.pop()
        self.assertEqual(queued[0][0], 0.1)


class MicrophoneDiagnosticsTests(unittest.TestCase):
    def test_rms_to_meter_percent_makes_real_microphone_levels_visible(self):
        self.assertEqual(rms_to_meter_percent(0.0), 0)
        self.assertEqual(rms_to_meter_percent(0.00001), 0)
        self.assertGreaterEqual(rms_to_meter_percent(0.001), 5)
        self.assertGreaterEqual(rms_to_meter_percent(0.01), 50)
        self.assertEqual(rms_to_meter_percent(1.0), 100)

    def test_microphone_health_reports_no_callbacks(self):
        message = microphone_health_message(block_count=0, peak_rms=0.0, chunk_count=0)

        self.assertIn("No microphone audio was received", message)
        self.assertIn("microphone permission", message)

    def test_microphone_health_reports_silent_input(self):
        message = microphone_health_message(block_count=10, peak_rms=0.00002, chunk_count=0)

        self.assertIn("Microphone input looks silent", message)
        self.assertIn("choose a different input device", message)

    def test_microphone_health_accepts_audible_input(self):
        self.assertIsNone(microphone_health_message(block_count=10, peak_rms=0.01, chunk_count=1))

    def test_engine_stop_emits_diagnostic_when_recording_was_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = MeetingTranscriberEngine(EngineConfig(output_file=os.path.join(tmp, "out.txt")))
            events = []
            engine.on_event(lambda event_type, payload: events.append((event_type, payload)))
            engine._audio_block_count = 10
            engine._audio_peak_rms = 0.00002
            engine._audio_chunk_count = 0

            engine.stop()

        messages = [payload["message"] for event_type, payload in events if event_type == "error"]
        self.assertTrue(any("Microphone input looks silent" in message for message in messages))

    def test_preferred_input_sample_rate_uses_target_when_supported(self):
        class FakeSoundDevice:
            @staticmethod
            def check_input_settings(device=None, channels=None, samplerate=None, dtype=None):
                self.assertEqual(samplerate, 16000)

        self.assertEqual(preferred_input_sample_rate(FakeSoundDevice, None, 16000), 16000)

    def test_preferred_input_sample_rate_falls_back_to_device_default(self):
        class FakeSoundDevice:
            @staticmethod
            def check_input_settings(device=None, channels=None, samplerate=None, dtype=None):
                raise RuntimeError("unsupported")

            @staticmethod
            def query_devices(device=None, kind=None):
                return {"default_samplerate": 48000}

        self.assertEqual(preferred_input_sample_rate(FakeSoundDevice, 9, 16000), 48000)

    def test_resample_audio_changes_length_for_model_sample_rate(self):
        import numpy as np

        source = np.ones(48000, dtype=np.float32)
        resampled = resample_audio(source, 48000, 16000)

        self.assertEqual(len(resampled), 16000)
        self.assertEqual(resampled.dtype, np.float32)


class AudioPreprocessorTests(unittest.TestCase):
    def test_dc_offset_moves_toward_zero(self):
        import numpy as np

        processor = AudioPreprocessor(16000)
        audio = np.full(16000, 0.2, dtype=np.float32)

        cleaned = processor.process(audio)

        self.assertLess(abs(float(np.mean(cleaned[-4000:]))), 0.01)

    def test_high_pass_reduces_low_frequency_drift(self):
        import numpy as np

        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        low = (0.1 * np.sin(2 * np.pi * 20 * t)).astype(np.float32)
        mid = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        low_cleaned = AudioPreprocessor(sample_rate).process(low)
        mid_cleaned = AudioPreprocessor(sample_rate).process(mid)

        self.assertLess(float(np.std(low_cleaned)), float(np.std(mid_cleaned)) * 0.8)

    def test_quiet_noise_is_attenuated(self):
        import numpy as np

        rng = np.random.default_rng(7)
        noise = rng.normal(0.0, 0.0002, 16000).astype(np.float32)

        cleaned = AudioPreprocessor(16000).process(noise)

        self.assertLess(float(np.sqrt(np.mean(np.square(cleaned)))), 0.0002)

    def test_loud_speech_like_signal_is_preserved(self):
        import numpy as np

        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        speech = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        cleaned = AudioPreprocessor(sample_rate).process(speech)

        self.assertGreater(float(np.sqrt(np.mean(np.square(cleaned)))), 0.02)

    def test_limiter_caps_peaks(self):
        import numpy as np

        cleaned = AudioPreprocessor(16000).process(np.array([2.0, -2.0, 0.0], dtype=np.float32))

        self.assertLessEqual(float(np.max(np.abs(cleaned))), 0.98)

    def test_chunk_activity_rejects_noise_and_accepts_speech(self):
        import numpy as np

        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        speech = (0.04 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        noise = np.zeros(sample_rate, dtype=np.float32)

        self.assertFalse(audio_chunk_has_activity(noise))
        self.assertTrue(audio_chunk_has_activity(speech))

    def test_soft_speech_survives_cleanup_and_activity_gate(self):
        import numpy as np

        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        soft_speech = (0.0015 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        cleaned = AudioPreprocessor(sample_rate).process(soft_speech)

        self.assertTrue(audio_chunk_has_activity(cleaned))


class TranscriptionFilterTests(unittest.TestCase):
    def test_rejects_high_no_speech_probability(self):
        segment = type("Segment", (), {"no_speech_prob": 0.9})()

        self.assertFalse(transcription_segment_is_usable(segment, "hello"))

    def test_rejects_low_log_probability(self):
        segment = type("Segment", (), {"avg_logprob": -2.0})()

        self.assertFalse(transcription_segment_is_usable(segment, "hello"))

    def test_accepts_normal_segment(self):
        segment = type("Segment", (), {"no_speech_prob": 0.1, "avg_logprob": -0.2})()

        self.assertTrue(transcription_segment_is_usable(segment, "hello"))


class EngineConfigTests(unittest.TestCase):
    def test_missing_model_paths_return_clear_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = EngineConfig(
                whisper_model_dir=os.path.join(tmp, "missing-whisper"),
                output_file=os.path.join(tmp, "out.txt"),
            )

            errors = config.validate()

        self.assertIn("Whisper model path does not exist", "\n".join(errors))
        self.assertNotIn("Speaker", "\n".join(errors))

    def test_empty_model_folder_returns_incomplete_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            whisper = os.path.join(tmp, "whisper")
            os.mkdir(whisper)
            config = EngineConfig(
                whisper_model_dir=whisper,
                output_file=os.path.join(tmp, "out.txt"),
            )

            errors = "\n".join(config.validate())

        self.assertIn("Whisper model is incomplete: expected model.bin", errors)
        self.assertNotIn("Speaker", errors)

    def test_minimal_required_model_files_pass_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            whisper = os.path.join(tmp, "whisper")
            os.mkdir(whisper)
            open(os.path.join(whisper, "model.bin"), "wb").close()
            config = EngineConfig(
                whisper_model_dir=whisper,
                output_file=os.path.join(tmp, "out.txt"),
            )

            errors = config.validate()

        self.assertEqual(errors, [])

    def test_defaults_pin_english_and_accuracy_chunks(self):
        config = EngineConfig()

        self.assertEqual(config.language, "en")
        self.assertEqual(config.chunk_seconds, 5.0)
        self.assertEqual(config.overlap_seconds, 0.5)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.compute_type, "int8")

    def test_default_cpu_threads_stays_bounded_for_laptops(self):
        self.assertEqual(default_cpu_thread_count(2), 2)
        self.assertEqual(default_cpu_thread_count(16), 8)
        self.assertEqual(default_cpu_thread_count(None), min(8, os.cpu_count() or 4))

    def test_effective_cpu_threads_preserves_explicit_user_value(self):
        self.assertEqual(effective_cpu_threads(6, "cpu"), 6)
        self.assertEqual(effective_cpu_threads(0, "cuda"), 0)

    def test_validate_rejects_loopback_without_wasapi_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            whisper = os.path.join(tmp, "whisper")
            os.mkdir(whisper)
            open(os.path.join(whisper, "model.bin"), "wb").close()
            config = EngineConfig(
                whisper_model_dir=whisper,
                output_file=os.path.join(tmp, "out.txt"),
                capture_mode=CAPTURE_MODE_LOOPBACK,
                system_device_index=3,
            )

            with patch("transcriber_engine.has_wasapi_output_device", return_value=False):
                errors = config.validate()

        self.assertTrue(any("WASAPI output" in error for error in errors))

    def test_has_wasapi_output_device_false_on_non_windows(self):
        with patch("os.name", "posix"):
            self.assertFalse(has_wasapi_output_device(0))

    def test_has_wasapi_output_device_checks_hostapi_name(self):
        fake_sd = type(sys)("sounddevice")

        def query_devices(index):
            return {"hostapi": 1}

        fake_sd.query_devices = query_devices
        fake_sd.query_hostapis = lambda: [{"name": "MME"}, {"name": "Windows WASAPI"}]

        with patch("os.name", "nt"), patch.dict(sys.modules, {"sounddevice": fake_sd}):
            self.assertTrue(has_wasapi_output_device(0))

    def test_whisper_model_load_uses_cpu_thread_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            whisper = os.path.join(tmp, "whisper")
            os.mkdir(whisper)
            open(os.path.join(whisper, "model.bin"), "wb").close()
            calls = []

            class FakeWhisperModel:
                def __init__(self, *args, **kwargs):
                    calls.append((args, kwargs))

            fake_module = type(sys)("faster_whisper")
            fake_module.WhisperModel = FakeWhisperModel
            config = EngineConfig(
                whisper_model_dir=whisper,
                output_file=os.path.join(tmp, "out.txt"),
                cpu_threads=6,
            )

            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                MeetingTranscriberEngine(config)._load_whisper_model()

        self.assertEqual(calls[0][1]["device"], "cpu")
        self.assertEqual(calls[0][1]["cpu_threads"], 6)
        self.assertEqual(calls[0][1]["compute_type"], "int8")

    def test_whisper_model_load_allows_cuda_trial_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            whisper = os.path.join(tmp, "whisper")
            os.mkdir(whisper)
            open(os.path.join(whisper, "model.bin"), "wb").close()
            calls = []

            class FakeWhisperModel:
                def __init__(self, *args, **kwargs):
                    calls.append((args, kwargs))

            fake_module = type(sys)("faster_whisper")
            fake_module.WhisperModel = FakeWhisperModel
            config = EngineConfig(
                whisper_model_dir=whisper,
                output_file=os.path.join(tmp, "out.txt"),
                device="cuda",
                compute_type="float16",
            )

            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                MeetingTranscriberEngine(config)._load_whisper_model()

        self.assertEqual(calls[0][1]["device"], "cuda")
        self.assertEqual(calls[0][1]["compute_type"], "float16")
        self.assertNotIn("cpu_threads", calls[0][1])

    def test_openvino_whisper_model_loads_pipeline_and_returns_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_case = self
            model_dir = os.path.join(tmp, "openvino")
            os.mkdir(model_dir)
            open(os.path.join(model_dir, "openvino_encoder_model.xml"), "w", encoding="utf-8").close()
            calls = []

            class FakeWhisperPipeline:
                def __init__(self, path, device):
                    calls.append((path, device))

                def generate(self, samples):
                    test_case.assertEqual(samples, [0.1, 0.2])
                    return " hello "

            fake_module = type(sys)("openvino_genai")
            fake_module.WhisperPipeline = FakeWhisperPipeline
            config = EngineConfig(
                whisper_model_dir=model_dir,
                output_file=os.path.join(tmp, "out.txt"),
                device="openvino:GPU",
                compute_type="fp16",
            )

            with patch.dict(sys.modules, {"openvino_genai": fake_module}):
                model = MeetingTranscriberEngine(config)._load_whisper_model()
                segments, _ = model.transcribe([0.1, 0.2])

        self.assertEqual(calls, [(model_dir, "GPU")])
        self.assertEqual(segments[0].text, "hello")


class TranscriptionBacklogTests(unittest.TestCase):
    def test_audio_chunk_duration_uses_sample_count_and_rate(self):
        chunk = AudioChunk(1, 0.0, [0.0] * 32000, 16000)

        self.assertEqual(audio_chunk_duration_seconds(chunk), 2.0)

    def test_queued_audio_seconds_ignores_stop_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = MeetingTranscriberEngine(EngineConfig(output_file=os.path.join(tmp, "out.txt")))
            engine._transcription_queue.put(AudioChunk(1, 0.0, [0.0] * 16000, 16000))
            engine._transcription_queue.put(None)

            seconds = engine._queued_audio_seconds()

        self.assertEqual(seconds, 1.0)

    def test_fanout_preserves_all_chunks_when_transcription_backlog_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = MeetingTranscriberEngine(EngineConfig(output_file=os.path.join(tmp, "out.txt")))
            for index in range(4):
                engine._raw_chunk_queue.put(AudioChunk(index, float(index), [0.0] * 16000, 16000))
            engine._raw_chunk_queue.put(None)
            events = []
            engine.on_event(lambda event_type, payload: events.append((event_type, payload)))

            engine._fanout_loop()
            remaining = []
            while not engine._transcription_queue.empty():
                item = engine._transcription_queue.get_nowait()
                remaining.append(None if item is None else item.index)

        self.assertEqual(remaining, [0, 1, 2, 3, None])
        self.assertEqual(engine._pending_transcription_chunks, {0, 1, 2, 3})
        self.assertTrue(any(payload.get("message") == "Finishing transcription" for event_type, payload in events if event_type == "status"))
        self.assertFalse(any("Catching up: skipped" in payload.get("message", "") for event_type, payload in events if event_type == "log"))

    def test_lag_uses_oldest_pending_capture_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = MeetingTranscriberEngine(EngineConfig(output_file=os.path.join(tmp, "out.txt")))
            events = []
            engine.on_event(lambda event_type, payload: events.append((event_type, payload)))
            engine._transcription_queue.put(AudioChunk(1, 0.0, [0.0] * 16000, 16000, captured_at=70.0))
            current = AudioChunk(2, 1.0, [0.0] * 16000, 16000, captured_at=90.0)

            with patch("transcriber_engine.time.monotonic", return_value=100.0):
                engine._emit_transcription_lag(active=True, current_chunk=current)

        lag_events = [payload for event_type, payload in events if event_type == "lag"]
        self.assertEqual(lag_events[-1]["behind_seconds"], 30.0)
        self.assertTrue(lag_events[-1]["active"])

    def test_transcript_save_failure_does_not_abort_future_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = MeetingTranscriberEngine(EngineConfig(output_file=os.path.join(tmp, "out.txt")))
            events = []
            engine.on_event(lambda event_type, payload: events.append((event_type, payload)))
            engine.store.add_transcript(1, 0.0, 1.0, "First sentence.")
            calls = {"count": 0}

            def flaky_refresh(rows):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise PermissionError("locked")

            engine.writer.refresh = flaky_refresh

            engine._save_transcript_snapshot(final=False)
            engine.store.add_transcript(2, 1.0, 2.0, "Second sentence.")
            engine._save_transcript_snapshot(final=False)

        messages = [payload.get("message", "") for event_type, payload in events if event_type == "error"]
        self.assertEqual(calls["count"], 2)
        self.assertEqual(len([message for message in messages if "Transcript save failed" in message]), 1)

    def test_final_save_failure_writes_recovery_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = MeetingTranscriberEngine(EngineConfig(output_file=os.path.join(tmp, "meeting_transcript.txt")))
            events = []
            engine.on_event(lambda event_type, payload: events.append((event_type, payload)))
            engine.store.add_transcript(1, 0.0, 1.0, "Recovered sentence.")

            with patch.object(engine.writer, "refresh", side_effect=PermissionError("locked")):
                recovery_path = engine._save_transcript_snapshot(final=True)

            with open(recovery_path, "r", encoding="utf-8") as handle:
                content = handle.read()

        messages = [payload.get("message", "") for event_type, payload in events if event_type == "error"]
        self.assertTrue(str(recovery_path).endswith(".recovery.txt"))
        self.assertIn("[00:00:00] Recovered sentence.", content)
        self.assertTrue(any(str(recovery_path) in message for message in messages))


class MicMeterGeometryTests(unittest.TestCase):
    def test_segment_count_grows_with_width_and_bar_width_stays_constant(self):
        narrow_count, narrow_width, _ = meter_bar_geometry(220)
        wide_count, wide_width, _ = meter_bar_geometry(440)

        self.assertGreater(wide_count, narrow_count)
        self.assertEqual(narrow_width, 7)
        self.assertEqual(wide_width, 7)


if __name__ == "__main__":
    unittest.main()
