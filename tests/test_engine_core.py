import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from transcriber_engine import (
    AudioChunk,
    AudioPreprocessor,
    AtomicTranscriptWriter,
    DuplicateSuppressor,
    EngineConfig,
    InputBlockBuffer,
    MeetingTranscriberEngine,
    TranscriptStore,
    audio_chunk_has_activity,
    audio_chunk_duration_seconds,
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

    def test_defaults_pin_english_and_shorter_chunks(self):
        config = EngineConfig()

        self.assertEqual(config.language, "en")
        self.assertEqual(config.chunk_seconds, 3.0)
        self.assertEqual(config.overlap_seconds, 0.25)


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

    def test_stale_chunks_are_dropped_and_newest_chunks_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = MeetingTranscriberEngine(EngineConfig(output_file=os.path.join(tmp, "out.txt")))
            for index in range(4):
                engine._transcription_queue.put(AudioChunk(index, float(index), [], 16000))
                engine._pending_transcription_chunks.add(index)

            dropped = engine._drop_stale_transcription_chunks(max_queued=2)
            remaining = []
            while not engine._transcription_queue.empty():
                remaining.append(engine._transcription_queue.get_nowait().index)

        self.assertEqual(dropped, 2)
        self.assertEqual(remaining, [2, 3])
        self.assertEqual(engine._pending_transcription_chunks, {2, 3})


class MicMeterGeometryTests(unittest.TestCase):
    def test_segment_count_grows_with_width_and_bar_width_stays_constant(self):
        narrow_count, narrow_width, _ = meter_bar_geometry(220)
        wide_count, wide_width, _ = meter_bar_geometry(440)

        self.assertGreater(wide_count, narrow_count)
        self.assertEqual(narrow_width, 7)
        self.assertEqual(wide_width, 7)


if __name__ == "__main__":
    unittest.main()
