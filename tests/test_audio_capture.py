import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from scipy.io import wavfile

import audio_capture
from audio_capture import RecordingError, delete_recording, record_consultation


def fake_sounddevice(signal=None, fail_with=None):
    """A stand-in for the sounddevice module (no microphone needed)."""
    module = types.ModuleType("sounddevice")

    def check_input_settings(**kwargs):
        if fail_with:
            raise fail_with

    def rec(frames, samplerate, channels, dtype):
        base = np.zeros((frames, channels), dtype="float32") if signal is None else signal
        return base[:frames] if len(base) >= frames else np.resize(base, (frames, channels))

    module.check_input_settings = check_input_settings
    module.rec = rec
    module.wait = lambda: None
    return module


class RecordingTests(unittest.TestCase):
    def test_saves_16bit_mono_wav_with_timestamped_name(self):
        tone = (0.5 * np.sin(np.linspace(0, 100, 8000))).astype("float32").reshape(-1, 1)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"sounddevice": fake_sounddevice(tone)}):
            path = record_consultation(duration=1, fs=8000, data_dir=tmp)
            self.assertTrue(os.path.basename(path).startswith("consultation_"))
            self.assertTrue(path.endswith(".wav"))
            rate, data = wavfile.read(path)
        self.assertEqual(rate, 8000)
        self.assertEqual(data.dtype, np.int16)
        self.assertEqual(data.shape, (8000,))
        self.assertGreater(int(np.abs(data).max()), 10000)  # signal survived conversion

    def test_creates_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"sounddevice": fake_sounddevice()}):
            target = os.path.join(tmp, "nested", "data")
            path = record_consultation(duration=1, fs=8000, data_dir=target)
            self.assertTrue(os.path.exists(path))

    def test_microphone_error_becomes_recording_error(self):
        broken = fake_sounddevice(fail_with=OSError("No default input device"))
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"sounddevice": broken}):
            with self.assertRaises(RecordingError) as ctx:
                record_consultation(duration=1, data_dir=tmp)
            self.assertIn("No default input device", str(ctx.exception))
            self.assertEqual(os.listdir(tmp), [])  # nothing half-written

    def test_missing_audio_library_becomes_recording_error(self):
        with mock.patch.dict(sys.modules, {"sounddevice": None}):  # forces ImportError
            with self.assertRaises(RecordingError):
                record_consultation(duration=1)


class DeleteTests(unittest.TestCase):
    def test_delete_existing_and_missing(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        self.assertTrue(delete_recording(path))
        self.assertFalse(os.path.exists(path))
        self.assertFalse(delete_recording(path))   # already gone
        self.assertFalse(delete_recording(None))   # nothing recorded


if __name__ == "__main__":
    unittest.main()
