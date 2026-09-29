"""Microphone recording for AegisScribe.

Recordings are saved locally as 16-bit mono WAV files, named by timestamp
(e.g. data/consultation_20260929_143015.wav) so one consultation never
overwrites another. Call delete_recording() once you no longer need the audio.
"""
import os
from datetime import datetime

DATA_DIR = "data"


class RecordingError(RuntimeError):
    """Raised when audio cannot be captured (no microphone, permission denied...)."""


def record_consultation(duration=10, fs=44100, data_dir=DATA_DIR):
    """Record `duration` seconds from the default microphone.

    Returns the path of the saved WAV file. Raises RecordingError on failure.
    """
    try:
        # Imported here so the app still starts on machines without PortAudio.
        import numpy as np
        import sounddevice as sd
        from scipy.io.wavfile import write
    except (ImportError, OSError) as exc:
        raise RecordingError(f"Audio libraries are not available: {exc}") from exc

    try:
        sd.check_input_settings(samplerate=fs, channels=1)
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()  # block until the buffer is full
    except Exception as exc:  # noqa: BLE001 - sounddevice raises many error types
        raise RecordingError(f"Could not access the microphone: {exc}") from exc

    # float32 WAVs do not play in every browser; 16-bit PCM works everywhere.
    pcm = (np.clip(recording, -1.0, 1.0) * 32767).astype(np.int16)

    os.makedirs(data_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(data_dir, f"consultation_{stamp}.wav")
    write(save_path, fs, pcm)
    return save_path


def delete_recording(path):
    """Delete a recording. Returns True if a file was removed."""
    try:
        os.remove(path)
        return True
    except (FileNotFoundError, TypeError, OSError):
        return False


if __name__ == "__main__":
    # Quick hardware test: records 5 seconds.
    print("Recording 5 seconds...")
    print("Saved:", record_consultation(duration=5))
