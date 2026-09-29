"""AegisScribe environment check. Run:  python heartbeat.py

Every check is independent, so one missing library does not hide the others.
"""
import platform
import shutil

print("--- AegisScribe Universal Heartbeat ---")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")

problems = []


def check(label, fn):
    """Run fn(); print '✅ label: result' or '❌ label: error'."""
    try:
        print(f"✅ {label}: {fn()}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {label}: {exc}")
        problems.append(label)


def _version(module):
    def get():
        return __import__(module).__version__
    return get


for lib, module in [("NumPy", "numpy"), ("Torch", "torch"), ("Whisper", "whisper"),
                    ("Transformers", "transformers"), ("Streamlit", "streamlit")]:
    check(f"{lib} version", _version(module))


def ram():
    import psutil

    gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    if gb < 8:
        print("⚠️  Low RAM: MedGemma needs about 8 GB. Consider a GPU or a smaller setup.")
    return f"{gb} GB"


def cuda():
    import torch

    if torch.cuda.is_available():
        return f"available ({torch.cuda.get_device_name(0)}) -> 4-bit MedGemma"
    return "not available -> MedGemma will load unquantised on CPU (needs ~8-9 GB free RAM)"


def ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("not found on PATH (Whisper needs it to read audio)")
    return path


def bitsandbytes():
    import bitsandbytes  # noqa: F401

    return "importable"


def microphone():
    import sounddevice as sd

    return sd.query_devices(kind="input")["name"]


def whisper_engine():
    import whisper

    whisper.load_model("tiny", device="cpu")
    return "engine online"


check("System RAM", ram)
check("CUDA GPU", cuda)
check("FFmpeg", ffmpeg)
check("bitsandbytes (only needed with a GPU)", bitsandbytes)
check("Microphone", microphone)
check("Whisper tiny", whisper_engine)

print()
print("All checks passed. Run: streamlit run app.py" if not problems
      else f"{len(problems)} check(s) need attention: " + ", ".join(problems))
