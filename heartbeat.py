import torch
import numpy
import whisper
import platform
import psutil # You might need to run: pip install psutil

print("--- AegisScribe Universal Heartbeat ---")
print(f"OS: {platform.system()} {platform.release()}")
print(f"NumPy Version: {numpy.__version__}")
print(f"Torch Version: {torch.__version__}")
print(f"Whisper Version: {whisper.__version__}")

# Check CPU Health for Phase 3
ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
print(f"System RAM: {ram_gb} GB")

if ram_gb < 8:
    print("⚠️ WARNING: Low RAM. Use 'tiny' models only.")
else:
    print("✅ Status: System ready for CPU-Inference.")

# Verify Whisper is actually loading properly
try:
    # We load 'tiny' just to see if the DLLs are working
    test_model = whisper.load_model("tiny", device="cpu")
    print("✅ Whisper: Engine online.")
except Exception as e:
    print(f"❌ Whisper Error: {e}")