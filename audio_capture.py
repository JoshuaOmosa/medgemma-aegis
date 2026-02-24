import sounddevice as sd
from scipy.io.wavfile import write
import os
import time

def record_consultation(duration=10, fs=44100):
    print(f"--- Phase 2: Secure Local Recording ---")
    
    # 1. Check for Microphone
    try:
        # We start the recording buffer
        print(f"🎤 [RECORDING] {duration} seconds remaining...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        
        # 2. Visual Feedback (Countdown)
        for i in range(duration, 0, -1):
            print(f"Time left: {i}s", end="\r")
            time.sleep(1)
            
        sd.wait()  # Ensure recording buffer is full
        
        # 3. Secure Local Storage
        save_path = "data/raw_consultation.wav"
        os.makedirs("data", exist_ok=True)
        
        # Write the file
        write(save_path, fs, recording) 
        
        print(f"\n✅ Recording saved safely to: {save_path}")
        return save_path

    except Exception as e:
        print(f"\n❌ Hardware Error: Could not access microphone. {e}")
        return None

if __name__ == "__main__":
    # Test a 5-second clip to verify it works
    record_consultation(duration=5)