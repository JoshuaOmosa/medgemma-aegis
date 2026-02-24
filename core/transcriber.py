import whisper
import os

def run_transcription(audio_path):
    print(f"--- 🩺 Phase 2: Transcribing Audio ---")
    # Using 'tiny' to keep your 7.35GB RAM happy
    model = whisper.load_model("tiny")
    
    # We guide the AI with a 'prompt' to improve medical accuracy
    result = model.transcribe(
        audio_path, 
        initial_prompt="A clinical consultation regarding patient symptoms and history."
    )
    
    print("✅ Transcription Complete.")
    return result["text"]

if __name__ == "__main__":
    # This is a test run - make sure you have a file in your recordings folder
    # test_file = "recordings/consultation.wav"
    # print(run_transcription(test_file))
    pass