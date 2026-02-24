import os
import torch
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from functools import partialmethod

# 1. STOP THE CRASHES: Disable progress bars that trigger Windows 'WinError 6'
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def process_medical_session(audio_path):
    print("--- 🩺 Phase 1: Transcribing ---")
    st_model = whisper.load_model("tiny", device="cpu")
    # verbose=None is critical to prevent console handle errors
    result = st_model.transcribe(audio_path, fp16=False, verbose=None)
    transcript = result["text"].strip()
    
    print("--- 🧠 Phase 2: Generating SOAP Note with MedGemma 1.5 ---")
    model_id = "google/medgemma-1.5-4b-it"
    
    try:
        # Optimized 4-bit Config for 8GB RAM
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # Uses modern hardware acceleration
            bnb_4bit_quant_type="nf4",             # Better for clinical accuracy
            bnb_4bit_use_double_quant=True         # Saves an extra ~250MB of RAM
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        scribe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        # CORRECT MedGemma 1.5 Prompt Template
        # Note: Must use these specific tags or the model 'hallucinates'
        prompt = (
            f"<bos><start_of_turn>user\n"
            f"You are a clinical scribe. Transform the following transcript into a "
            f"structured SOAP note (Subjective, Objective, Assessment, Plan).\n\n"
            f"Transcript: {transcript}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        
        # Generation: Keep temperature low for medical facts
        raw_output = scribe(prompt, max_new_tokens=512, temperature=0.1)[0]['generated_text']
        
        # Clean output to only show the note
        soap_note = raw_output.split("model\n")[-1].strip()
        soap_note = soap_note.replace("<end_of_turn>", "").replace("<eos>", "")

    except Exception as e:
        print(f"MedGemma Error: {e}")
        # Fast fallback in case of OOM (Out of Memory)
        return transcript, "Memory limit reached. Try a shorter recording."

    return transcript, soap_note