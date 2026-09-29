from transformers import pipeline

def generate_medical_note(transcript):
    print(f"--- 🧠 Phase 3: Generating SOAP Note ---")
    
    # We use a compact, efficient model for your RAM constraints
    # 'distilgpt2' is a placeholder; for the final demo, we can try MedGemma-2B
    scribe = pipeline("text-generation", model="distilgpt2")
    
    prompt = f"Summarize the following medical transcript into Subjective, Objective, Assessment, and Plan:\n\n{transcript}"
    
    # Generating the note
    note = scribe(prompt, max_new_tokens=200, clean_up_tokenization_spaces=True)
    return note[0]['generated_text']