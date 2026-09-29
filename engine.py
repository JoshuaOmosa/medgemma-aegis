"""AegisScribe engine: Whisper transcription + MedGemma SOAP-note generation.

Heavy libraries (torch, whisper, transformers) are imported lazily and both
models are cached after the first load, so the app starts fast and only the
first "Generate" click pays the loading cost.

Settings can be changed with environment variables (no code edits needed):

    AEGIS_WHISPER_MODEL   Whisper size: tiny | base | small ...   (default: tiny)
    AEGIS_LANGUAGE        Force a language code, e.g. "en", "sw"   (default: auto)
    AEGIS_WHISPER_PROMPT  Optional Whisper initial prompt          (default: none)
    AEGIS_LLM_ID          Hugging Face model id                    (default: google/medgemma-1.5-4b-it)
    AEGIS_MAX_NEW_TOKENS  Max length of the generated note         (default: 512)
    AEGIS_TRUST_REMOTE_CODE  Set to 1 to allow custom model code   (default: off)
"""
import logging
import os
from functools import lru_cache, partialmethod
from typing import Tuple

log = logging.getLogger("aegisscribe.engine")

# Progress bars can crash Windows consoles ("WinError 6"), so switch them off.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
try:
    from tqdm import tqdm

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
except ImportError:  # tqdm is optional here
    pass

# ---- Settings ---------------------------------------------------------------
WHISPER_MODEL = os.getenv("AEGIS_WHISPER_MODEL", "tiny")
WHISPER_LANGUAGE = os.getenv("AEGIS_LANGUAGE") or None
WHISPER_PROMPT = os.getenv("AEGIS_WHISPER_PROMPT") or None
LLM_ID = os.getenv("AEGIS_LLM_ID", "google/medgemma-1.5-4b-it")
MAX_NEW_TOKENS = int(os.getenv("AEGIS_MAX_NEW_TOKENS", "512"))
TRUST_REMOTE_CODE = os.getenv("AEGIS_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")

SOAP_PROMPT = """You are a clinical scribe. Convert the consultation transcript below into a structured SOAP note.

Rules:
- Use exactly these four headings, in this order: Subjective, Objective, Assessment, Plan.
- Use only information stated in the transcript. Do not invent symptoms, findings, measurements, diagnoses or medications.
- If a section has no supporting information, write "Not documented" under it.
- Be concise and use clinical language.

Transcript:
{transcript}"""


class SoapGenerationError(RuntimeError):
    """Raised when a SOAP note cannot be produced. The message is user-friendly."""


# ---- Model loading (cached) ---------------------------------------------------
@lru_cache(maxsize=None)
def get_whisper():
    """Load Whisper once and reuse it."""
    import whisper

    log.info("Loading Whisper '%s' on CPU", WHISPER_MODEL)
    return whisper.load_model(WHISPER_MODEL, device="cpu")


@lru_cache(maxsize=None)
def get_scribe():
    """Load MedGemma once and reuse it.

    With a CUDA GPU the model is loaded in 4-bit (bitsandbytes). Without one,
    bitsandbytes is not reliable, so the model is loaded unquantised in
    bfloat16 instead, which needs roughly 8-9 GB of free RAM.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    log.info("Loading %s", LLM_ID)
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=TRUST_REMOTE_CODE)

    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=TRUST_REMOTE_CODE,
        )
    else:
        log.warning("No CUDA GPU found: loading MedGemma unquantised in bfloat16 on CPU")
        model = AutoModelForCausalLM.from_pretrained(
            LLM_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=TRUST_REMOTE_CODE,
        )

    return pipeline("text-generation", model=model, tokenizer=tokenizer)


# ---- Public API ---------------------------------------------------------------
def transcribe_audio(audio_path: str) -> str:
    """Speech to text with Whisper. Returns the transcript (may be empty)."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    result = get_whisper().transcribe(
        audio_path,
        fp16=False,
        verbose=None,  # avoids console-handle errors on Windows
        language=WHISPER_LANGUAGE,
        initial_prompt=WHISPER_PROMPT,
    )
    return result["text"].strip()


def generate_soap_note(transcript: str) -> str:
    """Turn a transcript into a SOAP note. Raises SoapGenerationError on failure."""
    transcript = (transcript or "").strip()
    if not transcript:
        raise SoapGenerationError(
            "No speech was detected in the recording, so there is nothing to summarise. "
            "Check the microphone and try again."
        )
    try:
        scribe = get_scribe()
        messages = [{"role": "user", "content": SOAP_PROMPT.format(transcript=transcript)}]
        # Greedy decoding: deterministic output, which is what we want for clinical text.
        output = scribe(messages, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return _extract_reply(output)
    except SoapGenerationError:
        raise
    except Exception as exc:  # noqa: BLE001 - we translate everything into a readable message
        log.exception("SOAP generation failed")
        raise SoapGenerationError(_explain(exc)) from exc


def process_medical_session(audio_path: str) -> Tuple[str, str]:
    """Backwards-compatible one-shot helper: audio -> (transcript, note).

    On failure the note field contains an explanatory message. The app itself
    calls transcribe_audio() and generate_soap_note() separately so it can show
    proper errors.
    """
    transcript = transcribe_audio(audio_path)
    try:
        return transcript, generate_soap_note(transcript)
    except SoapGenerationError as exc:
        return transcript, f"[Note generation failed] {exc}"


# ---- Internals ----------------------------------------------------------------
def _extract_reply(output) -> str:
    """Pull the assistant's text out of a text-generation pipeline result."""
    generated = output[0]["generated_text"]

    if isinstance(generated, list):  # chat format: a list of {"role", "content"}
        reply = next(
            (m.get("content") for m in reversed(generated) if m.get("role") == "assistant"), ""
        )
        if isinstance(reply, list):  # content split into parts
            reply = "".join(p.get("text", "") for p in reply if isinstance(p, dict))
    else:  # plain string: keep whatever follows the last model turn marker
        reply = generated.split("<start_of_turn>model")[-1]

    reply = (reply or "").replace("<end_of_turn>", "").replace("<eos>", "").strip()
    if not reply:
        raise SoapGenerationError("The model returned an empty note. Please try again.")
    return reply


def _explain(exc: Exception) -> str:
    """Turn a low-level exception into advice a clinician can act on."""
    text = str(exc)
    lowered = text.lower()
    if isinstance(exc, MemoryError) or "out of memory" in lowered or "not enough memory" in lowered:
        return (
            "Not enough memory to run MedGemma. Close other applications, record a shorter clip, "
            f"or use a machine with more RAM or a GPU. (Details: {text})"
        )
    if any(word in lowered for word in ("gated", "401", "403", "restricted", "access to model")):
        return (
            f"Could not download {LLM_ID}. It is a gated model: accept its terms on Hugging Face "
            f"and run `huggingface-cli login`. (Details: {text})"
        )
    if "bitsandbytes" in lowered:
        return (
            "4-bit loading failed (bitsandbytes). It needs a supported NVIDIA GPU; "
            f"see the README troubleshooting section. (Details: {text})"
        )
    return f"Note generation failed: {type(exc).__name__}: {text}"
