import inspect
import os
from datetime import datetime

import streamlit as st

from audio_capture import RecordingError, delete_recording, record_consultation
from engine import SoapGenerationError, generate_soap_note, transcribe_audio
from fhir_export import build_fhir_bundle, to_json
from soap_utils import missing_sections

st.set_page_config(page_title="AegisScribe", page_icon="🛡️", layout="wide")
st.title("🛡️ AegisScribe")
st.caption("Secure Local Clinical Assistant | Data Status: Local Only")


def _stretch():
    """Full-width button arguments, whichever way this Streamlit version spells them."""
    params = inspect.signature(st.button).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


STRETCH = _stretch()

# 1. Session state: keeps results on screen across Streamlit reruns.
#    "transcript" and "soap_note" are also the keys of the editable text boxes,
#    so a clinician's corrections are kept.
for key, default in {
    "transcript": "",
    "soap_note": "",
    "audio_path": None,
    "reviewed": False,
}.items():
    st.session_state.setdefault(key, default)

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    duration = st.slider("Recording Duration (sec)", 5, 60, 15)
    patient_id = st.text_input("Patient ID (optional)", help="Only used in the FHIR export.")
    delete_after = st.checkbox(
        "Delete audio after transcribing",
        value=True,
        help="Removes the recording from disk once the transcript exists.",
    )
    st.info("Recordings are stored in the local `data/` folder.")

# Main UI
col_rec, col_gen = st.columns(2)

with col_rec:
    if st.button("🎤 Start New Consultation", **STRETCH):
        try:
            with st.spinner(f"Recording for {duration} seconds... speak now"):
                path = record_consultation(duration=duration)
            old = st.session_state.audio_path
            if old and old != path:
                delete_recording(old)
            st.session_state.audio_path = path
            # A new recording invalidates any previous note.
            st.session_state.transcript = ""
            st.session_state.soap_note = ""
            st.session_state.reviewed = False
            st.success(f"Captured: {path}")
        except RecordingError as exc:
            st.error(str(exc))

with col_gen:
    if st.button("🪄 Generate SOAP Note", **STRETCH):
        audio_path = st.session_state.audio_path
        if not audio_path or not os.path.exists(audio_path):
            st.error("No recording found! Please record first.")
        else:
            transcribed = False
            try:
                with st.spinner("Transcribing audio..."):
                    transcript = transcribe_audio(audio_path)
                transcribed = True
                # Set before the LLM step so the transcript survives a note failure.
                st.session_state.transcript = transcript
                st.session_state.soap_note = ""
                st.session_state.reviewed = False
                with st.spinner("Drafting SOAP note with MedGemma (the first run is slow)..."):
                    st.session_state.soap_note = generate_soap_note(transcript)
            except SoapGenerationError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Transcription failed: {exc}")
            finally:
                if transcribed and delete_after:
                    delete_recording(audio_path)
                    st.session_state.audio_path = None

audio_path = st.session_state.audio_path
if audio_path and os.path.exists(audio_path):
    st.audio(audio_path)

# Display Results
if st.session_state.transcript or st.session_state.soap_note:
    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Transcript")
        st.text_area("Review Transcription:", key="transcript", height=400)
    with res_col2:
        st.subheader("AI SOAP Note")
        st.text_area("Edit SOAP Note:", key="soap_note", height=400)

    note = st.session_state.soap_note
    if not note.strip():
        st.info("No note has been generated yet. Fix the problem shown above and click Generate again.")
    else:
        missing = missing_sections(note)
        if missing:
            st.warning(
                "The note is missing or has empty section(s): " + ", ".join(missing)
                + ". Check it against the transcript before exporting."
            )

    st.divider()
    st.subheader("Export")
    st.caption(
        "AI-drafted notes can be wrong. Export is a FHIR R4 document bundle marked "
        "'preliminary'; it has not been validated against a specific EHR."
    )
    reviewed = st.checkbox("I have reviewed and corrected this note", key="reviewed")
    include_transcript = st.checkbox("Include transcript in export", value=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = build_fhir_bundle(
        soap_note=note,
        transcript=st.session_state.transcript,
        patient_id=patient_id,
        include_transcript=include_transcript,
    )
    st.download_button(
        "💾 Download FHIR JSON",
        data=to_json(bundle),
        file_name=f"aegisscribe_note_{stamp}.json",
        mime="application/fhir+json",
        disabled=not (reviewed and note.strip()),
        **STRETCH,
    )
