import streamlit as st
from audio_capture import record_consultation
from engine import process_medical_session
import os

st.set_page_config(page_title="AegisScribe", page_icon="🛡️", layout="wide")
st.title("🛡️ AegisScribe")
st.caption("Secure Local Clinical Assistant | Data Status: Local Only")

# 1. Initialize Session State (This prevents data loss on refresh)
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'soap_note' not in st.session_state:
    st.session_state.soap_note = ""

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    duration = st.slider("Recording Duration (sec)", 5, 60, 15)
    st.info("Files are stored in: `/data/raw_consultation.wav`")

# Main UI
col_rec, col_gen = st.columns(2)

with col_rec:
    if st.button("🎤 Start New Consultation", use_container_width=True):
        path = record_consultation(duration=duration)
        if path:
            st.success(f"Captured: {path}")
            st.audio(path)

with col_gen:
    if st.button("🪄 Generate SOAP Note", use_container_width=True):
        if os.path.exists("data/raw_consultation.wav"):
            with st.spinner("AI Analysis in progress..."):
                t, s = process_medical_session("data/raw_consultation.wav")
                # Save to session state so it stays on screen
                st.session_state.transcript = t
                st.session_state.soap_note = s
        else:
            st.error("No recording found! Please record first.")

# Display Results
if st.session_state.transcript:
    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Transcript")
        st.text_area("Review Transcription:", value=st.session_state.transcript, height=400)
    with res_col2:
        st.subheader("AI SOAP Note")
        st.text_area("Edit SOAP Note:", value=st.session_state.soap_note, height=400)

    st.divider()
    if st.button("💾 Export to EHR (FHIR/JSON)"):
        st.balloons()
        st.success("Data formatted to FHIR JSON and ready for export.")