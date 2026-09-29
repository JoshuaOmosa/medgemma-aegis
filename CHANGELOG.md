# Changelog

## Unreleased

### Fixed
- Models (Whisper, MedGemma) are now loaded once and cached instead of on every click.
- `temperature=0.1` was silently ignored because sampling was off; generation is now explicitly greedy (deterministic).
- Failures no longer hide behind "Memory limit reached". Real causes are reported (memory, gated model, bitsandbytes, or the raw error) and the transcript is always kept.
- Whisper no longer receives a made-up prompt on silence; empty transcripts skip the LLM entirely.
- Edits to the transcript and note now persist across Streamlit reruns.
- Recordings are 16-bit WAV (play in every browser), timestamped (no more overwriting), and can be deleted automatically after transcription.
- Missing microphone/audio libraries now show an error instead of failing silently.
- Sidebar no longer names a wrong file path.

### Added
- Real FHIR R4 export (`fhir_export.py`): document Bundle with a `preliminary` Composition, download button gated behind a "reviewed" checkbox.
- Warning when a SOAP section is missing or empty (`soap_utils.py`).
- CPU fallback (bfloat16) when no CUDA GPU is present; 4-bit only with a GPU.
- Prompt now restricts the model to the transcript ("Not documented" for empty sections).
- Configuration through `AEGIS_*` environment variables.
- `requirements.txt`, `requirements-dev.txt`, 38 unit tests, expanded `heartbeat.py`, full README.

### Changed
- `trust_remote_code` is now **off by default** (set `AEGIS_TRUST_REMOTE_CODE=1` if your model revision needs it).
- MedGemma is called with chat-format messages instead of a hand-built prompt string.
- `record_consultation()` raises `RecordingError` instead of returning `None`.
- `process_medical_session()` still works for compatibility; the app uses `transcribe_audio()` and `generate_soap_note()` separately.

### Moved
- `core/` -> `legacy/core/` (unused early drafts).
- `Models/`, `Http/`, `Providers/` -> `unrelated/laravel-queue-system/` (a separate PHP project).
