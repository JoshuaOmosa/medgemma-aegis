import os
import tempfile
import unittest
from unittest import mock

import engine
from engine import SoapGenerationError


class FakeWhisper:
    def __init__(self, text):
        self.text, self.kwargs = text, None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return {"text": self.text}


def fake_scribe(reply, raises=None):
    """A stand-in for the transformers text-generation pipeline (chat format)."""
    calls = []

    def scribe(messages, **kwargs):
        calls.append((messages, kwargs))
        if raises:
            raise raises
        return [{"generated_text": messages + [{"role": "assistant", "content": reply}]}]

    scribe.calls = calls
    return scribe


class TranscribeTests(unittest.TestCase):
    def test_returns_stripped_text_and_passes_settings(self):
        whisper = FakeWhisper("  Patient has a cough.  ")
        with tempfile.NamedTemporaryFile(suffix=".wav") as f, \
                mock.patch.object(engine, "get_whisper", return_value=whisper):
            self.assertEqual(engine.transcribe_audio(f.name), "Patient has a cough.")
        self.assertFalse(whisper.kwargs["fp16"])

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            engine.transcribe_audio("/nonexistent/file.wav")


class GenerateNoteTests(unittest.TestCase):
    def test_returns_clean_assistant_reply_and_uses_greedy_decoding(self):
        scribe = fake_scribe("Subjective: Cough<end_of_turn>\n")
        with mock.patch.object(engine, "get_scribe", return_value=scribe):
            note = engine.generate_soap_note("Patient has a cough.")
        self.assertEqual(note, "Subjective: Cough")
        messages, kwargs = scribe.calls[0]
        self.assertIn("Patient has a cough.", messages[0]["content"])
        self.assertIs(kwargs["do_sample"], False)
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["max_new_tokens"], engine.MAX_NEW_TOKENS)

    def test_prompt_forbids_invention(self):
        self.assertIn("Do not invent", engine.SOAP_PROMPT)
        self.assertIn("Not documented", engine.SOAP_PROMPT)

    def test_transcript_with_braces_is_safe(self):
        scribe = fake_scribe("ok")
        with mock.patch.object(engine, "get_scribe", return_value=scribe):
            engine.generate_soap_note("Dose is {5 mg} twice daily")
        self.assertIn("{5 mg}", scribe.calls[0][0][0]["content"])

    def test_empty_transcript_never_loads_the_model(self):
        with mock.patch.object(engine, "get_scribe") as get_scribe:
            for blank in ("", "   ", None):
                with self.assertRaises(SoapGenerationError) as ctx:
                    engine.generate_soap_note(blank)
                self.assertIn("No speech", str(ctx.exception))
            get_scribe.assert_not_called()

    def test_plain_string_output_is_handled(self):
        raw = "<bos><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\nSubjective: x<end_of_turn>"
        scribe = lambda *a, **k: [{"generated_text": raw}]  # noqa: E731
        with mock.patch.object(engine, "get_scribe", return_value=scribe):
            self.assertEqual(engine.generate_soap_note("hi"), "Subjective: x")

    def test_empty_model_reply_is_an_error(self):
        with mock.patch.object(engine, "get_scribe", return_value=fake_scribe("  ")):
            with self.assertRaises(SoapGenerationError):
                engine.generate_soap_note("hi")


class ErrorMessageTests(unittest.TestCase):
    def raise_with(self, exc):
        with mock.patch.object(engine, "get_scribe", return_value=fake_scribe("", raises=exc)):
            with self.assertRaises(SoapGenerationError) as ctx:
                engine.generate_soap_note("hi")
        return str(ctx.exception)

    def test_out_of_memory_is_explained(self):
        msg = self.raise_with(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
        self.assertIn("Not enough memory", msg)
        self.assertIn("CUDA out of memory", msg)  # real cause preserved

    def test_gated_model_is_explained(self):
        msg = self.raise_with(OSError("401 Client Error: gated repo"))
        self.assertIn("huggingface-cli login", msg)

    def test_bitsandbytes_is_explained(self):
        self.assertIn("bitsandbytes", self.raise_with(ImportError("No package named bitsandbytes")))

    def test_unknown_error_keeps_type_and_message(self):
        msg = self.raise_with(ValueError("weird failure"))
        self.assertIn("ValueError", msg)
        self.assertIn("weird failure", msg)

    def test_load_failure_is_not_swallowed_as_memory_error(self):
        with mock.patch.object(engine, "get_scribe", side_effect=OSError("disk exploded")):
            with self.assertRaises(SoapGenerationError) as ctx:
                engine.generate_soap_note("hi")
        self.assertIn("disk exploded", str(ctx.exception))


class ProcessSessionTests(unittest.TestCase):
    def test_backwards_compatible_tuple(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f, \
                mock.patch.object(engine, "get_whisper", return_value=FakeWhisper("hello")), \
                mock.patch.object(engine, "get_scribe", return_value=fake_scribe("Subjective: hi")):
            self.assertEqual(engine.process_medical_session(f.name), ("hello", "Subjective: hi"))

    def test_failure_keeps_transcript_and_explains(self):
        with tempfile.NamedTemporaryFile(suffix=".wav") as f, \
                mock.patch.object(engine, "get_whisper", return_value=FakeWhisper("hello")), \
                mock.patch.object(engine, "get_scribe", side_effect=RuntimeError("out of memory")):
            transcript, note = engine.process_medical_session(f.name)
        self.assertEqual(transcript, "hello")
        self.assertIn("Note generation failed", note)


class CachingTests(unittest.TestCase):
    def test_loaders_are_cached(self):
        self.assertTrue(hasattr(engine.get_whisper, "cache_info"))
        self.assertTrue(hasattr(engine.get_scribe, "cache_info"))


if __name__ == "__main__":
    unittest.main()
