"""Build a FHIR R4 document Bundle from a reviewed SOAP note.

The bundle contains:
  * a Composition (the note itself, one section per SOAP heading),
  * a Device (AegisScribe, as the author of the draft),
  * optionally a Patient carrying only the identifier you supply.

The Composition status is always ``preliminary``: the text was drafted by an AI
and, until a clinician signs it off inside their own EHR workflow, it is not a
final record.

NOTE: this output has not been run through an official FHIR validator, and a
real EHR integration will need its own profile/terminology mapping. Treat it as
a solid starting point, not a certified interface.
"""
import html
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from soap_utils import parse_soap

LOINC_PROGRESS_NOTE = {
    "system": "http://loinc.org",
    "code": "11506-3",
    "display": "Progress note",
}
XHTML_NS = "http://www.w3.org/1999/xhtml"


def _urn() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


def _narrative(text: str) -> Dict[str, str]:
    """Wrap plain text as a FHIR narrative (escaped XHTML)."""
    body = (text or "").strip() or "Not documented"
    lines = "<br/>".join(html.escape(line, quote=False) for line in body.splitlines())
    return {"status": "generated", "div": f'<div xmlns="{XHTML_NS}"><p>{lines}</p></div>'}


def _section(title: str, text: str) -> Dict[str, Any]:
    return {"title": title, "text": _narrative(text)}


def build_fhir_bundle(
    soap_note: str,
    transcript: str = "",
    patient_id: Optional[str] = None,
    include_transcript: bool = False,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Return a FHIR ``Bundle`` (type ``document``) as a Python dict."""
    timestamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")

    composition_url, device_url = _urn(), _urn()

    sections = [_section(name, body) for name, body in parse_soap(soap_note).items()]
    if not sections:  # the model ignored the SOAP headings: keep the note whole
        sections = [_section("Clinical note", soap_note)]
    if include_transcript and transcript.strip():
        sections.append(_section("Source transcript", transcript))

    composition: Dict[str, Any] = {
        "resourceType": "Composition",
        "status": "preliminary",
        "type": {"coding": [LOINC_PROGRESS_NOTE]},
        "date": timestamp,
        "author": [{"reference": device_url}],
        "title": "Consultation note (AI-drafted, pending clinician review)",
        "section": sections,
    }

    device = {
        "resourceType": "Device",
        "status": "active",
        "deviceName": [{"name": "AegisScribe", "type": "user-friendly-name"}],
    }

    entries = [
        {"fullUrl": composition_url, "resource": composition},
        {"fullUrl": device_url, "resource": device},
    ]

    patient_id = (patient_id or "").strip()
    if patient_id:
        patient_url = _urn()
        composition["subject"] = {"reference": patient_url}
        entries.append({
            "fullUrl": patient_url,
            "resource": {"resourceType": "Patient", "identifier": [{"value": patient_id}]},
        })

    return {
        "resourceType": "Bundle",
        "identifier": {"system": "urn:ietf:rfc:3986", "value": _urn()},
        "type": "document",
        "timestamp": timestamp,
        "entry": entries,
    }


def to_json(bundle: Dict[str, Any]) -> str:
    """Serialise a bundle for download."""
    return json.dumps(bundle, indent=2, ensure_ascii=False)
