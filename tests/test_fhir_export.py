import json
import unittest
from datetime import datetime, timezone

from fhir_export import build_fhir_bundle, to_json

NOTE = "Subjective: Cough\nObjective: 38C\nAssessment: URTI\nPlan: Fluids <rest>"
NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)


def resources(bundle):
    return {e["resource"]["resourceType"]: e["resource"] for e in bundle["entry"]}


class FhirBundleTests(unittest.TestCase):
    def test_document_bundle_shape(self):
        bundle = build_fhir_bundle(NOTE, now=NOW)
        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(bundle["type"], "document")
        self.assertEqual(bundle["timestamp"], "2026-09-29T12:00:00+00:00")
        self.assertIn("value", bundle["identifier"])
        # A document bundle must start with its Composition.
        self.assertEqual(bundle["entry"][0]["resource"]["resourceType"], "Composition")

    def test_composition_is_preliminary_with_four_sections(self):
        comp = resources(build_fhir_bundle(NOTE, now=NOW))["Composition"]
        self.assertEqual(comp["status"], "preliminary")
        self.assertEqual([s["title"] for s in comp["section"]],
                         ["Subjective", "Objective", "Assessment", "Plan"])
        self.assertEqual(comp["type"]["coding"][0]["system"], "http://loinc.org")

    def test_narrative_is_html_escaped(self):
        comp = resources(build_fhir_bundle(NOTE, now=NOW))["Composition"]
        plan_div = comp["section"][3]["text"]["div"]
        self.assertIn("&lt;rest&gt;", plan_div)
        self.assertNotIn("<rest>", plan_div)

    def test_references_resolve_to_bundle_entries(self):
        bundle = build_fhir_bundle(NOTE, patient_id="P-001", now=NOW)
        urls = {e["fullUrl"] for e in bundle["entry"]}
        comp = resources(bundle)["Composition"]
        self.assertIn(comp["author"][0]["reference"], urls)
        self.assertIn(comp["subject"]["reference"], urls)

    def test_patient_only_when_id_given(self):
        without = resources(build_fhir_bundle(NOTE, now=NOW))
        self.assertNotIn("Patient", without)
        self.assertNotIn("subject", without["Composition"])
        blank = resources(build_fhir_bundle(NOTE, patient_id="   ", now=NOW))
        self.assertNotIn("Patient", blank)
        with_id = resources(build_fhir_bundle(NOTE, patient_id="P-001", now=NOW))
        self.assertEqual(with_id["Patient"]["identifier"][0]["value"], "P-001")

    def test_unstructured_note_kept_whole(self):
        comp = resources(build_fhir_bundle("Patient seems fine.", now=NOW))["Composition"]
        self.assertEqual([s["title"] for s in comp["section"]], ["Clinical note"])
        self.assertIn("Patient seems fine.", comp["section"][0]["text"]["div"])

    def test_transcript_included_only_on_request(self):
        off = resources(build_fhir_bundle(NOTE, transcript="hello", now=NOW))["Composition"]
        on = resources(build_fhir_bundle(NOTE, transcript="hello", include_transcript=True, now=NOW))["Composition"]
        self.assertNotIn("Source transcript", [s["title"] for s in off["section"]])
        self.assertEqual(on["section"][-1]["title"], "Source transcript")

    def test_json_round_trip_and_unicode(self):
        bundle = build_fhir_bundle("Subjective: Maumivu ya kichwa é\nPlan: x", now=NOW)
        text = to_json(bundle)
        self.assertIn("é", text)  # not escaped to \u00e9
        self.assertEqual(json.loads(text)["resourceType"], "Bundle")


if __name__ == "__main__":
    unittest.main()
