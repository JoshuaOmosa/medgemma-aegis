import unittest

from soap_utils import SECTIONS, missing_sections, parse_soap

PLAIN = """Subjective: Cough for three days.
Objective: Temp 38.1 C.
Assessment: Likely viral URTI.
Plan: Rest and fluids."""

MARKDOWN = """Here is the SOAP note:

**Subjective:**
Cough for three days.
Worse at night.

**Objective:**
Temp 38.1 C.

## Assessment
Likely viral URTI.

### Plan
- Rest and fluids
- Review in 1 week
"""


class ParseSoapTests(unittest.TestCase):
    def test_plain_inline_headings(self):
        result = parse_soap(PLAIN)
        self.assertEqual(list(result), list(SECTIONS))
        self.assertEqual(result["Subjective"], "Cough for three days.")
        self.assertEqual(result["Plan"], "Rest and fluids.")

    def test_markdown_headings_and_preamble_ignored(self):
        result = parse_soap(MARKDOWN)
        self.assertEqual(list(result), list(SECTIONS))
        self.assertEqual(result["Subjective"], "Cough for three days.\nWorse at night.")
        self.assertIn("Review in 1 week", result["Plan"])
        self.assertNotIn("Here is", result["Subjective"])

    def test_prefixed_headings(self):
        result = parse_soap("S - Subjective:\nHeadache\nO - Objective:\nBP 120/80\n"
                            "A - Assessment:\nTension headache\nP - Plan:\nParacetamol")
        self.assertEqual(result["Objective"], "BP 120/80")
        self.assertEqual(result["Plan"], "Paracetamol")

    def test_sentence_starting_with_plan_is_not_a_heading(self):
        note = "Subjective: Fever\nObjective: 38C\nAssessment: Malaria\nPlan:\nPlan to review in two weeks."
        result = parse_soap(note)
        self.assertEqual(result["Plan"], "Plan to review in two weeks.")

    def test_bold_closing_before_colon(self):
        self.assertEqual(parse_soap("**Subjective**: Sore throat")["Subjective"], "Sore throat")

    def test_empty_and_none(self):
        self.assertEqual(parse_soap(""), {})
        self.assertEqual(parse_soap(None), {})


class MissingSectionsTests(unittest.TestCase):
    def test_complete_note(self):
        self.assertEqual(missing_sections(PLAIN), [])

    def test_missing_and_empty_sections_reported_in_order(self):
        note = "Subjective: Pain\nObjective:\nPlan: Rest"
        self.assertEqual(missing_sections(note), ["Objective", "Assessment"])

    def test_free_text_reports_everything_missing(self):
        self.assertEqual(missing_sections("The patient is fine."), list(SECTIONS))


if __name__ == "__main__":
    unittest.main()
