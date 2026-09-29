"""Helpers for working with SOAP-formatted text.

Language models format headings in many ways (``**Subjective:**``, ``## Plan``,
``S - Subjective``...). These helpers split a note into its four sections and
report which ones are missing, so the UI can warn the clinician instead of
silently showing an incomplete note.
"""
import re
from typing import Dict, List, Optional

SECTIONS = ("Subjective", "Objective", "Assessment", "Plan")

# A heading is one of the four SOAP words, optionally decorated with markdown
# (#, **, __) or an "S -" style prefix, and followed by either a colon or the
# end of the line. Requiring ":" or end-of-line stops ordinary sentences such
# as "Plan to review in two weeks" from being mistaken for a heading.
_HEADING = re.compile(
    r"^[ \t]*(?:\#{1,6}[ \t]*)?(?:[*_]{1,3})?[ \t]*"
    r"(?:[SOAP][ \t]*[-\u2013\u2014:.)][ \t]*)?"
    r"(?P<name>Subjective|Objective|Assessment|Plan)"
    r"[*_]*(?:[ \t]*:[*_]*[ \t]*(?P<inline>[^\n]*)|[ \t]*$)",
    re.IGNORECASE | re.MULTILINE,
)


def parse_soap(text: Optional[str]) -> Dict[str, str]:
    """Split a SOAP note into ``{"Subjective": ..., "Objective": ..., ...}``.

    Only sections that were found are returned, in the order they appear.
    Text before the first heading (e.g. "Here is the note:") is ignored.
    """
    text = text or ""
    matches = list(_HEADING.finditer(text))
    sections: Dict[str, str] = {}
    for i, match in enumerate(matches):
        name = match.group("name").capitalize()
        body_start = match.start("inline") if match.group("inline") is not None else match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections[name] = f"{sections[name]}\n{body}".strip() if name in sections else body
    return sections


def missing_sections(text: Optional[str]) -> List[str]:
    """Return the SOAP sections that are absent or empty, in SOAP order."""
    found = parse_soap(text)
    return [name for name in SECTIONS if not found.get(name)]
