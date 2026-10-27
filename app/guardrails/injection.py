"""Prompt injection. The applicant note is wrapped as data and never placed in the system
prompt (app/agent/graph.py); this output check catches the rest: a notice that approves,
promises, reveals its instructions, or copies the note into the letter.
"""

import re

from app.schemas import CheckResult, NoticeDraft

FOLLOWED_INSTRUCTION = re.compile(
    # Approval claims only: "history of approved credit" is a legitimate reason wording.
    r"\b(you are|you're|you have been|you've been|it has been|is|was) approved\b"
    r"|\bwe (will|can|would|have) approved?\b|\bapproved for\b"
    r"|\bcongratulations\b|\bguarantee\w*\b|\bpromise\w*\b|\breconsider\w*\b"
    r"|\bsystem prompt\b|\bcase_data\b|\bapplicant_note\b|\binstructions?\b"
    r"|\bignore (all|any|previous|prior)\b",
    re.IGNORECASE,
)
COPY_WORDS = 6  # copying this many consecutive words from the note counts as echoing it


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def check_injection(draft: NoticeDraft, applicant_note: str | None) -> CheckResult:
    text = " ".join([draft.paragraph, *(r.text for r in draft.reasons)])
    hits = [f"'{m.group(0)}'" for m in FOLLOWED_INSTRUCTION.finditer(text)]
    if applicant_note and _ngrams(applicant_note, COPY_WORDS) & _ngrams(text, COPY_WORDS):
        hits.append("copies the applicant note")
    return CheckResult(
        name="prompt_injection", passed=not hits, detail="; ".join(hits), on_fail="escalate"
    )
