import textstat

from app.schemas import CheckResult, NoticeDraft

MAX_GRADE = 8.0


def reading_grade(text: str) -> float:
    return float(textstat.flesch_kincaid_grade(text))


def check_reading_level(draft: NoticeDraft) -> CheckResult:
    grade = reading_grade(draft.paragraph)
    return CheckResult(
        name="reading_level",
        passed=grade <= MAX_GRADE,
        detail=f"Flesch-Kincaid grade {grade:.1f} (max {MAX_GRADE:g})",
    )
