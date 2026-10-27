"""Every output check, run in a fixed order, each as its own MLflow span."""

import mlflow
from mlflow.entities import SpanType

from app.guardrails.faithfulness import check_banned_wording, check_faithfulness
from app.guardrails.injection import check_injection
from app.guardrails.pii import check_personal_data
from app.guardrails.protected import check_protected_attributes
from app.guardrails.reading_level import check_reading_level
from app.guardrails.required_content import check_required_content
from app.repositories.reason_codes import get_reason_catalog
from app.schemas import CheckResult, DenialCase, NoticeDraft


def run_checks(
    case: DenialCase, draft: NoticeDraft, notice: str, applicant_note: str | None = None
) -> list[CheckResult]:
    catalog = get_reason_catalog()
    texts = [draft.paragraph, *(r.text for r in draft.reasons)]
    results = []
    for name, check in [
        ("faithfulness", lambda: check_faithfulness(case, draft)),
        ("banned_wording", lambda: check_banned_wording(draft, catalog)),
        ("protected_attributes", lambda: check_protected_attributes(draft, catalog)),
        ("personal_data", lambda: check_personal_data(texts)),
        ("prompt_injection", lambda: check_injection(draft, applicant_note)),
        ("required_content", lambda: check_required_content(draft, notice)),
        ("reading_level", lambda: check_reading_level(draft)),
    ]:
        with mlflow.start_span(name=f"check.{name}", span_type=SpanType.EVALUATOR) as span:
            result = check()
            span.set_outputs(result.model_dump())
        results.append(result)
    return results
