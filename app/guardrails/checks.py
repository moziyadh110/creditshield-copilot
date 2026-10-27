"""Every output check, run in a fixed order. Day 3 adds the remaining checks here."""

import mlflow
from mlflow.entities import SpanType

from app.guardrails.faithfulness import check_banned_wording, check_faithfulness
from app.repositories.reason_codes import get_reason_catalog
from app.schemas import CheckResult, DenialCase, NoticeDraft


def run_checks(case: DenialCase, draft: NoticeDraft) -> list[CheckResult]:
    results = []
    for name, check in [
        ("faithfulness", lambda: check_faithfulness(case, draft)),
        ("banned_wording", lambda: check_banned_wording(draft, get_reason_catalog())),
    ]:
        with mlflow.start_span(name=f"check.{name}", span_type=SpanType.EVALUATOR) as span:
            result = check()
            span.set_outputs(result.model_dump())
        results.append(result)
    return results
