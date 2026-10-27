"""Faithfulness: the draft's reasons must be exactly the case's SHAP reason features, in rank
order, and no reason text may use a wording banned for its feature. A code check, so it
can't be argued around the way an LLM judge can.
"""

from app.repositories.reason_codes import ReasonCatalog
from app.schemas import CheckResult, DenialCase, NoticeDraft


def check_faithfulness(case: DenialCase, draft: NoticeDraft) -> CheckResult:
    expected = [r.feature for r in case.reasons]
    got = [r.feature for r in draft.reasons]
    if got == expected:
        return CheckResult(name="faithfulness", passed=True)
    problems = []
    if invented := [f for f in got if f not in expected]:
        problems.append(f"invented reasons {invented}")
    if missing := [f for f in expected if f not in got]:
        problems.append(f"missing reasons {missing}")
    if not problems:
        problems.append(f"wrong order: expected {expected}, got {got}")
    return CheckResult(name="faithfulness", passed=False, detail="; ".join(problems))


def check_banned_wording(draft: NoticeDraft, catalog: ReasonCatalog) -> CheckResult:
    hits = []
    for r in draft.reasons:
        rc = catalog.features.get(r.feature)
        for phrase in rc.banned if rc else []:
            if phrase.lower() in r.text.lower():
                hits.append(f"{r.feature}: '{phrase}'")
    return CheckResult(name="banned_wording", passed=not hits, detail="; ".join(hits))
