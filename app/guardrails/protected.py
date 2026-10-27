"""Protected attributes: no notice may mention an ECOA-protected basis (Reg B 1002.2(z);
15 U.S.C. 1691(a)) or the age-group cutoff logic. Checks the free text with a term and
pattern list, and the structured fields against the reason catalog.
"""

import re

from app.repositories.reason_codes import ReasonCatalog
from app.schemas import CheckResult, NoticeDraft

# Word-boundary patterns, case-insensitive. Kept narrow enough that ordinary credit wording
# ("a single late payment", "your oldest account") does not trip them.
PROTECTED_PATTERNS = {
    "age": r"\bages?\b|\baged\b|\byears? old\b|\byoung(er)?\b|\belderly\b|\bsenior citizens?\b"
    r"|\bretire(e|d|ment)?\b|\bdate of birth\b|\bbirth ?date\b",
    "sex": r"\bsex\b|\bgender\b|\bmale\b|\bfemale\b|\bwom[ae]n\b|\bm[ae]n\b|\bpregnan\w*",
    "marital status": r"\bmarital\b|\bmarried\b|\bunmarried\b|\bdivorced?\b|\bwidow(ed|er)?\b"
    r"|\bspouse\b|\bhusband\b|\bwife\b|\bsingle (parent|mother|father)\b",
    "race or color": r"\brac(e|ial)\b|\bethnic(ity)?\b|\bskin\b",
    "religion": r"\breligio(n|us)\b|\bchurch\b|\bmosque\b|\bsynagogue\b|\btemple\b",
    "national origin": r"\bnational origin\b|\bnationality\b|\bcitizenship\b|\bimmigra\w*"
    r"|\bforeign(er)?\b",
    "children or family": r"\bchild(ren)?\b|\bkids?\b|\bdependents?\b|\bfamily\b",
    "public assistance": r"\bpublic assistance\b|\bwelfare\b|\bfood stamps\b|\bsnap benefits\b",
    "cutoff logic": r"\bcut-?offs?\b|\bthresholds?\b|\bminimum score\b|\bqualifying score\b"
    r"|\bage group\b|\bscor(e|ing) (limit|bar)\b",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in PROTECTED_PATTERNS.items()}


def find_protected(text: str) -> list[str]:
    return [f"{kind}: '{m.group(0)}'" for kind, rx in _COMPILED.items() for m in rx.finditer(text)]


def check_protected_attributes(draft: NoticeDraft, catalog: ReasonCatalog) -> CheckResult:
    hits = find_protected(draft.paragraph)
    for r in draft.reasons:
        hits += [f"{r.feature}: {h}" for h in find_protected(r.text)]
        rc = catalog.features.get(r.feature)
        if rc is None:
            hits.append(f"structured field: unknown feature {r.feature!r}")
        elif rc.escalate:
            hits.append(f"structured field: escalate-only feature {r.feature!r} was worded")
    return CheckResult(
        name="protected_attributes", passed=not hits, detail="; ".join(hits), on_fail="escalate"
    )
