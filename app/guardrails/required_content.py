"""Required content (Reg B 1002.9(a)(2)): the rendered notice must contain the action taken,
the creditor's name and contact, the ECOA notice with the agency, and the specific reasons.
The template supplies all but the reasons; this check proves the render kept them and that
the LLM did not try to write legal text of its own.
"""

from app.agent.template import CREDITOR
from app.schemas import CheckResult, NoticeDraft

REQUIRED_IN_NOTICE = {
    "statement of action taken": "unable to approve your",
    "creditor name": CREDITOR,
    "creditor contact": "notices@example.com",
    "ECOA notice (1002.9(b)(1))": "The Federal Equal Credit Opportunity Act prohibits creditors",
    "administering agency": "Bureau of Consumer Financial Protection",
    "principal reasons heading": "Principal reason(s) for our decision:",
}
# Legal text only the template may write.
TEMPLATE_ONLY = ["Equal Credit Opportunity Act", "Bureau of Consumer Financial Protection"]


def check_required_content(draft: NoticeDraft, notice: str) -> CheckResult:
    problems = [
        f"missing {name}" for name, text in REQUIRED_IN_NOTICE.items() if text not in notice
    ]
    if draft.paragraph.strip() not in notice:
        problems.append("missing the reasons paragraph")
    problems += [
        f"paragraph writes template text '{t}'" for t in TEMPLATE_ONLY if t in draft.paragraph
    ]
    return CheckResult(
        name="required_content", passed=not problems, detail="; ".join(problems), on_fail="escalate"
    )
