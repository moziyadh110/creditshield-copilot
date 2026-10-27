"""Input allow-list: the agent sees reason features with their ranks, the product, the notice
date and, if present, the applicant note after Presidio masking. Everything else on the case
(applicant_id, impact scores, and any extra field such as age, sex or marital status) is
dropped before the prompt is built.
"""

from datetime import date

from app.guardrails.pii import mask_pii
from app.schemas import AgentInput, DenialCase


def build_agent_input(case: DenialCase, notice_date: date) -> AgentInput:
    return AgentInput(
        product=case.product,
        notice_date=notice_date,
        reasons=[{"rank": r.rank, "feature": r.feature} for r in case.reasons],
        applicant_note=mask_pii(case.applicant_note) if case.applicant_note else None,
    )
