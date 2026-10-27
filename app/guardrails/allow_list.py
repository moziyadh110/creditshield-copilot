"""Input allow-list: the agent sees reason features with their ranks, the product and the
notice date. Everything else on the case (applicant_id, impact scores, decision details) is
dropped before the prompt is built.

applicant_note is withheld until Presidio masking exists (Day 3): failing closed means an
unmasked note can never reach the provider by accident.
"""

from datetime import date

from app.schemas import AgentInput, DenialCase


def build_agent_input(case: DenialCase, notice_date: date) -> AgentInput:
    return AgentInput(
        product=case.product,
        notice_date=notice_date,
        reasons=[{"rank": r.rank, "feature": r.feature} for r in case.reasons],
        applicant_note=None,
    )
