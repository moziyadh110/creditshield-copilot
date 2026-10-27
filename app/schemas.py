"""Domain schemas shared by every layer.

Everything keys on the feature name (or the stable CSxx code reason_codes.yaml assigns to
it), never on CreditShield's RC01-RC04, which are only ranks.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.repositories.reason_codes import get_reason_catalog


class CaseReason(BaseModel):
    rank: int = Field(ge=1, le=4)
    feature: str
    impact_score: float


class DenialCase(BaseModel):
    """One CreditShield decision. applicant_id and impact_score never reach the LLM."""

    applicant_id: int
    decision: Literal["DENY", "MANUAL_REVIEW"]
    product: str
    reasons: list[CaseReason] = Field(min_length=1, max_length=4)
    applicant_note: str | None = None
    synthetic: bool = False

    @field_validator("reasons")
    @classmethod
    def _known_unique_ranked(cls, reasons: list[CaseReason]) -> list[CaseReason]:
        features = [r.feature for r in reasons]
        if len(features) != len(set(features)):
            raise ValueError("duplicate reason features")
        unknown = [f for f in features if f not in get_reason_catalog().features]
        if unknown:
            raise ValueError(f"reason features not in reason_codes.yaml: {unknown}")
        if [r.rank for r in reasons] != list(range(1, len(reasons) + 1)):
            raise ValueError("reason ranks must be 1..n in order")
        return reasons


class AgentInput(BaseModel):
    """The only data the LLM sees (input allow-list)."""

    product: str
    notice_date: date
    reasons: list[dict[str, int | str]]  # [{"rank": 1, "feature": "late_payment_rate"}, ...]
    applicant_note: str | None = None


class DraftReason(BaseModel):
    feature: str
    text: str = Field(min_length=3, max_length=300)


class NoticeDraft(BaseModel):
    """What the LLM returns: one plain line per reason, in rank order, plus the paragraph."""

    reasons: list[DraftReason] = Field(min_length=1, max_length=4)
    paragraph: str = Field(min_length=20, max_length=1500)

    @model_validator(mode="after")
    def _unique_features(self) -> "NoticeDraft":
        features = [r.feature for r in self.reasons]
        if len(features) != len(set(features)):
            raise ValueError("each reason feature may appear only once")
        return self


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


Outcome = Literal["sent", "revised", "escalated"]
