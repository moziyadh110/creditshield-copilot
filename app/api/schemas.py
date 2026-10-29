"""Request and response bodies of the HTTP API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import CheckResult, NoticeDraft


class NoticeRequest(BaseModel):
    applicant_id: int


class NoticeSent(BaseModel):
    request_id: str
    applicant_id: int
    outcome: Literal["sent", "revised"]
    notice: str
    prompt_version: int
    model: str
    cache_hit: bool
    run_id: int
    trace_id: str


class SentToReview(BaseModel):
    request_id: str
    applicant_id: int
    status: Literal["sent to human review"] = "sent to human review"
    reason: str
    review_id: int
    run_id: int
    trace_id: str


class ReviewQueueItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    applicant_id: int
    decision: str
    reason: str
    status: str
    draft: dict[str, Any] | None
    check_results: list[dict[str, Any]]
    run_id: int | None
    trace_id: str | None
    created_at: datetime
    # A draft to start from: the LLM's draft, or catalog wording for a denial with no draft.
    suggested_draft: dict[str, Any] | None = None


class ApproveRequest(NoticeDraft):
    reviewer_note: str | None = Field(default=None, max_length=2000)


class RejectRequest(BaseModel):
    reviewer_note: str = Field(min_length=3, max_length=2000)


class ReviewResolved(BaseModel):
    review_id: int
    status: Literal["approved", "rejected"]
    notice: str | None
    check_results: list[CheckResult]
