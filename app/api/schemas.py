"""Request and response bodies of the HTTP API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


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
