"""Routes call services only: no database, LLM or MLflow code here."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.auth import require_api_key
from app.api.schemas import (
    ApproveRequest,
    NoticeRequest,
    NoticeSent,
    RejectRequest,
    ReviewQueueItem,
    ReviewResolved,
    SentToReview,
)
from app.logs import request_id
from app.repositories.cases import CaseNotFound
from app.schemas import NoticeDraft
from app.services import reviews
from app.services.notices import draft_notice

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/notices",
    dependencies=[Depends(require_api_key)],
    response_model=NoticeSent,
    responses={202: {"model": SentToReview}, 404: {"description": "Unknown applicant"}},
)
async def create_notice(body: NoticeRequest, request: Request):
    deps = request.app.state.deps
    rid = request_id.get()
    try:
        r = await draft_notice(
            body.applicant_id,
            provider=deps.provider,
            search=deps.search,
            sessions=deps.sessions,
            trace_tags={"request_id": rid},
        )
    except CaseNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    if r.outcome == "escalated":
        queued = SentToReview(
            request_id=rid, applicant_id=r.applicant_id, reason=r.escalation_reason,
            review_id=r.review_id, run_id=r.run_id, trace_id=r.trace_id,
        )  # fmt: skip
        return JSONResponse(status_code=202, content=queued.model_dump())
    return NoticeSent(
        request_id=rid,
        applicant_id=r.applicant_id,
        outcome=r.outcome,
        notice=r.notice,
        prompt_version=r.prompt_version,
        model=r.model,
        cache_hit=r.cache_hit,
        run_id=r.run_id,
        trace_id=r.trace_id,
    )


@router.get(
    "/review-queue",
    dependencies=[Depends(require_api_key)],
    response_model=list[ReviewQueueItem],
)
async def review_queue(request: Request):
    return await reviews.pending(request.app.state.deps.sessions)


def _resolved(r: reviews.Resolved) -> ReviewResolved:
    return ReviewResolved(
        review_id=r.review_id, status=r.status, notice=r.notice, check_results=r.checks
    )


@router.post(
    "/review-queue/{review_id}/approve",
    dependencies=[Depends(require_api_key)],
    response_model=ReviewResolved,
    responses={
        404: {"description": "Unknown review item or applicant"},
        409: {"description": "Already resolved, or not a denial"},
        422: {"description": "The draft fails a check; the item stays pending"},
    },
)
async def approve_review(review_id: int, body: ApproveRequest, request: Request):
    draft = NoticeDraft(reasons=body.reasons, paragraph=body.paragraph)
    try:
        r = await reviews.approve(
            request.app.state.deps.sessions, review_id, draft, body.reviewer_note
        )
    except (reviews.ReviewNotFound, CaseNotFound) as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except reviews.ReviewConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    except reviews.ChecksFailed as e:
        failed = [c.model_dump() for c in e.checks if not c.passed]
        raise HTTPException(status_code=422, detail={"failed_checks": failed}) from None
    return _resolved(r)


@router.post(
    "/review-queue/{review_id}/reject",
    dependencies=[Depends(require_api_key)],
    response_model=ReviewResolved,
    responses={404: {"description": "Unknown review item"}, 409: {"description": "Resolved"}},
)
async def reject_review(review_id: int, body: RejectRequest, request: Request):
    try:
        r = await reviews.reject(request.app.state.deps.sessions, review_id, body.reviewer_note)
    except reviews.ReviewNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except reviews.ReviewConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from None
    return _resolved(r)
