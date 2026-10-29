"""Routes call services only: no database, LLM or MLflow code here."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.auth import require_api_key
from app.api.schemas import NoticeRequest, NoticeSent, ReviewQueueItem, SentToReview
from app.logs import request_id
from app.repositories.cases import CaseNotFound
from app.services.notices import draft_notice, pending_reviews

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
    return await pending_reviews(request.app.state.deps.sessions)
