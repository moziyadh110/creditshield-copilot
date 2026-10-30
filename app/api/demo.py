"""POST /demo/notices: the live demo behind the project site. Mounted only when DEMO_ENABLED."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.logs import request_id
from app.services.demo import RateLimiter, UnknownCase, demo_case
from app.services.notices import draft_notice

router = APIRouter(prefix="/demo")
site = APIRouter()


class DemoRequest(BaseModel):
    case_id: str = Field(max_length=40)
    applicant_note: str | None = Field(default=None, max_length=1000)


class DemoResult(BaseModel):
    request_id: str
    case_id: str
    case: dict[str, Any]
    agent_input: dict[str, Any]  # what the LLM received: allow-listed, note masked by Presidio
    output: dict[str, Any]
    prompt_version: int | None
    cache_hit: bool
    trace_id: str


def _client(request: Request) -> str:
    # Behind Cloud Run's proxy the client is the first X-Forwarded-For entry. It can be forged,
    # which only dodges this limit: DAILY_BUDGET_USD still caps the spend.
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (request.client or ("?",))[0]


@site.get("/", include_in_schema=False)
async def to_site():
    """The backend's own address: send visitors to the demo site."""
    return RedirectResponse(get_settings().demo_site_url)


@router.post("/notices", response_model=DemoResult, responses={429: {}, 404: {}})
async def demo_notice(body: DemoRequest, request: Request):
    limiter: RateLimiter = request.app.state.demo_limiter
    if not limiter.allow(_client(request)):
        s = get_settings()
        raise HTTPException(
            429, f"Demo limit: {s.demo_per_minute} runs a minute, {s.demo_per_day} a day."
        )
    try:
        case = demo_case(body.case_id, body.applicant_note)
    except UnknownCase as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    deps = request.app.state.deps
    rid = request_id.get()
    r = await draft_notice(
        int(case.get("applicant_id") or 0),
        case=case,
        provider=deps.provider,
        search=deps.search,
        sessions=deps.sessions,
        trace_tags={"request_id": rid, "demo_case": body.case_id},
    )
    return DemoResult(
        request_id=rid, case_id=body.case_id, case=case, agent_input=r.agent_input,
        output=r.summary(), prompt_version=r.prompt_version, cache_hit=r.cache_hit,
        trace_id=r.trace_id,
    )  # fmt: skip
