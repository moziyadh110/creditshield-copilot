"""Resolve review-queue items. A reviewer either approves a draft, which must pass the same
checks as an LLM draft before it becomes the notice to send, or rejects the item, meaning it
is handled outside the Copilot (a MANUAL_REVIEW case goes to underwriting, a cnt_children
reason to a compliance-approved letter). A person can fix wording, but can't approve a
notice the guardrails would block."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import mlflow
from mlflow.entities import SpanType
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.template import render_notice
from app.guardrails.allow_list import build_agent_input
from app.guardrails.checks import run_checks
from app.llmops.tracing import setup_tracing
from app.repositories import review_queue
from app.repositories.cases import CaseNotFound, get_case
from app.repositories.reason_codes import get_reason_catalog
from app.schemas import CheckResult, NoticeDraft


class ReviewNotFound(LookupError):
    pass


class ReviewConflict(RuntimeError):
    """Already resolved, or the item can't be approved at all (not a denial)."""


@dataclass
class ChecksFailed(Exception):
    checks: list[CheckResult]


@dataclass
class Resolved:
    review_id: int
    status: str
    notice: str | None
    checks: list[CheckResult]


def case_features(applicant_id: int) -> list[str] | None:
    """The case's reason features in rank order, for prefilling a draft; None if the case
    can't be loaded (synthetic IDs, invalid data)."""
    try:
        return [r.feature for r in get_case(applicant_id).reasons]
    except (CaseNotFound, ValueError):
        return None


def suggested_draft(features: list[str]) -> dict[str, Any]:
    """Catalog wording for each reason: a starting point for items with no LLM draft."""
    catalog = get_reason_catalog().features
    reasons = [{"feature": f, "text": f"{catalog[f].reason}."} for f in features]
    lines = " ".join(r["text"] for r in reasons)
    paragraph = f"These are the main reasons for our decision. {lines}"
    return {"reasons": reasons, "paragraph": paragraph}


async def pending(sessions: async_sessionmaker) -> list[dict[str, Any]]:
    """Pending items, oldest first, each with a draft to start from: the LLM's draft if it
    made one, else catalog wording for a denial whose case loads, else None."""
    async with sessions() as s:
        items = await review_queue.list_pending(s)
    out = []
    for item in items:
        row = {c.key: getattr(item, c.key) for c in item.__table__.columns}
        features = case_features(item.applicant_id) if item.decision == "DENY" else None
        row["suggested_draft"] = item.draft or (suggested_draft(features) if features else None)
        out.append(row)
    return out


async def approve(
    sessions: async_sessionmaker,
    review_id: int,
    draft: NoticeDraft,
    reviewer_note: str | None = None,
    notice_date: date | None = None,
) -> Resolved:
    setup_tracing()
    async with sessions() as s, s.begin():
        item = await review_queue.get_for_update(s, review_id)
        if item is None:
            raise ReviewNotFound(f"review item {review_id} not found")
        if item.status != "pending":
            raise ReviewConflict(f"review item {review_id} is already {item.status}")
        if item.decision != "DENY":
            raise ReviewConflict(f"{item.decision} is not a denial: reject it, handle it outside")
        case = get_case(item.applicant_id)
        notice_date = notice_date or date.today()
        agent_input = await asyncio.to_thread(build_agent_input, case, notice_date)
        notice = render_notice(
            applicant_id=case.applicant_id,
            product=case.product,
            notice_date=notice_date,
            reasons_paragraph=draft.paragraph,
        )
        # One trace per approval. It has no "outcome" tag, so monitoring doesn't count it as
        # a notice request.
        with mlflow.start_span(name="review_approve", span_type=SpanType.CHAIN) as root:
            root.set_inputs({"review_id": review_id, "draft": draft.model_dump()})
            checks = await asyncio.to_thread(
                run_checks, case, draft, notice, agent_input.applicant_note
            )
            passed = all(c.passed for c in checks)
            root.set_outputs({"approved": passed})
            mlflow.update_current_trace(
                tags={"review_id": str(review_id), "case_id": str(case.applicant_id)}
            )
        if not passed:
            raise ChecksFailed(checks)
        item.status = "approved"
        item.final_draft = draft.model_dump()
        item.final_notice = notice
        item.reviewer_note = reviewer_note
        item.resolved_at = datetime.now(UTC)
    return Resolved(review_id, "approved", notice, checks)


async def reject(sessions: async_sessionmaker, review_id: int, reviewer_note: str) -> Resolved:
    async with sessions() as s, s.begin():
        item = await review_queue.get_for_update(s, review_id)
        if item is None:
            raise ReviewNotFound(f"review item {review_id} not found")
        if item.status != "pending":
            raise ReviewConflict(f"review item {review_id} is already {item.status}")
        item.status = "rejected"
        item.reviewer_note = reviewer_note
        item.resolved_at = datetime.now(UTC)
    return Resolved(review_id, "rejected", None, [])
