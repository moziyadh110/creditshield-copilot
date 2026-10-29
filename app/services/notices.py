"""Draft adverse action notices. run_notice runs the agent inside one MLflow trace and saves
nothing (the eval calls it directly); draft_notice is the service the API and CLI call: it
serves the response cache, enforces the daily budget, and writes the run log, the review
queue and the spend counter."""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import cache
from typing import Any

import mlflow
import mlflow.genai
from mlflow.entities import SpanType
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.graph import AgentDeps, NoticeState, build_graph
from app.agent.template import render_notice
from app.agent.tools import Toolbox
from app.config import get_settings
from app.db.models import Chunk, ReviewItem, RunLog
from app.db.session import make_async_engine, make_async_sessionmaker, make_sessionmaker
from app.guardrails.allow_list import build_agent_input
from app.guardrails.checks import run_checks
from app.llm.provider import LLMProvider, RunUsage, get_provider
from app.llmops.tracing import setup_tracing
from app.repositories import review_queue, run_log, spend
from app.repositories.cases import get_case
from app.repositories.reason_codes import get_reason_catalog
from app.retrieval.search import search_regulation
from app.schemas import AgentInput, CheckResult, DenialCase, NoticeDraft

log = logging.getLogger("copilot.notices")


@dataclass
class NoticeResult:
    applicant_id: int
    decision: str
    outcome: str
    notice: str | None
    escalation_reason: str | None
    checks: list[CheckResult]
    draft: NoticeDraft | None
    agent_input: dict[str, Any]
    retrieved_chunk_ids: list[str]
    prompt_version: int | None
    model: str
    usage: RunUsage
    latency_ms: int
    trace_id: str
    run_id: int | None = None
    review_id: int | None = None
    cache_hit: bool = False


@cache
def _sessions():
    return make_sessionmaker()


def pgvector_search(query: str) -> list[Chunk]:
    """Runs in a worker thread (Toolbox.search_regulation): the embedding is CPU-bound, so the
    query rides along on a sync connection there instead of blocking the event loop."""
    with _sessions()() as s:
        return search_regulation(s, query, k=3)


def _monitor_tags(
    decision: str,
    features: list[str],
    checks: list[CheckResult],
    escalation_reason: str | None,
    usage: RunUsage,
) -> dict[str, str]:
    """Per-trace facts the daily monitoring job reads (app/llmops/monitor.py)."""
    return {
        "decision": decision,
        "reason_features": ",".join(features),
        "checks": json.dumps({c.name: c.passed for c in checks}),
        "llm_calls": str(usage.calls),
        "cost_usd": f"{usage.cost_usd:.8f}",
        "provider_error": str("LLM provider" in (escalation_reason or "")).lower(),
    }


def _escalated(
    applicant_id: int, decision: str, reason: str, model: str, started: float, trace_id: str
) -> NoticeResult:
    """A run that escalates before any LLM call."""
    return NoticeResult(
        applicant_id=applicant_id,
        decision=decision,
        outcome="escalated",
        notice=None,
        escalation_reason=reason,
        checks=[],
        draft=None,
        agent_input={},
        retrieved_chunk_ids=[],
        prompt_version=None,
        model=model,
        usage=RunUsage(),
        latency_ms=int((time.perf_counter() - started) * 1000),
        trace_id=trace_id,
    )


async def run_notice(
    case: DenialCase | dict[str, Any],
    notice_date: date | None = None,
    *,
    provider: LLMProvider | None = None,
    search: Callable[[str], list[Any]] | None = None,
    prompt_uri: str | None = None,
    trace_tags: dict[str, str] | None = None,
) -> NoticeResult:
    """Run the agent on one case. A case given as a dict is validated first: missing or
    malformed data (a null SHAP value, an empty reason list, a null field, an over-long note)
    escalates to human review with no LLM call, never a crash or a guess."""
    settings = get_settings()
    setup_tracing()
    prompt_uri = prompt_uri or f"prompts:/{settings.prompt_name}@{settings.prompt_alias}"
    started = time.perf_counter()
    usage = RunUsage()
    tags = trace_tags or {}

    with mlflow.start_span(name="draft_notice", span_type=SpanType.AGENT) as root:
        raw = case if isinstance(case, dict) else None
        try:
            case = DenialCase.model_validate(case)
        except ValidationError as e:
            reason = f"invalid case data: {e.error_count()} error(s): " + "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
            )
            root.set_inputs({"case": raw})
            root.set_outputs({"outcome": "escalated", "notice": None})
            monitor = _monitor_tags(str(raw.get("decision")), [], [], reason, usage)
            mlflow.update_current_trace(
                tags={**tags, **monitor, "case_id": str(raw.get("applicant_id")),
                      "outcome": "escalated"}
            )  # fmt: skip
            return _escalated(
                int(raw.get("applicant_id") or 0),
                str(raw.get("decision") or "UNKNOWN"),
                reason,
                provider.model if provider else settings.llm_model,
                started,
                root.trace_id,
            )

        # Presidio masking and the registry lookup are blocking; threads keep the event loop
        # free (to_thread copies the context, so the prompt is still linked to this trace).
        agent_input = await asyncio.to_thread(build_agent_input, case, notice_date or date.today())
        root.set_inputs(agent_input.model_dump(mode="json"))
        prompt = await asyncio.to_thread(mlflow.genai.load_prompt, prompt_uri)
        provider = provider or get_provider()
        provider.configure(prompt.model_config or {})
        catalog = get_reason_catalog()
        toolbox = Toolbox(catalog, search or pgvector_search)
        deps = AgentDeps(provider, prompt.template, toolbox, catalog, usage, settings.llm_timeout_s)
        state: NoticeState = await build_graph(deps).ainvoke(
            {"case": case, "agent_input": agent_input}
        )
        root.set_outputs({"outcome": state["outcome"], "notice": state.get("notice")})
        monitor = _monitor_tags(
            case.decision,
            [r.feature for r in case.reasons],
            state.get("checks", []),
            state.get("escalation_reason"),
            usage,
        )
        mlflow.update_current_trace(
            tags={
                **tags,
                **monitor,
                "case_id": str(case.applicant_id),
                "prompt_version": str(prompt.version),
                "model": provider.model,
                "outcome": state["outcome"],
                "cache_hit": "false",
            }
        )

    return NoticeResult(
        applicant_id=case.applicant_id,
        decision=case.decision,
        outcome=state["outcome"],
        notice=state.get("notice"),
        escalation_reason=state.get("escalation_reason"),
        checks=state.get("checks", []),
        draft=state.get("draft"),
        agent_input=agent_input.model_dump(mode="json"),
        retrieved_chunk_ids=toolbox.retrieved_ids,
        prompt_version=int(prompt.version),
        model=provider.model,
        usage=usage,
        latency_ms=int((time.perf_counter() - started) * 1000),
        trace_id=root.trace_id,
    )


def _same_input(cached: dict[str, Any], current: AgentInput) -> bool:
    """The cache is keyed by case ID, prompt version and model; this also refuses a hit if
    the case itself changed (e.g. a new CreditShield decision under CASE_SOURCE=api)."""
    now = current.model_dump(mode="json")
    return {k: v for k, v in cached.items() if k != "notice_date"} == {
        k: v for k, v in now.items() if k != "notice_date"
    }


async def _from_cache(
    hit: RunLog, case: DenialCase, agent_input: AgentInput, tags: dict[str, str]
) -> NoticeResult | None:
    """Re-render the cached paragraph with today's date and run today's checks on it (no LLM
    call), in a trace tagged cache_hit. None if any check fails: a notice drafted before a
    guardrail changed must pass the current checks too, or it is drafted again. A rejected
    entry keeps its trace, tagged cache_rejected (monitoring counts only traces with an
    outcome)."""
    started = time.perf_counter()
    draft = NoticeDraft.model_validate(hit.draft)
    notice = render_notice(
        applicant_id=case.applicant_id,
        product=case.product,
        notice_date=agent_input.notice_date,
        reasons_paragraph=draft.paragraph,
    )
    # The checks run inside the trace: outside it, each check span became its own trace.
    with mlflow.start_span(name="draft_notice", span_type=SpanType.AGENT) as root:
        root.set_inputs(agent_input.model_dump(mode="json"))
        checks = await asyncio.to_thread(
            run_checks, case, draft, notice, agent_input.applicant_note
        )
        if not all(c.passed for c in checks):
            log.info("cache entry fails current checks", extra={"cached_run_id": hit.id})
            root.set_outputs({"cache_rejected": True, "cached_run_id": hit.id})
            mlflow.update_current_trace(
                tags={**tags, "case_id": str(case.applicant_id), "cache_rejected": "true"}
            )
            return None
        root.set_outputs({"outcome": hit.outcome, "notice": notice, "cached_run_id": hit.id})
        features = [r.feature for r in case.reasons]
        mlflow.update_current_trace(
            tags={
                **tags,
                **_monitor_tags(case.decision, features, checks, None, RunUsage()),
                "case_id": str(case.applicant_id),
                "prompt_version": str(hit.prompt_version),
                "model": hit.model,
                "outcome": hit.outcome,
                "cache_hit": "true",
            }
        )
    return NoticeResult(
        applicant_id=case.applicant_id,
        decision=case.decision,
        outcome=hit.outcome,
        notice=notice,
        escalation_reason=None,
        checks=checks,
        draft=draft,
        agent_input=agent_input.model_dump(mode="json"),
        retrieved_chunk_ids=hit.retrieved_chunk_ids,
        prompt_version=hit.prompt_version,
        model=hit.model,
        usage=RunUsage(),
        latency_ms=int((time.perf_counter() - started) * 1000),
        trace_id=root.trace_id,
        cache_hit=True,
    )


def _over_budget(spent: float, budget: float, model: str, case: DenialCase, tags) -> NoticeResult:
    """Fail closed: past the daily cap a case goes to review instead of calling the LLM."""
    started = time.perf_counter()
    reason = f"daily LLM budget reached (${spent:.4f} of ${budget:.2f}): no LLM call"
    with mlflow.start_span(name="draft_notice", span_type=SpanType.AGENT) as root:
        root.set_outputs({"outcome": "escalated", "notice": None})
        features = [r.feature for r in case.reasons]
        mlflow.update_current_trace(
            tags={**tags, **_monitor_tags(case.decision, features, [], reason, RunUsage()),
                  "case_id": str(case.applicant_id), "outcome": "escalated",
                  "budget_blocked": "true"}
        )  # fmt: skip
    return _escalated(case.applicant_id, case.decision, reason, model, started, root.trace_id)


async def draft_notice(
    applicant_id: int,
    notice_date: date | None = None,
    *,
    case: DenialCase | dict[str, Any] | None = None,
    provider: LLMProvider | None = None,
    search: Callable[[str], list[Any]] | None = None,
    sessions: async_sessionmaker | None = None,
    trace_tags: dict[str, str] | None = None,
) -> NoticeResult:
    """case, provider, search and sessions can be injected (the API passes the ones its
    lifespan created; tests pass fakes). By default the case comes from CASE_SOURCE, the
    provider from config, the search from pgvector, and a database engine is opened for this
    call only."""
    engine = None
    if sessions is None:
        engine = make_async_engine()
        sessions = make_async_sessionmaker(engine)
    try:
        return await _draft(
            case or get_case(applicant_id), notice_date or date.today(), provider, search,
            sessions, trace_tags or {},
        )  # fmt: skip
    finally:
        if engine is not None:
            await engine.dispose()


async def _draft(case, notice_date, provider, search, sessions, tags) -> NoticeResult:
    settings = get_settings()
    setup_tracing()
    provider = provider or get_provider()
    # Pin the version the alias points at now, so the cache key and the run agree.
    alias_uri = f"prompts:/{settings.prompt_name}@{settings.prompt_alias}"
    prompt = await asyncio.to_thread(mlflow.genai.load_prompt, alias_uri, link_to_model=False)
    version = int(prompt.version)
    model = (prompt.model_config or {}).get("model", provider.model)
    today = datetime.now(UTC).date()

    r = None
    if isinstance(case, DenialCase) or _valid(case):
        case = DenialCase.model_validate(case)
        agent_input = await asyncio.to_thread(build_agent_input, case, notice_date)
        async with sessions() as s:
            hit = await run_log.latest_sent(s, case.applicant_id, version, model)
            spent = await spend.spent_on(s, today)
        budget = settings.daily_budget_usd
        if hit is not None and _same_input(hit.agent_input, agent_input):
            r = await _from_cache(hit, case, agent_input, tags)
        if r is None and budget is not None and spent >= budget and case.decision == "DENY":
            r = _over_budget(spent, budget, model, case, tags)
    if r is None:
        r = await run_notice(
            case,
            notice_date,
            provider=provider,
            search=search,
            prompt_uri=f"prompts:/{settings.prompt_name}/{version}",
            trace_tags=tags,
        )
    await _save(r, sessions, today)
    log.info(
        "notice",
        extra={
            "applicant_id": r.applicant_id,
            "outcome": r.outcome,
            "cache_hit": r.cache_hit,
            "prompt_version": r.prompt_version,
            "model": r.model,
            "llm_calls": r.usage.calls,
            "cost_usd": round(r.usage.cost_usd, 6),
            "latency_ms": r.latency_ms,
            "trace_id": r.trace_id,
            "run_id": r.run_id,
            "review_id": r.review_id,
        },
    )
    return r


def _valid(case: dict[str, Any]) -> bool:
    try:
        DenialCase.model_validate(case)
    except ValidationError:
        return False
    return True


async def _save(r: NoticeResult, sessions: async_sessionmaker, today: date) -> None:
    """Run log, review item and spend counter in one transaction."""
    settings = get_settings()
    draft = r.draft.model_dump() if r.draft else None
    checks = [c.model_dump() for c in r.checks]
    async with sessions() as s, s.begin():
        row = await run_log.add_run(
            s,
            RunLog(
                applicant_id=r.applicant_id,
                decision=r.decision,
                outcome=r.outcome,
                escalation_reason=r.escalation_reason,
                agent_input=r.agent_input,
                retrieved_chunk_ids=r.retrieved_chunk_ids,
                draft=draft,
                notice=r.notice,
                check_results=checks,
                prompt_name=settings.prompt_name,
                prompt_version=r.prompt_version,
                model=r.model,
                llm_calls=r.usage.calls,
                prompt_tokens=r.usage.prompt_tokens,
                completion_tokens=r.usage.completion_tokens,
                cost_usd=r.usage.cost_usd,
                latency_ms=r.latency_ms,
                trace_id=r.trace_id,
                cache_hit=r.cache_hit,
            ),
        )
        r.run_id = row.id
        if r.outcome == "escalated":
            fields = {
                "decision": r.decision,
                "reason": r.escalation_reason,
                "check_results": checks,
                "draft": draft,
                "run_id": r.run_id,
                "trace_id": r.trace_id,
            }
            # One pending item per applicant: a repeat request refreshes it to the latest
            # run instead of queuing the same case again.
            item = await review_queue.pending_for(s, r.applicant_id)
            if item is None:
                item = await review_queue.add_item(
                    s, ReviewItem(applicant_id=r.applicant_id, **fields)
                )
            else:
                for k, v in fields.items():
                    setattr(item, k, v)
            r.review_id = item.id
        if r.usage.cost_usd > 0:
            await spend.add_spend(s, today, r.usage.cost_usd)
