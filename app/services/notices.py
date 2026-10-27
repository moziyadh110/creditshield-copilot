"""Draft adverse action notices. run_notice runs the agent inside one MLflow trace and saves
nothing (the eval calls it directly); draft_notice also writes the run log and, on
escalation, the review queue."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from functools import cache
from typing import Any

import mlflow
import mlflow.genai
from mlflow.entities import SpanType
from pydantic import ValidationError

from app.agent.graph import AgentDeps, NoticeState, build_graph
from app.agent.tools import Toolbox
from app.config import get_settings
from app.db.models import Chunk, ReviewItem, RunLog
from app.db.session import make_sessionmaker
from app.guardrails.allow_list import build_agent_input
from app.llm.provider import LLMProvider, RunUsage, get_provider
from app.llmops.tracing import setup_tracing
from app.repositories import review_queue, run_log
from app.repositories.cases import get_case
from app.repositories.reason_codes import get_reason_catalog
from app.retrieval.search import search_regulation
from app.schemas import CheckResult, DenialCase, NoticeDraft


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


@cache
def _sessions():
    return make_sessionmaker()


def _pgvector_search(query: str) -> list[Chunk]:
    with _sessions()() as s:
        return search_regulation(s, query, k=3)


async def run_notice(
    case: DenialCase | dict[str, Any],
    notice_date: date | None = None,
    *,
    provider: LLMProvider | None = None,
    search: Callable[[str], list[Any]] | None = None,
    prompt_uri: str | None = None,
) -> NoticeResult:
    """Run the agent on one case. A case given as a dict is validated first: missing or
    malformed data (a null SHAP value, an empty reason list, a null field) escalates to human
    review with no LLM call, never a crash or a guess."""
    settings = get_settings()
    setup_tracing()
    prompt_uri = prompt_uri or f"prompts:/{settings.prompt_name}@{settings.prompt_alias}"
    started = time.perf_counter()
    usage = RunUsage()

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
            mlflow.update_current_trace(
                tags={"case_id": str(raw.get("applicant_id")), "outcome": "escalated"}
            )
            return NoticeResult(
                applicant_id=int(raw.get("applicant_id") or 0),
                decision=str(raw.get("decision") or "UNKNOWN"),
                outcome="escalated",
                notice=None,
                escalation_reason=reason,
                checks=[],
                draft=None,
                agent_input={},
                retrieved_chunk_ids=[],
                prompt_version=None,
                model=provider.model if provider else settings.llm_model,
                usage=usage,
                latency_ms=int((time.perf_counter() - started) * 1000),
                trace_id=root.trace_id,
            )

        agent_input = build_agent_input(case, notice_date or date.today())
        root.set_inputs(agent_input.model_dump(mode="json"))
        # Loaded inside the trace, so the prompt version is linked to it.
        prompt = mlflow.genai.load_prompt(prompt_uri)
        provider = provider or get_provider()
        provider.configure(prompt.model_config or {})
        catalog = get_reason_catalog()
        toolbox = Toolbox(catalog, search or _pgvector_search)
        deps = AgentDeps(provider, prompt.template, toolbox, catalog, usage, settings.llm_timeout_s)
        state: NoticeState = await build_graph(deps).ainvoke(
            {"case": case, "agent_input": agent_input}
        )
        root.set_outputs({"outcome": state["outcome"], "notice": state.get("notice")})
        mlflow.update_current_trace(
            tags={
                "case_id": str(case.applicant_id),
                "prompt_version": str(prompt.version),
                "model": provider.model,
                "outcome": state["outcome"],
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


async def draft_notice(
    applicant_id: int,
    notice_date: date | None = None,
    *,
    case: DenialCase | dict[str, Any] | None = None,
    provider: LLMProvider | None = None,
    search: Callable[[str], list[Any]] | None = None,
) -> NoticeResult:
    """case, provider and search can be injected (tests); by default the case comes from
    CASE_SOURCE, the provider from config and the search from pgvector."""
    settings = get_settings()
    r = await run_notice(
        case or get_case(applicant_id), notice_date, provider=provider, search=search
    )
    draft = r.draft.model_dump() if r.draft else None
    checks = [c.model_dump() for c in r.checks]
    with _sessions()() as s, s.begin():
        row = run_log.add_run(
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
            ),
        )
        r.run_id = row.id
        if r.outcome == "escalated":
            r.review_id = review_queue.add_item(
                s,
                ReviewItem(
                    applicant_id=r.applicant_id,
                    decision=r.decision,
                    reason=r.escalation_reason,
                    check_results=checks,
                    draft=draft,
                    run_id=r.run_id,
                    trace_id=r.trace_id,
                ),
            ).id
    return r
