"""Draft one adverse action notice end to end: one MLflow trace and one run-log row per run."""

import time
from dataclasses import dataclass
from datetime import date

import mlflow
import mlflow.genai
from mlflow.entities import SpanType

from app.agent.graph import AgentDeps, NoticeState, build_graph
from app.agent.tools import Toolbox
from app.config import get_settings
from app.db.models import RunLog
from app.db.session import make_sessionmaker
from app.guardrails.allow_list import build_agent_input
from app.llm.provider import RunUsage, get_provider
from app.llmops.tracing import setup_tracing
from app.repositories import run_log
from app.repositories.cases import get_case
from app.repositories.reason_codes import get_reason_catalog
from app.retrieval.search import search_regulation
from app.schemas import CheckResult


@dataclass
class NoticeResult:
    applicant_id: int
    outcome: str
    notice: str | None
    escalation_reason: str | None
    checks: list[CheckResult]
    prompt_version: int
    model: str
    usage: RunUsage
    latency_ms: int
    trace_id: str
    run_id: int


async def draft_notice(applicant_id: int, notice_date: date | None = None) -> NoticeResult:
    settings = get_settings()
    setup_tracing()
    sessions = make_sessionmaker()
    case = get_case(applicant_id)
    agent_input = build_agent_input(case, notice_date or date.today())
    started = time.perf_counter()

    def search(query: str):
        with sessions() as s:
            return search_regulation(s, query, k=3)

    with mlflow.start_span(name="draft_notice", span_type=SpanType.AGENT) as root:
        root.set_inputs(agent_input.model_dump(mode="json"))
        # Loaded inside the trace, so the prompt version is linked to it.
        prompt = mlflow.genai.load_prompt(
            f"prompts:/{settings.prompt_name}@{settings.prompt_alias}"
        )
        provider = get_provider()
        provider.configure(prompt.model_config or {})
        usage = RunUsage()
        toolbox = Toolbox(get_reason_catalog(), search)
        graph = build_graph(
            AgentDeps(provider, prompt.template, toolbox, get_reason_catalog(), usage)
        )
        state: NoticeState = await graph.ainvoke({"case": case, "agent_input": agent_input})
        root.set_outputs({"outcome": state["outcome"], "notice": state.get("notice")})
        mlflow.update_current_trace(
            tags={
                "case_id": str(applicant_id),
                "prompt_version": str(prompt.version),
                "model": provider.model,
                "outcome": state["outcome"],
            }
        )
        trace_id = root.trace_id

    latency_ms = int((time.perf_counter() - started) * 1000)
    draft = state.get("draft")
    checks = state.get("checks", [])
    with sessions() as s, s.begin():
        row = run_log.add_run(
            s,
            RunLog(
                applicant_id=applicant_id,
                decision=case.decision,
                outcome=state["outcome"],
                escalation_reason=state.get("escalation_reason"),
                agent_input=agent_input.model_dump(mode="json"),
                retrieved_chunk_ids=toolbox.retrieved_ids,
                draft=draft.model_dump() if draft else None,
                notice=state.get("notice"),
                check_results=[c.model_dump() for c in checks],
                prompt_name=settings.prompt_name,
                prompt_version=int(prompt.version),
                model=provider.model,
                llm_calls=usage.calls,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=usage.cost_usd,
                latency_ms=latency_ms,
                trace_id=trace_id,
            ),
        )
        run_id = row.id

    return NoticeResult(
        applicant_id=applicant_id,
        outcome=state["outcome"],
        notice=state.get("notice"),
        escalation_reason=state.get("escalation_reason"),
        checks=checks,
        prompt_version=int(prompt.version),
        model=provider.model,
        usage=usage,
        latency_ms=latency_ms,
        trace_id=trace_id,
        run_id=run_id,
    )
