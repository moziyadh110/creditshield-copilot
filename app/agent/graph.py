"""The notice agent as an async LangGraph flow.

route ─ MANUAL_REVIEW or escalate-only feature ─────────────────────────────► escalate
  └─ DENY ► retrieve ► draft ⇄ tools ► validate ─ invalid (1 re-ask) ─► draft
                                          └─ valid ► check ─ fail (1 revision) ─► draft
                                                       └─ pass ► send
A second invalid output, a second failed check, any check whose failure is "escalate"
(protected attributes, personal data, injection, required content), or a tool error or
timeout sends the case to human review. Nothing is guessed and nothing is sent.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agent.template import render_notice
from app.agent.tools import TOOL_SCHEMAS, Toolbox, ToolFailure
from app.guardrails.checks import run_checks
from app.llm.provider import LLMProvider, LLMResponse, RunUsage
from app.repositories.reason_codes import ReasonCatalog
from app.schemas import AgentInput, CheckResult, DenialCase, NoticeDraft

MAX_TOOL_ROUNDS = 4
RULES_QUERY = "statement of specific reasons for adverse action must indicate the principal reasons"


class NoticeState(TypedDict, total=False):
    case: DenialCase
    agent_input: AgentInput
    messages: list[dict[str, Any]]
    raw_output: str | None
    draft: NoticeDraft | None
    checks: list[CheckResult]
    reasks: int
    revisions: int
    outcome: str
    escalation_reason: str | None
    notice: str | None


@dataclass
class AgentDeps:
    provider: LLMProvider
    system_prompt: str
    toolbox: Toolbox
    catalog: ReasonCatalog
    usage: RunUsage
    llm_timeout_s: float = 60.0


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def build_user_message(agent_input: AgentInput, rules: list[dict[str, str]]) -> str:
    """The only user-turn content. The case, including any masked applicant note, is JSON data
    inside <case_data>; nothing from the case ever goes into the system prompt."""
    context = "\n\n".join(f"[{c['section']}]\n{c['text']}" for c in rules)
    return (
        f"<case_data>\n{agent_input.model_dump_json(indent=2)}\n</case_data>\n\n"
        f"Regulation context (retrieved):\n{context}"
    )


def build_graph(deps: AgentDeps):
    async def chat(messages: list[dict], **kwargs) -> LLMResponse:
        try:
            resp = await asyncio.wait_for(
                deps.provider.chat(messages, **kwargs), timeout=deps.llm_timeout_s
            )
        except TimeoutError as e:
            raise ToolFailure(f"LLM provider timed out after {deps.llm_timeout_s:g}s") from e
        except Exception as e:  # provider/network errors: escalate, never crash or guess
            raise ToolFailure(f"LLM provider error: {type(e).__name__}: {e}") from e
        deps.usage.add(resp)
        return resp

    def route(state: NoticeState) -> str:
        case = state["case"]
        if case.decision == "MANUAL_REVIEW":
            return "escalate"
        if any(deps.catalog.features[r.feature].escalate for r in case.reasons):
            return "escalate"
        return "retrieve"

    async def retrieve(state: NoticeState) -> NoticeState:
        try:
            rules = await deps.toolbox.search_regulation(RULES_QUERY)
        except ToolFailure as e:
            return {"escalation_reason": f"tool failure: {e}"}
        messages = [
            {"role": "system", "content": deps.system_prompt},
            {"role": "user", "content": build_user_message(state["agent_input"], rules)},
        ]
        return {"messages": messages, "reasks": 0, "revisions": 0}

    async def draft(state: NoticeState) -> NoticeState:
        messages = list(state["messages"])
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                resp = await chat(messages, tools=TOOL_SCHEMAS)
                messages.append(resp.message)
                calls = resp.message.get("tool_calls") or []
                if not calls:
                    return {"messages": messages, "raw_output": resp.message.get("content") or ""}
                for call in calls:
                    fn = call["function"]
                    result = await deps.toolbox.call(fn["name"], fn.get("arguments", "{}"))
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            # Tool budget used up: ask for the answer with tools switched off.
            resp = await chat(messages, json_output=True)
        except ToolFailure as e:
            return {"escalation_reason": f"tool failure: {e}"}
        messages.append(resp.message)
        return {"messages": messages, "raw_output": resp.message.get("content") or ""}

    def after_tools(state: NoticeState) -> str:
        return "escalate" if state.get("escalation_reason") else "next"

    def validate(state: NoticeState) -> NoticeState:
        try:
            return {"draft": NoticeDraft.model_validate_json(_strip_fences(state["raw_output"]))}
        except (ValidationError, json.JSONDecodeError) as e:
            if state["reasks"] >= 1:
                return {"draft": None, "escalation_reason": f"invalid output after re-ask: {e}"}
            feedback = (
                "Your reply was not valid for the required JSON shape. Error:\n"
                f"{e}\nReply again with JSON only."
            )
            messages = state["messages"] + [{"role": "user", "content": feedback}]
            return {"draft": None, "messages": messages, "reasks": state["reasks"] + 1}

    def after_validate(state: NoticeState) -> str:
        if state.get("draft") is not None:
            return "check"
        return "escalate" if state.get("escalation_reason") else "draft"

    def check(state: NoticeState) -> NoticeState:
        case, inp, draft_ = state["case"], state["agent_input"], state["draft"]
        notice = render_notice(
            applicant_id=case.applicant_id,
            product=case.product,
            notice_date=inp.notice_date,
            reasons_paragraph=draft_.paragraph,
        )
        results = run_checks(case, draft_, notice, inp.applicant_note)
        failed = [r for r in results if not r.passed]
        if not failed:
            return {"checks": results, "notice": notice}
        summary = "; ".join(f"{r.name}: {r.detail}" for r in failed)
        if any(r.on_fail == "escalate" for r in failed):
            return {"checks": results, "escalation_reason": f"blocked by checks: {summary}"}
        if state["revisions"] >= 1:
            return {
                "checks": results,
                "escalation_reason": f"checks failed after revision: {summary}",
            }
        feedback = f"Your draft failed these checks: {summary}\nFix them and reply with JSON only."
        messages = state["messages"] + [{"role": "user", "content": feedback}]
        return {"checks": results, "messages": messages, "revisions": state["revisions"] + 1}

    def after_check(state: NoticeState) -> str:
        if all(r.passed for r in state["checks"]):
            return "send"
        return "escalate" if state.get("escalation_reason") else "draft"

    def send(state: NoticeState) -> NoticeState:
        return {"outcome": "revised" if state["revisions"] else "sent"}

    def escalate(state: NoticeState) -> NoticeState:
        case = state["case"]
        reason = state.get("escalation_reason")
        if reason is None and case.decision == "MANUAL_REVIEW":
            reason = "MANUAL_REVIEW is not a denial: sent to review with no LLM call"
        elif reason is None:
            reason = "escalate-only reason feature (e.g. cnt_children): no LLM call"
        # A notice that was rendered but blocked must never look sendable.
        return {"outcome": "escalated", "escalation_reason": reason, "notice": None}

    g = StateGraph(NoticeState)
    for name, fn in [
        ("retrieve", retrieve),
        ("draft", draft),
        ("validate", validate),
        ("check", check),
        ("send", send),
        ("escalate", escalate),
    ]:
        g.add_node(name, fn)
    g.add_conditional_edges(START, route, ["retrieve", "escalate"])
    g.add_conditional_edges("retrieve", after_tools, {"next": "draft", "escalate": "escalate"})
    g.add_conditional_edges("draft", after_tools, {"next": "validate", "escalate": "escalate"})
    g.add_conditional_edges("validate", after_validate, ["check", "draft", "escalate"])
    g.add_conditional_edges("check", after_check, ["send", "draft", "escalate"])
    g.add_edge("send", END)
    g.add_edge("escalate", END)
    return g.compile()
