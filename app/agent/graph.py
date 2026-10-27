"""The notice agent as an async LangGraph flow.

route ─ MANUAL_REVIEW or escalate-only feature ─────────────────────────────► escalate
  └─ DENY ► retrieve ► draft ⇄ tools ► validate ─ invalid (1 re-ask) ─► draft
                                          └─ valid ► check ─ fail (1 revision) ─► draft
                                                       └─ pass ► send
A second invalid output or a second failed check escalates to human review.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agent.template import render_notice
from app.agent.tools import TOOL_SCHEMAS, Toolbox
from app.guardrails.checks import run_checks
from app.llm.provider import LLMProvider, RunUsage
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


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def build_graph(deps: AgentDeps):
    def route(state: NoticeState) -> str:
        case = state["case"]
        if case.decision == "MANUAL_REVIEW":
            return "escalate"
        if any(deps.catalog.features[r.feature].escalate for r in case.reasons):
            return "escalate"
        return "retrieve"

    async def retrieve(state: NoticeState) -> NoticeState:
        rules = await deps.toolbox.search_regulation(RULES_QUERY)
        context = "\n\n".join(f"[{c['section']}]\n{c['text']}" for c in rules)
        user = (
            f"<case_data>\n{state['agent_input'].model_dump_json(indent=2)}\n</case_data>\n\n"
            f"Regulation context (retrieved):\n{context}"
        )
        messages = [
            {"role": "system", "content": deps.system_prompt},
            {"role": "user", "content": user},
        ]
        return {"messages": messages, "reasks": 0, "revisions": 0}

    async def draft(state: NoticeState) -> NoticeState:
        messages = list(state["messages"])
        for _ in range(MAX_TOOL_ROUNDS):
            resp = await deps.provider.chat(messages, tools=TOOL_SCHEMAS)
            deps.usage.add(resp)
            messages.append(resp.message)
            calls = resp.message.get("tool_calls") or []
            if not calls:
                return {"messages": messages, "raw_output": resp.message.get("content") or ""}
            for call in calls:
                fn = call["function"]
                result = await deps.toolbox.call(fn["name"], fn.get("arguments", "{}"))
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        # Tool budget used up: ask for the answer with tools switched off.
        resp = await deps.provider.chat(messages, json_output=True)
        deps.usage.add(resp)
        messages.append(resp.message)
        return {"messages": messages, "raw_output": resp.message.get("content") or ""}

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
        results = run_checks(state["case"], state["draft"])
        failed = [r for r in results if not r.passed]
        if not failed:
            return {"checks": results}
        summary = "; ".join(f"{r.name}: {r.detail}" for r in failed)
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
        case, inp = state["case"], state["agent_input"]
        notice = render_notice(
            applicant_id=case.applicant_id,
            product=case.product,
            notice_date=inp.notice_date,
            reasons_paragraph=state["draft"].paragraph,
        )
        return {"notice": notice, "outcome": "revised" if state["revisions"] else "sent"}

    def escalate(state: NoticeState) -> NoticeState:
        case = state["case"]
        reason = state.get("escalation_reason")
        if reason is None and case.decision == "MANUAL_REVIEW":
            reason = "MANUAL_REVIEW is not a denial: sent to review with no LLM call"
        elif reason is None:
            reason = "escalate-only reason feature (e.g. cnt_children): no LLM call"
        return {"outcome": "escalated", "escalation_reason": reason}

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
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "validate")
    g.add_conditional_edges("validate", after_validate, ["check", "draft", "escalate"])
    g.add_conditional_edges("check", after_check, ["send", "draft", "escalate"])
    g.add_edge("send", END)
    g.add_edge("escalate", END)
    return g.compile()
