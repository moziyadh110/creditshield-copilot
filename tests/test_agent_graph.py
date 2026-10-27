"""The LangGraph flow with a scripted fake LLM: no API key, no network, no database."""

import asyncio
import json
from datetime import date

import pytest

from app.agent.graph import AgentDeps, build_graph
from app.agent.tools import Toolbox
from app.guardrails.allow_list import build_agent_input
from app.llm.cost import Usage
from app.llm.provider import LLMResponse, RunUsage
from app.repositories.reason_codes import get_reason_catalog
from app.schemas import DenialCase

FEATURES = ["late_payment_rate", "bureau_debt_ratio"]


def make_case(decision="DENY", features=FEATURES) -> DenialCase:
    reasons = [{"rank": i, "feature": f, "impact_score": 0.1} for i, f in enumerate(features, 1)]
    return DenialCase(applicant_id=1, decision=decision, product="Cash loans", reasons=reasons)


def good_draft(features=FEATURES) -> str:
    reasons = [{"feature": f, "text": f"Plain reason about {f}."} for f in features]
    return json.dumps(
        {"reasons": reasons, "paragraph": "These are the main reasons for our decision."}
    )


def tool_call(name, args) -> dict:
    fn = {"name": name, "arguments": json.dumps(args)}
    return {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": fn}]}


class FakeProvider:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, messages, *, tools=None, json_output=False):
        self.calls.append(list(messages))  # copy: the agent keeps appending to its list
        reply = self.replies.pop(0)
        message = reply if isinstance(reply, dict) else {"role": "assistant", "content": reply}
        return LLMResponse(message=message, usage=Usage(100, 10), cost_usd=0.0001, model="fake")


class FakeChunk:
    def __init__(self, section_id):
        self.section_id, self.text, self.source_url = section_id, "rule text", "https://example"


def run(case, replies):
    provider = FakeProvider(replies)
    catalog = get_reason_catalog()
    toolbox = Toolbox(catalog, lambda q: [FakeChunk("1002.9(b)(2)")])
    deps = AgentDeps(provider, "system prompt", toolbox, catalog, RunUsage())
    graph = build_graph(deps)
    state = asyncio.run(
        graph.ainvoke({"case": case, "agent_input": build_agent_input(case, date(2026, 10, 27))})
    )
    return state, provider


def test_manual_review_goes_to_review_without_llm_call():
    state, provider = run(make_case(decision="MANUAL_REVIEW"), [])
    assert state["outcome"] == "escalated"
    assert provider.calls == []


def test_escalate_only_feature_never_reaches_llm():
    state, provider = run(make_case(features=["cnt_children", "late_payment_rate"]), [])
    assert state["outcome"] == "escalated"
    assert provider.calls == []


def test_happy_path_with_tool_call_sends_notice():
    state, provider = run(
        make_case(), [tool_call("get_reason_codes", {"features": FEATURES}), good_draft()]
    )
    assert state["outcome"] == "sent"
    assert "These are the main reasons" in state["notice"]
    assert "Equal Credit Opportunity Act" in state["notice"]  # from the fixed template
    tool_reply = json.loads(provider.calls[1][-1]["content"])
    assert tool_reply["late_payment_rate"]["code"] == "CS33"


def test_invalid_json_is_re_asked_once_then_succeeds():
    state, provider = run(make_case(), ["not json", good_draft()])
    assert state["outcome"] == "sent"
    assert "not valid" in provider.calls[1][-1]["content"]


def test_invalid_json_twice_escalates():
    state, _ = run(make_case(), ["not json", '{"reasons": []}'])
    assert state["outcome"] == "escalated"
    assert "invalid output" in state["escalation_reason"]


def test_failed_check_is_revised_once():
    reordered = good_draft(list(reversed(FEATURES)))
    state, provider = run(make_case(), [reordered, good_draft()])
    assert state["outcome"] == "revised"
    assert "wrong order" in provider.calls[1][-1]["content"]


def test_failed_check_twice_escalates():
    invented = good_draft(["late_payment_rate", "amt_credit"])
    state, _ = run(make_case(), [invented, invented])
    assert state["outcome"] == "escalated"
    assert "invented reasons ['amt_credit']" in state["escalation_reason"]


def test_bad_tool_arguments_go_back_to_the_model_as_data():
    state, provider = run(make_case(), [tool_call("get_reason_codes", {}), good_draft()])
    assert state["outcome"] == "sent"
    assert "bad arguments" in provider.calls[1][-1]["content"]


def test_agent_input_is_allow_listed():
    inp = build_agent_input(make_case(), date(2026, 10, 27)).model_dump()
    assert set(inp) == {"product", "notice_date", "reasons", "applicant_note"}
    assert inp["reasons"] == [
        {"rank": 1, "feature": FEATURES[0]},
        {"rank": 2, "feature": FEATURES[1]},
    ]
    assert inp["applicant_note"] is None


@pytest.mark.parametrize(
    "reasons, error",
    [
        ([{"rank": 1, "feature": "age_years", "impact_score": 0.1}], "not in reason_codes.yaml"),
        ([{"rank": 1, "feature": "amt_credit", "impact_score": 0.1}] * 2, "duplicate"),
        ([{"rank": 2, "feature": "amt_credit", "impact_score": 0.1}], "ranks"),
    ],
)
def test_denial_case_rejects_bad_reasons(reasons, error):
    with pytest.raises(ValueError, match=error):
        DenialCase(applicant_id=1, decision="DENY", product="Cash loans", reasons=reasons)
