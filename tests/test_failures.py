"""Tool errors and timeouts escalate to review instead of crashing or guessing."""

import asyncio
import json
from datetime import date

import httpx
import pytest
from openai import APIConnectionError
from sqlalchemy.exc import OperationalError

from app.agent.graph import AgentDeps, build_graph
from app.agent.tools import Toolbox, ToolFailure
from app.guardrails.allow_list import build_agent_input
from app.llm.cost import Usage
from app.llm.provider import LLMResponse, RunUsage
from app.repositories.creditshield import CreditShieldClient, NotADenial
from app.repositories.reason_codes import get_reason_catalog
from app.schemas import DenialCase

CASE = DenialCase(
    applicant_id=1,
    decision="DENY",
    product="Cash loans",
    reasons=[{"rank": 1, "feature": "late_payment_rate", "impact_score": 0.3}],
)


class Chunk:
    section_id, text, source_url = "1002.9(b)(2)", "rule text", "https://example"


class RaisingProvider:
    model = "fake"

    async def chat(self, messages, **kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com"))


class HangingProvider:
    model = "fake"

    async def chat(self, messages, **kwargs):
        await asyncio.sleep(30)
        return LLMResponse({"role": "assistant", "content": "{}"}, Usage(0, 0), 0.0, "fake")


def run(provider, search=lambda q: [Chunk()], search_timeout_s=10.0):
    catalog = get_reason_catalog()
    toolbox = Toolbox(catalog, search, search_timeout_s=search_timeout_s)
    deps = AgentDeps(provider, "system", toolbox, catalog, RunUsage(), llm_timeout_s=0.2)
    inp = build_agent_input(CASE, date(2026, 10, 27))
    return asyncio.run(build_graph(deps).ainvoke({"case": CASE, "agent_input": inp}))


def test_llm_error_escalates():
    state = run(RaisingProvider())
    assert state["outcome"] == "escalated"
    assert "LLM provider error: APIConnectionError" in state["escalation_reason"]
    assert state.get("notice") is None


def test_llm_timeout_escalates():
    state = run(HangingProvider())
    assert state["outcome"] == "escalated"
    assert "timed out" in state["escalation_reason"]


def test_pgvector_error_escalates_before_any_llm_call():
    def broken(query):
        raise OperationalError("SELECT", {}, Exception("connection refused"))

    state = run(RaisingProvider(), search=broken)
    assert state["outcome"] == "escalated"
    assert "regulation search failed: OperationalError" in state["escalation_reason"]


def test_pgvector_hang_escalates():
    def slow(query):
        import time

        time.sleep(2)
        return [Chunk()]

    state = run(RaisingProvider(), search=slow, search_timeout_s=0.2)
    assert state["outcome"] == "escalated"
    assert "regulation search timed out" in state["escalation_reason"]


# --- CreditShield API ---------------------------------------------------------------------

DENY_BODY = {
    "decision": "DENY",
    "adverse_action_reasons": [
        {"rank": 1, "reason_code": "RC01", "reason": "x", "feature": "late_payment_rate",
         "impact_score": 0.42},
    ],
}  # fmt: skip


def client_with(handler) -> CreditShieldClient:
    return CreditShieldClient(
        "http://cs", "key", timeout_s=0.2, transport=httpx.MockTransport(handler)
    )


def test_creditshield_response_keeps_only_rank_feature_and_impact():
    client = client_with(lambda req: httpx.Response(200, json=DENY_BODY))
    case = asyncio.run(client.predict(5, "Cash loans", {"age_years": 70}))
    assert case.reasons[0].model_dump() == {
        "rank": 1,
        "feature": "late_payment_rate",
        "impact_score": 0.42,
    }
    assert "age" not in json.dumps(case.model_dump())


@pytest.mark.parametrize(
    "handler, message",
    [
        (lambda req: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=req)), "timed out"),
        (lambda req: httpx.Response(500, json={"detail": "redis down"}), "API error"),
        (
            lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused", request=req)),
            "API error",
        ),
    ],
)
def test_creditshield_failures_raise_tool_failure(handler, message):
    with pytest.raises(ToolFailure, match=message):
        asyncio.run(client_with(handler).predict(5, "Cash loans", {}))


def test_creditshield_approval_is_not_a_denial():
    body = {"decision": "APPROVE", "adverse_action_reasons": []}
    with pytest.raises(NotADenial):
        asyncio.run(client_with(lambda req: httpx.Response(200, json=body)).predict(5, "x", {}))
