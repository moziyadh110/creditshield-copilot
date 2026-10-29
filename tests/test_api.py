"""The HTTP API end to end with a scripted LLM: auth, 200 vs 202, the response cache, the
daily budget, request IDs and the review queue.

Needs the pgvector database (docker compose up -d db; alembic upgrade head). MLflow uses the
temporary SQLite store from conftest, with the prompts registered into it here. Cases use
synthetic applicant IDs, and the scripted LLM costs $0, so the real spend counter is untouched.
"""

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import get_settings
from app.db.session import make_sessionmaker
from app.llm.cost import Usage
from app.llm.provider import LLMResponse
from app.llmops.prompts import register
from app.repositories.cases import CaseNotFound
from app.schemas import DenialCase

KEY = "test-key"
SENT, MANUAL, BUDGET, STALE = 990101, 990102, 990103, 990104
CASES = {
    i: DenialCase(
        applicant_id=i,
        decision=decision,
        product="Cash loans",
        reasons=[{"rank": 1, "feature": "late_payment_rate", "impact_score": 0.3}],
    )
    for i, decision in [
        (SENT, "DENY"),
        (MANUAL, "MANUAL_REVIEW"),
        (BUDGET, "DENY"),
        (STALE, "DENY"),
    ]
}
GOOD_DRAFT = json.dumps(
    {
        "reasons": [{"feature": "late_payment_rate", "text": "You paid late on past loans."}],
        "paragraph": "These are the main reasons for our decision. You paid late on past loans.",
    }
)


class Chunk:
    section_id, text, source_url = "1002.9(b)(2)", "rule text", "https://example"


class ScriptedProvider:
    """Answers every call with GOOD_DRAFT and counts the calls."""

    model = "fake"

    def __init__(self):
        self.calls = 0

    def configure(self, model_config):
        self.model = model_config.get("model", self.model)

    async def chat(self, messages, **kwargs):
        self.calls += 1
        message = {"role": "assistant", "content": GOOD_DRAFT}
        return LLMResponse(message, Usage(10, 5), 0.0, self.model)


def _get_case(applicant_id):
    try:
        return CASES[applicant_id]
    except KeyError:
        raise CaseNotFound(f"applicant {applicant_id} is not in denials.parquet") from None


@pytest.fixture(scope="module")
def db():
    sessions = make_sessionmaker()
    try:
        with sessions() as s:
            s.execute(text("select 1 from daily_spend limit 1"))
    except OperationalError:
        pytest.skip("pgvector database not reachable on DATABASE_URL")
    register("notice-reasons")
    _clean(sessions)  # a crashed earlier run must not leave a cached notice behind
    yield sessions
    _clean(sessions)


def _clean(sessions):
    with sessions() as s, s.begin():
        for table in ("review_queue", "run_log"):
            sql = f"delete from {table} where applicant_id = any(:ids)"
            s.execute(text(sql), {"ids": [*CASES]})


@pytest.fixture
def api(db, monkeypatch):
    from app.api.main import create_app

    settings = get_settings().model_copy(
        update={"api_key": SecretStr(KEY), "daily_budget_usd": None}
    )
    monkeypatch.setattr("app.api.auth.get_settings", lambda: settings)
    monkeypatch.setattr("app.services.notices.get_settings", lambda: settings)
    monkeypatch.setattr("app.services.notices.get_case", _get_case)
    provider = ScriptedProvider()
    with TestClient(create_app(provider=provider, search=lambda q: [Chunk()])) as client:
        client.headers["X-API-Key"] = KEY
        yield client, provider, settings


def test_health_is_public(api):
    client, _, _ = api
    assert client.get("/health", headers={"X-API-Key": ""}).status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_protected_routes_need_the_key(api, headers):
    client, _, _ = api
    client.headers.pop("X-API-Key")
    assert client.post("/notices", json={"applicant_id": SENT}, headers=headers).status_code == 401
    assert client.get("/review-queue", headers=headers).status_code == 401


def test_no_configured_key_fails_closed(api, monkeypatch):
    client, _, settings = api
    unset = settings.model_copy(update={"api_key": None})
    monkeypatch.setattr("app.api.auth.get_settings", lambda: unset)
    assert client.get("/review-queue").status_code == 503


def test_notice_then_cache_hit(api):
    client, provider, _ = api
    first = client.post("/notices", json={"applicant_id": SENT})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["outcome"] == "sent" and body["cache_hit"] is False
    assert "You paid late on past loans." in body["notice"]
    calls = provider.calls

    again = client.post("/notices", json={"applicant_id": SENT})
    assert again.status_code == 200
    cached = again.json()
    assert cached["cache_hit"] is True and provider.calls == calls  # no LLM call
    assert cached["notice"] == body["notice"] and cached["trace_id"] != body["trace_id"]
    assert cached["prompt_version"] == body["prompt_version"]


def test_request_id_is_echoed_or_created(api):
    client, _, _ = api
    r = client.post("/notices", json={"applicant_id": SENT}, headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123" and r.json()["request_id"] == "abc123"
    assert len(client.get("/health").headers["X-Request-ID"]) == 32


def test_manual_review_is_202_and_queued(api):
    client, provider, _ = api
    calls = provider.calls
    r = client.post("/notices", json={"applicant_id": MANUAL})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "sent to human review" and "MANUAL_REVIEW" in body["reason"]
    assert provider.calls == calls
    queued = {item["id"]: item for item in client.get("/review-queue").json()}
    assert queued[body["review_id"]]["applicant_id"] == MANUAL


def test_budget_reached_fails_closed_without_llm_call(api, monkeypatch):
    client, provider, settings = api
    capped = settings.model_copy(update={"daily_budget_usd": 0.0})
    monkeypatch.setattr("app.services.notices.get_settings", lambda: capped)
    calls = provider.calls
    r = client.post("/notices", json={"applicant_id": BUDGET})
    assert r.status_code == 202 and "daily LLM budget reached" in r.json()["reason"]
    assert provider.calls == calls


def test_unknown_applicant_is_404(api):
    client, _, _ = api
    assert client.post("/notices", json={"applicant_id": 1}).status_code == 404


def test_over_long_applicant_note_escalates_without_llm_call(api, monkeypatch):
    client, provider, _ = api
    long_note = CASES[BUDGET].model_dump() | {"applicant_note": "x" * 1001}
    monkeypatch.setattr("app.services.notices.get_case", lambda i: long_note)
    calls = provider.calls
    r = client.post("/notices", json={"applicant_id": BUDGET})
    assert r.status_code == 202 and "applicant_note" in r.json()["reason"]
    assert provider.calls == calls


def test_cached_notice_that_fails_current_checks_is_redrafted(api, db):
    """A notice cached before a guardrail changed must pass today's checks to be served."""
    import mlflow.genai

    from app.db.models import RunLog
    from app.guardrails.allow_list import build_agent_input

    client, provider, _ = api
    case = CASES[STALE]
    version = int(mlflow.genai.load_prompt("prompts:/notice-reasons@production").version)
    leaky = {
        "reasons": [{"feature": "late_payment_rate", "text": "You paid late on past loans."}],
        "paragraph": "Applicants your age often pay late, which is the main reason here.",
    }
    with db() as s, s.begin():
        s.add(
            RunLog(
                applicant_id=STALE, decision="DENY", outcome="sent",
                agent_input=build_agent_input(case, date.today()).model_dump(mode="json"),
                retrieved_chunk_ids=[], draft=leaky, notice="old notice", check_results=[],
                prompt_name="notice-reasons", prompt_version=version, model="deepseek-flash",
                llm_calls=1, prompt_tokens=0, completion_tokens=0, cost_usd=0.0, latency_ms=0,
            )
        )  # fmt: skip
    calls = provider.calls
    r = client.post("/notices", json={"applicant_id": STALE})
    assert r.status_code == 200 and r.json()["cache_hit"] is False
    assert provider.calls > calls and "your age" not in r.json()["notice"]
