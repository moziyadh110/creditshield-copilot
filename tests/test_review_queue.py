"""Day 3 done-criteria, end to end through the service and the real database: a forced check
failure and a forced timeout both land in the review queue, and nothing is sent.

Needs the pgvector database (docker compose up -d db; alembic upgrade head). MLflow uses the
temporary SQLite store from conftest, with prompt v1 registered into it here.
"""

import asyncio
import json

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from app.db.models import ReviewItem, RunLog
from app.db.session import make_sessionmaker
from app.llm.cost import Usage
from app.llm.provider import LLMResponse
from app.llmops.prompts import register
from app.schemas import DenialCase
from app.services.notices import draft_notice

APPLICANT = 990001  # synthetic id, not in denials.parquet
CASE = DenialCase(
    applicant_id=APPLICANT,
    decision="DENY",
    product="Cash loans",
    reasons=[{"rank": 1, "feature": "late_payment_rate", "impact_score": 0.3}],
)


class Chunk:
    section_id, text, source_url = "1002.9(b)(2)", "rule text", "https://example"


class ScriptedProvider:
    model = "fake"

    def __init__(self, content=None, hang=False):
        self.content, self.hang = content, hang

    def configure(self, model_config):
        pass

    async def chat(self, messages, **kwargs):
        if self.hang:
            await asyncio.sleep(120)
        return LLMResponse(
            {"role": "assistant", "content": self.content}, Usage(10, 5), 0.0, "fake"
        )


@pytest.fixture(scope="module")
def sessions():
    s = make_sessionmaker()
    try:
        with s() as conn:
            conn.execute(text("select 1 from review_queue limit 1"))
    except OperationalError:
        pytest.skip("pgvector database not reachable on DATABASE_URL")
    register("notice-reasons")
    yield s
    with s() as conn, conn.begin():
        conn.execute(text("delete from review_queue where applicant_id = :a"), {"a": APPLICANT})
        conn.execute(text("delete from run_log where applicant_id = :a"), {"a": APPLICANT})


def queued(sessions, review_id) -> ReviewItem:
    with sessions() as s:
        return s.scalars(select(ReviewItem).where(ReviewItem.id == review_id)).one()


def test_forced_check_failure_lands_in_queue(sessions):
    leaky = json.dumps(
        {
            "reasons": [{"feature": "late_payment_rate", "text": "You paid late on past loans."}],
            "paragraph": "Applicants your age often pay late, which is the main reason here.",
        }
    )
    r = asyncio.run(
        draft_notice(
            APPLICANT, case=CASE, provider=ScriptedProvider(leaky), search=lambda q: [Chunk()]
        )
    )
    assert r.outcome == "escalated" and r.notice is None
    item = queued(sessions, r.review_id)
    assert item.status == "pending" and "protected_attributes" in item.reason
    assert item.draft["paragraph"].startswith("Applicants your age")
    with sessions() as s:
        assert s.get(RunLog, r.run_id).notice is None


def test_forced_timeout_lands_in_queue(sessions, monkeypatch):
    monkeypatch.setattr("app.services.notices.get_settings", _settings_with_timeout(0.2))
    r = asyncio.run(
        draft_notice(
            APPLICANT, case=CASE, provider=ScriptedProvider(hang=True), search=lambda q: [Chunk()]
        )
    )
    assert r.outcome == "escalated" and r.notice is None
    item = queued(sessions, r.review_id)
    assert "timed out" in item.reason and item.run_id == r.run_id


def _settings_with_timeout(seconds):
    from app.config import get_settings

    settings = get_settings().model_copy(update={"llm_timeout_s": seconds})
    return lambda: settings


def test_no_llm_key_sends_denials_to_review(sessions):
    from app.config import get_settings
    from app.llm.provider import UnconfiguredProvider, get_provider_or_unconfigured

    provider = get_provider_or_unconfigured(
        get_settings().model_copy(update={"deepseek_api_key": None})
    )
    assert isinstance(provider, UnconfiguredProvider)
    r = asyncio.run(
        draft_notice(APPLICANT, case=CASE, provider=provider, search=lambda q: [Chunk()])
    )
    assert r.outcome == "escalated" and r.notice is None
    assert "No API key set" in queued(sessions, r.review_id).reason
