"""The Streamlit review page against a fake API and fake monitoring history (Streamlit's
AppTest runs the script headless): which actions each kind of item offers, a refused
approval showing its failed checks, and the monitoring tab with and without data."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parent.parent / "app" / "review" / "page.py")
DRAFT = {
    "reasons": [{"feature": "late_payment_rate", "text": "You paid late on past loans with us."}],
    "paragraph": "These are the main reasons for our decision. You paid late on past loans.",
}


def item(id, decision, draft):
    return {
        "id": id, "applicant_id": 100000 + id, "decision": decision, "reason": "test reason",
        "status": "pending", "draft": draft, "suggested_draft": draft, "run_id": 1,
        "created_at": "2026-10-29T12:00:00+00:00", "trace_id": "tr-1",
        "check_results": [{"name": "banned_wording", "passed": False, "detail": "x",
                           "on_fail": "revise"}],
    }  # fmt: skip


QUEUE = [item(1, "DENY", DRAFT), item(2, "MANUAL_REVIEW", None)]


class Response:
    def __init__(self, status_code, body):
        self.status_code, self._body, self.text = status_code, body, str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class FakeAPI:
    posts: list = []

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def get(self, path):
        return Response(200, QUEUE)

    def post(self, path, json):
        FakeAPI.posts.append((path, json))
        failed = [{"name": "protected_attributes", "passed": False, "detail": "age"}]
        return Response(422, {"detail": {"failed_checks": failed}})


HISTORY = pd.DataFrame(
    {"requests": [200.0], "escalation_rate": [0.26], "escalation_baseline": [0.25],
     "cost_usd_p50": [0.0005], "cost_usd_p95": [0.001], "latency_ms_p50": [2000.0],
     "latency_ms_p95": [3600.0], "reason_psi": [0.01], "pass_rate.faithfulness": [1.0],
     "alerts": [""]},
    index=pd.Index(["2026-10-29"], name="day"),
)  # fmt: skip


@pytest.fixture
def page():
    def run(history=HISTORY):
        FakeAPI.posts = []
        with (
            patch("httpx.Client", FakeAPI),
            patch("app.llmops.monitor.history", lambda: history),
        ):
            at = AppTest.from_file(PAGE, default_timeout=60).run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    return run


def test_a_denial_can_be_edited_and_a_refused_approval_shows_the_failed_checks(page):
    at = page()
    assert at.metric[0].value == "2"
    assert [b.label for b in at.button] == [
        "Run checks and approve",
        "Reject: handle outside the Copilot",
    ]
    assert at.text_input[0].value == DRAFT["reasons"][0]["text"]
    with patch("httpx.Client", FakeAPI):
        at.button[0].click().run()
    path, body = FakeAPI.posts[0]
    assert path == "/review-queue/1/approve" and body["reasons"] == DRAFT["reasons"]
    assert "fails these checks" in at.error[0].value
    assert any("protected_attributes" in t.value.to_string() for t in at.table)


def test_manual_review_offers_only_reject(page):
    at = page()
    with patch("httpx.Client", FakeAPI):
        at.selectbox[0].select(at.selectbox[0].options[1]).run()
    assert [b.label for b in at.button] == ["Reject: handle outside the Copilot"]
    assert "not a denial" in at.info[0].value


def test_monitoring_tab_with_and_without_runs(page):
    at = page()
    labels = {m.label: m.value for m in at.metric}
    assert labels["Escalation rate"] == "26.0%" and labels["Latency p95"] == "3.6 s"
    empty = page(history=pd.DataFrame())
    assert any("make monitor" in i.value for i in empty.info)
