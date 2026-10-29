"""Resolving the review queue through the API: an approved draft must pass every check, a
MANUAL_REVIEW case can only be rejected, and a resolved item can't be resolved again.

Needs the pgvector database (docker compose up -d db; alembic upgrade head). Items are
inserted directly with synthetic applicant IDs and removed afterwards.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from app.config import get_settings
from app.db.models import ReviewItem
from app.db.session import make_sessionmaker
from app.repositories.cases import CaseNotFound
from app.schemas import DenialCase

KEY = "test-key"
DENY, MANUAL, CHILDREN = 990201, 990202, 990203
CASES = {
    DENY: DenialCase(
        applicant_id=DENY,
        decision="DENY",
        product="Cash loans",
        reasons=[
            {"rank": 1, "feature": "late_payment_rate", "impact_score": 0.3},
            {"rank": 2, "feature": "amt_credit", "impact_score": 0.2},
        ],
    ),
    MANUAL: DenialCase(
        applicant_id=MANUAL,
        decision="MANUAL_REVIEW",
        product="Cash loans",
        reasons=[{"rank": 1, "feature": "late_payment_rate", "impact_score": 0.3}],
    ),
    CHILDREN: DenialCase(
        applicant_id=CHILDREN,
        decision="DENY",
        product="Cash loans",
        reasons=[{"rank": 1, "feature": "cnt_children", "impact_score": 0.3}],
    ),
}
GOOD = {
    "reasons": [
        {"feature": "late_payment_rate", "text": "You paid late on past loans with us."},
        {"feature": "amt_credit", "text": "The amount of credit you asked for is too high."},
    ],
    "paragraph": "These are the main reasons for our decision. First, you paid late on past "
    "loans with us. Second, the amount of credit you asked for is too high.",
}


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
            s.execute(text("select final_notice from review_queue limit 1"))
    except OperationalError:
        pytest.skip("pgvector database not reachable on DATABASE_URL")
    yield sessions
    _clean(sessions)


def _clean(sessions):
    with sessions() as s, s.begin():
        sql = "delete from review_queue where applicant_id = any(:ids)"
        s.execute(text(sql), {"ids": [*CASES]})


def _queue(sessions, applicant_id, draft=None) -> int:
    case = CASES[applicant_id]
    with sessions() as s, s.begin():
        item = ReviewItem(
            applicant_id=applicant_id, decision=case.decision, reason="test", check_results=[],
            draft=draft,
        )  # fmt: skip
        s.add(item)
        s.flush()
        return item.id


def _item(sessions, review_id) -> ReviewItem:
    with sessions() as s:
        return s.scalars(select(ReviewItem).where(ReviewItem.id == review_id)).one()


@pytest.fixture
def api(db, monkeypatch):
    from app.api.main import create_app

    _clean(db)
    settings = get_settings().model_copy(update={"api_key": SecretStr(KEY)})
    monkeypatch.setattr("app.api.auth.get_settings", lambda: settings)
    monkeypatch.setattr("app.services.reviews.get_case", _get_case)
    with TestClient(create_app(provider=object(), search=lambda q: [])) as client:
        client.headers["X-API-Key"] = KEY
        yield client, db


def test_an_approved_draft_becomes_the_notice_once(api):
    client, db = api
    rid = _queue(db, DENY, draft=GOOD)
    r = client.post(f"/review-queue/{rid}/approve", json=GOOD | {"reviewer_note": "fine"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "approved" and GOOD["paragraph"] in body["notice"]
    assert all(c["passed"] for c in body["check_results"])
    item = _item(db, rid)
    assert item.status == "approved" and item.final_notice == body["notice"]
    assert item.draft == GOOD and item.resolved_at is not None  # the LLM draft is kept
    assert client.post(f"/review-queue/{rid}/approve", json=GOOD).status_code == 409
    assert rid not in {i["id"] for i in client.get("/review-queue").json()}


def test_an_edit_that_fails_a_check_is_refused_and_stays_pending(api):
    client, db = api
    rid = _queue(db, DENY, draft=GOOD)
    leaky = GOOD | {"paragraph": GOOD["paragraph"] + " Applicants your age often pay late."}
    r = client.post(f"/review-queue/{rid}/approve", json=leaky)
    assert r.status_code == 422
    assert "protected_attributes" in {c["name"] for c in r.json()["detail"]["failed_checks"]}
    missing = {"reasons": GOOD["reasons"][:1], "paragraph": GOOD["paragraph"]}
    r = client.post(f"/review-queue/{rid}/approve", json=missing)
    assert "faithfulness" in {c["name"] for c in r.json()["detail"]["failed_checks"]}
    assert _item(db, rid).status == "pending"


def test_cnt_children_can_never_be_approved(api):
    client, db = api
    rid = _queue(db, CHILDREN)
    worded = {
        "reasons": [{"feature": "cnt_children", "text": "The number of people you support."}],
        "paragraph": "These are the main reasons for our decision. The number of people you "
        "support.",
    }
    r = client.post(f"/review-queue/{rid}/approve", json=worded)
    assert r.status_code == 422
    failed = {c["name"]: c["detail"] for c in r.json()["detail"]["failed_checks"]}
    assert "escalate-only feature 'cnt_children'" in failed["protected_attributes"]


def test_manual_review_can_only_be_rejected_with_a_note(api):
    client, db = api
    rid = _queue(db, MANUAL)
    assert client.post(f"/review-queue/{rid}/approve", json=GOOD).status_code == 409
    assert client.post(f"/review-queue/{rid}/reject", json={"reviewer_note": ""}).status_code == 422
    r = client.post(f"/review-queue/{rid}/reject", json={"reviewer_note": "sent to underwriting"})
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    item = _item(db, rid)
    assert item.status == "rejected" and item.reviewer_note == "sent to underwriting"
    assert item.final_notice is None
    again = client.post(f"/review-queue/{rid}/reject", json={"reviewer_note": "again"})
    assert again.status_code == 409


def test_queue_suggests_catalog_wording_when_there_is_no_llm_draft(api):
    client, db = api
    no_draft, manual = _queue(db, DENY), _queue(db, MANUAL)
    queued = {i["id"]: i for i in client.get("/review-queue").json()}
    suggested = queued[no_draft]["suggested_draft"]
    assert [r["feature"] for r in suggested["reasons"]] == ["late_payment_rate", "amt_credit"]
    assert queued[manual]["suggested_draft"] is None


def test_unknown_review_item_is_404(api):
    client, _ = api
    assert client.post("/review-queue/0/approve", json=GOOD).status_code == 404
    assert client.post("/review-queue/0/reject", json={"reviewer_note": "x" * 3}).status_code == 404
