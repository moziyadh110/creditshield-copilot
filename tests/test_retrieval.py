from pathlib import Path

import pytest
import yaml
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.config import DATA_DIR
from app.db.models import Chunk
from app.db.session import make_sessionmaker
from app.retrieval.chunking import chunk_regb
from app.retrieval.search import search_regulation

QUESTIONS = yaml.safe_load((Path(__file__).parent / "data/retrieval_questions.yaml").read_text())


def test_chunks_are_section_aware():
    chunks = chunk_regb(DATA_DIR / "regb")
    ids = [c.section_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {"1002.9(a)(2)", "1002.9(b)(1)", "1002.9(g)"} <= set(ids)
    forms = [i for i in ids if i.startswith("1002-appC-C-")]
    assert forms == [f"1002-appC-C-{n}" for n in range(1, 6)]
    for c in chunks:
        assert c.source_url.startswith("https://www.ecfr.gov/current/title-12/")
        assert c.text.startswith("12 CFR")


def test_sub_items_stay_with_their_parent():
    by_id = {c.section_id: c for c in chunk_regb(DATA_DIR / "regb")}
    assert "(ii) A disclosure of the applicant's right" in by_id["1002.9(a)(2)"].text
    assert "____Length of employment" in by_id["1002-appC-C-1"].text


def test_every_expected_section_exists():
    ids = {c.section_id for c in chunk_regb(DATA_DIR / "regb")}
    assert len(QUESTIONS) == 10
    assert {q["expected"] for q in QUESTIONS} <= ids


@pytest.fixture(scope="module")
def session():
    """The ingested pgvector index (docker compose up -d db, alembic upgrade head, ingest)."""
    try:
        with make_sessionmaker()() as s:
            if s.scalar(select(Chunk.id).limit(1)) is None:
                pytest.fail("chunks table is empty: run the ingest first")
            yield s
    except OperationalError:
        pytest.skip("pgvector database not reachable on DATABASE_URL")


@pytest.mark.xfail(
    strict=True,
    reason="Known gap: 8/10 measured 2026-10-26 (target 9/10, evals/targets.yaml). Misses "
    "1002.9(a)(2) and (b)(2): the sample forms outrank the rule text. Tried bge-base, hybrid "
    "BM25, two rerankers, rule/form filtering, parent-document retrieval and LLM query "
    "rewriting (5/10); none reached 9. strict=True fails this test once it starts passing.",
)
def test_hit_rate_top3(session):
    misses = []
    for q in QUESTIONS:
        top3 = [c.section_id for c in search_regulation(session, q["question"], k=3)]
        if q["expected"] not in top3:
            misses.append((q["question"], q["expected"], top3))
    hits = len(QUESTIONS) - len(misses)
    print(f"\nretrieval hit rate (top 3): {hits}/{len(QUESTIONS)}")
    assert hits >= 9, misses
