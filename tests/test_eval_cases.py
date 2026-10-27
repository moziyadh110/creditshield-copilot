"""The eval set: right group sizes, reproducible from denials.parquet, duplicates rejected."""

import collections
import json

import pytest

from app.schemas import DenialCase
from evals.build_cases import build
from evals.cases import CASES_PATH, GROUPS, DuplicateCase, load_cases


def test_groups_match_the_plan():
    rows = load_cases()
    assert collections.Counter(r["group"] for r in rows) == GROUPS
    assert all(r["synthetic"] for r in rows if r["group"] in {"injection", "proxy_bait"})


def test_committed_file_is_reproducible():
    assert load_cases() == build()


def test_cases_that_should_get_a_notice_are_valid():
    for r in load_cases():
        if r["expect"] != "escalate":
            DenialCase.model_validate(r["case"])


@pytest.mark.parametrize("field", ["case_id", "content"])
def test_duplicates_are_rejected(tmp_path, field):
    rows = [json.loads(line) for line in CASES_PATH.read_text().splitlines()[:2]]
    if field == "case_id":
        rows[1]["case_id"] = rows[0]["case_id"]
    else:
        rows[1]["case"] = {**rows[0]["case"], "applicant_id": 1}  # same LLM input, other id
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(DuplicateCase):
        load_cases(path)
