"""Load evals/cases.jsonl. Rejects duplicate case ids, and duplicate cases: two cases whose
LLM-visible content (decision, product, ranked reason features, applicant note) is the same
would count one input twice."""

import json
from pathlib import Path

CASES_PATH = Path(__file__).parent / "cases.jsonl"
GROUPS = {"typical": 20, "edge": 8, "must_escalate": 7, "injection": 8, "proxy_bait": 7}


class DuplicateCase(ValueError):
    pass


def _content_key(case: dict) -> str:
    reasons = [(r.get("rank"), r.get("feature")) for r in case.get("reasons") or []]
    return json.dumps(
        [case.get("decision"), case.get("product"), reasons, case.get("applicant_note")]
    )


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids, contents = set(), {}
    for row in rows:
        if row["case_id"] in ids:
            raise DuplicateCase(f"duplicate case_id {row['case_id']}")
        ids.add(row["case_id"])
        key = _content_key(row["case"])
        if key in contents:
            raise DuplicateCase(f"{row['case_id']} duplicates {contents[key]}")
        contents[key] = row["case_id"]
    return rows
