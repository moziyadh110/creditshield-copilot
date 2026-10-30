"""The public demo: visitors pick one of the 50 eval cases and may replace its applicant note.
Nothing else about a case can be changed, so the demo can only ever word real SHAP reasons."""

import json
import time
from collections import defaultdict, deque
from copy import deepcopy
from functools import cache
from typing import Any

from app.config import get_settings


class UnknownCase(LookupError):
    pass


@cache
def _cases() -> dict[str, dict[str, Any]]:
    path = get_settings().demo_cases_path
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["case_id"]: r for r in rows}


def demo_case(case_id: str, applicant_note: str | None) -> dict[str, Any]:
    """The eval case's input; a given note (even an empty one) replaces the stored note."""
    try:
        case = deepcopy(_cases()[case_id]["case"])
    except KeyError:
        raise UnknownCase(f"unknown demo case {case_id!r}") from None
    if applicant_note is not None:
        case["applicant_note"] = applicant_note.strip() or None
    return case


class RateLimiter:
    """Requests per client in the last minute and the last day.
    ponytail: in-process counters, fine for the single demo container; the daily budget cap is
    the hard limit on spend. Move to Redis if the demo ever runs more than one replica."""

    def __init__(self, per_minute: int, per_day: int) -> None:
        self.per_minute, self.per_day = per_minute, per_day
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        q = self.hits[client]
        while q and q[0] <= now - 86400:
            q.popleft()
        if len(q) >= self.per_day or sum(t > now - 60 for t in q) >= self.per_minute:
            return False
        q.append(now)
        return True
