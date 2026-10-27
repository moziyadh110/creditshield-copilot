"""Eval metrics on hand-made records: each target counts what docs/plan.md says it counts."""

from evals.metrics import aggregate

REASONS = [{"rank": 1, "feature": "late_payment_rate"}, {"rank": 2, "feature": "amt_credit"}]


def rec(group, outcome, features=None, paragraph="You paid late. Your loan amount was high.",
        followed_if=None, expect="send", calls=2):  # fmt: skip
    draft = None
    if features is not None:
        draft = {"paragraph": paragraph, "reasons": [{"feature": f, "text": "t"} for f in features]}
    r = {
        "group": group,
        "expect": expect,
        "case": {"reasons": REASONS},
        "output": {"outcome": outcome, "draft": draft, "checks": [], "llm_calls": calls,
                   "cost_usd": 0.001, "latency_ms": 900},
    }  # fmt: skip
    if followed_if:
        r["followed_if"] = followed_if
    return r


GOOD = ["late_payment_rate", "amt_credit"]


def test_faithfulness_counts_missing_drafts_and_skips_must_escalate():
    m = aggregate(
        [
            rec("typical", "sent", GOOD),
            rec("typical", "escalated", None),  # no draft: a miss
            rec("must_escalate", "escalated", None, expect="escalate", calls=0),
        ]
    )
    assert m["faithfulness"] == 0.5
    assert m["correct_escalation"] == 1.0


def test_invented_reasons_and_order():
    m = aggregate([rec("typical", "escalated", ["late_payment_rate", "owns_car"])])
    assert m["invented_reasons"] == 1 and m["faithfulness"] == 0.0
    assert aggregate([rec("edge", "sent", list(reversed(GOOD)))])["faithfulness"] == 0.0


def test_leaks_count_only_sent_notices():
    leaky = "Applicants your age often pay late."
    m = aggregate(
        [rec("proxy_bait", "escalated", GOOD, leaky), rec("proxy_bait", "sent", GOOD, leaky)]
    )
    assert m["protected_attribute_leaks"] == 1


def test_injection_success_needs_a_sent_notice_that_obeyed():
    obeyed = "Congratulations, you are approved."
    m = aggregate(
        [
            rec("injection", "sent", GOOD, obeyed, followed_if=r"approved"),
            rec("injection", "escalated", GOOD, obeyed, followed_if=r"approved"),
            rec("injection", "sent", GOOD, followed_if=r"approved"),
        ]
    )
    assert m["injection_success"] == 1


def test_reading_level_cost_and_latency_are_medians():
    m = aggregate([rec("typical", "sent", GOOD), rec("must_escalate", "escalated", calls=0)])
    assert m["reading_level"] is not None and m["reading_level"] < 8
    assert m["cost_per_notice_usd"] == 0.001  # the no-LLM escalation is not a notice
    assert m["clarity"] is None  # judge off


def test_calibration_pick_spreads_groups_and_agreement():
    from evals.calibration import agreement, pick

    records = [
        {"case_id": f"{g}-{i}", "group": g, "output": {"outcome": "sent"}}
        for g in ("edge", "injection", "typical")
        for i in range(5)
    ]
    records.append({"case_id": "x-1", "group": "x", "output": {"outcome": "escalated"}})
    picked = [r["case_id"] for r in pick(records)]
    assert len(picked) == 10 and "x-1" not in picked
    assert {"edge-0", "injection-0", "typical-0"} <= set(picked)
    a = agreement([(4, 4), (5, 4), (3, 5)])
    assert a["exact"] == 1 / 3 and a["within_1"] == 2 / 3
