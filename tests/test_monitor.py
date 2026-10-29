"""The daily monitor's metrics and alert rules, on hand-made trace tags (no MLflow server)."""

import json
import math

import pytest

from app.llmops import monitor as mon


def trace(outcome="sent", decision="DENY", features="late_payment_rate", checks=None, **tags):
    checks = {"faithfulness": True, "protected_attributes": True} if checks is None else checks
    return {
        "outcome": outcome,
        "decision": decision,
        "reason_features": features,
        "checks": json.dumps(checks),
        "llm_calls": "1",
        "cost_usd": "0.0005",
        "provider_error": "false",
        "latency_ms": 2000.0,
        **tags,
    }


def test_psi_is_zero_for_the_same_mix_and_floors_missing_features():
    ref = {"a": 0.5, "b": 0.5}
    assert mon.psi(ref, ref) == 0
    # "b" vanished: its share is floored at 1e-4 instead of dividing by zero
    expected = (1 - 0.5) * math.log(1 / 0.5) + (1e-4 - 0.5) * math.log(1e-4 / 0.5)
    assert mon.psi(ref, {"a": 1.0}) == pytest.approx(expected)
    assert mon.psi_status(0.05) == "ok"
    assert mon.psi_status(0.10) == "investigate"
    assert mon.psi_status(0.25) == "alert"


def test_daily_metrics():
    rows = [
        trace(),
        trace(outcome="escalated", provider_error="true", llm_calls="0", cost_usd="0"),
        trace(checks={"faithfulness": True, "protected_attributes": False}, outcome="escalated"),
        trace(decision="MANUAL_REVIEW", outcome="escalated", checks={}, llm_calls="0"),
        trace(cache_hit="true", llm_calls="0", cost_usd="0", latency_ms=100.0),
    ]
    m = mon.daily_metrics(rows, {"late_payment_rate": 1.0})
    assert m["requests"] == 5 and m["cache_hits"] == 1
    assert m["escalation_rate"] == 3 / 5
    assert m["protected_blocks"] == 1
    assert m["provider_error_rate"] == 1 / 3  # DENY, not cached: rows 1-3
    assert m["pass_rate.protected_attributes"] == 3 / 4  # the empty checks dict ran nothing
    assert m["cost_usd_p50"] == 0.0005  # only requests that called the LLM
    assert m["reason_psi"] == 0


def test_alert_rules():
    quiet = {"escalation_rate": 0.4, "protected_blocks": 0, "budget_blocked": 3, "reason_psi": 0.1}
    assert mon.find_alerts(quiet, baseline=0.25, budget=None) == []  # 40% <= 2 x 25%
    loud = {"escalation_rate": 0.6, "protected_blocks": 1, "budget_blocked": 3, "reason_psi": 0.3}
    names = [a.name for a in mon.find_alerts(loud, baseline=0.25, budget=1.0)]
    assert names == ["escalation_rate", "protected_block", "budget", "reason_psi"]


def test_reference_escalation_rate_is_what_denials_parquet_escalates_by_design():
    shares, by_design = mon.reference()
    assert sum(shares.values()) == pytest.approx(1)
    assert by_design == pytest.approx(0.25)  # 50 MANUAL_REVIEW of 200, no cnt_children reason
