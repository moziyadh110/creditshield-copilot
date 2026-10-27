"""Release gate and rollback against the temporary MLflow store, with eval runs faked."""

import mlflow.genai
import pytest

from app.config import get_settings
from app.llmops import gate

TARGETS = gate.load_targets()
GOOD = {
    "faithfulness": 1.0,
    "invented_reasons": 0,
    "protected_attribute_leaks": 0,
    "injection_success": 0,
    "correct_escalation": 1.0,
    "reading_level": 6.5,
    "clarity": 4.4,
}


def test_good_runs_pass():
    assert gate.check_targets([GOOD] * 3, TARGETS) == []


def test_too_few_runs_block():
    assert "needs 3" in gate.check_targets([GOOD], TARGETS)[0]


def test_count_target_must_hold_on_every_run():
    misses = gate.check_targets([GOOD, GOOD, {**GOOD, "protected_attribute_leaks": 1}], TARGETS)
    assert misses == ["protected_attribute_leaks: 1 (run 3), target == 0 on every run"]


def test_rate_target_is_judged_on_the_mean():
    runs = [GOOD, GOOD, {**GOOD, "faithfulness": 0.95}]  # mean 0.983 >= 0.98
    assert gate.check_targets(runs, TARGETS) == []
    runs = [GOOD, {**GOOD, "faithfulness": 0.95}, {**GOOD, "faithfulness": 0.95}]
    assert gate.check_targets(runs, TARGETS)[0].startswith("faithfulness: mean 0.967")


def test_unmeasured_target_blocks_and_report_targets_never_do():
    runs = [{**GOOD, "clarity": None, "latency_ms": 99999}] * 3
    assert gate.check_targets(runs, TARGETS) == ["clarity: not measured on every run"]


@pytest.fixture
def two_versions(monkeypatch):
    """(old, new): two versions of a throwaway prompt in the temporary store, @production -> old.
    Its own name, so the real notice-reasons history other tests register stays untouched."""
    settings = get_settings().model_copy(update={"prompt_name": "gate-test"})
    monkeypatch.setattr(gate, "get_settings", lambda: settings)
    old, new = (
        int(mlflow.genai.register_prompt(name="gate-test", template=t).version)
        for t in ("old text", "new text")
    )
    mlflow.genai.set_prompt_alias("gate-test", alias="production", version=old)
    return old, new


def test_promote_refuses_a_version_that_misses_a_target(two_versions, monkeypatch):
    old, new = two_versions
    monkeypatch.setattr(gate, "eval_runs", lambda v, n: [{**GOOD, "clarity": 3.1}] * 3)
    assert gate.promote(new) is False
    assert gate.resolve_version("production") == old


def test_promote_then_rollback(two_versions, monkeypatch):
    old, new = two_versions
    monkeypatch.setattr(gate, "eval_runs", lambda v, n: [GOOD] * 3)
    assert gate.promote(new) is True
    assert gate.resolve_version("production") == new
    assert gate.resolve_version("previous") == old
    assert gate.rollback() is True
    assert gate.resolve_version("production") == old
