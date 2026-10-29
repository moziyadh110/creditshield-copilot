"""Daily monitoring:  uv run python -m app.llmops.monitor [--day YYYY-MM-DD] [--test-alert]

Reads one UTC day of production traces (the live experiment, never the eval one) with
mlflow.search_traces and reports:

  escalation rate        share of requests sent to human review
  check pass rates       per guardrail check, over the drafts that reached the checks
  cost and latency       p50 / p95 per notice (cost over requests that called the LLM)
  provider error rate    share of LLM-bound requests that failed on a provider error or timeout
  reason-mix PSI         today's reason features against data/denials.parquet

PSI follows CreditShield's approach, reimplemented here (no code is copied): shares per reason
feature, a 1e-4 floor, 0.10 = investigate, 0.25 = alert.

Alerts: escalation rate above 2x baseline (the median of the last 7 days' rates; until 7 days
exist, the rate denials.parquet escalates by design: MANUAL_REVIEW cases plus escalate-only
reasons), any protected-attribute block, the daily budget reached (only when DAILY_BUDGET_USD
is set) and PSI >= 0.25. Each alert is a structured log line, an `alert.<name>` tag on the
day's MLflow run, and a POST to ALERT_WEBHOOK_URL when that is set. --test-alert fires one
test alert through the same path.
"""

import argparse
import json
import logging
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time

import httpx
import mlflow
import numpy as np
import pandas as pd

from app.config import get_settings
from app.logs import setup_logging
from app.repositories.cases import load_denials
from app.repositories.reason_codes import get_reason_catalog

log = logging.getLogger("copilot.monitor")

PSI_FLOOR = 1e-4
PSI_INVESTIGATE, PSI_ALERT = 0.10, 0.25
BASELINE_DAYS = 7
ESCALATION_ALERT_FACTOR = 2.0


@dataclass
class Alert:
    name: str
    message: str


def reason_shares(feature_lists: list[list[str]]) -> dict[str, float]:
    counts = Counter(f for features in feature_lists for f in features)
    total = sum(counts.values())
    return {f: n / total for f, n in counts.items()} if total else {}


def psi(reference: dict[str, float], current: dict[str, float]) -> float:
    """Population stability index over the union of features, each share floored at 1e-4."""
    total = 0.0
    for f in reference.keys() | current.keys():
        e = max(reference.get(f, 0.0), PSI_FLOOR)
        a = max(current.get(f, 0.0), PSI_FLOOR)
        total += (a - e) * math.log(a / e)
    return total


def psi_status(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "alert" if value >= PSI_ALERT else "investigate" if value >= PSI_INVESTIGATE else "ok"


def reference() -> tuple[dict[str, float], float]:
    """(reason shares, by-design escalation rate) of data/denials.parquet."""
    cases = list(load_denials().values())
    catalog = get_reason_catalog()
    by_design = sum(
        c.decision == "MANUAL_REVIEW"
        or any(catalog.features[r.feature].escalate for r in c.reasons)
        for c in cases
    )
    return reason_shares([[r.feature for r in c.reasons] for c in cases]), by_design / len(cases)


def day_traces(day: date) -> list[dict[str, str | float]]:
    """Tags (plus latency_ms) of each production request traced on `day` (UTC)."""
    s = get_settings()
    exp = mlflow.get_experiment_by_name(s.mlflow_experiment)
    if exp is None:
        return []
    start = int(datetime.combine(day, time(), UTC).timestamp() * 1000)
    end = start + 86_400_000
    traces = mlflow.search_traces(
        locations=[exp.experiment_id],
        filter_string=f"trace.timestamp_ms >= {start} AND trace.timestamp_ms < {end}",
        return_type="list",
        include_spans=False,
        max_results=100_000,
    )
    # Only request traces carry an outcome (not, e.g., a rejected cache entry's trace).
    return [
        {**t.info.tags, "latency_ms": float(t.info.execution_duration or 0)}
        for t in traces
        if "outcome" in t.info.tags
    ]


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def daily_metrics(rows: list[dict], ref_shares: dict[str, float]) -> dict[str, float | None]:
    n = len(rows)
    escalated = [r for r in rows if r.get("outcome") == "escalated"]
    llm_bound = [r for r in rows if r.get("decision") == "DENY" and r.get("cache_hit") != "true"
                 and r.get("budget_blocked") != "true"]  # fmt: skip
    checks = [json.loads(r["checks"]) for r in rows if r.get("checks")]
    names = sorted({name for c in checks for name in c})
    costs = [float(r["cost_usd"]) for r in rows if int(r.get("llm_calls", 0) or 0) > 0]
    latencies = [float(r["latency_ms"]) for r in rows]
    features = [r["reason_features"].split(",") for r in rows if r.get("reason_features")]
    current = reason_shares(features)
    metrics: dict[str, float | None] = {
        "requests": n,
        "cache_hits": sum(r.get("cache_hit") == "true" for r in rows),
        "escalation_rate": len(escalated) / n if n else None,
        "protected_blocks": sum(c.get("protected_attributes") is False for c in checks),
        "budget_blocked": sum(r.get("budget_blocked") == "true" for r in rows),
        "provider_error_rate": (
            sum(r.get("provider_error") == "true" for r in llm_bound) / len(llm_bound)
            if llm_bound
            else None
        ),
        "cost_usd_total": sum(costs),
        "cost_usd_p50": _pct(costs, 50),
        "cost_usd_p95": _pct(costs, 95),
        "latency_ms_p50": _pct(latencies, 50),
        "latency_ms_p95": _pct(latencies, 95),
        "reason_psi": psi(ref_shares, current) if current else None,
    }
    for name in names:
        ran = [c[name] for c in checks if name in c]
        metrics[f"pass_rate.{name}"] = sum(ran) / len(ran)
    return metrics


def history() -> pd.DataFrame:
    """One row per monitored day (the latest run wins), indexed by ISO day, oldest first:
    the daily metrics, escalation_baseline, and an `alerts` column joining that day's alert
    messages. Empty if `make monitor` has never run."""
    exp = mlflow.get_experiment_by_name(get_settings().mlflow_monitor_experiment)
    if exp is None:
        return pd.DataFrame()
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    if runs.empty or "tags.day" not in runs:
        return pd.DataFrame()
    runs = runs.dropna(subset=["tags.day"]).drop_duplicates("tags.day", keep="last")
    alert_cols = [c for c in runs if c.startswith("tags.alert.")]
    out = runs[[c for c in runs if c.startswith("metrics.")]]
    out.columns = [c.removeprefix("metrics.") for c in out.columns]
    out = out.assign(
        alerts=runs[alert_cols].apply(lambda r: "; ".join(r.dropna()), axis=1) if alert_cols else ""
    )
    return out.set_index(runs["tags.day"].rename("day")).sort_index()


def escalation_baseline(day: date, by_design: float) -> tuple[float, str]:
    """Median escalation rate of the last 7 monitored days before `day`; until 7 exist, the
    rate denials.parquet escalates by design."""
    runs = history()
    rates = runs["escalation_rate"].dropna() if "escalation_rate" in runs else []
    recent = [float(r) for d, r in rates.items() if d < day.isoformat()][-BASELINE_DAYS:]
    if len(recent) < BASELINE_DAYS:
        return by_design, f"denials.parquet by design ({len(recent)} of 7 days monitored)"
    return statistics.median(recent), "median of the last 7 days"


def find_alerts(m: dict, baseline: float, budget: float | None) -> list[Alert]:
    alerts = []
    rate = m["escalation_rate"]
    if rate is not None and rate > ESCALATION_ALERT_FACTOR * baseline:
        msg = f"escalation rate {rate:.1%} is above 2x baseline {baseline:.1%}"
        alerts.append(Alert("escalation_rate", msg))
    if m["protected_blocks"]:
        msg = f"{m['protected_blocks']} draft(s) blocked for a protected attribute"
        alerts.append(Alert("protected_block", msg))
    if budget is not None and m["budget_blocked"]:
        msg = f"daily LLM budget ${budget:.2f} reached: {m['budget_blocked']} denial(s) to review"
        alerts.append(Alert("budget", msg))
    if psi_status(m["reason_psi"]) == "alert":
        alerts.append(Alert("reason_psi", f"reason-mix PSI {m['reason_psi']:.3f} >= {PSI_ALERT}"))
    return alerts


def fire(alert: Alert, day: date) -> None:
    payload = {"alert": alert.name, "message": alert.message, "day": day.isoformat(),
               "source": "creditshield-copilot monitor"}  # fmt: skip
    log.warning(
        "alert", extra={"alert": alert.name, "detail": alert.message, "day": payload["day"]}
    )
    mlflow.set_tag(f"alert.{alert.name}", alert.message)
    url = get_settings().alert_webhook_url
    if url is not None:
        try:
            httpx.post(url.get_secret_value(), json=payload, timeout=5).raise_for_status()
        except httpx.HTTPError as e:  # a dead webhook must not hide the alert
            log.error("alert webhook failed", extra={"alert": alert.name, "error": str(e)})


def report(day: date, m: dict, baseline: float, baseline_source: str, alerts: list[Alert]) -> str:
    def fmt(name, v):
        if v is None:
            return "n/a"
        if name.endswith("_rate") or name.startswith("pass_rate."):
            return f"{v:.1%}"
        if name.startswith("cost_usd"):
            return f"${v:.5f}"
        if name.startswith("latency_ms"):
            return f"{v / 1000:.2f} s"
        return f"{v:.3f}" if isinstance(v, float) and not v.is_integer() else f"{v:g}"

    lines = [f"Monitoring for {day.isoformat()} (UTC)", ""]
    lines += [f"  {k:<36} {fmt(k, v)}" for k, v in m.items()]
    lines += [
        "",
        f"  escalation baseline: {baseline:.1%} ({baseline_source}); alert above "
        f"{ESCALATION_ALERT_FACTOR * baseline:.1%}",
        f"  reason-mix PSI status: {psi_status(m['reason_psi'])}",
        "",
        f"Alerts: {len(alerts)}",
    ]
    lines += [f"  ALERT {a.name}: {a.message}" for a in alerts]
    return "\n".join(lines)


def run(day: date, test_alert: bool = False) -> tuple[dict, list[Alert]]:
    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    ref_shares, by_design = reference()
    m = daily_metrics(day_traces(day), ref_shares)
    baseline, source = escalation_baseline(day, by_design)
    alerts = find_alerts(m, baseline, s.daily_budget_usd)
    if test_alert:
        alerts.append(Alert("test", "test alert from make monitor --test-alert"))
    mlflow.set_experiment(s.mlflow_monitor_experiment)
    with mlflow.start_run(run_name=f"monitor-{day.isoformat()}"):
        mlflow.set_tags({"day": day.isoformat(), "baseline_source": source,
                         "reason_psi_status": psi_status(m["reason_psi"])})  # fmt: skip
        mlflow.log_metrics({k: float(v) for k, v in m.items() if v is not None})
        mlflow.log_metrics({"escalation_baseline": baseline, "alerts": len(alerts)})
        for a in alerts:
            fire(a, day)
    print(report(day, m, baseline, source, alerts))
    return m, alerts


def main() -> None:
    parser = argparse.ArgumentParser(prog="monitor")
    parser.add_argument("--day", type=date.fromisoformat, default=datetime.now(UTC).date())
    parser.add_argument("--test-alert", action="store_true", help="fire one test alert")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")  # MLflow prints emoji; Windows pipes are cp1252
    setup_logging()
    run(args.day, args.test_alert)


if __name__ == "__main__":
    main()
