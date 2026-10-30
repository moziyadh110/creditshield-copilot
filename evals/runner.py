"""Run the 50-case eval:  uv run python -m evals.runner --prompt candidate [--llm recorded]

Each mode is one MLflow run in the eval experiment: every case is drafted into a trace
linked to the prompt version it loaded, then mlflow.genai.evaluate scores those traces with
the code checks and the clarity judge as @scorer functions. The run carries the target
metrics, targets.yaml and the pass/fail per target.
Writes evals/results/<mode>/v<version>-<stamp>.{json,md,cases.jsonl}.

Modes:
  e2e         the whole pipeline, retrieval from pgvector (needs the database)
  generation  the same cases with fixed retrieved context (FIXED_SECTIONS), so drafting is
              scored without retrieval noise

The pipeline's guardrails stay on in both modes: the eval measures what would be sent.

LLM calls (--llm):
  live      real API calls (default). A full-set live run also records every LLM call to
            evals/recorded/v<version>/<model>/<mode>-<stamp>.jsonl (see evals/recording.py)
  recorded  no API calls: replays the latest `repeated_runs` recordings for the version, one
            eval run each, into the separate "-recorded" experiment and
            evals/results/recorded/. Fails with "re-record needed" if none match. CI uses this
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import mlflow
import mlflow.genai
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from app.config import DATA_DIR, get_settings
from app.llmops.gate import eval_experiment, load_targets, passes, resolve_version
from app.llmops.tracing import setup_tracing
from app.retrieval.chunking import chunk_regb
from app.services.notices import run_notice
from evals import metrics as m
from evals.cases import load_cases
from evals.judge import JUDGE_MODEL_CONFIG, judge_clarity
from evals.recording import Live, Replay, recording_dir, recordings

RESULTS_DIR = Path(__file__).parent / "results"
NOTICE_DATE = date(2026, 10, 28)  # fixed, so the LLM input (and CI recordings) never drift
FIXED_SECTIONS = ("1002.9(a)(2)", "1002.9(b)(2)", "1002-appC-C-1")


def fixed_search():
    chunks = [c for c in chunk_regb(DATA_DIR / "regb") if c.section_id in FIXED_SECTIONS]
    assert len(chunks) == len(FIXED_SECTIONS)
    return lambda query: chunks


def _error_output(e: Exception) -> dict:
    return {"outcome": "error", "escalation_reason": f"{type(e).__name__}: {e}", "draft": None,
            "notice": None, "checks": [], "retrieved": [], "llm_calls": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "cost_usd": 0.0, "latency_ms": 0, "model": ""}  # fmt: skip


def make_scorers(records: dict[str, dict], judge: bool, llm: Live | Replay):
    """Scorers look the case up by case_id: the trace carries only what the LLM saw."""

    def code(name, fn):
        @scorer(name=name)
        def _score(expectations):
            return fn(records[expectations["case_id"]])

        return _score

    scorers = [
        code("faithfulness", m.faithful),
        code("invented_reasons", m.invented),
        code("protected_leak", m.leaked),
        code("injection_followed", m.injection_followed),
        code("correct_escalation", m.escalated_correctly),
        code("reading_grade", m.grade),
    ]

    @scorer(name="clarity")
    def clarity(expectations):
        rec = records[expectations["case_id"]]
        if not m.is_sent(rec):
            return None
        score, rationale, cost = asyncio.run(
            _judge(llm, expectations["case_id"], rec["output"]["draft"]["paragraph"])
        )
        rec |= {"clarity": score, "clarity_rationale": rationale, "judge_cost_usd": cost}
        return None if score is None else Feedback(value=score, rationale=rationale)

    return scorers + [clarity] if judge else scorers


async def _judge(llm: Live | Replay, case_id: str, paragraph: str):
    provider = llm.judge(case_id)
    try:
        return await judge_clarity(paragraph, provider)
    finally:
        await provider.aclose()


async def draft_all(
    cases: list[dict], prompt_uri: str, search, workers: int, llm: Live | Replay
) -> dict:
    """Run the pipeline on every case, `workers` at a time, on one event loop. One live
    provider (one HTTP client) for all of them, closed before the loop ends."""
    limit = asyncio.Semaphore(workers)

    async def one(row):
        async with limit:
            try:
                r = await run_notice(
                    row["case"],
                    NOTICE_DATE,
                    provider=llm.pipeline(row["case_id"]),
                    search=search,
                    prompt_uri=prompt_uri,
                )
                return row["case_id"], r.summary(), r.trace_id
            except Exception as e:  # recorded as a failed case, never silently dropped
                return row["case_id"], _error_output(e), None

    try:
        done = await asyncio.gather(*map(one, cases))
    finally:
        await llm.aclose()
    return {cid: (out, tid) for cid, out, tid in done}


def run_mode(
    mode: str,
    version: int,
    cases: list[dict],
    *,
    judge: bool,
    full_set: bool,
    workers: int,
    replay: Replay | None = None,
) -> dict:
    """Drafting runs here, not as evaluate's predict_fn: inside evaluate's worker threads
    MLflow 3.15 deadlocked on lazy imports (2026-10-27). evaluate then scores the traces."""
    s = get_settings()
    prompt_uri = f"prompts:/{s.prompt_name}/{version}"
    search = fixed_search() if mode == "generation" else None
    llm = replay or Live(record=full_set)

    targets = load_targets()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=f"eval-{mode}-v{version}-{stamp}") as run:
        mlflow.set_tags(
            {
                "eval": "true",
                "eval_mode": mode,
                "prompt_name": s.prompt_name,
                "prompt_version": str(version),
                "full_set": str(full_set).lower(),
                "judge": str(judge).lower(),
                "llm": "recorded" if replay else "live",
            }
        )
        mlflow.log_params(
            {"n_cases": len(cases), "notice_date": NOTICE_DATE.isoformat(),
             "judge_model": JUDGE_MODEL_CONFIG["model"] if judge else "none",
             "fixed_sections": ",".join(FIXED_SECTIONS) if search else "pgvector"}
        )  # fmt: skip
        if replay:
            mlflow.log_param("recording", replay.path.name)
        drafted = asyncio.run(draft_all(cases, prompt_uri, search, workers, llm))
        mlflow.flush_trace_async_logging()
        records = {row["case_id"]: {**row, "output": drafted[row["case_id"]][0]} for row in cases}
        data = []
        for row in cases:
            trace_id = drafted[row["case_id"]][1]
            trace = mlflow.get_trace(trace_id) if trace_id else None
            if trace is None:  # a crashed case: still in the metrics, just not scored in MLflow
                continue
            exp = {"case_id": row["case_id"], "group": row["group"], "expect": row["expect"]}
            data.append({"trace": trace, "expectations": exp})
        mlflow.genai.evaluate(data=data, scorers=make_scorers(records, judge, llm))

        records = list(records.values())
        metrics = m.aggregate(records)
        models = sorted({r["output"]["model"] for r in records if r["output"]["llm_calls"]})
        mlflow.set_tag("model", ",".join(models))
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, int | float)})
        table = [
            {
                "target": name,
                "value": metrics.get(name),
                "rule": f"{t['op']} {t['value']:g}" if "op" in t else "report",
                "passed": passes(t, metrics.get(name)) if "op" in t else None,
                "gates": t["kind"] != "report" and t.get("gate", True),
            }
            for name, t in targets["targets"].items()
            if name in metrics
        ]
        result = {
            "run_id": run.info.run_id,
            "mode": mode,
            "prompt_version": version,
            "models": models,
            "full_set": full_set,
            "judge": judge,
            "created": stamp,
            "metrics": metrics,
            "targets": table,
        }
        if replay:
            result["replay"] = {"missing": replay.missing, "drift": replay.drift}
            mlflow.log_metrics(
                {"replay_missing": len(replay.missing), "replay_drift": len(replay.drift)}
            )
        elif full_set:
            llm.write(recording_dir(version, _prompt_model(prompt_uri)) / f"{mode}-{stamp}.jsonl")
        out_dir = RESULTS_DIR / "recorded" / mode if replay else RESULTS_DIR / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / f"v{version}-{stamp}"
        files = {
            base.with_suffix(".json"): json.dumps(result, indent=2),
            base.with_suffix(".md"): results_markdown(result),
            Path(f"{base}.cases.jsonl"): "".join(json.dumps(r) + "\n" for r in records),
        }
        for path, text in files.items():
            path.write_text(text, encoding="utf-8")
            mlflow.log_artifact(str(path))
        mlflow.log_dict(targets, "targets.yaml")
    return result


def _prompt_model(prompt_uri: str) -> str:
    prompt = mlflow.genai.load_prompt(prompt_uri, link_to_model=False)
    return (prompt.model_config or {}).get("model", get_settings().llm_model)


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def results_markdown(r: dict) -> str:
    met = r["metrics"]
    lines = [
        f"# Eval: prompt v{r['prompt_version']}, {r['mode']}",
        "",
        f"Run `{r['run_id']}` · {r['created']} UTC · model {', '.join(r['models']) or 'n/a'}"
        f" · {met['n_cases']} cases{'' if r['full_set'] else ' (subset)'}"
        f" · judge {'on' if r['judge'] else 'off'}",
        "",
        "| Target | Value | Rule | Result |",
        "| --- | --- | --- | --- |",
    ]
    for t in r["targets"]:
        result = "report" if t["passed"] is None else ("pass" if t["passed"] else "**MISS**")
        if t["passed"] is not None and not t["gates"]:
            result += " (not gating)"
        lines.append(f"| {t['target']} | {_fmt(t['value'])} | {t['rule']} | {result} |")
    lines += ["", "| Group | Outcomes |", "| --- | --- |"]
    for g, counts in met["outcomes_by_group"].items():
        lines.append(f"| {g} | {', '.join(f'{k} {v}' for k, v in sorted(counts.items()))} |")
    failed = met["checks_failed_in_final_draft"]
    lines += [
        "",
        f"Sent rate on cases expected to send: {_fmt(met['sent_rate_expected_send'])}",
        "Checks failed on the final draft: "
        + (", ".join(f"{k} {v}" for k, v in sorted(failed.items())) or "none"),
        f"Total cost: ${met['total_cost_usd']:.4f} pipeline + ${met['judge_cost_usd']:.4f} judge",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals.runner")
    parser.add_argument("--prompt", default="candidate", help="prompt version number or alias")
    parser.add_argument("--mode", choices=["e2e", "generation", "both"], default="both")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--limit", type=int, help="first N cases only (smoke test, never gates)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--llm", choices=["live", "recorded"], default="live")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", str(args.workers))
    setup_tracing()
    recorded = args.llm == "recorded"
    mlflow.set_experiment(eval_experiment(recorded))
    version = resolve_version(args.prompt)
    cases = load_cases()
    full_set = args.limit is None
    if not full_set:
        cases = cases[: args.limit]
    opts = {"judge": not args.no_judge, "full_set": full_set, "workers": args.workers}
    failed = False
    for mode in ["e2e", "generation"] if args.mode == "both" else [args.mode]:
        if not recorded:
            print(results_markdown(run_mode(mode, version, cases, **opts)))
            continue
        model = _prompt_model(f"prompts:/{get_settings().prompt_name}/{version}")
        paths = recordings(version, model, mode)[-load_targets()["repeated_runs"] :]
        if not paths:
            where = recording_dir(version, model)
            print(f"re-record needed: no {mode} recordings for prompt v{version} in {where}")
            failed = True
        for path in paths:
            replay = Replay(path)
            print(results_markdown(run_mode(mode, version, cases, **opts, replay=replay)))
            print(f"replayed {path.name}: {len(replay.drift)} request(s) differ from it")
            if replay.missing:
                print(f"re-record needed: {len(replay.missing)} call(s) missing from {path.name}")
                failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
