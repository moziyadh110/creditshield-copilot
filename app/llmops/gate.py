"""Release gate and rollback for the notice prompt.

  uv run python -m app.llmops.gate check [VERSION]    exit 1 if VERSION misses a target
  uv run python -m app.llmops.gate check --recorded   the same, on CI's replayed eval runs
  uv run python -m app.llmops.gate promote VERSION    gate, then @previous <- @production <- VERSION
  uv run python -m app.llmops.gate rollback           @production <- @previous

VERSION is a number or an alias (default: candidate). The gate reads the latest
`repeated_runs` full end-to-end eval runs logged for that prompt version and applies
evals/targets.yaml: count targets must hold on every run, rate targets on the mean, report
targets (and gate: false) never block. A model change is a new prompt version (its
model_config), so it goes through the same gate.

CI runs `check --recorded`: the same targets and code, applied to eval runs that replayed
recorded LLM responses. Those are logged to a separate "-recorded" experiment, so a replay
can never count towards a real promotion.
"""

import argparse
import operator
import statistics
import sys
from pathlib import Path

import mlflow
import mlflow.genai
import yaml
from mlflow.exceptions import MlflowException

from app.config import ROOT_DIR, get_settings

TARGETS_PATH = ROOT_DIR / "evals" / "targets.yaml"
OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq}


def load_targets(path: Path = TARGETS_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def eval_experiment(recorded: bool = False) -> str:
    name = get_settings().mlflow_eval_experiment
    return f"{name}-recorded" if recorded else name


def gating(targets: dict) -> dict[str, dict]:
    return {
        name: t
        for name, t in targets["targets"].items()
        if t["kind"] in ("count", "rate") and t.get("gate", True)
    }


def passes(target: dict, value: float | None) -> bool:
    return value is not None and OPS[target["op"]](value, target["value"])


def check_targets(runs: list[dict[str, float | None]], targets: dict) -> list[str]:
    """The targets missed across `runs` (one metrics dict per eval run). Empty = release."""
    needed = targets["repeated_runs"]
    if len(runs) < needed:
        return [f"needs {needed} full end-to-end eval runs, found {len(runs)}"]
    misses = []
    for name, t in gating(targets).items():
        values = [run.get(name) for run in runs]
        rule = f"{t['op']} {t['value']:g}"
        if any(v is None for v in values):
            misses.append(f"{name}: not measured on every run")
        elif t["kind"] == "count":
            bad = [f"{v:g} (run {i})" for i, v in enumerate(values, 1) if not passes(t, v)]
            if bad:
                misses.append(f"{name}: {', '.join(bad)}, target {rule} on every run")
        elif not passes(t, mean := statistics.fmean(values)):
            misses.append(f"{name}: mean {mean:.3g}, target {rule}")
    return misses


def resolve_version(ref: str) -> int:
    name = get_settings().prompt_name
    uri = f"prompts:/{name}/{ref}" if ref.isdigit() else f"prompts:/{name}@{ref}"
    return int(mlflow.genai.load_prompt(uri, link_to_model=False).version)


def eval_runs(version: int, n: int, recorded: bool = False) -> list[dict[str, float | None]]:
    s = get_settings()
    df = mlflow.search_runs(
        experiment_names=[eval_experiment(recorded)],
        filter_string=(
            f"tags.prompt_name = '{s.prompt_name}' and tags.prompt_version = '{version}' "
            "and tags.eval_mode = 'e2e' and tags.full_set = 'true' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=n,
    )
    cols = [c for c in df.columns if c.startswith("metrics.")]
    return [
        {c.removeprefix("metrics."): (None if v != v else float(v)) for c, v in row.items()}
        for row in df[cols].to_dict("records")
    ]


def check(version: int, recorded: bool = False) -> list[str]:
    targets = load_targets()
    misses = check_targets(eval_runs(version, targets["repeated_runs"], recorded), targets)
    label = f"prompt version {version}" + (" (recorded responses)" if recorded else "")
    print(f"{label}: " + ("meets every target" if not misses else "BLOCKED"))
    for m in misses:
        print(f"  missed {m}")
    return misses


def _alias_version(alias: str) -> int | None:
    try:
        return resolve_version(alias)
    except MlflowException:
        return None


def promote(version: int) -> bool:
    if check(version):
        return False
    name = get_settings().prompt_name
    current = _alias_version("production")
    if current == version:
        print(f"version {version} is already @production")
        return True
    if current is not None:
        mlflow.genai.set_prompt_alias(name, alias="previous", version=current)
    mlflow.genai.set_prompt_alias(name, alias="production", version=version)
    print(f"{name}@production -> {version}" + (f" (@previous -> {current})" if current else ""))
    return True


def rollback() -> bool:
    name = get_settings().prompt_name
    previous = _alias_version("previous")
    if previous is None:
        print(f"{name} has no @previous version to roll back to")
        return False
    current = _alias_version("production")
    mlflow.genai.set_prompt_alias(name, alias="production", version=previous)
    print(f"{name}@production -> {previous} (was {current})")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gate")
    sub = parser.add_subparsers(dest="command", required=True)
    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("version", nargs="?", default="candidate")
    check_cmd.add_argument("--recorded", action="store_true", help="CI's replayed eval runs")
    sub.add_parser("promote").add_argument("version")
    sub.add_parser("rollback")
    args = parser.parse_args(argv)
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    if args.command == "rollback":
        return 0 if rollback() else 1
    version = resolve_version(args.version)
    if args.command == "check":
        return 1 if check(version, args.recorded) else 0
    return 0 if promote(version) else 1


if __name__ == "__main__":
    sys.exit(main())
