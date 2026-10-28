"""Calibrate the clarity judge against hand scores.

  uv run python -m evals.calibration sheet RESULTS.cases.jsonl   write evals/calibration/sheet.csv
  (the user replaces each ? in your_score with 1-5, using the rubric in evals/judge.py)
  uv run python -m evals.calibration compare                     write evals/calibration/report.md

The sheet is blind: it holds the paragraphs only, never the judge's scores. Ten sent
notices are picked deterministically, spread across case groups.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from evals.judge import RUBRIC

DIR = Path(__file__).parent / "calibration"
SHEET = DIR / "sheet.csv"
REPORT = DIR / "report.md"
N = 10
PLACEHOLDER = "?"


def pick(records: list[dict], n: int = N) -> list[dict]:
    """Round-robin over groups in case_id order, so every group with sent notices appears."""
    sent = [r for r in records if r["output"]["outcome"] in ("sent", "revised")]
    groups: dict[str, list[dict]] = {}
    for r in sorted(sent, key=lambda r: r["case_id"]):
        groups.setdefault(r["group"], []).append(r)
    picked: list[dict] = []
    while len(picked) < n and any(groups.values()):
        for g in sorted(groups):
            if groups[g] and len(picked) < n:
                picked.append(groups[g].pop(0))
    return picked


def write_sheet(results: Path) -> None:
    records = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    DIR.mkdir(exist_ok=True)
    with SHEET.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Score first and a ? placeholder, so filling it in never touches the quoted paragraph.
        w.writerow(["case_id", "your_score", "your_note", "source", "paragraph"])
        for r in pick(records):
            w.writerow(
                [r["case_id"], PLACEHOLDER, "", results.name, r["output"]["draft"]["paragraph"]]
            )
    (DIR / "rubric.md").write_text("# Clarity rubric (same text the judge gets)\n\n```\n"
                                   + RUBRIC + "\n```\n", encoding="utf-8")  # fmt: skip
    print(f"wrote {SHEET}: replace each ? in your_score with 1-5, then run compare")


def agreement(pairs: list[tuple[int, int]]) -> dict[str, float]:
    """pairs of (human, judge) scores."""
    return {
        "n": len(pairs),
        "exact": sum(h == j for h, j in pairs) / len(pairs),
        "within_1": sum(abs(h - j) <= 1 for h, j in pairs) / len(pairs),
        "mean_abs_diff": statistics.fmean(abs(h - j) for h, j in pairs),
        "human_mean": statistics.fmean(h for h, _ in pairs),
        "judge_mean": statistics.fmean(j for _, j in pairs),
    }


def compare() -> None:
    rows = list(csv.DictReader(SHEET.open(encoding="utf-8")))
    missing = [r["case_id"] for r in rows if r["your_score"].strip() in ("", PLACEHOLDER)]
    if missing:
        raise SystemExit(f"no hand score yet for {', '.join(missing)}")
    judged = {}
    for source in {r["source"] for r in rows}:
        path = next(Path(__file__).parent.glob(f"results/*/{source}"))
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            judged[rec["case_id"]] = rec.get("clarity")
    pairs, lines = [], []
    for r in rows:
        human, judge = int(r["your_score"]), judged.get(r["case_id"])
        if judge is None:
            raise SystemExit(f"{r['case_id']} has no judge score in {r['source']}")
        pairs.append((human, judge))
        lines.append(f"| {r['case_id']} | {human} | {judge} | {r['your_note']} |")
    a = agreement(pairs)
    REPORT.write_text(
        "\n".join(
            [
                "# Judge calibration",
                "",
                f"{a['n']} sent notices scored by hand (blind) and by the judge "
                f"(deepseek-flash, evals/judge.py), source `{rows[0]['source']}`.",
                "",
                f"- Exact agreement: {a['exact']:.0%}",
                f"- Within 1 point: {a['within_1']:.0%}",
                f"- Mean absolute difference: {a['mean_abs_diff']:.2f}",
                f"- Mean score: hand {a['human_mean']:.2f}, judge {a['judge_mean']:.2f}",
                "",
                "| Case | Hand | Judge | Note |",
                "| --- | --- | --- | --- |",
                *lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(REPORT.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals.calibration")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sheet").add_argument("results", type=Path)
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "sheet":
        write_sheet(args.results)
    else:
        compare()


if __name__ == "__main__":
    main()
