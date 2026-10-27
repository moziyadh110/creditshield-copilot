"""copilot draft --applicant <id>: draft one notice and print it with its check report."""

import argparse
import asyncio

from app.services.notices import draft_notice


def _print_result(r) -> None:
    print("=" * 72)
    if r.notice:
        print(r.notice)
    else:
        print(f"No notice sent. Escalated to human review: {r.escalation_reason}")
    print("=" * 72)
    print("Checks:")
    for c in r.checks or []:
        print(
            f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f": {c.detail}" if c.detail else "")
        )
    if not r.checks:
        print("  (none run)")
    print(f"Outcome: {r.outcome}  |  prompt v{r.prompt_version}  |  model {r.model}")
    u = r.usage
    print(
        f"LLM calls {u.calls}  |  tokens {u.prompt_tokens} in / {u.completion_tokens} out  |  "
        f"cost ${u.cost_usd:.5f}  |  {r.latency_ms} ms"
    )
    print(f"Run log id {r.run_id}  |  MLflow trace {r.trace_id}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="copilot")
    sub = parser.add_subparsers(dest="command", required=True)
    draft = sub.add_parser("draft", help="draft the adverse action notice for one applicant")
    draft.add_argument("--applicant", type=int, required=True)
    args = parser.parse_args()
    if args.command == "draft":
        _print_result(asyncio.run(draft_notice(args.applicant)))


if __name__ == "__main__":
    main()
