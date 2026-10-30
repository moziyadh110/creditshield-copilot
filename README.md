# CreditShield Copilot

[![CI](https://github.com/moziyadh110/creditshield-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/moziyadh110/creditshield-copilot/actions/workflows/ci.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)

**The model decides, the LLM only explains, and nothing ships unless every reason traces back
to a SHAP feature.**

When a lender in the US denies credit, it must send an *adverse action notice* that states
the principal reasons (Equal Credit Opportunity Act, Regulation B). CreditShield Copilot
writes the reasons part of that letter for every denial from
[CreditShield](https://github.com/moziyadh110/fair-lend-ml)'s credit model:

- an LLM agent puts the model's top 4 SHAP reasons into plain words,
- seven checks in code must pass before anything is sent,
- anything that fails, or that the system shouldn't word at all, goes to a human review queue.

**[Open the live demo](https://moziyadh110.github.io/creditshield-copilot/)**: pick any of
the 50 eval cases and watch its recorded run go through the pipeline step by step: the
CreditShield decision, routing, what the LLM was given, the Reg B lookup, the draft, the
seven checks, and the letter or the review queue. Real, unedited outputs; no install needed.

[![Demo page: a prompt-injection case, the allow-listed input the LLM saw and the seven checks it passed](docs/demo.png)](https://moziyadh110.github.io/creditshield-copilot/#injection-01)

> **Simplified, not legal advice.** A portfolio project on the public Home Credit dataset,
> with synthetic applicant IDs and a fictional creditor. It has had no compliance review.

**Built with** Python 3.13, LangGraph, DeepSeek (OpenAI-compatible, swappable), pgvector with
local `bge-small` embeddings, Presidio, FastAPI, MLflow 3.15 (prompt registry, tracing,
evaluation), Streamlit, Docker Compose and GitHub Actions.

## Contents

[Example](#example) · [How it works](#how-it-works) · [Results](#results) ·
[Guardrails](#guardrails) · [Quickstart](#quickstart) · [Development](#development) ·
[Design decisions](#design-decisions) · [Limitations](#responsible-ai-and-limitations)

## Example

CreditShield denied applicant #100336. This is everything the LLM saw: the reason features
in rank order, the product and the date. No score, no cutoff, no feature values, no age.

```json
{"product": "Cash loans", "notice_date": "2026-10-29",
 "reasons": [{"rank": 1, "feature": "bureau_debt_ratio"}, {"rank": 2, "feature": "active_loans"},
             {"rank": 3, "feature": "avg_cc_utilization"}, {"rank": 4, "feature": "max_cc_utilization"}],
 "applicant_note": null}
```

The agent returns one sentence per feature plus a paragraph. The faithfulness check compares
the features to the SHAP output in code, the other six checks read the text, and only then is
the paragraph put into the fixed letter (prompt v1, unedited):

```text
CreditShield Lending (demo)
Date: 2026-10-29
To: Applicant #100336

Re: Your application for a cash loan

Thank you for your recent application. We regret that we are unable to approve your
request for a cash loan.

Principal reason(s) for our decision:
Thank you for your application. We could not approve it, and the main reasons are listed
below. The amount you owe on credit accounts with other lenders is high compared with your
credit limits. You have a number of credit accounts that are currently active with other
lenders. Your balances on your credit cards with us are high compared with your credit
limits, and one of your credit cards with us was used at or near its credit limit.

[Consumer reporting agency disclosure (FCRA section 615): not included in this demo.]

NOTICE: The Federal Equal Credit Opportunity Act prohibits creditors from discriminating
against credit applicants on the basis of race, color, religion, ... (fixed ECOA text)
```

The LLM wrote only the paragraph under "Principal reason(s)". The template supplies the action
taken, the ECOA notice and the contact details, so the model can't get the legal text wrong.

## How it works

```mermaid
flowchart LR
    CS[CreditShield<br/>decision + top 4 SHAP features] --> AL[Input allow-list<br/>Presidio masking]
    AL --> AG[LangGraph agent<br/>DeepSeek + Reg B retrieval]
    AG --> CK{7 code checks}
    CK -- pass --> T[Fixed legal template] --> N[Notice sent]
    CK -- fail twice, or a<br/>protected attribute --> Q[(Review queue)]
    AL -- MANUAL_REVIEW,<br/>cnt_children --> Q
    Q --> RP[Review page<br/>edits re-run the checks]
    AG -.-> ML[MLflow<br/>prompt registry · traces · eval · gate · monitoring]
```

1. **Input.** A decision comes from `data/denials.parquet` (200 exported CreditShield
   decisions) or, optionally, the live CreditShield API. An allow-list drops everything the
   LLM doesn't need, and a unit test checks what actually reaches it.
2. **Routing.** Only DENY decisions get a letter. MANUAL_REVIEW cases aren't denials, so they
   go to review with no LLM call. So does any case with `cnt_children` as a reason, because
   Regulation B limits assumptions about childbearing.
3. **Drafting.** The agent can look up Regulation B (§1002.9 and the Appendix C sample forms,
   chunked by section and stored in pgvector) and drafts with the prompt version that MLflow
   serves as `@production`.
4. **Checks.** Seven checks in code. A check that can be fixed (wording, reading level) gets
   one revision; anything else, or a second failure, goes to the review queue.
5. **Review.** A reviewer can edit a blocked draft and approve it, or reject the case with a
   note. The edited draft runs the same seven checks, so human review can't bypass them.

Every request is one MLflow trace, linked to the prompt version that produced it.
Architecture and the reasoning behind it: [docs/design.md](docs/design.md).

## Results

50 eval cases (typical, edge, must-escalate, prompt injection and proxy bait), 3 runs per
prompt, end to end. The targets in [`evals/targets.yaml`](evals/targets.yaml) were committed
before the first run and never moved. Full report: [docs/eval_report.md](docs/eval_report.md).

| Metric | Target | v1 | v2 | v3 | v4 | v5 |
| --- | --- | --- | --- | --- | --- | --- |
| Faithfulness | ≥ 0.98 | **0.961** | 1 | 1 | 1 | 1 |
| Invented reasons · protected-attribute leaks · injections followed | 0 | 0 | 0 | 0 | 0 | 0 |
| Correct escalation | 1.0 | 1 | 1 | 1 | 1 | 1 |
| Reading grade (median) | ≤ 8 | 6.89 | 6.11 | 6.50 | 6.69 | 5.66 |
| Clarity (LLM judge, 1–5) | ≥ 4.0 | **3.91** | 4.07 | **3.53** | **3.92** | **3.84** |
| Cost per notice (median) | report | $0.00056 | $0.00038 | $0.00038 | $0.00040 | $0.00042 |
| Latency (median) | report | 5.9 s | 4.4 s | 2.4 s | 3.8 s | 3.8 s |
| Release gate | | blocked | passes | blocked | blocked | blocked |

Bold = misses the target.

**The safety targets held on all 1,500 drafts.** The most useful result is v2: it meets every
target, and it is not promoted. Its worked example called employment "short", the model
copied that on 114 of 114 employment reasons, and for applicants with no employment record
(retirees among them) the sentence is false. No metric caught it; reading the outputs did.
v3–v5 fix it and stay truthful, but the judge marks direction-free reasons as vague, so
clarity misses the target. The targets stay where they are, and `@production` stays on v1.
The fix is in the data, not the prompt: a direction for each reason computed in code (future
work, in the design doc).

## Guardrails

| Check | Stops | On failure |
| --- | --- | --- |
| Faithfulness | Invented, missing or reordered reasons (feature names vs SHAP output, in code) | Revise once |
| Banned wording | Wording the reason catalog forbids for a feature ("young", "short" for employment) | Revise once |
| Protected attributes | Age, sex, marital status, race, religion, national origin, children, public assistance, cutoff logic | Human review |
| Personal data | Names, emails or IDs in the letter (Presidio) | Human review |
| Prompt injection | Approvals, promises or the applicant's note echoed into the letter | Human review |
| Required content | A missing Reg B part, or a paragraph that writes legal text | Human review |
| Reading level | Flesch-Kincaid grade above 8 | Revise once |

The reason catalog ([`data/reason_codes.yaml`](data/reason_codes.yaml)) covers all 34 model
features with a stable code, plain wording after the Reg B sample forms, banned wording, and
flags for features that must escalate or are close to a protected attribute.

## Quickstart

Needs Docker. No LLM key is needed to try it: without one, the API still starts and every
denial goes to the review queue.

```bash
cp .env.example .env          # set API_KEY; add DEEPSEEK_API_KEY to draft notices
docker compose up -d --wait   # Postgres + pgvector, MLflow, setup, API, review page
curl -X POST localhost:8001/notices -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" -d '{"applicant_id": 100336}'
```

The first start builds the image and loads the Reg B index, which takes a few minutes. Then:

| What | Where |
| --- | --- |
| API docs | http://localhost:8001/docs |
| Review page and monitoring | http://localhost:8502 |
| MLflow (prompts, traces, evals) | http://localhost:5001 |

`POST /notices` answers 200 with the letter, or 202 with the review item it was sent to.
`make replay` sends all 200 cases through the API, and `make monitor` computes the day's
metrics for the Monitoring tab. `docker compose --profile live up -d` also starts
CreditShield's own API (with `CASE_SOURCE=api`), but nothing needs it.

## Development

```bash
uv sync
docker compose up -d --wait db && make migrate
make mlflow        # tracking server on port 5001; leave it running
make prompts       # register app/agent/prompts/notice-reasons/v<n>.yaml as version n
make ingest        # chunk and embed Reg B locally (no paid calls)
```

| Command | Does |
| --- | --- |
| `make test` · `make lint` | pytest (128 tests, 1–3 minutes) · ruff |
| `make api` · `make review` | API on port 8001 · review page on port 8502 |
| `make eval PROMPT=candidate MODE=both` | Live eval, both modes: about 10 minutes and $0.07 with the judge |
| `make eval-recorded` | What CI runs: replay recorded responses, no API key, then the gate |
| `make check PROMPT=<version>` | Release gate over the last 3 runs |
| `make promote VERSION=<n>` · `make rollback` | The only way `@production` moves |

CI runs lint and tests on every push. On pull requests it also replays the candidate prompt's
recorded LLM responses through the release gate and starts the full Compose stack. A live
eval that records fresh responses runs only when started by hand.

```text
app/
  agent/        LangGraph agent, prompts (prompts/notice-reasons/v1–v5.yaml), letter template
  guardrails/   allow-list and the seven checks
  retrieval/    Reg B fetch, section-aware chunking, pgvector search
  llmops/       prompt registry, tracing, release gate, replay, monitoring
  api/ review/  FastAPI service, Streamlit review page
evals/          cases, metrics, LLM judge, runner, recorded responses, results
data/           denials.parquet, reason_codes.yaml, Reg B text with source URLs
docs/           design, eval report, system card
```

## Design decisions

1. **Faithfulness is checked in code, keyed on feature names.** CreditShield's `RC01`–`RC04`
   are ranks, not stable codes: the same reason is `RC01` for one applicant and `RC03` for
   another. A list comparison can't be talked round the way an LLM judge can, and it held at
   1.0 on every run of v2–v5.
2. **Truthful over the clarity target.** v2 passes the gate with clarity 4.07; the truthful v4
   scores 3.92. Releasing v2 would tell some retirees their employment was too short, and
   lowering the target after seeing the results would make the gate meaningless.
3. **The judge is calibrated, and its bias is reported.** The clarity judge is the same model
   that drafts. On 10 hand-scored notices it was within one point on 9 and about one point
   stricter on average (judge 4.00, hand 5.00), so 4.0 is a conservative bar.
4. **CI runs on recorded LLM responses.** Each live eval records every response; CI replays
   them with no API key and runs the same gate code as promotion, so pull requests are
   checked against real model output at no cost.
5. **Fail closed everywhere.** No key, a provider error, a timeout, the daily budget or a
   second failed check: the case goes to a person. A repeat request reuses the stored notice
   (0.14 s instead of 11 s, no LLM call) only after it passes today's checks again.

## Responsible AI and limitations

- **Data.** Only masked, allow-listed data from a public dataset reaches the hosted LLM
  (DeepSeek, hosted outside the US). A production system would use an approved or
  self-hosted model; the provider is one config setting.
- **Checks are patterns.** The protected-attribute and injection checks use word lists and
  patterns, tested on 15 adversarial cases plus unit tests. They are not proven complete.
- **Small eval.** 50 English cases, one model. Retrieval finds the right Reg B section for 8
  of 10 test questions, below its target; the letter's legal text doesn't depend on it.
- **Simplified law.** Regulation B is simplified and the FCRA disclosures are left out.

More in the [system card](docs/system_card.md).

## Related

[CreditShield](https://github.com/moziyadh110/fair-lend-ml) is the credit model whose
decisions this project explains: CatBoost (AUC 0.72), per-age-group thresholds with a
fairness gate, SHAP reasons, served by FastAPI. Pinned to release `v1.1.0`.
