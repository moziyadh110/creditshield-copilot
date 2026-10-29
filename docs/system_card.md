# System card: CreditShield Copilot

**Simplified, not legal advice.** This is a portfolio demo of how an LLM can explain a
credit decision under rules it must not break. It implements a simplified reading of
Regulation B and has had no compliance review. Don't use it to send real notices.

## What it does

Turns a credit denial from CreditShield's model into the "principal reasons" section of an
adverse action notice (12 CFR 1002.9(a)(2), Appendix C). The model's top 4 SHAP features
are the reasons; the LLM only words them, and a fixed template supplies the rest of the
letter.

## Intended use

- Drafting the reasons section for **DENY** decisions, for a person or downstream system to
  send, with every draft checked in code first.
- Showing how to put guardrails, evaluation, release gating and monitoring around an LLM in
  a regulated setting.

## Out of scope

- **Making or changing credit decisions.** The LLM never sees the score, the cutoff or any
  applicant value.
- **MANUAL_REVIEW cases.** They are not denials, so they go to review without an LLM call.
- **`cnt_children` reasons.** Never worded by the system, by the LLM or a reviewer; they go
  to review and must be handled with a compliance-approved letter.
- **Real applicants.** The data is the public Home Credit dataset with synthetic IDs
  ("Applicant #<id>") and a fictional creditor ("CreditShield Lending (demo)").
- **The FCRA section 615 disclosures** a real notice needs when a consumer report was used.
  The template marks where they would go.

## Data

- **Input.** 200 decisions exported from CreditShield (`data/denials.parquet`): applicant
  ID, decision, product, and the top 4 reason features with impact scores. No names,
  addresses, dates of birth or free text exist in the source data.
- **What reaches the LLM.** Only the reason features and ranks, the product, the notice
  date, and an optional applicant note. The note exists only in synthetic eval cases; it is
  masked by Presidio and passed as data, never as instructions. Applicant IDs, impact scores,
  feature values and all other fields are dropped by an allow-list, which a unit test checks.
- **Where it goes.** The LLM is DeepSeek's hosted API (`deepseek-flash`), which runs outside
  the US. That is acceptable only because the data is masked, allow-listed and public. **A
  production system would use an approved or self-hosted model** behind the same provider
  interface (`app/llm/provider.py` switches by config to OpenAI, Anthropic or a local Ollama
  model, all through their OpenAI-compatible endpoints).
- **Embeddings** run locally (`BAAI/bge-small-en-v1.5`); the regulation text never leaves the
  machine.
- **Stored.** Postgres holds the run log, review queue, regulation chunks and spend counter;
  MLflow holds traces, eval runs and the prompt registry. Traces contain the masked agent
  input and the drafts.

## Safeguards

- **Faithfulness in code.** A notice is sent only if its reason features match the SHAP
  features exactly, in rank order.
- **Seven output checks** (faithfulness, banned wording, protected attributes, personal data,
  prompt injection, required content, reading level). Any failure the system can't fix with
  one revision goes to human review; nothing is guessed.
- **Protected attributes kept out of the input**, not only filtered from the output. Age is
  never a reason, and credit-history length is never worded as age (tested by 7 proxy-bait
  cases).
- **Human review can't bypass the checks.** A reviewer's edited draft runs the same 7
  checks before it can be approved.
- **Fail closed.** No LLM key, a provider error, a timeout or the daily budget all send the
  case to review.
- **Release gate.** A prompt version reaches production only if it meets every target in
  `evals/targets.yaml` over 3 runs. The targets were committed before the first eval.

## Evaluation results

Fifty cases, 3 runs per prompt and mode, 1,500 drafts in all (details in
[eval_report.md](eval_report.md)):

- **Safety targets held on every draft:** 0 invented reasons, 0 protected-attribute leaks,
  0 injections followed, and every must-escalate case went to review.
- **No prompt passes every target in a truthful form.** v2 does (clarity 4.07), but it
  claims a direction the data can't support: it called employment "short" for applicants
  with no employment record, which includes retirees. v3–v5 are truthful and score 3.53–3.92
  on clarity against a target of 4.0. The targets were not moved, so `@production` stays on
  v1, which the gate also blocks (faithfulness 0.961, clarity 3.91). In this repo every
  version is a demo; in production none would be released.

## Known limitations

- **Self-judging bias.** The clarity judge is `deepseek-flash`, the same model that drafts.
  Calibrated against 10 hand-scored notices, it scored about one point lower than the human
  reader (judge 4.00, hand 5.00; within one point on 9 of 10). The hand scores were all 5s,
  so the calibration shows the offset but can't show whether the judge ranks notices the
  way a person would.
- **Direction-free reasons read as vague.** Because the LLM never sees values, it can say
  that a factor counted against the applicant but not which way. Reg B's sample reasons
  often state a direction ("insufficient income").
- **Pattern-based checks.** The protected-attribute and injection checks use word lists and
  patterns. They catch what was tested (8 injection and 7 proxy-bait cases, plus unit tests)
  but would miss a paraphrase no pattern covers.
- **Small eval set.** 50 cases, one model, English only.
- **Retrieval** finds the right Reg B section for 8 of 10 test questions, below the 9 of 10
  target. The notice's legal content doesn't depend on it, because the template supplies it.
- **Not deterministic.** Temperature 0 still varies between API calls.
- **Upstream fairness.** The Copilot explains CreditShield's decisions and doesn't audit
  them. CreditShield documents its own fairness gate (age-group thresholds, with applicants
  62+ never given a stricter cutoff).

## Before any real use

Compliance and legal review of the template, the reason catalog and every banned-wording
list; an approved or self-hosted model; the FCRA disclosures; a larger eval set scored by
compliance reviewers; access control on the review page; and `DAILY_BUDGET_USD` set.
