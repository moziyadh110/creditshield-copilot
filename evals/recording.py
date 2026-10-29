"""Record LLM responses in the live eval and replay them in CI, so CI needs no API key.

A live full-set eval writes every LLM call it makes (the pipeline's and the judge's) to
evals/recorded/v<prompt version>/<model>/<mode>-<stamp>.jsonl: one line per call, keyed by
case and call number. A replay gives each case its recorded responses in the same order, so
the rest of the pipeline (retrieval, guardrails, template, metrics) runs for real on today's
code. Each line also stores a hash of the request; a replayed request that hashes differently
is counted as drift (the response is still replayed, but the run reports it).

Recordings belong to one prompt version and one model. If the version under test has none,
or a case asks for more calls than were recorded, the replay fails with "re-record needed"
instead of passing on stale answers. The manual live eval in CI is what records them.
"""

import asyncio
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from app.config import get_settings
from app.llm.cost import Usage
from app.llm.provider import LLMResponse, get_provider

RECORDED_DIR = Path(__file__).parent / "recorded"


class RecordingMissing(RuntimeError):
    pass


def recording_dir(version: int, model: str) -> Path:
    return RECORDED_DIR / f"v{version}" / model


def recordings(version: int, model: str, mode: str) -> list[Path]:
    """Recorded runs for one prompt version, model and mode, oldest first."""
    return sorted(recording_dir(version, model).glob(f"{mode}-*.jsonl"))


def request_key(provider: Any, messages: list[dict], tools: list[dict] | None, json_output: bool):
    payload = {
        "model": provider.model,
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
        "messages": messages,
        "tools": tools,
        "json_output": json_output,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class RecordingProvider:
    """Passes calls through to a live provider and appends each response to `lines`."""

    def __init__(self, inner: Any, case_id: str, lines: list[dict]):
        self.inner, self.case_id, self.lines, self.calls = inner, case_id, lines, 0

    @property
    def model(self) -> str:
        return self.inner.model

    def configure(self, model_config: dict[str, Any]) -> None:
        self.inner.configure(model_config)

    async def chat(self, messages, *, tools=None, json_output=False) -> LLMResponse:
        line = {
            "case_id": self.case_id,
            "call": self.calls,
            "key": request_key(self.inner, messages, tools, json_output),
        }
        self.calls += 1
        try:
            r = await self.inner.chat(messages, tools=tools, json_output=json_output)
        except BaseException as e:  # a timeout cancels the call: replay it as a timeout too
            timeout = isinstance(e, TimeoutError | asyncio.CancelledError)
            self.lines.append(
                line | {"error": "timeout" if timeout else f"{type(e).__name__}: {e}"}
            )
            raise
        self.lines.append(
            line
            | {"message": r.message, "usage": asdict(r.usage), "cost_usd": r.cost_usd}
            | {"model": r.model}
        )
        return r

    async def aclose(self) -> None:
        await self.inner.aclose()


class ReplayProvider:
    """Answers each call with the case's next recorded response. No network, no API key."""

    def __init__(self, case_id: str, recorded: list[dict], run: "Replay"):
        s = get_settings()
        self.model, self.temperature, self.max_tokens = s.llm_model, s.llm_temperature, 800
        self.case_id, self.recorded, self.run, self.calls = case_id, recorded, run, 0

    def configure(self, model_config: dict[str, Any]) -> None:
        self.model = model_config.get("model", self.model)
        self.temperature = model_config.get("temperature", self.temperature)
        self.max_tokens = model_config.get("max_tokens", self.max_tokens)

    async def chat(self, messages, *, tools=None, json_output=False) -> LLMResponse:
        n, self.calls = self.calls, self.calls + 1
        if n >= len(self.recorded):
            self.run.missing.append(f"{self.case_id} call {n}")
            raise RecordingMissing(f"re-record needed: no recorded call {n} for {self.case_id}")
        rec = self.recorded[n]
        if rec["key"] != request_key(self, messages, tools, json_output):
            self.run.drift.append(f"{self.case_id} call {n}")
        with mlflow.start_span(name="llm.chat", span_type=SpanType.CHAT_MODEL) as span:
            span.set_inputs({"messages": messages, "replayed": True})
            span.set_attributes({SpanAttributeKey.MODEL: self.model, "replayed": True})
            if "error" in rec:
                if rec["error"] == "timeout":
                    raise TimeoutError("recorded timeout")
                raise RuntimeError(f"recorded provider error: {rec['error']}")
            span.set_outputs(rec["message"])
        return LLMResponse(
            message=rec["message"],
            usage=Usage(**rec["usage"]),
            cost_usd=rec["cost_usd"],
            model=rec["model"],
        )

    async def aclose(self) -> None:
        pass


class Live:
    """Live LLM calls. With record=True every call is kept for write()."""

    def __init__(self, record: bool):
        self.record, self.lines, self._shared = record, [], None

    def pipeline(self, case_id: str):
        # One provider (one HTTP client) for every case; the wrapper only adds recording.
        self._shared = self._shared or get_provider()
        return RecordingProvider(self._shared, case_id, self.lines) if self.record else self._shared

    def judge(self, case_id: str):
        p = get_provider()  # the judge runs on its own event loop, so it needs its own client
        return RecordingProvider(p, f"judge:{case_id}", self.lines) if self.record else p

    async def aclose(self) -> None:
        if self._shared is not None:
            await self._shared.aclose()
            self._shared = None

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = sorted(self.lines, key=lambda r: (r["case_id"], r["call"]))
        path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


class Replay:
    """Replays one recorded run. `missing` and `drift` list the calls that didn't line up."""

    def __init__(self, path: Path):
        self.path, self.missing, self.drift = path, [], []
        self.by_case: dict[str, list[dict]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            self.by_case[rec["case_id"]].append(rec)
        for calls in self.by_case.values():
            calls.sort(key=lambda r: r["call"])

    def pipeline(self, case_id: str) -> ReplayProvider:
        return ReplayProvider(case_id, self.by_case.get(case_id, []), self)

    def judge(self, case_id: str) -> ReplayProvider:
        return self.pipeline(f"judge:{case_id}")

    async def aclose(self) -> None:
        pass
