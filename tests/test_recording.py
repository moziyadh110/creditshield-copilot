"""Record and replay of LLM calls (evals/recording.py), with a fake provider: no network."""

import asyncio

import pytest

from app.llm.cost import Usage
from app.llm.provider import LLMResponse
from evals.recording import Live, RecordingProvider, Replay

MESSAGES = [{"role": "user", "content": "hello"}]


class FakeProvider:
    def __init__(self, fail=None):
        self.model, self.temperature, self.max_tokens, self.fail = "fake", 0, 800, fail
        self.calls = 0

    def configure(self, model_config):
        self.model = model_config.get("model", self.model)

    async def chat(self, messages, *, tools=None, json_output=False):
        self.calls += 1
        if self.fail:
            raise self.fail
        return LLMResponse({"role": "assistant", "content": f"answer {self.calls}"},
                           Usage(10, 5, 2), 0.001, self.model)  # fmt: skip

    async def aclose(self):
        pass


def _record(tmp_path, fail=None):
    live = Live(record=True)
    provider = RecordingProvider(FakeProvider(fail), "case-1", live.lines)
    provider.configure({"model": "fake", "temperature": 0, "max_tokens": 800})

    async def two_calls():
        for _ in range(2):
            await provider.chat(MESSAGES, json_output=True)

    if fail:
        with pytest.raises(type(fail)):
            asyncio.run(two_calls())
    else:
        asyncio.run(two_calls())
    path = tmp_path / "e2e-1.jsonl"
    live.write(path)
    return path


def test_replay_returns_the_recorded_responses_in_order(tmp_path):
    replay = Replay(_record(tmp_path))
    provider = replay.pipeline("case-1")
    provider.configure({"model": "fake", "temperature": 0, "max_tokens": 800})
    got = [asyncio.run(provider.chat(MESSAGES, json_output=True)) for _ in range(2)]
    assert [r.message["content"] for r in got] == ["answer 1", "answer 2"]
    assert got[0].usage == Usage(10, 5, 2) and got[0].cost_usd == 0.001
    assert replay.missing == [] and replay.drift == []


def test_a_changed_request_is_replayed_but_counted_as_drift(tmp_path):
    replay = Replay(_record(tmp_path))
    provider = replay.pipeline("case-1")
    provider.configure({"model": "fake", "temperature": 0, "max_tokens": 800})
    r = asyncio.run(provider.chat([{"role": "user", "content": "changed"}], json_output=True))
    assert r.message["content"] == "answer 1"
    assert replay.drift == ["case-1 call 0"]


def test_an_unrecorded_call_means_re_record_needed(tmp_path):
    replay = Replay(_record(tmp_path))
    with pytest.raises(RuntimeError, match="re-record needed"):
        asyncio.run(replay.pipeline("case-2").chat(MESSAGES))
    assert replay.missing == ["case-2 call 0"]


def test_a_recorded_timeout_replays_as_a_timeout(tmp_path):
    replay = Replay(_record(tmp_path, fail=TimeoutError()))
    with pytest.raises(TimeoutError):
        asyncio.run(replay.pipeline("case-1").chat(MESSAGES, json_output=True))
