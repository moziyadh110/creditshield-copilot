"""Swappable LLM provider. Every supported provider speaks the OpenAI chat format, so one
async client covers DeepSeek (default), OpenAI, Anthropic's compatibility endpoint and Ollama.

Each call is one MLflow CHAT_MODEL span carrying token usage and cost.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey
from openai import AsyncOpenAI

from app.config import Settings, get_settings
from app.llm.cost import Usage, cost_usd

DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1/",
    "ollama": "http://localhost:11434/v1",
}


@dataclass
class LLMResponse:
    message: dict[str, Any]  # assistant message: content and/or tool_calls
    usage: Usage
    cost_usd: float
    model: str


@dataclass
class RunUsage:
    """Totals across every LLM call in one notice run."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, r: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += r.usage.prompt_tokens
        self.completion_tokens += r.usage.completion_tokens
        self.cost_usd += r.cost_usd


class LLMProvider(Protocol):
    model: str

    def configure(self, model_config: dict[str, Any]) -> None: ...

    async def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, json_output: bool = False
    ) -> LLMResponse: ...


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = 800
        key = settings.deepseek_api_key if self.provider == "deepseek" else settings.llm_api_key
        if key is None and self.provider != "ollama":
            raise RuntimeError(f"No API key set for LLM provider {self.provider!r}")
        base_url = (
            settings.llm_base_url
            if self.provider == "deepseek"
            else DEFAULT_BASE_URLS[self.provider]
        )
        self.client = client or AsyncOpenAI(
            api_key=key.get_secret_value() if key else "ollama",
            base_url=base_url,
            timeout=settings.llm_timeout_s,
            max_retries=1,
        )

    def configure(self, model_config: dict[str, Any]) -> None:
        """Apply the model config stored with the registered prompt version."""
        self.model = model_config.get("model", self.model)
        self.temperature = model_config.get("temperature", self.temperature)
        self.max_tokens = model_config.get("max_tokens", self.max_tokens)

    async def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, json_output: bool = False
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        if self.provider == "deepseek":
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        with mlflow.start_span(name="llm.chat", span_type=SpanType.CHAT_MODEL) as span:
            span.set_inputs(
                {"messages": messages, "tools": [t["function"]["name"] for t in tools or []]}
            )
            span.set_attributes(
                {SpanAttributeKey.MODEL: self.model, "temperature": self.temperature}
            )
            resp = await self.client.chat.completions.create(**kwargs)
            message = resp.choices[0].message.model_dump(exclude_none=True)
            usage = Usage.from_openai(resp.usage)
            cost = cost_usd(resp.model or self.model, usage)
            span.set_outputs(message)
            span.set_attributes(
                {
                    SpanAttributeKey.CHAT_USAGE: {
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                    },
                    SpanAttributeKey.LLM_COST: {"total_cost": cost},
                    "finish_reason": resp.choices[0].finish_reason,
                }
            )
        return LLMResponse(
            message=message, usage=usage, cost_usd=cost, model=resp.model or self.model
        )


def get_provider(settings: Settings | None = None) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(settings or get_settings())
