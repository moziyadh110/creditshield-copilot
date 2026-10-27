from dataclasses import dataclass
from typing import Any

from app.config import LLM_PRICES_USD_PER_M


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cache_hit_tokens: int = 0

    @classmethod
    def from_openai(cls, usage: Any) -> "Usage":
        if usage is None:
            return cls(0, 0)
        # DeepSeek reports prompt_cache_hit_tokens; OpenAI reports prompt_tokens_details.
        hit = getattr(usage, "prompt_cache_hit_tokens", None)
        if hit is None and getattr(usage, "prompt_tokens_details", None) is not None:
            hit = usage.prompt_tokens_details.cached_tokens
        return cls(usage.prompt_tokens, usage.completion_tokens, hit or 0)


def cost_usd(model: str, usage: Usage) -> float:
    """Cost of one call; 0.0 for models without a price entry (e.g. a local Ollama model)."""
    price = LLM_PRICES_USD_PER_M.get(model)
    if price is None:
        return 0.0
    miss = usage.prompt_tokens - usage.cache_hit_tokens
    return (
        usage.cache_hit_tokens * price["input_cache_hit"]
        + miss * price["input_cache_miss"]
        + usage.completion_tokens * price["output"]
    ) / 1_000_000
