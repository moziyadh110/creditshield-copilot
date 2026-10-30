"""All settings come from the environment (or a local .env). No secret has a default."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# USD per 1M tokens at DeepSeek's PEAK rate (off-peak is half), from
# https://api-docs.deepseek.com/quick_start/pricing/ on 2026-10-27. Peak never under-reports.
# MLflow's own pricing (LiteLLM's table) doesn't know these models, so cost is computed here.
LLM_PRICES_USD_PER_M = {
    "deepseek-flash": {"input_cache_hit": 0.006, "input_cache_miss": 0.30, "output": 1.20},
    "deepseek-v4-pro": {"input_cache_hit": 0.044, "input_cache_miss": 1.32, "output": 3.96},
}


class Settings(BaseSettings):
    # Blank values in .env (e.g. DAILY_BUDGET_USD=) fall back to the defaults below.
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", extra="ignore", env_ignore_empty=True
    )

    database_url: str = "postgresql+psycopg://copilot:copilot@localhost:5433/copilot"

    llm_provider: Literal["deepseek", "openai", "anthropic", "ollama"] = "deepseek"
    llm_model: str = "deepseek-flash"
    llm_base_url: str = "https://api.deepseek.com"
    llm_temperature: float = 0.0
    deepseek_api_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None  # for providers other than DeepSeek
    llm_timeout_s: float = 60.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    case_source: Literal["file", "api"] = "file"
    denials_path: Path = DATA_DIR / "denials.parquet"
    reason_codes_path: Path = DATA_DIR / "reason_codes.yaml"
    regb_dir: Path = DATA_DIR / "regb"
    creditshield_api_url: str = "http://localhost:8000"
    creditshield_api_key: SecretStr | None = None

    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment: str = "creditshield-copilot"
    # Eval runs and their traces live apart, so monitoring never counts eval traffic.
    mlflow_eval_experiment: str = "creditshield-copilot-eval"
    # One run per day from `make monitor`: the daily metrics and any alerts.
    mlflow_monitor_experiment: str = "creditshield-copilot-monitoring"
    prompt_name: str = "notice-reasons"
    prompt_alias: str = "production"  # "candidate" during eval

    # None means no cap; cost is still logged on every run.
    daily_budget_usd: float | None = None

    # X-API-Key for the HTTP API. Unset = every protected route answers 503 (fail closed).
    api_key: SecretStr | None = None
    # Where the review page finds the API (it sends api_key with every call).
    copilot_api_url: str = "http://localhost:8001"

    # Optional: alerts from `make monitor` are also POSTed here as JSON (e.g. a Slack webhook).
    alert_webhook_url: SecretStr | None = None

    # Public demo (the Hugging Face Space): POST /demo/notices needs no API key, accepts only
    # the eval cases (plus an optional applicant note) and is rate-limited per client IP.
    # Off by default; the demo deployment also sets DAILY_BUDGET_USD.
    demo_enabled: bool = False
    demo_cases_path: Path = ROOT_DIR / "evals" / "cases.jsonl"
    demo_site_url: str = "https://moziyadh110.github.io/creditshield-copilot/"  # GET / goes here
    demo_origins: list[str] = ["https://moziyadh110.github.io"]
    demo_per_minute: int = 5
    demo_per_day: int = 40


@lru_cache
def get_settings() -> Settings:
    return Settings()
