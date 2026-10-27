"""All settings come from the environment (or a local .env). No secret has a default."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    # Blank values in .env (e.g. DAILY_BUDGET_USD=) fall back to the defaults below.
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", extra="ignore", env_ignore_empty=True
    )

    database_url: str = "postgresql+psycopg://copilot:copilot@localhost:5433/copilot"

    llm_provider: Literal["deepseek", "openai", "anthropic", "ollama"] = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com"
    llm_temperature: float = 0.0
    deepseek_api_key: SecretStr | None = None

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    case_source: Literal["file", "api"] = "file"
    denials_path: Path = DATA_DIR / "denials.parquet"
    reason_codes_path: Path = DATA_DIR / "reason_codes.yaml"
    regb_dir: Path = DATA_DIR / "regb"
    creditshield_api_url: str = "http://localhost:8000"
    creditshield_api_key: SecretStr | None = None

    mlflow_tracking_uri: str = "http://localhost:5001"

    # None means no cap; cost is still logged on every run.
    daily_budget_usd: float | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
