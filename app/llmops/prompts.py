"""Register prompt source files in the MLflow prompt registry:  uv run python -m app.llmops.prompts

Each app/agent/prompts/<name>/v<n>.yaml becomes a registry version only if its template or
model config differs from the latest registered version, so re-running is a no-op. The first
version registered gets @production when no @production alias exists yet; a newer version gets
@candidate, and only the release gate (app/llmops/gate.py) can make it @production.
"""

from pathlib import Path

import mlflow
import mlflow.genai
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from app.config import get_settings

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agent" / "prompts"


def _source_files(name: str) -> list[Path]:
    return sorted((PROMPTS_DIR / name).glob("v*.yaml"), key=lambda p: int(p.stem[1:]))


def _latest(name: str):
    try:
        versions = list(MlflowClient().search_prompt_versions(name))
    except MlflowException:
        return None
    return max(versions, key=lambda v: int(v.version), default=None)


def _model_config(version) -> dict:
    cfg = version.model_config
    return cfg if isinstance(cfg, dict) else (cfg.to_dict() if cfg else {})


def register(name: str) -> list[int]:
    registered = []
    for path in _source_files(name):
        src = yaml.safe_load(path.read_text(encoding="utf-8"))
        latest = _latest(name)
        if (
            latest is not None
            and latest.template == src["template"]
            and _model_config(latest) == src["model_config"]
        ):
            continue
        v = mlflow.genai.register_prompt(
            name=name,
            template=src["template"],
            commit_message=src["commit_message"],
            model_config=src["model_config"],
            tags={"source_file": path.relative_to(PROMPTS_DIR.parent.parent.parent).as_posix()},
        )
        registered.append(v.version)
        print(f"registered {name} version {v.version} from {path.name}")
    try:
        mlflow.genai.load_prompt(f"prompts:/{name}@production", link_to_model=False)
    except MlflowException:
        first = registered[0] if registered else 1
        mlflow.genai.set_prompt_alias(name, alias="production", version=first)
        print(f"set {name}@production -> version {first}")
    # The newest version is what the eval and the release gate look at next.
    if registered and registered[-1] != int(
        mlflow.genai.load_prompt(f"prompts:/{name}@production", link_to_model=False).version
    ):
        mlflow.genai.set_prompt_alias(name, alias="candidate", version=registered[-1])
        print(f"set {name}@candidate -> version {registered[-1]}")
    if not registered:
        print(f"{name}: no changes, nothing registered")
    return registered


def main() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    register(settings.prompt_name)


if __name__ == "__main__":
    main()
