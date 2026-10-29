"""Prompt registration: v<n>.yaml becomes registry version n, and re-running registers nothing."""

import shutil

import mlflow.genai
import yaml

from app.llmops import prompts


def test_register_is_idempotent_and_keeps_file_numbers(tmp_path, monkeypatch):
    name = "notice-reasons-register-test"
    shutil.copytree(prompts.PROMPTS_DIR / "notice-reasons", tmp_path / name)
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)
    files = prompts._source_files(name)

    assert prompts.register(name) == list(range(1, len(files) + 1))
    assert prompts.register(name) == []  # 2026-10-29: v1 and v2 used to be registered again

    for n, path in enumerate(files, 1):
        assert path.stem == f"v{n}"
        registered = mlflow.genai.load_prompt(f"prompts:/{name}/{n}", link_to_model=False)
        assert registered.template == yaml.safe_load(path.read_text(encoding="utf-8"))["template"]
    production = mlflow.genai.load_prompt(f"prompts:/{name}@production", link_to_model=False)
    candidate = mlflow.genai.load_prompt(f"prompts:/{name}@candidate", link_to_model=False)
    assert (int(production.version), int(candidate.version)) == (1, len(files))
