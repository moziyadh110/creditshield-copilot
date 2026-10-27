from app.config import Settings


def test_blank_env_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("DAILY_BUDGET_USD", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    settings = Settings(_env_file=None)
    assert settings.daily_budget_usd is None
    assert settings.deepseek_api_key is None


def test_secrets_are_not_printed(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    settings = Settings(_env_file=None)
    assert "sk-test-123" not in repr(settings)
    assert settings.deepseek_api_key.get_secret_value() == "sk-test-123"


def test_temperature_defaults_to_zero():
    assert Settings(_env_file=None).llm_temperature == 0.0
