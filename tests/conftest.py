import os

# Before mlflow is imported: see evals/__init__.py (telemetry thread vs import lock).
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _temporary_mlflow_store(tmp_path_factory):
    """Tests trace into a throwaway SQLite store, never the real tracking server."""
    db = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    yield
