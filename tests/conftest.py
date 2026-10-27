import mlflow
import pytest


@pytest.fixture(autouse=True, scope="session")
def _temporary_mlflow_store(tmp_path_factory):
    """Tests trace into a throwaway SQLite store, never the real tracking server."""
    db = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    yield
