from functools import cache

import mlflow
import mlflow.langchain

from app.config import get_settings


@cache
def setup_tracing() -> None:
    """Point MLflow at the tracking server and trace every LangGraph node, once per process."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    # run_tracer_inline keeps the LangGraph node spans inside our root span under asyncio.
    mlflow.langchain.autolog(run_tracer_inline=True, silent=True)
