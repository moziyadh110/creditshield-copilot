import os

# Set before mlflow is imported. MLflow 3.15's telemetry thread and the autologging that
# mlflow.genai.evaluate switches on deadlock on Python's import lock (seen 2026-10-27: the
# eval hung before its first case). Telemetry off avoids it, and sends nothing out.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
