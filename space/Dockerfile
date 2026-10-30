# syntax=docker/dockerfile:1
# The live demo as one container for a Hugging Face Space (Docker SDK): a throwaway Postgres
# with pgvector, MLflow on SQLite, and the Copilot API with the public /demo route on 7860.
# Copy this file and README.md into the Space; it builds the GitHub repo's main branch.

FROM pgvector/pgvector:pg17
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 user
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy HF_HOME=/opt/hf

# Changes whenever main gets a new commit, so a rebuild never reuses a stale clone.
ADD https://api.github.com/repos/moziyadh110/creditshield-copilot/commits/main /tmp/main.json
RUN git clone --depth 1 https://github.com/moziyadh110/creditshield-copilot /app
WORKDIR /app
RUN uv sync --frozen --no-dev --python 3.13 \
    && .venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5')" \
    && chown -R user /app

USER user
ENV PATH=/app/.venv/bin:/usr/lib/postgresql/17/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    DATABASE_URL=postgresql+psycopg://user@localhost:5432/copilot \
    MLFLOW_TRACKING_URI=sqlite:////home/user/mlflow.db \
    DEMO_ENABLED=true \
    DAILY_BUDGET_USD=0.10
EXPOSE 7860
CMD ["sh", "space/start.sh"]
