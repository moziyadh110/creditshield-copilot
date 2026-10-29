# syntax=docker/dockerfile:1
# The Copilot API image. Build stage: uv installs the locked dependencies (CPU-only PyTorch on
# Linux) and downloads the embedding model. Runtime stage: the venv, the model and the app,
# run as a non-root user with no network needed for embeddings.

FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never HF_HOME=/opt/hf
WORKDIR /app

# Dependencies first, so a code change doesn't reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBEDDING_MODEL}')"


FROM python:3.13-slim
RUN useradd --create-home --uid 1000 copilot \
    && mkdir /mlartifacts && chown copilot /mlartifacts
WORKDIR /app
COPY --from=build /opt/hf /opt/hf
COPY --from=build /app/.venv /app/.venv
COPY alembic.ini ./
COPY alembic ./alembic
COPY data ./data
COPY app ./app

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    MLFLOW_DISABLE_TELEMETRY=true
USER copilot
EXPOSE 8001
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
