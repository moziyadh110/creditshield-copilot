# Each target is a thin wrapper: on Windows without make, run the uv command directly.
.PHONY: db up down mlflow migrate prompts fetch-regb ingest draft api review replay monitor eval eval-recorded eval-cases check promote rollback test lint

db:
	docker compose up -d --wait db

# The whole stack in Docker: db, MLflow (5001), setup, the API (8001) and the review page
# (8502). Stop `make mlflow` first.
up:
	docker compose up -d --wait --build

down:
	docker compose down

# MLflow UI on http://localhost:5001. psycopg2 is named explicitly: SQLAlchemy 2.1 defaults
# postgresql:// to psycopg 3, which breaks MLflow's prompt-version lookups.
mlflow:
	uv run mlflow server --backend-store-uri postgresql+psycopg2://copilot:copilot@localhost:5433/mlflow --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5001

migrate:
	uv run alembic upgrade head

prompts:
	uv run python -m app.llmops.prompts

fetch-regb:
	uv run python -m app.retrieval.fetch_regb

ingest:
	uv run python -m app.retrieval.ingest

draft:
	uv run copilot draft --applicant $(APPLICANT)

# The HTTP API on http://localhost:8001 (docs at /docs); needs API_KEY in .env.
api:
	uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8001

# The review page and monitoring tab on http://localhost:8502; calls the API with API_KEY.
review:
	uv run streamlit run app/review/page.py --server.port 8502 --server.address 127.0.0.1 \n		--browser.gatherUsageStats false

# A day of demo traffic: every case in denials.parquet through the running API.
replay:
	uv run python -m app.llmops.replay

# Daily metrics and alerts from today's traces; ARGS=--test-alert fires a test alert.
monitor:
	uv run python -m app.llmops.monitor $(ARGS)

# PROMPT is a version number or alias; MODE is e2e, generation or both.
PROMPT ?= candidate
MODE ?= both
eval:
	uv run python -m evals.runner --prompt $(PROMPT) --mode $(MODE)

# What CI runs: the candidate's recorded responses, no API key, then the gate on those runs.
eval-recorded:
	uv run python -m evals.runner --prompt candidate --mode e2e --llm recorded
	uv run python -m app.llmops.gate check candidate --recorded

eval-cases:
	uv run python -m evals.build_cases

check:
	uv run python -m app.llmops.gate check $(PROMPT)

promote:
	uv run python -m app.llmops.gate promote $(VERSION)

rollback:
	uv run python -m app.llmops.gate rollback

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
