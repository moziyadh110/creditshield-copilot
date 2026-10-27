# Each target is a thin wrapper: on Windows without make, run the uv command directly.
.PHONY: db mlflow migrate prompts fetch-regb ingest draft eval eval-cases check promote rollback test lint

db:
	docker compose up -d --wait db

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

# PROMPT is a version number or alias; MODE is e2e, generation or both.
PROMPT ?= candidate
MODE ?= both
eval:
	uv run python -m evals.runner --prompt $(PROMPT) --mode $(MODE)

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
