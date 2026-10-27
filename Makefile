# Each target is a thin wrapper: on Windows without make, run the uv command directly.
.PHONY: db migrate fetch-regb ingest test lint

db:
	docker compose up -d --wait db

migrate:
	uv run alembic upgrade head

fetch-regb:
	uv run python -m app.retrieval.fetch_regb

ingest:
	uv run python -m app.retrieval.ingest

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
