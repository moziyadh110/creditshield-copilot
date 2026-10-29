"""The Copilot HTTP API:  uv run uvicorn app.api.main:app --port 8001

The lifespan handler creates the LLM client, the embedding model and the database pool once
at startup and closes them on shutdown; every request shares them.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.api.routes import router
from app.config import get_settings
from app.db.session import make_async_engine, make_async_sessionmaker
from app.llm.provider import LLMProvider, get_provider
from app.llmops.tracing import setup_tracing
from app.logs import request_id, setup_logging
from app.repositories.cases import load_denials
from app.retrieval.embeddings import get_model
from app.services.notices import pgvector_search

log = logging.getLogger("copilot.api")


@dataclass
class Deps:
    provider: LLMProvider
    search: Callable[[str], list[Any]]
    sessions: async_sessionmaker
    engine: AsyncEngine


def create_app(
    provider: LLMProvider | None = None, search: Callable[[str], list[Any]] | None = None
) -> FastAPI:
    """provider and search can be injected (tests); by default they come from config."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        setup_tracing()
        if search is None:
            await asyncio.to_thread(get_model)  # load the embedding model before any request
        if get_settings().case_source == "file":
            await asyncio.to_thread(load_denials)
        engine = make_async_engine()
        app.state.deps = Deps(
            provider=provider or get_provider(),
            search=search or pgvector_search,
            sessions=make_async_sessionmaker(engine),
            engine=engine,
        )
        log.info("startup")
        try:
            yield
        finally:
            if provider is None:
                await app.state.deps.provider.client.close()
            await engine.dispose()
            log.info("shutdown")

    app = FastAPI(title="CreditShield Copilot", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Request ID (the caller's X-Request-ID, or a new one) on every log line, the MLflow
        trace and the response; one structured log line per request."""
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return response
        except Exception:
            log.exception("unhandled error", extra={"path": request.url.path})
            raise
        finally:
            request_id.reset(token)

    app.include_router(router)
    return app


app = create_app()
