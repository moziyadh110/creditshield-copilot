from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings


def make_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url, pool_pre_ping=True)


def make_sessionmaker(engine: Engine | None = None) -> sessionmaker:
    return sessionmaker(engine or make_engine(), expire_on_commit=False)


def make_async_engine(url: str | None = None) -> AsyncEngine:
    """Same database as DATABASE_URL, through asyncpg. (psycopg's async mode can't run on
    Windows' default Proactor event loop; asyncpg can.)"""
    url = make_url(url or get_settings().database_url).set(drivername="postgresql+asyncpg")
    return create_async_engine(url, pool_pre_ping=True)


def make_async_sessionmaker(engine: AsyncEngine | None = None) -> async_sessionmaker:
    return async_sessionmaker(engine or make_async_engine(), expire_on_commit=False)
