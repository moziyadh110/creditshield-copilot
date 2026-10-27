"""Reg B chunks in pgvector."""

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import Chunk


def replace_all(session: Session, chunks: list[Chunk]) -> None:
    """Rebuild the index from scratch in the caller's transaction."""
    session.execute(delete(Chunk))
    session.add_all(chunks)


def nearest(session: Session, embedding: list[float], k: int = 3) -> list[Chunk]:
    """Top-k chunks by cosine distance; ties broken by section_id so results are deterministic."""
    stmt = (
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(embedding), Chunk.section_id)
        .limit(k)
    )
    return list(session.scalars(stmt))
