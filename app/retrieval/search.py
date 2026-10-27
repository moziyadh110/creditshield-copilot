from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.repositories import chunks as chunk_repo
from app.retrieval.embeddings import embed_query


def search_regulation(session: Session, query: str, k: int = 3) -> list[Chunk]:
    return chunk_repo.nearest(session, embed_query(query), k)
