"""Rebuild the pgvector index from data/regb/:  uv run python -m app.retrieval.ingest

Reads only the committed text files and the local embedding model: no network, no paid calls.
"""

from app.config import get_settings
from app.db.models import Chunk
from app.db.session import make_sessionmaker
from app.repositories import chunks as chunk_repo
from app.retrieval.chunking import chunk_regb
from app.retrieval.embeddings import embed_documents


def main() -> None:
    settings = get_settings()
    reg_chunks = chunk_regb(settings.regb_dir)
    vectors = embed_documents([c.text for c in reg_chunks])
    rows = [
        Chunk(
            section_id=c.section_id,
            title=c.title,
            source_url=c.source_url,
            text=c.text,
            embedding=v,
            embedding_model=settings.embedding_model,
        )
        for c, v in zip(reg_chunks, vectors, strict=True)
    ]
    with make_sessionmaker()() as session, session.begin():
        chunk_repo.replace_all(session, rows)
    print(f"Ingested {len(rows)} chunks with {settings.embedding_model}")


if __name__ == "__main__":
    main()
