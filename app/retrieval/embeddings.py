"""Local sentence-transformers embeddings. Runs on this machine; never a paid API call."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings

# bge v1.5 models expect this prefix on queries (not on documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def get_model(name: str | None = None) -> SentenceTransformer:
    return SentenceTransformer(name or get_settings().embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    prefix = BGE_QUERY_PREFIX if "bge" in get_settings().embedding_model.lower() else ""
    return get_model().encode(prefix + text, normalize_embeddings=True).tolist()
