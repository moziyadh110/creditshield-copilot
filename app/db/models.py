from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# BAAI/bge-small-en-v1.5 output size. Changing the embedding model means a new migration.
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Chunk(Base):
    """One section of Reg B text (e.g. 1002.9(a)(2)) or one Appendix C sample form."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    section_id: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
