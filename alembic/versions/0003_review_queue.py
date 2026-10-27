"""review queue table

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "review_queue",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("applicant_id", sa.BigInteger, nullable=False),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("check_results", sa.JSON, nullable=False),
        sa.Column("draft", sa.JSON),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("run_log.id")),
        sa.Column("trace_id", sa.String(64)),
        sa.Column("status", sa.String(16), server_default="pending", nullable=False),
        sa.Column("reviewer_note", sa.Text),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_review_queue_applicant_id", "review_queue", ["applicant_id"])
    op.create_index("ix_review_queue_status", "review_queue", ["status"])


def downgrade() -> None:
    op.drop_index("ix_review_queue_status", table_name="review_queue")
    op.drop_index("ix_review_queue_applicant_id", table_name="review_queue")
    op.drop_table("review_queue")
