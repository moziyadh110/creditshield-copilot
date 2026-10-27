"""run log table

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "run_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("applicant_id", sa.BigInteger, nullable=False),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("outcome", sa.String(16), nullable=False),
        sa.Column("escalation_reason", sa.Text),
        sa.Column("agent_input", sa.JSON, nullable=False),
        sa.Column("retrieved_chunk_ids", sa.JSON, nullable=False),
        sa.Column("draft", sa.JSON),
        sa.Column("notice", sa.Text),
        sa.Column("check_results", sa.JSON, nullable=False),
        sa.Column("prompt_name", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.Integer),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("llm_calls", sa.Integer, nullable=False),
        sa.Column("prompt_tokens", sa.Integer, nullable=False),
        sa.Column("completion_tokens", sa.Integer, nullable=False),
        sa.Column("cost_usd", sa.Float, nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("trace_id", sa.String(64)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_run_log_applicant_id", "run_log", ["applicant_id"])
    op.create_index("ix_run_log_trace_id", "run_log", ["trace_id"])


def downgrade() -> None:
    op.drop_index("ix_run_log_trace_id", table_name="run_log")
    op.drop_index("ix_run_log_applicant_id", table_name="run_log")
    op.drop_table("run_log")
