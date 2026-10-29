"""daily spend table and a cache-hit flag on the run log

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "run_log",
        sa.Column("cache_hit", sa.Boolean, server_default=sa.false(), nullable=False),
    )
    op.create_table(
        "daily_spend",
        sa.Column("day", sa.Date, primary_key=True),
        sa.Column("cost_usd", sa.Float, server_default="0", nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("daily_spend")
    op.drop_column("run_log", "cache_hit")
