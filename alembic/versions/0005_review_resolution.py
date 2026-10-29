"""review resolution: the reviewer's approved draft and the notice built from it

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("review_queue", sa.Column("final_draft", sa.JSON, nullable=True))
    op.add_column("review_queue", sa.Column("final_notice", sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column("review_queue", "final_notice")
    op.drop_column("review_queue", "final_draft")
