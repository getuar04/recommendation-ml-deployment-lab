"""add interactions user_id+timestamp composite index

Revision ID: 0b6779c4f8fb
Revises: d8d3f6d8fc93
Create Date: 2026-08-04 12:40:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '0b6779c4f8fb'
down_revision: Union[str, Sequence[str], None] = 'd8d3f6d8fc93'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Phase A: supports app.db.repositories.recent_interactions_for_ranking()'s
    WHERE user_id = ? ORDER BY timestamp DESC LIMIT N query on the VIDEO recommendation hot
    path with a single index covering both the filter and the sort, instead of relying on
    the existing separate single-column ix_interactions_user_id/ix_interactions_timestamp
    indexes. Additive only -- no existing column, table, or index is touched or dropped."""
    op.create_index(
        "ix_interactions_user_id_timestamp",
        "interactions",
        ["user_id", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_interactions_user_id_timestamp", table_name="interactions")
