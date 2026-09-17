"""add contents content_type is_active composite index

Revision ID: b818afd45ec5
Revises: f3a7c1d9e6b2
Create Date: 2026-09-11 23:15:37.374712

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'b818afd45ec5'
down_revision: Union[str, Sequence[str], None] = 'f3a7c1d9e6b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Supports app.api.candidate_routes.py's
    `SELECT * FROM contents WHERE is_active = true AND content_type = 'VIDEO'` query (the
    VIDEO candidate-generation stand-in, POST /candidates/generate) with a single index
    covering both filter predicates, instead of a full table scan (contents previously had
    no index on either column). Additive only -- no existing column, table, or index is
    touched or dropped."""
    op.create_index(
        "ix_contents_content_type_is_active",
        "contents",
        ["content_type", "is_active"],
    )


def downgrade() -> None:
    op.drop_index("ix_contents_content_type_is_active", table_name="contents")
