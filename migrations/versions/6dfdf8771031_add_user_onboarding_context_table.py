"""add user onboarding context table

Revision ID: 6dfdf8771031
Revises: b818afd45ec5
Create Date: 2026-09-14 09:50:50.905474

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6dfdf8771031'
down_revision: Union[str, Sequence[str], None] = 'b818afd45ec5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """New, additive table only -- no existing table/column is touched. `user_id` is the
    primary key (one context row per user, matching user_demographic_context/
    session_search_intent's existing single-row-per-user convention) -- no separate index
    needed since every lookup is a point read by user_id."""
    op.create_table(
        "user_onboarding_context",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("region", sa.String(length=8), nullable=True),
        sa.Column("language", sa.String(length=16), nullable=True),
        sa.Column("interests_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("user_onboarding_context")
