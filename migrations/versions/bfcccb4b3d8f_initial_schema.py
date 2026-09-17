"""initial schema

Revision ID: bfcccb4b3d8f
Revises: 
Create Date: 2026-07-30 09:58:40.699095

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bfcccb4b3d8f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Reflects the real schema as it existed before Phase 2: users, interactions,
    contents (pre-lifecycle shape). Previously created ad hoc by Base.metadata.create_all()
    at application startup; this revision is the first real migration and matches that
    shape exactly so upgrading an existing database is a no-op change to already-present
    tables (Alembic will simply record this revision as applied -- see README for the
    `alembic stamp` note on adopting migrations against a database that already has these
    tables from the old create_all() startup behavior)."""
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="ACTIVE"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_user_id", "users", ["user_id"], unique=True)

    op.create_table(
        "contents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("content_id", sa.String(), nullable=False),
        sa.Column("creator_id", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("popularity_score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_contents_content_id", "contents", ["content_id"], unique=True)

    op.create_table(
        "interactions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("content_id", sa.String(), nullable=False),
        sa.Column("creator_id", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("watch_time_seconds", sa.Float(), nullable=True),
        sa.Column("content_duration_seconds", sa.Float(), nullable=True),
        sa.Column("watch_percentage", sa.Float(), nullable=True),
        sa.Column("liked", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("shared", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("favorited", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("commented", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("creator_followed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_interactions_event_id", "interactions", ["event_id"], unique=True)
    op.create_index("ix_interactions_user_id", "interactions", ["user_id"])
    op.create_index("ix_interactions_content_id", "interactions", ["content_id"])
    op.create_index("ix_interactions_creator_id", "interactions", ["creator_id"])
    op.create_index("ix_interactions_category", "interactions", ["category"])
    op.create_index("ix_interactions_timestamp", "interactions", ["timestamp"])


def downgrade() -> None:
    op.drop_table("interactions")
    op.drop_table("contents")
    op.drop_table("users")
