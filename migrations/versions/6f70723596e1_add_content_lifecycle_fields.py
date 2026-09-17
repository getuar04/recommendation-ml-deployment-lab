"""add content lifecycle fields

Revision ID: 6f70723596e1
Revises: bfcccb4b3d8f
Create Date: 2026-07-30 09:58:41.617930

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6f70723596e1'
down_revision: Union[str, Sequence[str], None] = 'bfcccb4b3d8f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Additive only: existing content rows are preserved and safely backfilled, never
    dropped or recreated. New columns get server defaults so existing rows immediately
    have valid values (contentType='VIDEO', isActive=true, updatedAt=createdAt)."""
    op.add_column("contents", sa.Column("content_type", sa.String(), nullable=False, server_default="VIDEO"))
    op.add_column("contents", sa.Column("duration_seconds", sa.Float(), nullable=True))
    op.add_column("contents", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column("contents", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE contents SET updated_at = created_at WHERE updated_at IS NULL")
    with op.batch_alter_table("contents") as batch_op:
        batch_op.alter_column("updated_at", nullable=False)


def downgrade() -> None:
    op.drop_column("contents", "updated_at")
    op.drop_column("contents", "is_active")
    op.drop_column("contents", "duration_seconds")
    op.drop_column("contents", "content_type")
