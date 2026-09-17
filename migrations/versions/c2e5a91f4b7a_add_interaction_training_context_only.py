"""add interaction training-context-only flag

Revision ID: c2e5a91f4b7a
Revises: a1c4f7e9b2d3
Create Date: 2026-08-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2e5a91f4b7a'
down_revision: Union[str, Sequence[str], None] = 'a1c4f7e9b2d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Additive only: existing interaction rows are preserved, never dropped or recreated.
    server_default=false backfills every existing row immediately -- real event ingestion
    never sets this field, so it stays false for every production row past and future unless
    a training-data generator explicitly opts a row in. See app.db.models.Interaction's own
    comment and app.ml.dataset_builder.build_dataset for the full mechanism."""
    op.add_column(
        "interactions",
        sa.Column("is_training_context_only", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("interactions", "is_training_context_only")
