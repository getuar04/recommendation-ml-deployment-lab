"""add interaction content_pending flag

Revision ID: e4b6a2c8f1d5
Revises: b2b771bd8b9b
Create Date: 2026-09-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e4b6a2c8f1d5'
down_revision: Union[str, Sequence[str], None] = 'b2b771bd8b9b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Additive only: existing interaction rows are preserved, never dropped or recreated.
    server_default=false backfills every existing row immediately (every pre-existing
    interaction was, by definition, stored while its Content row was already locally known).
    See app.db.models.Interaction.content_pending's own comment and
    app.services.event_service.store_event for the full eventual-consistency mechanism this
    backs (Task: local-existence blocking validation)."""
    op.add_column(
        "interactions",
        sa.Column("content_pending", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("interactions", "content_pending")
