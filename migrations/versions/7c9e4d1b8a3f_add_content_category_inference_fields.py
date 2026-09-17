"""add content category inference fields

Revision ID: 7c9e4d1b8a3f
Revises: 6dfdf8771031
Create Date: 2026-09-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7c9e4d1b8a3f'
down_revision: Union[str, Sequence[str], None] = '6dfdf8771031'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Additive only: both columns are nullable with no server default, so every existing
    content row reads back as category_confidence=NULL / category_source=NULL --
    semantically "not locally inferred" (either a pre-existing row from before this feature
    existed, or a row whose category was explicitly supplied by the caller), which
    app.services.content_enrichment_service and app.api.candidate_routes both treat as
    ordinary, unremarkable state, never a crash. No existing row is dropped or recreated."""
    op.add_column("contents", sa.Column("category_confidence", sa.Float(), nullable=True))
    op.add_column("contents", sa.Column("category_source", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column("contents", "category_source")
    op.drop_column("contents", "category_confidence")
