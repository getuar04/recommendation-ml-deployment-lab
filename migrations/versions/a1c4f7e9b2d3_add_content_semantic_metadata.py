"""add content semantic metadata

Revision ID: a1c4f7e9b2d3
Revises: 0b6779c4f8fb
Create Date: 2026-08-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1c4f7e9b2d3'
down_revision: Union[str, Sequence[str], None] = '0b6779c4f8fb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Additive only: all five columns are nullable with no server default, so every
    existing content row reads back as title=NULL / empty token lists (app.db.models.Content
    decodes NULL the same as "[]") -- semantically identical to "no semantic metadata was
    ever provided", which app.ml.dataset_builder and the online recommendation path both
    treat as neutral (see FeatureHistory.features()'s 0.5-affinity/has_semantic_history=False
    defaults) rather than a crash. No existing row is dropped or recreated."""
    op.add_column("contents", sa.Column("title", sa.String(length=200), nullable=True))
    op.add_column("contents", sa.Column("hashtags_json", sa.Text(), nullable=True))
    op.add_column("contents", sa.Column("topics_json", sa.Text(), nullable=True))
    op.add_column("contents", sa.Column("entities_json", sa.Text(), nullable=True))
    op.add_column("contents", sa.Column("subgenres_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("contents", "subgenres_json")
    op.drop_column("contents", "entities_json")
    op.drop_column("contents", "topics_json")
    op.drop_column("contents", "hashtags_json")
    op.drop_column("contents", "title")
