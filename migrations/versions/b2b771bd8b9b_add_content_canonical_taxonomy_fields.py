"""add content canonical taxonomy fields

Revision ID: b2b771bd8b9b
Revises: 7c9e4d1b8a3f
Create Date: 2026-09-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2b771bd8b9b'
down_revision: Union[str, Sequence[str], None] = '7c9e4d1b8a3f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Additive only, mirroring 7c9e4d1b8a3f's own precedent: all three columns are nullable
    with no server default and no CHECK/enum constraint, so every existing content row reads
    back as primary_category=NULL / subcategory=NULL / taxonomy_version=NULL -- "no canonical
    taxonomy classification exists for this row yet", true for every row at the time of this
    migration. No existing row is dropped, recreated, or backfilled, and the pre-existing
    `category` column (the currently-active VIDEO model's only categorical feature) is not
    touched in any way. Deliberately no proposed-taxonomy enum: these columns accept any
    future string value, since no canonical category list is product-approved yet."""
    op.add_column("contents", sa.Column("primary_category", sa.String(length=64), nullable=True))
    op.add_column("contents", sa.Column("subcategory", sa.String(length=64), nullable=True))
    op.add_column("contents", sa.Column("taxonomy_version", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("contents", "taxonomy_version")
    op.drop_column("contents", "subcategory")
    op.drop_column("contents", "primary_category")
