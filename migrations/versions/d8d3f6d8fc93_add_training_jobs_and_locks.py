"""add training jobs and locks

Revision ID: d8d3f6d8fc93
Revises: 6f70723596e1
Create Date: 2026-07-30 09:58:42.632318

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd8d3f6d8fc93'
down_revision: Union[str, Sequence[str], None] = '6f70723596e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """New, additive tables only -- no existing table is touched."""
    op.create_table(
        "training_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("model_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="PENDING"),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("artifact_path", sa.String(), nullable=True),
        sa.Column("metadata_path", sa.String(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
    )
    op.create_index("ix_training_jobs_job_id", "training_jobs", ["job_id"], unique=True)

    op.create_table(
        "training_locks",
        sa.Column("model_type", sa.String(), primary_key=True),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("training_locks")
    op.drop_table("training_jobs")
