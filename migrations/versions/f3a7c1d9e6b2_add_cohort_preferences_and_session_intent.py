"""add cohort preferences, user demographic context, and persistent session search intent

Revision ID: f3a7c1d9e6b2
Revises: c2e5a91f4b7a
Create Date: 2026-09-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3a7c1d9e6b2'
down_revision: Union[str, Sequence[str], None] = 'c2e5a91f4b7a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """New, additive tables only -- no existing table is touched. Replaces the removed
    LOCAL_POC_REGIONAL_CATEGORY_COHORT_PRIOR hardcoded table (app.ml.reranker) with a
    data-driven, versioned persistence layer (app.services.cohort_aggregation_service /
    app.services.cohort_preference_provider), plus a persistent/shared backing store for
    session search intent (app.services.session_intent_provider) replacing the old
    single-process in-memory dict."""
    op.create_table(
        "user_demographic_context",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column("region", sa.String(length=8), nullable=True),
        sa.Column("age_bucket", sa.String(length=16), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "recommendation_cohort_preferences",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("cohort_type", sa.String(length=16), nullable=False),
        sa.Column("region", sa.String(length=8), nullable=True),
        sa.Column("age_bucket", sa.String(length=16), nullable=True),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("preference_score", sa.Float(), nullable=False),
        sa.Column("sample_users", sa.Integer(), nullable=False),
        sa.Column("sample_interactions", sa.Integer(), nullable=False),
        sa.Column("positive_count", sa.Integer(), nullable=False),
        sa.Column("negative_count", sa.Integer(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_recommendation_cohort_preferences_version", "recommendation_cohort_preferences", ["version"])
    op.create_index(
        "ix_cohort_pref_lookup", "recommendation_cohort_preferences",
        ["version", "cohort_type", "region", "age_bucket"],
    )

    op.create_table(
        "session_search_intent",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column("intent_json", sa.Text(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_session_search_intent_expires_at", "session_search_intent", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_session_search_intent_expires_at", table_name="session_search_intent")
    op.drop_table("session_search_intent")
    op.drop_index("ix_cohort_pref_lookup", table_name="recommendation_cohort_preferences")
    op.drop_index("ix_recommendation_cohort_preferences_version", table_name="recommendation_cohort_preferences")
    op.drop_table("recommendation_cohort_preferences")
    op.drop_table("user_demographic_context")
