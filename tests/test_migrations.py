"""Deterministic, isolated Alembic migration tests. Every test creates and cleans up only
its own temporary SQLite file -- never `data/*.db`, never a real PostgreSQL database."""
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

REPO_ROOT = Path(__file__).resolve().parents[1]


def _alembic_config(db_path: Path) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return config


@pytest.fixture
def migration_db(tmp_path):
    db_path = tmp_path / "migration_test.db"
    yield db_path
    db_path.unlink(missing_ok=True)


def test_upgrade_head_creates_expected_tables_and_columns(migration_db):
    command.upgrade(_alembic_config(migration_db), "head")
    engine = create_engine(f"sqlite:///{migration_db}")
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {"users", "contents", "interactions", "training_jobs", "training_locks"} <= tables

    content_columns = {c["name"] for c in inspector.get_columns("contents")}
    assert {"content_type", "duration_seconds", "is_active", "updated_at"} <= content_columns
    assert {"title", "hashtags_json", "topics_json", "entities_json", "subgenres_json"} <= content_columns

    job_columns = {c["name"] for c in inspector.get_columns("training_jobs")}
    assert {"job_id", "status", "requested_at", "result_json"} <= job_columns

    assert any(c["name"] == "model_type" for c in inspector.get_columns("training_locks"))

    # Phase A: composite index backing the bounded VIDEO ranking hot-path query.
    interaction_indexes = {idx["name"] for idx in inspector.get_indexes("interactions")}
    assert "ix_interactions_user_id_timestamp" in interaction_indexes
    engine.dispose()


def test_unique_constraints_are_enforced_after_migration(migration_db):
    command.upgrade(_alembic_config(migration_db), "head")
    engine = create_engine(f"sqlite:///{migration_db}")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO users (user_id, status, created_at) VALUES ('u-1', 'ACTIVE', '2026-01-01')"
        ))
    with engine.connect() as conn, pytest.raises(IntegrityError), conn.begin():
        conn.execute(text(
            "INSERT INTO users (user_id, status, created_at) VALUES ('u-1', 'ACTIVE', '2026-01-02')"
        ))
    engine.dispose()


def test_downgrade_to_base_then_upgrade_returns_to_head(migration_db):
    config = _alembic_config(migration_db)
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    engine = create_engine(f"sqlite:///{migration_db}")
    # "base" removes every table this project owns; Alembic's own bookkeeping table
    # (alembic_version) is expected to remain (empty of any stamped revision).
    remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
    assert remaining == set()
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{migration_db}")
    tables = set(inspect(engine).get_table_names())
    assert {"users", "contents", "interactions", "training_jobs", "training_locks"} <= tables
    interaction_indexes = {idx["name"] for idx in inspect(engine).get_indexes("interactions")}
    assert "ix_interactions_user_id_timestamp" in interaction_indexes
    engine.dispose()


def test_content_lifecycle_fields_are_backfilled_on_existing_rows(migration_db):
    """A row inserted at revision 0001 (before the content-lifecycle columns exist) must
    end up with sane, non-null values after upgrading through 0002 -- proving the backfill
    (contentType='VIDEO', isActive=true, updatedAt=createdAt) actually runs, not just that
    the columns exist."""
    config = _alembic_config(migration_db)
    command.upgrade(config, "bfcccb4b3d8f")
    engine = create_engine(f"sqlite:///{migration_db}")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO contents (content_id, creator_id, category, popularity_score, created_at) "
            "VALUES ('legacy-1', 'cr-1', 'FOOD', 0.5, '2026-01-01T00:00:00')"
        ))
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{migration_db}")
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT content_type, is_active, updated_at FROM contents WHERE content_id='legacy-1'"
        )).one()
    assert row.content_type == "VIDEO"
    assert row.is_active in (1, True)
    assert row.updated_at == "2026-01-01T00:00:00"
    engine.dispose()


def test_content_semantic_metadata_columns_are_additive_and_nullable(migration_db):
    """A row inserted before the semantic-metadata migration exists must still read back
    cleanly (all five new columns NULL) after upgrading to head -- proving the migration is
    purely additive and never touches/backfills existing rows."""
    config = _alembic_config(migration_db)
    command.upgrade(config, "0b6779c4f8fb")
    engine = create_engine(f"sqlite:///{migration_db}")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO contents (content_id, creator_id, category, content_type, popularity_score, "
            "is_active, created_at, updated_at) VALUES ('legacy-2', 'cr-1', 'SPORT', 'VIDEO', 0.5, 1, "
            "'2026-01-01T00:00:00', '2026-01-01T00:00:00')"
        ))
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{migration_db}")
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT title, hashtags_json, topics_json, entities_json, subgenres_json "
            "FROM contents WHERE content_id='legacy-2'"
        )).one()
    assert tuple(row) == (None, None, None, None, None)
    engine.dispose()

    command.downgrade(config, "0b6779c4f8fb")
    engine = create_engine(f"sqlite:///{migration_db}")
    content_columns = {c["name"] for c in inspect(engine).get_columns("contents")}
    assert not {"title", "hashtags_json", "topics_json", "entities_json", "subgenres_json"} & content_columns
    engine.dispose()


def test_content_canonical_taxonomy_columns_are_additive_and_nullable(migration_db):
    """A row inserted before the canonical-taxonomy-foundation migration exists must still
    read back cleanly (all three new columns NULL) after upgrading to head -- proving the
    migration is purely additive and never touches/backfills existing rows, exactly like
    0b6779c4f8fb's own semantic-metadata migration before it."""
    config = _alembic_config(migration_db)
    command.upgrade(config, "7c9e4d1b8a3f")
    engine = create_engine(f"sqlite:///{migration_db}")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO contents (content_id, creator_id, category, content_type, popularity_score, "
            "is_active, created_at, updated_at) VALUES ('legacy-3', 'cr-1', 'SPORT', 'VIDEO', 0.5, 1, "
            "'2026-01-01T00:00:00', '2026-01-01T00:00:00')"
        ))
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{migration_db}")
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT primary_category, subcategory, taxonomy_version, category "
            "FROM contents WHERE content_id='legacy-3'"
        )).one()
    assert (row.primary_category, row.subcategory, row.taxonomy_version) == (None, None, None)
    assert row.category == "SPORT"  # untouched legacy column
    engine.dispose()

    command.downgrade(config, "7c9e4d1b8a3f")
    engine = create_engine(f"sqlite:///{migration_db}")
    content_columns = {c["name"] for c in inspect(engine).get_columns("contents")}
    assert not {"primary_category", "subcategory", "taxonomy_version"} & content_columns
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{migration_db}")
    content_columns = {c["name"] for c in inspect(engine).get_columns("contents")}
    assert {"primary_category", "subcategory", "taxonomy_version"} <= content_columns
    engine.dispose()


def test_content_canonical_taxonomy_columns_accept_arbitrary_unapproved_values(migration_db):
    """No CHECK/enum constraint exists on primary_category/subcategory/taxonomy_version --
    an arbitrary, not-yet-product-approved string must store without error, proving these
    columns don't silently encode the (unapproved) proposed taxonomy list."""
    command.upgrade(_alembic_config(migration_db), "head")
    engine = create_engine(f"sqlite:///{migration_db}")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO contents (content_id, creator_id, category, content_type, popularity_score, "
            "is_active, created_at, updated_at, primary_category, subcategory, taxonomy_version) "
            "VALUES ('future-1', 'cr-1', 'SPORT', 'VIDEO', 0.5, 1, '2026-01-01T00:00:00', "
            "'2026-01-01T00:00:00', 'SOME_FUTURE_UNAPPROVED_VALUE', 'ANOTHER_FUTURE_VALUE', 'v2-hypothetical')"
        ))
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT primary_category, subcategory, taxonomy_version FROM contents WHERE content_id='future-1'"
        )).one()
    assert tuple(row) == ("SOME_FUTURE_UNAPPROVED_VALUE", "ANOTHER_FUTURE_VALUE", "v2-hypothetical")
    engine.dispose()
