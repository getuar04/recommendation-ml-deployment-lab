"""Fast, database-less Alembic sanity check: exactly one head, and every revision module
imports/loads without error. Never opens a database connection -- `app.db.database.engine`
is never touched, so this needs no `DATABASE_URL` at all and is safe to run in the CI
`validate` stage before any service container exists.

The full upgrade/downgrade/schema-assertion contract is exercised separately: against
SQLite by `tests/test_migrations.py` (already part of the normal pytest suite), and against
PostgreSQL by `scripts/ci/check_postgres_integration.py` (the CI `integration` stage). This
script exists as a sub-second gate before either of those heavier checks runs.

Run from the repository root: `python -m scripts.ci.check_migrations`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]


def alembic_config(repo_root: Path = REPO_ROOT) -> Config:
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "migrations"))
    return config


def check_single_head(script_dir: ScriptDirectory) -> list[str]:
    heads = script_dir.get_heads()
    if len(heads) != 1:
        return [f"expected exactly 1 Alembic head, found {len(heads)}: {sorted(heads)}"]
    return []


def check_revisions_load(script_dir: ScriptDirectory) -> list[str]:
    errors: list[str] = []
    for revision in script_dir.walk_revisions():
        if revision.module is None:
            errors.append(f"revision {revision.revision} failed to load its module")
    return errors


def main() -> int:
    script_dir = ScriptDirectory.from_config(alembic_config())
    errors = check_single_head(script_dir) + check_revisions_load(script_dir)
    if errors:
        print("Alembic migration sanity check FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    heads = script_dir.get_heads()
    revision_count = sum(1 for _ in script_dir.walk_revisions())
    print(f"OK: single Alembic head ({heads[0]}), {revision_count} revision(s) import cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
