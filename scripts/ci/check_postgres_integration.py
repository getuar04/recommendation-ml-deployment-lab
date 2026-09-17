"""PostgreSQL-specific migration/integration validation -- the one thing the SQLite-only
pytest suite can never prove (`tests/conftest.py` unconditionally forces
`DATABASE_URL=sqlite://` at import time, for every test, by design -- see its own
docstring-equivalent comment). Deliberately run as a plain script
(`python -m scripts.ci.check_postgres_integration`), never through pytest: routing it through
pytest would either be silently overridden back to SQLite by conftest, or require
special-casing conftest for one script and risking the existing, working SQLite isolation
every other test relies on. This script imports nothing from `tests/` and never participates
in pytest collection.

Requires `DATABASE_URL` to already point at a real, disposable PostgreSQL database (a GitLab
CI service container, or an operator-provided throwaway one) -- refuses to run against
SQLite or an unset URL, and never prints the credential portion of the URL.

What this proves, in order:
1. The database becomes reachable within a bounded number of retries.
2. `alembic upgrade head` succeeds against real PostgreSQL (not just SQLite).
3. The resulting schema has the expected tables/columns/indexes.
4. `alembic downgrade base` removes every project table.
5. With the schema at `base`, a normal (non-migration) `import app.main` against this same
   PostgreSQL URL creates NO tables -- proving `Base.metadata.create_all()` is genuinely
   unreachable for a `postgresql://` `DATABASE_URL`, empirically, not merely "currently
   written that way" (see `app/main.py`'s `if DATABASE_URL.startswith("sqlite"):` gate).
6. `alembic upgrade head` again succeeds (re-upgrade after downgrade).
7. A focused HTTP smoke check (`TestClient`, real PostgreSQL-backed session) proves the
   application actually serves a request end-to-end against this engine/driver, not just
   that the schema looks right.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONNECT_RETRY_SECONDS = 30
CONNECT_RETRY_INTERVAL = 1.0
SUBPROCESS_TIMEOUT_SECONDS = 60

REQUIRED_TABLES = {"users", "contents", "interactions", "training_jobs", "training_locks"}
REQUIRED_CONTENT_COLUMNS = {"content_type", "duration_seconds", "is_active", "updated_at"}


def redacted(url: str) -> str:
    """Masks the password portion of a SQLAlchemy-style DB URL for safe logging."""
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", url)


def require_postgres_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url.startswith("postgresql"):
        print(
            f"DATABASE_URL must point at PostgreSQL (got {redacted(url) or '<unset>'}); "
            "refusing to run against SQLite or an unset URL.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return url


def wait_for_ready(url: str, *, retry_seconds: float = CONNECT_RETRY_SECONDS) -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError

    deadline = time.monotonic() + retry_seconds
    engine = create_engine(url)
    last_error: Exception | None = None
    try:
        while time.monotonic() < deadline:
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                print(f"PostgreSQL ready at {redacted(url)}")
                return
            except OperationalError as exc:
                last_error = exc
                time.sleep(CONNECT_RETRY_INTERVAL)
    finally:
        engine.dispose()
    print(f"PostgreSQL did not become ready within {retry_seconds}s: {last_error}", file=sys.stderr)
    raise SystemExit(1)


def alembic_config(url: str, repo_root: Path = REPO_ROOT):
    from alembic.config import Config
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def assert_schema_at_head(url: str) -> None:
    from sqlalchemy import create_engine, inspect
    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        missing_tables = REQUIRED_TABLES - tables
        if missing_tables:
            raise AssertionError(f"missing expected tables after upgrade: {sorted(missing_tables)}")

        content_columns = {c["name"] for c in inspector.get_columns("contents")}
        missing_columns = REQUIRED_CONTENT_COLUMNS - content_columns
        if missing_columns:
            raise AssertionError(f"contents table missing columns: {sorted(missing_columns)}")

        interaction_indexes = {idx["name"] for idx in inspector.get_indexes("interactions")}
        if "ix_interactions_user_id_timestamp" not in interaction_indexes:
            raise AssertionError("interactions table missing ix_interactions_user_id_timestamp index")

        user_id_is_unique = any(
            "user_id" in idx["column_names"] and idx["unique"] for idx in inspector.get_indexes("users")
        ) or any("user_id" in uc["column_names"] for uc in inspector.get_unique_constraints("users"))
        if not user_id_is_unique:
            raise AssertionError("users.user_id has no unique constraint or unique index")
    finally:
        engine.dispose()


def assert_schema_at_base(url: str) -> None:
    from sqlalchemy import create_engine, inspect
    engine = create_engine(url)
    try:
        remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
        if remaining:
            raise AssertionError(f"downgrade to base left unexpected tables: {sorted(remaining)}")
    finally:
        engine.dispose()


def _run_subprocess(script: str, *, url: str, timeout: int = SUBPROCESS_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["DATABASE_URL"] = url
    env.setdefault("APP_ENV", "test")
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=timeout, check=False,
    )


def assert_no_ddl_on_normal_startup(url: str) -> None:
    """Schema must already be at `base` (no project tables) before this is called. Proves
    `app.main`'s SQLite-only `create_all()` gate empirically, in a fresh process, rather than
    trusting a source-code grep."""
    result = _run_subprocess("import app.main", url=url)
    if result.returncode != 0:
        raise AssertionError(f"importing app.main against PostgreSQL failed:\n{result.stderr[-2000:]}")
    assert_schema_at_base(url)


def run_smoke_http_check(url: str) -> None:
    """One real HTTP round-trip through a PostgreSQL-backed session -- proves the
    engine/driver/session actually serve a request, not just that the schema looks right."""
    script = (
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "client = TestClient(app)\n"
        "health = client.get('/api/v1/recommendation-ml-service/health')\n"
        "assert health.status_code in (200, 503), ('health', health.status_code, health.text)\n"
        "created = client.post('/api/v1/recommendation-ml-service/users', json={'userId': 'ci-postgres-smoke-user'})\n"
        "assert created.status_code in (201, 409), ('create user', created.status_code, created.text)\n"
        "print('SMOKE_CHECK_OK')\n"
    )
    result = _run_subprocess(script, url=url)
    if result.returncode != 0 or "SMOKE_CHECK_OK" not in result.stdout:
        raise AssertionError(
            f"PostgreSQL smoke HTTP check failed:\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )


def main() -> int:
    url = require_postgres_url()
    print(f"Target: {redacted(url)}")
    wait_for_ready(url)

    from alembic import command

    config = alembic_config(url)

    print("Step 1/5: alembic upgrade head")
    command.upgrade(config, "head")
    assert_schema_at_head(url)
    print("  OK: schema matches expected tables/columns/indexes at head.")

    print("Step 2/5: alembic downgrade base")
    command.downgrade(config, "base")
    assert_schema_at_base(url)
    print("  OK: downgrade to base removed every project table.")

    print("Step 3/5: normal app startup issues no DDL against PostgreSQL")
    assert_no_ddl_on_normal_startup(url)
    print("  OK: app.main does not call Base.metadata.create_all() for a PostgreSQL DATABASE_URL.")

    print("Step 4/5: alembic upgrade head (re-upgrade after downgrade)")
    command.upgrade(config, "head")
    assert_schema_at_head(url)
    print("  OK: re-upgrade to head succeeded and schema is correct again.")

    print("Step 5/5: focused PostgreSQL integration smoke test (real HTTP request)")
    run_smoke_http_check(url)
    print("  OK: health + user-creation requests succeeded against PostgreSQL.")

    print("\nPostgreSQL migration/integration check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
