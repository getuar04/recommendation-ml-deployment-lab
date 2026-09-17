"""Production-oriented Alembic migration runner, invoked once at container startup (see
Dockerfile's CMD) BEFORE uvicorn starts serving -- so the application can never begin
serving requests against a PostgreSQL schema that is behind the code. This is exactly the
failure this script exists to prevent: the app container previously started (and reported
healthy) with no migration step at all, while the database was 2 revisions behind head,
and the first real request failed with `psycopg.errors.UndefinedColumn: column
contents.category_confidence does not exist`.

Skips entirely for a SQLite DATABASE_URL: mirrors app.main's own "SQLite bypasses Alembic"
carve-out (`Base.metadata.create_all()` there is the sole schema-management path for the
isolated SQLite dev/test path, never Alembic) -- there is no persistent PostgreSQL schema
to protect on that path.

Idempotent: `alembic upgrade head` is a no-op once the schema is already current (the
normal case on every restart after the first). Race-safe across multiple concurrently
starting replicas via a PostgreSQL session-level advisory lock held for the duration of
the upgrade: a second replica starting at the same moment blocks on the lock instead of
racing Alembic's own DDL, then finds the schema already at head once it acquires it.

Never resets, drops, or recreates anything: only ever calls `alembic upgrade head`, never
`downgrade`/`create_all`/DB creation. Any failure (unreachable database, a broken
revision, a conflicting concurrent DDL change) exits non-zero -- the Dockerfile CMD chains
this script with `&&` so a failed migration means uvicorn never starts, instead of the
service silently serving against an incompatible schema.

Usage: python -m scripts.run_migrations
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# A fixed, stable 64-bit-safe key for pg_advisory_lock -- must be identical across every
# process that might race this script (every app replica starting concurrently), and
# distinct from any other advisory lock this codebase might ever take. Derived from a
# fixed string (not randomly generated), so it is identical across processes/restarts
# with nothing to share out-of-band: zlib.crc32(b"recommendation-ml-service:alembic-migrations").
ADVISORY_LOCK_KEY = 2596996162

CONNECT_RETRY_SECONDS = 30.0
CONNECT_RETRY_INTERVAL = 1.0


def redacted(url: str) -> str:
    """Masks the password portion of a SQLAlchemy-style DB URL for safe logging."""
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:***@", url)


def wait_for_ready(url: str, *, retry_seconds: float = CONNECT_RETRY_SECONDS) -> None:
    """Defense in depth beyond docker-compose's own `depends_on: condition: service_healthy`
    (which only proves Postgres accepts connections, e.g. via pg_isready) -- matters for any
    deployment target that does not offer an equivalent readiness gate."""
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
                print(f"run_migrations: PostgreSQL ready at {redacted(url)}")
                return
            except OperationalError as exc:
                last_error = exc
                time.sleep(CONNECT_RETRY_INTERVAL)
    finally:
        engine.dispose()
    raise RuntimeError(f"PostgreSQL did not become ready within {retry_seconds}s: {last_error}")


def alembic_config(url: str, repo_root: Path = REPO_ROOT):
    from alembic.config import Config
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    return config


def upgrade_with_advisory_lock(url: str) -> None:
    from sqlalchemy import create_engine, text

    lock_engine = create_engine(url)
    # AUTOCOMMIT: pg_advisory_lock/unlock are session-scoped, not transaction-scoped --
    # each call must run standalone, not inside an implicit transaction this connection
    # would otherwise leave open (and never commit) until close.
    lock_conn = lock_engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        print(f"run_migrations: acquiring advisory lock {ADVISORY_LOCK_KEY} "
              "(blocks here if another replica is migrating concurrently)...")
        lock_conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": ADVISORY_LOCK_KEY})
        try:
            from alembic import command
            print("run_migrations: alembic upgrade head")
            command.upgrade(alembic_config(url), "head")
            print("run_migrations: schema is at head.")
        finally:
            lock_conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": ADVISORY_LOCK_KEY})
    finally:
        lock_conn.close()
        lock_engine.dispose()


def main() -> int:
    from app.core.config import DATABASE_URL

    if DATABASE_URL.startswith("sqlite"):
        print("run_migrations: DATABASE_URL is SQLite -- Alembic is not used for this path "
              "(app.main's own Base.metadata.create_all() covers it); nothing to do.")
        return 0

    print(f"run_migrations: target {redacted(DATABASE_URL)}")
    wait_for_ready(DATABASE_URL)
    upgrade_with_advisory_lock(DATABASE_URL)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # fail-fast: any error here must stop container startup
        print(f"run_migrations: FAILED -- {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
