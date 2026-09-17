"""DEMO/LOCAL-ONLY reset: deletes only the known demo users' persisted interactions (never
anyone else's data, never Content/creator rows, never the model artifact) so a re-run of
`scripts.seed_demo_users` reproduces the exact baseline interaction counts its own
`assert len(b.events) == N` checks encode.

Deliberately narrow -- it does NOT drop tables, does NOT touch the model artifact (a separate
volume from Postgres; training/promotion is untouched by this script entirely), and does NOT
affect any user outside the fixed demo set below. Search intent is in-process memory
(app.services.session_intent_provider.InMemorySessionIntentProvider), not persisted here at
all -- it self-expires within its TTL, or clears instantly on `docker compose restart app`;
this script cannot reach into a running server's memory and does not try to.

ENVIRONMENT GUARD: refuses to run unless APP_ENV is unset/"local"/"test" AND DATABASE_URL
does not look like a real remote host (no non-local hostname) -- see `_guard_environment()`.
Requires an explicit --yes to actually delete anything; without it, prints what WOULD be
deleted and exits 0.

Usage:
    python -m scripts.reset_demo            # dry run, prints what would be deleted
    python -m scripts.reset_demo --yes       # actually deletes
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit

from scripts.refresh_demo_requests import NOT_INTERESTED_DEMO_USER
from scripts.seed_demo_users import build_all_users

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "postgres", "0.0.0.0", ""}


class ResetGuardError(RuntimeError):
    pass


def _guard_environment() -> None:
    app_env = os.getenv("APP_ENV", "local").strip().lower() or "local"
    if app_env not in ("local", "test"):
        raise ResetGuardError(
            f"Refusing to run: APP_ENV={app_env!r} does not look like a demo/local environment "
            "(only 'local'/'test' are allowed). This guard exists so this script can never "
            "accidentally run against a production-like deployment."
        )
    database_url = os.getenv("DATABASE_URL", "")
    host = urlsplit(database_url.replace("postgresql+psycopg", "postgresql", 1)).hostname or ""
    if database_url.startswith("sqlite"):
        return  # sqlite (tests, isolated demo) is always local by construction
    if host.lower() not in _LOCAL_HOSTS:
        raise ResetGuardError(
            f"Refusing to run: DATABASE_URL host {host!r} does not look local (expected one of "
            f"{sorted(_LOCAL_HOSTS)}). This guard exists so this script can never accidentally "
            "run against a remote/production database."
        )


def _demo_user_ids() -> list[str]:
    users = build_all_users(datetime.now(timezone.utc))
    ids = [user_id for user_id, _ in users.values()]
    ids.append(NOT_INTERESTED_DEMO_USER)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="Actually delete (default: dry run).")
    args = parser.parse_args()

    try:
        _guard_environment()
    except ResetGuardError as exc:
        print(f"RESET REFUSED: {exc}", file=sys.stderr)
        return 1

    # Imported after the environment guard passes -- app.db.database reads DATABASE_URL at
    # import time, so nothing DB-related is touched until the guard above has already cleared.
    from sqlalchemy import delete

    from app.db.database import SessionLocal
    from app.db.models import Interaction

    user_ids = _demo_user_ids()
    db = SessionLocal()
    try:
        existing = db.query(Interaction).filter(Interaction.user_id.in_(user_ids)).count()
        print(f"demo users: {user_ids}")
        print(f"persisted interactions for these users: {existing}")
        if not args.yes:
            print("DRY RUN -- pass --yes to actually delete. Nothing was changed.")
            return 0
        db.execute(delete(Interaction).where(Interaction.user_id.in_(user_ids)))
        db.commit()
        print(f"deleted {existing} interaction rows for the demo user set.")
        print("Content/creator rows and the model artifact were NOT touched.")
        print("Next: python -m scripts.seed_demo_users --base-url <url>  to restore the baseline.")
        print("If a stale search intent is still visible, restart the app container "
              "(docker compose restart app) to clear in-process session state.")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
