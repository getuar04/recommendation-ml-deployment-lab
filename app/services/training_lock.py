"""Restart-safe, per-model-type training lock backed by the `training_locks` table.

Only one VIDEO training job and one LIVE training job may run at a time; VIDEO and LIVE use
separate rows (and write to separate artifact paths) so they may run concurrently. Locking
is transactional at the database level -- `model_type` is the table's primary key, so two
concurrent `acquire()` calls racing to insert the same row cannot both succeed, unlike a
Python-process-local lock (which is neither restart-safe nor multi-worker-safe).

A lock older than `TRAINING_STALE_LOCK_SECONDS` is treated as abandoned -- the process that
held it crashed or was killed without releasing it -- and is reclaimed on the next
`acquire()` attempt rather than blocking training forever.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import TRAINING_STALE_LOCK_SECONDS
from app.core.logging import logger
from app.db.models import TrainingLock

__all__ = ["TrainingAlreadyRunningError", "acquire", "release"]


class TrainingAlreadyRunningError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_stale(lock: TrainingLock) -> bool:
    locked_at = lock.locked_at
    if locked_at.tzinfo is None:
        locked_at = locked_at.replace(tzinfo=timezone.utc)
    return (_now() - locked_at).total_seconds() > TRAINING_STALE_LOCK_SECONDS


def acquire(db: Session, model_type: str, job_id: str) -> None:
    """Acquire the lock for `model_type`, reclaiming a stale lock if one is found.
    Raises `TrainingAlreadyRunningError` if a live lock is already held."""
    existing = db.scalar(select(TrainingLock).where(TrainingLock.model_type == model_type))
    if existing is not None:
        if not _is_stale(existing):
            raise TrainingAlreadyRunningError(f"A {model_type} training job is already running.")
        logger.warning("Reclaiming stale %s training lock held by job %s", model_type, existing.job_id)
        # Optimistic compare-and-swap: the UPDATE only takes effect if the row still has
        # exactly the (job_id, locked_at) we just read as stale. Two processes can observe
        # the same stale lock concurrently; without this WHERE clause, a blind overwrite
        # would let both "win" the reclaim and run training simultaneously. If another
        # process reclaims first, this affects zero rows and we must not proceed as the
        # winner.
        # Session.execute() is typed to return the general Result base, but for a Core UPDATE
        # statement like this one it's always a CursorResult (which has .rowcount) at runtime.
        result = cast(CursorResult, db.execute(
            update(TrainingLock)
            .where(
                TrainingLock.model_type == model_type,
                TrainingLock.job_id == existing.job_id,
                TrainingLock.locked_at == existing.locked_at,
            )
            .values(job_id=job_id, locked_at=_now())
        ))
        db.commit()
        if result.rowcount == 0:
            raise TrainingAlreadyRunningError(f"A {model_type} training job is already running.")
        return
    db.add(TrainingLock(model_type=model_type, job_id=job_id, locked_at=_now()))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise TrainingAlreadyRunningError(f"A {model_type} training job is already running.")


def release(db: Session, model_type: str, job_id: str) -> None:
    """Release the lock for `model_type` only if it is still held by `job_id` -- a job
    that lost a stale-lock race to a newer job must not release the newer job's lock."""
    lock = db.scalar(select(TrainingLock).where(TrainingLock.model_type == model_type))
    if lock is not None and lock.job_id == job_id:
        db.delete(lock)
        db.commit()
