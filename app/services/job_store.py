"""Training job tracking persisted in the database (`training_jobs` table).

Replaces the previous in-process dict: a job now survives a process restart because its
state lives in PostgreSQL (or SQLite in tests), not in memory. Status transitions are
validated against a fixed state machine (`_ALLOWED_TRANSITIONS`); an invalid transition is
rejected internally (logged, `InvalidJobTransitionError` raised to the caller) rather than
silently applied.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.db.models import TrainingJob

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "PENDING": {"RUNNING", "CANCELLED", "FAILED"},
    "RUNNING": {"SUCCEEDED", "FAILED", "CANCELLED"},
    "SUCCEEDED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}


class InvalidJobTransitionError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_job(db: Session, job_type: str, *, created_by: str | None = None) -> str:
    job_id = uuid.uuid4().hex
    db.add(TrainingJob(job_id=job_id, model_type=job_type, status="PENDING",
                        requested_at=_now(), created_by=created_by))
    db.commit()
    return job_id


def _transition(db: Session, job_id: str, new_status: str, **fields: Any) -> None:
    job = db.scalar(select(TrainingJob).where(TrainingJob.job_id == job_id))
    if job is None:
        logger.warning("Training job %s not found for transition to %s", job_id, new_status)
        return
    allowed = _ALLOWED_TRANSITIONS.get(job.status, set())
    if new_status not in allowed:
        logger.warning("Rejected invalid training job transition %s -> %s for job %s",
                        job.status, new_status, job_id)
        raise InvalidJobTransitionError(f"{job.status} -> {new_status} is not a valid transition")
    job.status = new_status
    for key, value in fields.items():
        setattr(job, key, value)
    db.commit()


def mark_running(db: Session, job_id: str) -> None:
    _transition(db, job_id, "RUNNING", started_at=_now())


def mark_succeeded(db: Session, job_id: str, result: dict[str, Any]) -> None:
    _transition(
        db, job_id, "SUCCEEDED", finished_at=_now(),
        model_version=result.get("modelVersion"),
        artifact_path=result.get("_artifactPath"),
        metadata_path=result.get("_metadataPath"),
        result_json=json.dumps(result, default=str),
    )


def mark_failed(db: Session, job_id: str, error: str, message: str) -> None:
    _transition(db, job_id, "FAILED", finished_at=_now(), error_code=error, error_message=message)


def get_job(db: Session, job_id: str) -> dict[str, Any] | None:
    job = db.scalar(select(TrainingJob).where(TrainingJob.job_id == job_id))
    if job is None:
        return None
    updated_at = job.finished_at or job.started_at or job.requested_at
    result = json.loads(job.result_json) if job.status == "SUCCEEDED" and job.result_json else None
    return {
        "jobId": job.job_id,
        "jobType": job.model_type,
        "status": job.status,
        "result": result,
        "error": {"error": job.error_code, "message": job.error_message} if job.status == "FAILED" else None,
        "createdAt": job.requested_at.isoformat() if job.requested_at else None,
        "updatedAt": updated_at.isoformat() if updated_at else None,
        "requestedAt": job.requested_at.isoformat() if job.requested_at else None,
        "startedAt": job.started_at.isoformat() if job.started_at else None,
        "finishedAt": job.finished_at.isoformat() if job.finished_at else None,
    }
