"""Group C: training job persistence and status-transition validation."""
import pytest

from app.db.database import SessionLocal
from app.services import job_store
from app.services.job_store import InvalidJobTransitionError


def test_job_persists_through_pending_running_succeeded():
    db = SessionLocal()
    try:
        job_id = job_store.create_job(db, "VIDEO_TRAINING")
        assert job_store.get_job(db, job_id)["status"] == "PENDING"
        job_store.mark_running(db, job_id)
        assert job_store.get_job(db, job_id)["status"] == "RUNNING"
        job_store.mark_succeeded(db, job_id, {"modelVersion": "v1", "selectedModel": "LogisticRegression"})
    finally:
        db.close()

    # A fresh session/connection -- simulating a process restart reading persisted state --
    # must see the same completed job with its full result, not just an in-memory echo.
    restarted = SessionLocal()
    try:
        job = job_store.get_job(restarted, job_id)
        assert job["status"] == "SUCCEEDED"
        assert job["result"]["modelVersion"] == "v1"
        assert job["result"]["selectedModel"] == "LogisticRegression"
        assert job["finishedAt"] is not None
    finally:
        restarted.close()


def test_job_persists_failure_with_error_details():
    db = SessionLocal()
    try:
        job_id = job_store.create_job(db, "LIVE_TRAINING")
        job_store.mark_running(db, job_id)
        job_store.mark_failed(db, job_id, "INSUFFICIENT_LIVE_TRAINING_DATA", "not enough rows")
        job = job_store.get_job(db, job_id)
        assert job["status"] == "FAILED"
        assert job["error"] == {"error": "INSUFFICIENT_LIVE_TRAINING_DATA", "message": "not enough rows"}
        assert job["result"] is None
    finally:
        db.close()


def test_invalid_transition_is_rejected_and_job_stays_in_terminal_state():
    db = SessionLocal()
    try:
        job_id = job_store.create_job(db, "VIDEO_TRAINING")
        job_store.mark_running(db, job_id)
        job_store.mark_succeeded(db, job_id, {"modelVersion": "v1"})
        with pytest.raises(InvalidJobTransitionError):
            job_store.mark_running(db, job_id)  # SUCCEEDED -> RUNNING is not a valid transition
        assert job_store.get_job(db, job_id)["status"] == "SUCCEEDED"
    finally:
        db.close()


def test_pending_cannot_go_directly_to_succeeded_without_running():
    db = SessionLocal()
    try:
        job_id = job_store.create_job(db, "VIDEO_TRAINING")
        with pytest.raises(InvalidJobTransitionError):
            job_store._transition(db, job_id, "SUCCEEDED")
        assert job_store.get_job(db, job_id)["status"] == "PENDING"
    finally:
        db.close()


def test_unknown_job_lookup_returns_none():
    db = SessionLocal()
    try:
        assert job_store.get_job(db, "does-not-exist") is None
    finally:
        db.close()


def test_transition_on_an_unknown_job_id_is_a_safe_no_op():
    """_transition's job-not-found branch: a job_id with no matching row (e.g. a race, or a
    stale id) must log and return, never raise -- distinct from an INVALID transition on a
    real job (InvalidJobTransitionError)."""
    db = SessionLocal()
    try:
        job_store.mark_running(db, "does-not-exist")  # must not raise
        assert job_store.get_job(db, "does-not-exist") is None
    finally:
        db.close()
