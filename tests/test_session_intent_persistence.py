"""Persistent/shared session search-intent storage (spec §P):
app.services.session_intent_provider.DatabaseSessionIntentProvider, backed by the new
`session_search_intent` table (app.db.models.SessionSearchIntent). Mirrors
tests/test_session_intent_provider.py's InMemorySessionIntentProvider unit tests (same TTL
decay/expiry/replace/isolation invariants), but proves the additional guarantee the in-memory
implementation could never give: state survives a fresh provider instance reading the SAME
database, which is exactly what "shared across workers/processes" and "survives a restart"
reduce to at the unit-test level.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.db.database import SessionLocal
from app.db.models import SessionSearchIntent
from app.schemas.recommendation_schemas import SearchIntent
from app.services.session_intent_provider import (
    SEARCH_INTENT_TTL,
    DatabaseSessionIntentProvider,
)

FIXED_NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _intent(**overrides):
    payload = {"query": "Formula 1 pit stop", "entities": ["Formula 1"], "confidence": 1.0}
    payload.update(overrides)
    return SearchIntent.model_validate(payload)


def test_record_search_creates_an_immediately_active_intent(db):
    provider = DatabaseSessionIntentProvider()
    provider.record_search(db, "user-1", _intent(), now=FIXED_NOW)
    active = provider.get_active_intent(db, "user-1", now=FIXED_NOW)
    assert active is not None
    assert active.query == "Formula 1 pit stop"
    assert active.confidence == pytest.approx(1.0)


def test_confidence_decays_linearly_toward_zero_as_time_passes(db):
    provider = DatabaseSessionIntentProvider()
    provider.record_search(db, "user-1", _intent(confidence=1.0), now=FIXED_NOW)
    half_way = FIXED_NOW + SEARCH_INTENT_TTL / 2
    active = provider.get_active_intent(db, "user-1", now=half_way)
    assert active is not None
    assert 0.45 < active.confidence < 0.55


def test_intent_expires_at_ttl_boundary_and_is_lazily_evicted(db):
    """(22) An expired stored intent must stop affecting recommendation -- and the row is
    actually deleted (lazy eviction), not merely filtered on read."""
    provider = DatabaseSessionIntentProvider()
    provider.record_search(db, "user-1", _intent(), now=FIXED_NOW)
    just_before = FIXED_NOW + SEARCH_INTENT_TTL - timedelta(seconds=1)
    just_after = FIXED_NOW + SEARCH_INTENT_TTL + timedelta(seconds=1)
    assert provider.get_active_intent(db, "user-1", now=just_before) is not None
    assert provider.get_active_intent(db, "user-1", now=just_after) is None
    assert db.get(SessionSearchIntent, "user-1") is None  # actually deleted, not just filtered


def test_new_search_replaces_old_search_outright(db):
    provider = DatabaseSessionIntentProvider()
    provider.record_search(db, "user-1", _intent(query="Coldplay"), now=FIXED_NOW)
    provider.record_search(db, "user-1", _intent(query="Minecraft"), now=FIXED_NOW + timedelta(minutes=1))
    active = provider.get_active_intent(db, "user-1", now=FIXED_NOW + timedelta(minutes=1))
    assert active.query == "Minecraft"
    assert db.query(SessionSearchIntent).filter_by(user_id="user-1").count() == 1  # replaced, not appended


def test_user_isolation_search_for_one_user_does_not_leak_to_another(db):
    provider = DatabaseSessionIntentProvider()
    provider.record_search(db, "user-x", _intent(query="Coldplay"), now=FIXED_NOW)
    assert provider.get_active_intent(db, "user-y", now=FIXED_NOW) is None


def test_clear_removes_an_active_intent(db):
    provider = DatabaseSessionIntentProvider()
    provider.record_search(db, "user-1", _intent(), now=FIXED_NOW)
    provider.clear(db, "user-1")
    assert provider.get_active_intent(db, "user-1", now=FIXED_NOW) is None


# --------------------------------------------------------------------------- (24) survives re-instantiation

def test_intent_survives_a_fresh_provider_instance_reading_the_same_database(db):
    """(24, 20) The whole point of persistence: a SECOND, brand-new provider object (standing
    in for a different worker process, or the same process after a restart) reading the same
    database sees the intent the first instance recorded -- impossible with the old in-memory
    single-process dict."""
    first_provider = DatabaseSessionIntentProvider()
    first_provider.record_search(db, "user-1", _intent(query="Coldplay"), now=FIXED_NOW)

    second_provider = DatabaseSessionIntentProvider()
    active = second_provider.get_active_intent(db, "user-1", now=FIXED_NOW + timedelta(minutes=1))
    assert active is not None
    assert active.query == "Coldplay"


# --------------------------------------------------------------------------- (23) failure safety

def test_read_failure_degrades_to_no_active_intent_never_raises(db, monkeypatch):
    """(23) A transient store failure on the READ path must never break a recommendation
    response -- get_active_intent/peek swallow the exception and report 'no intent'."""
    provider = DatabaseSessionIntentProvider()
    provider.record_search(db, "user-1", _intent(), now=FIXED_NOW)

    def _broken_get(*args, **kwargs):
        raise RuntimeError("simulated database outage")

    monkeypatch.setattr(db, "get", _broken_get)
    assert provider.peek(db, "user-1", now=FIXED_NOW) is None
    assert provider.get_active_intent(db, "user-1", now=FIXED_NOW) is None


def test_read_with_db_none_degrades_to_no_active_intent_never_raises():
    """Regression: `recommend(None, request)` (offline/demo scoring, e.g.
    scripts/demo_production_recommendation_trace.py) is an established calling convention with
    no real DB session. `peek`/`get_active_intent` must swallow the resulting AttributeError
    from `db.get(...)` and report "no intent" -- the defensive `db.rollback()` this except
    block also performs must itself be skipped when there is no session to roll back, not
    raise a second AttributeError (`'NoneType' object has no attribute 'rollback'`)."""
    provider = DatabaseSessionIntentProvider()
    assert provider.peek(None, "user-1", now=FIXED_NOW) is None
    assert provider.get_active_intent(None, "user-1", now=FIXED_NOW) is None


# --------------------------------------------------------------------------- (25) stale intent cannot live forever

def test_stale_intent_cannot_influence_a_much_later_session_indefinitely(db):
    """(25) A search recorded long ago must not still be "active" arbitrarily far in the
    future -- TTL is absolute (SESSION_WINDOW-derived), not refreshed by mere lookups."""
    provider = DatabaseSessionIntentProvider()
    provider.record_search(db, "user-1", _intent(), now=FIXED_NOW)
    far_future = FIXED_NOW + timedelta(days=30)
    assert provider.get_active_intent(db, "user-1", now=far_future) is None
    # Repeated lookups before expiry must not push expires_at forward (no sliding window).
    provider.record_search(db, "user-2", _intent(), now=FIXED_NOW)
    for minute in range(1, 10):
        provider.get_active_intent(db, "user-2", now=FIXED_NOW + timedelta(minutes=minute))
    row = db.get(SessionSearchIntent, "user-2")
    assert row.expires_at.replace(tzinfo=timezone.utc) == FIXED_NOW + SEARCH_INTENT_TTL
