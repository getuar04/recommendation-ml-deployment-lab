"""Focused, direct-call coverage for app.services.user_registration_service: the internal
`_age_from_birthday` edge cases and the two IntegrityError-recovery branches in
apply_user_registered that tests/test_kafka_user_registered_consumer.py's Kafka-level tests
don't exercise (its own happy-path/validation-rejection tests never race a concurrent
writer). Race scenarios are simulated the same way tests/test_event_transactions.py already
proves app.services.event_service's own equivalent races: a real "winner" commit from a
separate session, then a monkeypatch that makes THIS call's own read lie about it (simulating
the narrow window where the read genuinely happened before the winner's commit).
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import User, UserOnboardingContext
from app.schemas.user_events import UserRegisteredEvent
from app.services import user_registration_service

BASE_DATA = {
    "userId": "reg-svc-user-1",
    "birthday": "1998-04-12",
    "interests": ["music"],
    "occurredAt": "2026-09-17T10:00:00Z",
}


def _event(**overrides):
    data = {**BASE_DATA, **overrides}
    return UserRegisteredEvent.model_validate({
        "type": "user.registered", "eventId": "evt-reg-svc-1",
        "target": {"type": "user", "id": data["userId"]}, "data": data,
    })


# --------------------------------------------------------------------- _age_from_birthday edge cases


def test_age_from_birthday_computes_exact_age_as_of_occurred_at():
    age = user_registration_service._age_from_birthday(
        date(1998, 4, 12), as_of=datetime(2026, 9, 17, tzinfo=timezone.utc),
    )
    assert age == 28


def test_age_from_birthday_in_the_future_relative_to_occurred_at_returns_none():
    """Never fabricated/negative -- a birthday after the event's own occurredAt (a genuinely
    malformed-looking but schema-valid date) degrades to 'no age context', matching
    app.core.cohort_context.age_bucket_for's own defensive posture."""
    age = user_registration_service._age_from_birthday(
        date(2030, 1, 1), as_of=datetime(2026, 9, 17, tzinfo=timezone.utc),
    )
    assert age is None


def test_age_from_birthday_outside_the_existing_onboarding_bound_returns_none():
    """A computed age above the existing 0-120 bound (app.schemas.recommendation_schemas.
    UserContext.age's own validation range) is treated as 'no age context', never rejected
    and never clamped/fabricated."""
    age = user_registration_service._age_from_birthday(
        date(1850, 1, 1), as_of=datetime(2026, 9, 17, tzinfo=timezone.utc),
    )
    assert age is None


def test_apply_user_registered_with_future_birthday_persists_no_age(client):
    event = _event(userId="reg-svc-future-birthday", birthday="2030-01-01")
    db = SessionLocal()
    try:
        user_registration_service.apply_user_registered(db, event)
    finally:
        db.close()
    check_db = SessionLocal()
    try:
        ctx = check_db.get(UserOnboardingContext, "reg-svc-future-birthday")
    finally:
        check_db.close()
    assert ctx is not None and ctx.age is None


# --------------------------------------------------------------------- _safe_region edge cases
#
# Task: user.registered contract audit re-audit finding: region IS now persisted when it
# safely fits the existing short-code contract, rather than being unconditionally discarded.


def test_safe_region_persists_a_short_compatible_value_normalized():
    assert user_registration_service._safe_region("al") == "AL"
    assert user_registration_service._safe_region("  us  ") == "US"


def test_safe_region_drops_a_value_too_long_for_the_existing_column():
    """A genuine free-form address -- the exact incompatibility this module's own docstring
    documents -- is dropped, never truncated."""
    assert user_registration_service._safe_region("221B Baker Street, London") is None


def test_safe_region_treats_none_and_blank_as_absent():
    assert user_registration_service._safe_region(None) is None
    assert user_registration_service._safe_region("") is None
    assert user_registration_service._safe_region("   ") is None


def test_safe_region_boundary_exactly_at_the_column_limit_is_accepted():
    eight_chars = "ABCDEFGH"
    assert len(eight_chars) == user_registration_service._REGION_MAX_LENGTH
    assert user_registration_service._safe_region(eight_chars) == eight_chars


def test_safe_region_one_character_over_the_boundary_is_dropped():
    nine_chars = "ABCDEFGHI"
    assert len(nine_chars) == user_registration_service._REGION_MAX_LENGTH + 1
    assert user_registration_service._safe_region(nine_chars) is None


# --------------------------------------------------------------------- concurrent-write races


def test_user_row_creation_race_is_recovered_not_crashed(client, monkeypatch):
    """Simulates a concurrent creator (e.g. an interaction event's own _ensure_user, or
    another instance of this same consumer) winning between this call's existence check and
    its own commit -- covers apply_user_registered's User-row IntegrityError recovery
    branch."""
    user_id = "race-user-creation"
    winner_db = SessionLocal()
    try:
        winner_db.add(User(user_id=user_id, status="ACTIVE"))
        winner_db.commit()
    finally:
        winner_db.close()

    def _stale_ensure_user(db_, uid):
        # Unconditionally adds, simulating a stale "doesn't exist" read even though the
        # winner above already committed this exact user_id first.
        db_.add(User(user_id=uid, status="ACTIVE"))

    monkeypatch.setattr(user_registration_service, "_ensure_user", _stale_ensure_user)

    event = _event(userId=user_id)
    db = SessionLocal()
    try:
        user_registration_service.apply_user_registered(db, event)  # must not raise
    finally:
        db.close()

    check_db = SessionLocal()
    try:
        rows = check_db.scalars(select(User).where(User.user_id == user_id)).all()
    finally:
        check_db.close()
    assert len(rows) == 1


def test_onboarding_context_creation_race_recovers_via_update(client, monkeypatch):
    """Simulates a concurrent context write (e.g. a simultaneous POST /users onboarding
    call, or another instance of this consumer) winning between this call's read and its own
    insert -- covers apply_user_registered's UserOnboardingContext IntegrityError recovery
    branch (rollback, re-fetch, update-on-top-of-the-winner's-row instead of crashing)."""
    user_id = "race-context-creation"
    winner_db = SessionLocal()
    try:
        winner_db.add(User(user_id=user_id, status="ACTIVE"))
        winner_db.add(UserOnboardingContext(user_id=user_id, age=99, region=None, language=None, interests_json=None))
        winner_db.commit()
    finally:
        winner_db.close()

    event = _event(userId=user_id, interests=["gaming"])
    db = SessionLocal()
    try:
        real_get = db.get
        calls = {"n": 0}

        def flaky_get(model, pk):
            calls["n"] += 1
            if model is UserOnboardingContext and calls["n"] == 1:
                # Simulate the narrow race window: this read happened before the winner's
                # commit was visible, even though (as proven above) it has already landed.
                return None
            return real_get(model, pk)

        monkeypatch.setattr(db, "get", flaky_get)
        user_registration_service.apply_user_registered(db, event)  # must not raise
        assert calls["n"] == 2  # the forced-stale first read, then the post-rollback recovery fetch
    finally:
        db.close()

    check_db = SessionLocal()
    try:
        rows = check_db.scalars(select(UserOnboardingContext).where(UserOnboardingContext.user_id == user_id)).all()
    finally:
        check_db.close()
    assert len(rows) == 1  # never duplicated
    assert rows[0].interests == ["GAMING"]  # this event's values applied on top of the winner's row
