"""Group 9 (Session 2): interaction-ingestion transaction and idempotency safety.

`store_event()`'s application-level idempotency check (`find_event()` before insert) is
race-prone on its own -- two concurrent requests for the same eventId can both pass it
before either commits. These tests force that exact race deterministically (by making the
initial check briefly lie) to exercise the database-level recovery path: a UNIQUE
constraint violation is caught, the whole failed transaction (including any pending
auto-created User row) is rolled back, and the already-committed row is returned instead of
letting the error surface as an internal 500.
"""
from datetime import datetime, timezone

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import Content, User
from app.schemas.event_schemas import EventCreate
from app.services import event_service
from tests.conftest import ensure_content


def _event(**overrides):
    payload = {
        "eventId": "evt-1", "userId": "u-1", "contentId": "c-1", "creatorId": "cr-1",
        "category": "FOOD", "eventType": "VIDEO_WATCHED", "watchTimeSeconds": 30,
        "contentDurationSeconds": 60, "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(overrides)
    return EventCreate(**payload)


def test_normal_new_event_is_stored(client):
    ensure_content(client, "c-1")
    db = SessionLocal()
    try:
        row, stored = event_service.store_event(db, _event())
        assert stored is True and row.event_id == "evt-1"
    finally:
        db.close()


def test_interactions_alias_behaves_identically_to_events(client):
    """Spec §6/§63: POST /api/v1/recommendation-ml-service/interactions is a backward-compatible alias for
    POST /api/v1/recommendation-ml-service/events -- same validation, idempotency, and derived watchPercentage."""
    ensure_content(client, "c-alias")
    payload = {"eventId": "evt-alias-1", "userId": "u-1", "contentId": "c-alias", "creatorId": "cr-1",
               "category": "FOOD", "eventType": "VIDEO_WATCHED", "watchTimeSeconds": 30,
               "contentDurationSeconds": 60, "timestamp": datetime.now(timezone.utc).isoformat()}
    first = client.post("/api/v1/recommendation-ml-service/interactions", json=payload)
    assert first.status_code == 201
    body = first.json()
    assert body["eventId"] == "evt-alias-1" and body["stored"] is True
    assert body["watchPercentage"] == 50.0

    # Idempotent across both routes: the same eventId submitted again via /events must be
    # recognised as already processed, proving both routes share one underlying store.
    duplicate = client.post("/api/v1/recommendation-ml-service/events", json=payload)
    assert duplicate.status_code == 201 and duplicate.json()["stored"] is False


def test_repeated_event_is_idempotent_without_a_race(client):
    ensure_content(client, "c-1")
    db = SessionLocal()
    try:
        first, first_stored = event_service.store_event(db, _event())
        second, second_stored = event_service.store_event(db, _event())
        assert first_stored is True and second_stored is False
        assert first.id == second.id
    finally:
        db.close()


def test_unknown_content_is_stored_pending_not_rejected(client):
    """Eventual-consistency audit (Task: local-existence blocking validation): RMS does not
    own Content -- a locally-unknown contentId can simply mean content.created hasn't been
    projected yet (out-of-order relative to this interaction), so a genuinely valid
    interaction must never be permanently rejected for that reason alone. The row is
    persisted with content_pending=True (Interaction.content_pending, app.db.models) instead
    of any invented creatorId/category/contentType -- every other column still comes only
    from the event payload itself, exactly like a resolved interaction."""
    db = SessionLocal()
    try:
        row, stored = event_service.store_event(db, _event(eventId="evt-ghost", contentId="ghost"))
        assert stored is True
        assert row.content_pending is True
        assert row.content_id == "ghost"
        assert row.category == "FOOD"
        assert row.creator_id == "cr-1"
    finally:
        db.close()


def test_content_arriving_later_is_validated_normally_for_subsequent_events(client):
    """Once the Content projection catches up, content-dependent checks (active/type-mismatch)
    apply in full to any LATER event for that content_id. Corrective pass (Task: content_pending
    evidence-leak audit): the earlier pending interaction previously stayed content_pending=True
    forever (undocumented as a real gap, not a design choice) -- app.api.content_routes.
    create_content now reconciles it via _resolve_pending_interactions, called once right after
    this Content row commits, so it becomes content_pending=False (evidence-eligible) too,
    proven below by re-reading it fresh from the DB rather than trusting the stale in-memory
    `pending_row` object from before reconciliation ran."""
    db = SessionLocal()
    try:
        pending_row, pending_stored = event_service.store_event(
            db, _event(eventId="evt-then-resolved-1", contentId="later-resolved"),
        )
        assert pending_stored is True and pending_row.content_pending is True
    finally:
        db.close()

    ensure_content(client, "later-resolved", category="FOOD")

    db = SessionLocal()
    try:
        reconciled = db.scalar(select(event_service.Interaction).where(event_service.Interaction.event_id == "evt-then-resolved-1"))
        assert reconciled.content_pending is False

        resolved_row, resolved_stored = event_service.store_event(
            db, _event(eventId="evt-then-resolved-2", contentId="later-resolved"),
        )
        assert resolved_stored is True and resolved_row.content_pending is False
    finally:
        db.close()


def test_interaction_for_a_never_registered_user_is_stored_not_rejected(client):
    """Interaction-before-user tolerance (pre-existing, unchanged by this task's content-side
    fix): _ensure_user auto-creates a minimal User row instead of rejecting -- confirmed
    explicitly here, alongside the equivalent content-side guarantee above."""
    ensure_content(client, "c-never-registered-user")
    db = SessionLocal()
    try:
        _row, stored = event_service.store_event(
            db, _event(eventId="evt-new-user", userId="never-registered-user", contentId="c-never-registered-user"),
        )
        assert stored is True
        assert db.scalar(select(User).where(User.user_id == "never-registered-user")) is not None
    finally:
        db.close()


def test_inactive_content_is_rejected(client):
    ensure_content(client, "c-inactive")
    db = SessionLocal()
    try:
        content = db.scalar(select(Content).where(Content.content_id == "c-inactive"))
        content.is_active = False
        db.commit()
        try:
            event_service.store_event(db, _event(eventId="evt-inactive", contentId="c-inactive"))
            assert False, "expected ContentInactiveError"
        except event_service.ContentInactiveError:
            pass
    finally:
        db.close()


def test_lost_race_on_event_id_falls_back_to_existing_row_not_a_crash(client, monkeypatch):
    """Force the exact race: the idempotency check reports 'not found' even though a
    concurrent request already committed the same eventId, so store_event must reach its
    own INSERT, hit the UNIQUE constraint, and recover -- not raise an unhandled error."""
    ensure_content(client, "race-content")
    event = _event(eventId="race-1", userId="u-race", contentId="race-content")

    db = SessionLocal()
    try:
        winner, winner_stored = event_service.store_event(db, event)
        assert winner_stored is True

        original_find_event = event_service.find_event
        calls = {"n": 0}

        def flaky_find_event(db_, event_id):
            calls["n"] += 1
            return None if calls["n"] == 1 else original_find_event(db_, event_id)

        monkeypatch.setattr(event_service, "find_event", flaky_find_event)

        loser, loser_stored = event_service.store_event(db, event)
        assert loser_stored is False
        assert loser.id == winner.id
        assert calls["n"] == 2  # the forced-false first check, then the post-rollback recovery fetch
    finally:
        db.close()


def test_lost_race_on_new_user_id_with_different_event_ids_is_stored_not_a_crash(client, monkeypatch):
    """Two concurrent events for the SAME brand-new userId but DIFFERENT eventIds can both
    pass _ensure_user's existence check before either commits (unlike the eventId race above,
    this is NOT a duplicate delivery -- both interactions are genuinely new and must both be
    persisted). The loser must retry its own Interaction insert against the now-existing user,
    not surface the User UNIQUE-constraint IntegrityError as an unhandled 500."""
    ensure_content(client, "race-content-3")
    ensure_content(client, "race-content-4")
    winner_event = _event(eventId="race-user-1", userId="brand-new-user-race", contentId="race-content-3")
    loser_event = _event(eventId="race-user-2", userId="brand-new-user-race", contentId="race-content-4")

    db = SessionLocal()
    try:
        _winner, winner_stored = event_service.store_event(db, winner_event)
        assert winner_stored is True

        original_ensure_user = event_service._ensure_user

        def flaky_ensure_user(db_, user_id):
            # Simulate the TOCTOU race directly: a stale read that still thinks the
            # brand-new user doesn't exist yet, even though the winner already committed it.
            db_.add(User(user_id=user_id, status="ACTIVE"))

        monkeypatch.setattr(event_service, "_ensure_user", flaky_ensure_user)

        loser, loser_stored = event_service.store_event(db, loser_event)
        assert loser_stored is True, "a genuinely new interaction was dropped instead of persisted"
        assert loser.event_id == "race-user-2"

        monkeypatch.setattr(event_service, "_ensure_user", original_ensure_user)
        users = db.scalars(select(User).where(User.user_id == "brand-new-user-race")).all()
        assert len(users) == 1, "the losing request's retried User row must not leave a duplicate"
    finally:
        db.close()


def test_no_partial_user_row_when_the_racing_insert_fails(client, monkeypatch):
    """A losing request's auto-created User row must not persist as an orphan when its
    interaction insert fails and rolls back -- both are in the same transaction."""
    ensure_content(client, "race-content-2")
    event = _event(eventId="race-2", userId="original-racer", contentId="race-content-2")

    db = SessionLocal()
    try:
        winner, _ = event_service.store_event(db, event)

        original_find_event = event_service.find_event
        calls = {"n": 0}

        def flaky_find_event(db_, event_id):
            calls["n"] += 1
            return None if calls["n"] == 1 else original_find_event(db_, event_id)

        monkeypatch.setattr(event_service, "find_event", flaky_find_event)

        second_event = _event(eventId="race-2", userId="brand-new-racer", contentId="race-content-2")
        row, stored = event_service.store_event(db, second_event)
        assert stored is False and row.id == winner.id

        assert db.scalar(select(User).where(User.user_id == "brand-new-racer")) is None
    finally:
        db.close()
