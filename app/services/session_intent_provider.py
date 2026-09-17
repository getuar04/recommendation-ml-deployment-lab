"""Search -> Session Intent Provider (VIDEO only).

Two implementations of the same `SessionIntentProvider` Protocol:

- `DatabaseSessionIntentProvider` (default -- SESSION_INTENT_STORE=DATABASE): persists each
  user's active search intent as one row in `session_search_intent` (this project's own
  existing PostgreSQL/SQLite infrastructure -- see app.db.models.SessionSearchIntent). Shared
  across every RMS worker/process and survives a restart, closing the exact gap the previous
  single-process in-memory-only implementation had. No new external dependency (no Redis/cache
  service) is introduced -- this is the same database every other piece of VIDEO state already
  lives in.
- `InMemorySessionIntentProvider` (SESSION_INTENT_STORE=MEMORY): the original single-process,
  in-memory, TTL-bounded store, kept available for tests/local debugging that want to avoid DB
  round-trips, or as an explicit operator opt-out.

PRODUCTION TARGET (unchanged by this pass): an explicit user search is expected to eventually
flow Event Tracking Service -> Kafka -> User Behavior Service, with UBS owning "current
session/search intent" the same way it will eventually own the rest of the behavior profile
(see app.services.user_profile_service / app.ml.dataset_builder.FeatureHistory.from_profile
for the equivalent, already-implemented boundary on the long-term-profile side). RMS's job,
unchanged by that future integration, is only to read the already-decayed, already-scoped
"active intent" for a user and hand it to the same app.ml.reranker.search_relevance() this
module already had -- nothing about matching/boosting changes; only *where the SearchIntent
being boosted with comes from* changes. That is why this file exposes a narrow Protocol
instead of hardwiring either implementation into the recommendation service: swapping in a
real UBS-backed client later is a one-file change.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Protocol

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import SESSION_INTENT_STORE
from app.core.logging import logger
from app.db.models import SessionSearchIntent
from app.ml.dataset_builder import SESSION_WINDOW
from app.schemas.recommendation_schemas import SearchIntent

# Reuses the exact same "current session" window already established by
# app.ml.dataset_builder's session_category_affinity/session_intent_confidence model
# features (SESSION_WINDOW=30min) -- one definition of "session" for the whole codebase,
# not a second, independently-tuned magic number. A search's explicit influence has fully
# decayed to zero by the time it would have left the session window anyway, so an explicit
# search and organic in-session behavior go stale on the same schedule.
SEARCH_INTENT_TTL = SESSION_WINDOW


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class SearchIntentRecord:
    intent: SearchIntent
    recorded_at: datetime

    @property
    def expires_at(self) -> datetime:
        return self.recorded_at + SEARCH_INTENT_TTL


class SessionIntentProvider(Protocol):
    """Contract a future UBS-backed client would implement identically. `db` is threaded
    through every method (even the in-memory implementation, which ignores it) so callers
    never need to know which implementation is active."""

    def record_search(self, db: Session, user_id: str, intent: SearchIntent, *, now: datetime | None = None) -> SearchIntentRecord: ...

    def get_active_intent(self, db: Session, user_id: str, *, now: datetime | None = None) -> SearchIntent | None: ...

    def peek(self, db: Session, user_id: str, *, now: datetime | None = None) -> SearchIntentRecord | None: ...

    def clear(self, db: Session, user_id: str) -> None: ...


def _decay(record: SearchIntentRecord, now: datetime) -> SearchIntent | None:
    """Confidence decays LINEARLY from the recorded value to 0 over SEARCH_INTENT_TTL, then
    the record is treated as absent -- simple, monotonic, easy to reason about and to test
    with an injected clock (no exponential/half-life tuning knobs to justify). Shared by both
    implementations so TTL-decay behavior is identical regardless of storage backend."""
    elapsed_seconds = (now - record.recorded_at).total_seconds()
    remaining_fraction = max(0.0, 1.0 - elapsed_seconds / SEARCH_INTENT_TTL.total_seconds())
    decayed_confidence = record.intent.confidence * remaining_fraction
    if decayed_confidence <= 0.0:
        return None
    return record.intent.model_copy(update={"confidence": decayed_confidence})


class DatabaseSessionIntentProvider:
    """Shared/persistent implementation (spec §P): one row per user in `session_search_intent`
    (app.db.models.SessionSearchIntent). A new search REPLACES the previous row outright
    (upsert-by-user_id -- matches the pre-existing in-memory "latest search wins" semantics).
    Lazy eviction only: an expired row is deleted the next time it's looked up for that user,
    exactly like the in-memory implementation's own eviction policy -- no background sweep
    process, bounded by "one row per user who has ever searched and not yet been read again
    after expiry".

    Failure safety (spec §P: "failure should not break recommendation"): every read
    (`peek`/`get_active_intent`) swallows any database exception and degrades to "no active
    intent" rather than propagating -- a transient DB issue must never turn into a broken
    recommendation response. `record_search`/`clear` are explicit, caller-initiated writes
    (a direct POST /users/{userId}/search-intent call) and are allowed to raise, mapped to a
    normal 5xx by the route layer like any other write failure -- unlike the read path, this
    is a request the caller is actively waiting on and expects an honest outcome for.
    """

    def record_search(self, db: Session, user_id: str, intent: SearchIntent, *, now: datetime | None = None) -> SearchIntentRecord:
        now = now or datetime.now(timezone.utc)
        expires_at = now + SEARCH_INTENT_TTL
        row = db.get(SessionSearchIntent, user_id)
        is_insert = row is None
        if row is None:
            row = SessionSearchIntent(user_id=user_id)
            db.add(row)
        row.intent_json = intent.model_dump_json(by_alias=True)
        row.recorded_at = now
        row.expires_at = expires_at
        if not is_insert:
            db.commit()
            return SearchIntentRecord(intent=intent, recorded_at=now)
        try:
            db.commit()
        except IntegrityError:
            # Lost a race: another concurrent first-time search for this same brand-new
            # user_id inserted and committed its row between our db.get() returning None and
            # our own commit -- "latest search wins" (this class's own upsert-by-user_id
            # semantics, see module docstring) must still hold for the loser, not surface as
            # an internal 500. Roll back the failed insert, then apply this exact write as an
            # UPDATE against the winner's now-persisted row -- mirrors
            # app.services.event_service.store_event's identical concurrent-insert recovery.
            db.rollback()
            row = db.get(SessionSearchIntent, user_id)
            assert row is not None  # the only way this commit could fail is the winner's row existing
            row.intent_json = intent.model_dump_json(by_alias=True)
            row.recorded_at = now
            row.expires_at = expires_at
            db.commit()
        return SearchIntentRecord(intent=intent, recorded_at=now)

    def peek(self, db: Session, user_id: str, *, now: datetime | None = None) -> SearchIntentRecord | None:
        now = now or datetime.now(timezone.utc)
        try:
            row = db.get(SessionSearchIntent, user_id)
            if row is None:
                return None
            if now >= _aware(row.expires_at):
                db.delete(row)
                db.commit()
                return None
            intent = SearchIntent.model_validate_json(row.intent_json)
            return SearchIntentRecord(intent=intent, recorded_at=_aware(row.recorded_at))
        except Exception as exc:  # noqa: BLE001 -- read path must never break recommendation
            # A failed statement (e.g. a missing table, or a lost connection) leaves the
            # SQLAlchemy Session's transaction in a state that requires an explicit rollback
            # before it can be reused -- without this, the NEXT operation on this same
            # request-scoped `db` (record_demographic_context, cohort resolution, ...) would
            # raise sqlalchemy.exc.PendingRollbackError instead of the caller's own, unrelated
            # error. Rolling back here keeps this failure contained to "no active intent",
            # exactly as documented, regardless of what runs next on this session.
            #
            # `db` itself can legitimately be None: `recommend(None, request)` (offline/demo
            # scoring, e.g. scripts/demo_production_recommendation_trace.py) is an established
            # calling convention with no real DB session -- that case already raised the
            # AttributeError caught here, so there is no session to roll back.
            if db is not None:
                db.rollback()
            logger.info("session intent read failed userId=%s failureType=%s", user_id, type(exc).__name__)
            return None

    def get_active_intent(self, db: Session, user_id: str, *, now: datetime | None = None) -> SearchIntent | None:
        now = now or datetime.now(timezone.utc)
        record = self.peek(db, user_id, now=now)
        if record is None:
            return None
        return _decay(record, now)

    def clear(self, db: Session, user_id: str) -> None:
        row = db.get(SessionSearchIntent, user_id)
        if row is not None:
            db.delete(row)
            db.commit()


class InMemorySessionIntentProvider:
    """Single-process, in-memory, TTL-bounded store (SESSION_INTENT_STORE=MEMORY). Kept for
    tests/local debugging that want to avoid DB round-trips, or as an explicit operator
    opt-out -- NOT the default (see DatabaseSessionIntentProvider for the production-shared
    behavior). `db` is accepted by every method for Protocol compatibility and otherwise
    ignored.

    - One record per user; a new search REPLACES the previous one outright.
    - Confidence decays LINEARLY (see `_decay` above).
    - Lazy eviction only -- no background sweep thread.
    - Thread-safe (a plain Lock around dict mutation/reads) since FastAPI may serve
      concurrent requests from multiple threads in-process; NOT multi-process/multi-instance
      safe -- a second RMS process has its own, independent store.
    - Cleared on every process restart -- there is no persistence here by design.
    """

    def __init__(self) -> None:
        self._records: dict[str, SearchIntentRecord] = {}
        self._lock = Lock()

    def record_search(self, db: Session, user_id: str, intent: SearchIntent, *, now: datetime | None = None) -> SearchIntentRecord:
        now = now or datetime.now(timezone.utc)
        record = SearchIntentRecord(intent=intent, recorded_at=now)
        with self._lock:
            self._records[user_id] = record
        return record

    def peek(self, db: Session, user_id: str, *, now: datetime | None = None) -> SearchIntentRecord | None:
        """Raw stored record (undecayed confidence), or None if absent/expired. Powers the
        read-only debug endpoint; `get_active_intent` is what reranking actually uses."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            record = self._records.get(user_id)
            if record is None:
                return None
            if now >= record.expires_at:
                del self._records[user_id]
                return None
            return record

    def get_active_intent(self, db: Session, user_id: str, *, now: datetime | None = None) -> SearchIntent | None:
        now = now or datetime.now(timezone.utc)
        record = self.peek(db, user_id, now=now)
        if record is None:
            return None
        return _decay(record, now)

    def clear(self, db: Session, user_id: str) -> None:
        with self._lock:
            self._records.pop(user_id, None)


def _build_provider() -> SessionIntentProvider:
    if SESSION_INTENT_STORE == "MEMORY":
        return InMemorySessionIntentProvider()
    return DatabaseSessionIntentProvider()


# Process-wide singleton -- see module docstring for what each implementation guarantees.
provider: SessionIntentProvider = _build_provider()
