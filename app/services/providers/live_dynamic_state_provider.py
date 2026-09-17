"""LIVE dynamic (real-time-ish) state: currentViewerCount, viewerGrowthRate, and recent
likes/gifts, derived from this service's own durable, already-existing local `interactions`
table -- never a new in-memory process-local structure.

Why a query over `Interaction`, not an in-memory `LiveDynamicState` object (the shape the
originating task suggested as an example): every LIVE_JOINED/LIVE_LEFT/LIVE_WATCHED/LIVE_LIKED/
LIVE_GIFT_SENT event is ALREADY durably persisted (`app.services.event_service.store_event`,
the same path `POST /events` and the optional Kafka consumer both use) before this module ever
runs. Deriving state by querying that table at read time, rather than maintaining a second,
parallel in-memory aggregate, gets three properties for free that this task explicitly cares
about and that process-local memory cannot provide: RESTART-SAFE (nothing to lose -- state is
always recomputed from durable rows), MULTI-INSTANCE-SAFE (every replica queries the same
Postgres rows, never its own private view), and no unbounded per-process memory growth (the
query window is always bounded, see LIVE_DYNAMIC_WINDOW_SECONDS/LIVE_VIEWER_STALE_SECONDS in
app.core.config). This is exactly "reuse durable persistence that already exists and is
appropriate" (the originating task's own preferred path over inventing new infrastructure).

Viewer-presence semantics (see LIVE_VIEWER_STALE_SECONDS's own config docstring for the
honesty caveat: no explicit client heartbeat-interval contract exists in this repository, so
that threshold is a documented assumption, not a verified platform contract): a user_id counts
as a CURRENT viewer of a stream if their most recent presence event for that stream
(LIVE_JOINED or LIVE_WATCHED -- the same two event types app.ml.live_session_builder's own
docstring already treats as "JOIN, then zero or more WATCHED pings") is NOT a LIVE_LEFT, and
that event's timestamp is within LIVE_VIEWER_STALE_SECONDS of "now". A LIVE_LEFT immediately
removes that user (their most recent event becomes LEFT). Deliberately NOT derived from total
historical joins/sessions/watches/likes/unique historical users -- see this module's own tests
for the distinction from `app.ml.live_dataset_builder.LiveHistory`'s (historical, not current)
aggregates.

Momentum (`viewerGrowthRate`): compares DISTINCT joiners in the recent half of
LIVE_DYNAMIC_WINDOW_SECONDS against the earlier half -- a join-RATE comparison (not a
concurrent-viewer-count-at-two-past-instants comparison, which would need re-deriving presence
at an arbitrary past instant and is not needed for a momentum signal). Clamped to
`LiveCandidate.viewer_growth_rate`'s own existing bounds (`[-1.0, 10.0]`,
app.schemas.live_schemas) so it always fits the pre-existing, unretrained LIVE model's feature
contract. Never divides by zero, never looks at events after "now", and naturally decays to
0.0 once no new joins occur within the window (both halves empty).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import LIVE_DYNAMIC_WINDOW_SECONDS, LIVE_VIEWER_STALE_SECONDS
from app.db.models import Interaction
from app.schemas.event_schemas import EventType

__all__ = ["NEUTRAL_DYNAMIC_STATE", "LiveDynamicState", "compute_dynamic_state"]

_JOINED = EventType.LIVE_JOINED.value
_LEFT = EventType.LIVE_LEFT.value
_WATCHED = EventType.LIVE_WATCHED.value
_LIKED = EventType.LIVE_LIKED.value
_GIFT_SENT = EventType.LIVE_GIFT_SENT.value
_PRESENCE_EVENT_TYPES = frozenset({_JOINED, _WATCHED})


@dataclass(frozen=True)
class LiveDynamicState:
    current_viewer_count: int
    viewer_growth_rate: float
    recent_likes: int
    recent_gifts: int


NEUTRAL_DYNAMIC_STATE = LiveDynamicState(current_viewer_count=0, viewer_growth_rate=0.0, recent_likes=0, recent_gifts=0)


def _event_type(row: Interaction) -> str:
    value = getattr(row, "event_type", "")
    return getattr(value, "value", value)


def _aware(ts: datetime) -> datetime:
    # A SQLite round-trip (this project's test DB) silently strips tzinfo even for a column
    # declared DateTime(timezone=True) -- same quirk app.ml.live_session_builder._utc and
    # app.ml.dataset_builder._utc already handle. Assumed UTC, never guessed from anything else.
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def _clamp(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, value))


def _state_for_stream(rows: list[Interaction], *, now: datetime) -> LiveDynamicState:
    # Most-recent presence/departure event per user_id -- rows are pre-sorted ascending by
    # timestamp, so a later row simply overwrites an earlier one for the same user_id.
    last_presence_event: dict[str, Interaction] = {}
    for row in rows:
        event_type = _event_type(row)
        if event_type in _PRESENCE_EVENT_TYPES or event_type == _LEFT:
            last_presence_event[row.user_id] = row

    stale_cutoff = now - timedelta(seconds=LIVE_VIEWER_STALE_SECONDS)
    current_viewer_count = sum(
        1 for row in last_presence_event.values()
        if _event_type(row) in _PRESENCE_EVENT_TYPES and _aware(row.timestamp) >= stale_cutoff
    )

    window_start = now - timedelta(seconds=LIVE_DYNAMIC_WINDOW_SECONDS)
    half_point = now - timedelta(seconds=LIVE_DYNAMIC_WINDOW_SECONDS / 2)
    recent_joiners: set[str] = set()
    earlier_joiners: set[str] = set()
    recent_likes = 0
    recent_gifts = 0
    for row in rows:
        ts = _aware(row.timestamp)
        if ts < window_start or ts > now:
            continue
        event_type = _event_type(row)
        if event_type == _JOINED:
            (recent_joiners if ts >= half_point else earlier_joiners).add(row.user_id)
        if bool(getattr(row, "liked", False)) or event_type == _LIKED:
            recent_likes += 1
        if event_type == _GIFT_SENT:
            recent_gifts += 1

    growth_rate = _clamp(
        (len(recent_joiners) - len(earlier_joiners)) / max(1, len(earlier_joiners)), low=-1.0, high=10.0,
    )

    return LiveDynamicState(
        current_viewer_count=current_viewer_count, viewer_growth_rate=growth_rate,
        recent_likes=recent_likes, recent_gifts=recent_gifts,
    )


def compute_dynamic_state(db: Session, stream_ids: Iterable[str], *, now: datetime) -> dict[str, LiveDynamicState]:
    """One bounded query covering every requested stream_id, never N+1. A stream_id with no
    rows in the lookback window (including one this service has no local Interaction history
    for at all, e.g. most explicit externally-supplied candidates) legitimately gets
    `NEUTRAL_DYNAMIC_STATE` -- genuinely absent local evidence, not a fabricated non-zero
    guess."""
    ids = list(dict.fromkeys(stream_ids))
    if not ids:
        return {}
    lookback = now - timedelta(seconds=max(LIVE_VIEWER_STALE_SECONDS, LIVE_DYNAMIC_WINDOW_SECONDS))
    # Eventual-consistency audit (Task: content_pending evidence-leak audit): this query
    # bypasses app.db.repositories entirely (its own direct select(Interaction)), so it must
    # apply the same content_pending exclusion those functions apply -- see
    # app.db.repositories.interactions()'s own comment for why. Without this, a not-yet-
    # resolved interaction referencing this stream_id would count toward its currentViewerCount/
    # viewerGrowthRate before RMS has ever confirmed this content_id is real LIVE content.
    rows = db.scalars(
        select(Interaction)
        .where(
            Interaction.content_id.in_(ids), Interaction.timestamp >= lookback, Interaction.timestamp <= now,
            Interaction.content_pending.is_(False),
        )
        .order_by(Interaction.timestamp.asc(), Interaction.id.asc())
    ).all()

    by_stream: dict[str, list[Interaction]] = defaultdict(list)
    for row in rows:
        by_stream[row.content_id].append(row)

    return {stream_id: _state_for_stream(by_stream.get(stream_id, []), now=now) for stream_id in ids}
