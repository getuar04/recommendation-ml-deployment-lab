"""LIVE viewing-session reconstruction from stored Interaction rows.

A VIDEO impression is one row = one event = one training example (see
`app.ml.dataset_builder`). A LIVE viewing is fundamentally different: a single logical
"watching this stream" episode is usually reported as several raw rows over time
(LIVE_JOINED, zero or more LIVE_WATCHED pings/engagement events, LIVE_LEFT), and a user may
reconnect mid-stream because of a network blip, the app going to the background, or simply
leaving and coming back on purpose. Session reconstruction turns that raw row stream into one
`LiveSession` per logical viewing episode, so LIVE dataset construction/serving can reason
about "how long did they actually watch" without double-counting reconnects or fabricating a
duration for a malformed sequence.

This intentionally does NOT reuse `app.ml.dataset_builder.FeatureHistory`/replay-saturation:
that machinery models *repeated exposure to the same static VIDEO content across many
separate impressions*, a different phenomenon from *one continuous LIVE viewing episode split
across several raw rows by reconnects*.

Reconnect / duration policy (deterministic, documented here so it stays the single source of
truth -- see `reconstruct_live_sessions` for the implementation):

- A LIVE_JOINED opens a viewing segment. A LIVE_LEFT closes the currently open segment and
  adds its duration to the session's `total_watch_seconds`.
- A LIVE_JOINED while a segment is already open (a duplicate JOIN, e.g. a client retry that
  used a new event_id) is a no-op: it never resets or extends the open segment.
- A LIVE_LEFT with no open segment (no matching JOIN was ever observed) contributes no
  duration -- there is nothing to measure -- and the session is flagged
  `exclusion_reason="UNMATCHED_LEFT"` for observability. It never fabricates a duration.
- A LIVE_JOINED that reopens within `LIVE_RECONNECT_GRACE_SECONDS` of this (user, content)
  pair's last LIVE_LEFT is a reconnect of the SAME logical session (`reconnect_count += 1`,
  watch time keeps accumulating in the same `LiveSession`); one that reopens after the grace
  window has elapsed starts a brand-new `LiveSession` instead.
- A session that ends with a still-open segment (a LIVE_JOINED with no matching LIVE_LEFT) never
  has that open segment's duration counted -- doing so would require using "now" as the
  segment's end, which is neither reproducible nor point-in-time-safe for training. The
  session is flagged `exclusion_reason="OPEN_ENDED"`; any already-closed segments/engagement
  it accumulated earlier are kept.
- Explicit watch-time precedence: a LIVE_LEFT row's own `watch_time_seconds` (when present) is
  authoritative for the segment it closes, used INSTEAD OF the join->leave timestamp delta,
  never added on top of it. A LIVE_WATCHED row's `watch_time_seconds` updates the currently
  open segment's provisional duration (the max of any readings seen so far, so an
  out-of-order/duplicate lower reading can never regress it); with no open segment, it is
  treated as a standalone "watched N seconds" report and folded straight into the session's
  total rather than silently dropped.
- Duplicate event ingestion (the exact same stored row appearing twice) is a no-op: rows are
  de-duplicated by `event_id` within each (user, content) group before processing.
- Rows are always processed in `(timestamp, event_id)` order regardless of how they were
  fetched, so out-of-order delivery/storage never changes the reconstructed result.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, NamedTuple

from app.core.config import LIVE_RECONNECT_GRACE_SECONDS

_JOINED = "LIVE_JOINED"
_LEFT = "LIVE_LEFT"
_WATCHED = "LIVE_WATCHED"
_LIKED = "LIVE_LIKED"
_SHARED = "LIVE_SHARED"
_COMMENTED = "LIVE_COMMENTED"
_GIFT_SENT = "LIVE_GIFT_SENT"
_CREATOR_FOLLOWED = "LIVE_CREATOR_FOLLOWED"
_NOT_INTERESTED = "LIVE_NOT_INTERESTED"
_IMPRESSION = "LIVE_IMPRESSION"


class _LiveSignals(NamedTuple):
    """LIVE-event-type aliases OR'd with the same boolean columns VIDEO rows use for the
    equivalent VIDEO event types -- mirrors `app.ml.feature_builder.interaction_signals`'
    "event-type aliases and explicit boolean columns are semantically equivalent" design, but
    recognizes the LIVE_* event types that function never covers (it only ever checks
    CONTENT_*/CREATOR_FOLLOWED). `gift_sent` has no dedicated boolean column on `Interaction`,
    so it is event-type-only."""

    liked: bool
    shared: bool
    commented: bool
    creator_followed: bool
    gift_sent: bool
    not_interested: bool


def _live_signals(row: Any) -> _LiveSignals:
    event_type = getattr(row, "event_type", "")
    event_type = getattr(event_type, "value", event_type)
    return _LiveSignals(
        liked=bool(getattr(row, "liked", False) or event_type == _LIKED),
        shared=bool(getattr(row, "shared", False) or event_type == _SHARED),
        commented=bool(getattr(row, "commented", False) or event_type == _COMMENTED),
        creator_followed=bool(getattr(row, "creator_followed", False) or event_type == _CREATOR_FOLLOWED),
        gift_sent=event_type == _GIFT_SENT,
        not_interested=event_type == _NOT_INTERESTED,
    )


@dataclass
class LiveSession:
    """One reconstructed LIVE viewing episode. See the module docstring for the reconnect/
    duration policy that produces these from raw `Interaction` rows."""

    user_id: str
    content_id: str
    creator_id: str | None
    category: str | None
    joined_at: datetime
    ended_at: datetime
    joined: bool = False
    impression: bool = False
    total_watch_seconds: float = 0.0
    reconnect_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    gift_count: int = 0
    creator_follow_count: int = 0
    not_interested_count: int = 0
    # Internal/testable observability only (not a spec-required field, but exposed rather
    # than silently swallowed) -- None means the session was reconstructed cleanly; otherwise
    # names the malformed/ambiguous pattern that was handled via the deterministic fallback
    # rules above instead of fabricating a duration or a label.
    exclusion_reason: str | None = None
    # Private bookkeeping, not part of the public contract.
    _open_segment_started_at: datetime | None = field(default=None, repr=False, compare=False)
    _pending_watched_seconds: float = field(default=0.0, repr=False, compare=False)
    _last_closed_at: datetime | None = field(default=None, repr=False, compare=False)


def _utc(value: datetime) -> datetime:
    # A SQLite round-trip (this project's test DB) silently strips tzinfo even for a column
    # declared DateTime(timezone=True) -- same quirk app.ml.dataset_builder._utc handles for
    # VIDEO. Assumed UTC (never guessed from anything else), matching that precedent exactly.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _sort_key(row: Any) -> tuple[datetime, str]:
    return (_utc(row.timestamp), str(getattr(row, "event_id", "")))


def _new_session(row: Any) -> LiveSession:
    ts = _utc(row.timestamp)
    return LiveSession(
        user_id=row.user_id, content_id=row.content_id,
        creator_id=getattr(row, "creator_id", None), category=getattr(row, "category", None),
        joined_at=ts, ended_at=ts,
    )


def _close_open_segment(session: LiveSession, *, at: datetime, explicit_seconds: float | None) -> None:
    started = session._open_segment_started_at
    if started is None:
        session.exclusion_reason = session.exclusion_reason or "UNMATCHED_LEFT"
        return
    duration = explicit_seconds if explicit_seconds is not None else max(
        session._pending_watched_seconds, (at - started).total_seconds(),
    )
    session.total_watch_seconds += duration
    session._open_segment_started_at = None
    session._pending_watched_seconds = 0.0
    session._last_closed_at = at


def reconstruct_live_sessions(rows: list[Any]) -> list[LiveSession]:
    """Reconstruct `LiveSession` objects from raw LIVE `Interaction` rows (or duck-typed
    equivalents exposing the same attributes). Rows for every (user_id, content_id) pair are
    processed independently, in strict `(timestamp, event_id)` order. Never raises on a
    malformed sequence -- see the module docstring for how each case (JOIN/LEFT ordering,
    duplicates, reconnects, unmatched events) is resolved deterministically."""
    grace = LIVE_RECONNECT_GRACE_SECONDS
    by_key: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in rows:
        by_key[(row.user_id, row.content_id)].append(row)

    sessions: list[LiveSession] = []
    for events in by_key.values():
        events.sort(key=_sort_key)
        seen_event_ids: set[str] = set()
        current: LiveSession | None = None
        for row in events:
            event_id = str(getattr(row, "event_id", ""))
            if event_id and event_id in seen_event_ids:
                continue  # duplicate event ingestion: the exact same row seen twice
            seen_event_ids.add(event_id)

            event_type = getattr(row, "event_type", "")
            event_type = getattr(event_type, "value", event_type)
            signals = _live_signals(row)
            ts = _utc(row.timestamp)

            if current is None:
                current = _new_session(row)
            elif (
                event_type == _JOINED
                and current._open_segment_started_at is None
                and current._last_closed_at is not None
                and (ts - current._last_closed_at).total_seconds() > grace
            ):
                sessions.append(current)
                current = _new_session(row)

            current.ended_at = ts

            if event_type == _JOINED:
                current.joined = True
                if current._open_segment_started_at is None:
                    if current._last_closed_at is not None:
                        current.reconnect_count += 1
                    current._open_segment_started_at = ts
                # else: duplicate JOIN while already open -- no-op, see module docstring.
            elif event_type == _LEFT:
                _close_open_segment(current, at=ts, explicit_seconds=getattr(row, "watch_time_seconds", None))
            elif event_type == _WATCHED:
                watch_time = getattr(row, "watch_time_seconds", None)
                if watch_time is not None:
                    # Explicit watch-time evidence implies the user was actually watching --
                    # stronger evidence of "joined-ness" than a bare LIVE_JOINED (which only
                    # means they opened the stream, possibly for 0 seconds). This also covers
                    # a client that only ever sends LIVE_WATCHED pings with no separate
                    # LIVE_JOINED/LIVE_LEFT pair at all.
                    current.joined = True
                    if current._open_segment_started_at is not None:
                        current._pending_watched_seconds = max(current._pending_watched_seconds, watch_time)
                    else:
                        current.total_watch_seconds = max(current.total_watch_seconds, watch_time)
            elif event_type == _IMPRESSION:
                current.impression = True

            if signals.liked:
                current.like_count += 1
            if signals.shared:
                current.share_count += 1
            if signals.commented:
                current.comment_count += 1
            if signals.gift_sent:
                current.gift_count += 1
            if signals.creator_followed:
                current.creator_follow_count += 1
            if signals.not_interested:
                current.not_interested_count += 1

        if current is not None:
            if current._open_segment_started_at is not None:
                current.exclusion_reason = current.exclusion_reason or "OPEN_ENDED"
            sessions.append(current)

    sessions.sort(key=lambda session: (session.ended_at, session.user_id, session.content_id))
    return sessions
