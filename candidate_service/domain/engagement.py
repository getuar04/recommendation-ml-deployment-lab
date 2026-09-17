"""Demo/V1 strong-engagement classification. Reuses this codebase's real event vocabulary
(app.db.models.Interaction.event_type) -- never invents a new one. A candidate is only ever
sourced from a POSITIVE engagement event; VIDEO_SKIPPED/CONTENT_NOT_INTERESTED never produce a
candidate, regardless of how the caller invokes this module (see
candidate_service.services.candidate_service, which filters on is_strong before this function
is even reached, so a negative event is never scored at all)."""
from __future__ import annotations

from candidate_service.domain.models import EngagementEvent

NEGATIVE_EVENT_TYPES = frozenset({"VIDEO_SKIPPED", "CONTENT_NOT_INTERESTED"})
STRONG_EVENT_TYPES = frozenset({"VIDEO_COMPLETED", "VIDEO_REWATCHED", "CONTENT_LIKED", "CONTENT_SHARED", "CONTENT_FAVORITED"})
STRONG_WATCH_PERCENTAGE_THRESHOLD = 80.0

# Bounded, deterministic, ordered by how decisive each signal is -- not tuned/learned.
_EVENT_ENGAGEMENT_SCORE = {
    "CONTENT_SHARED": 1.0,
    "CONTENT_FAVORITED": 0.95,
    "VIDEO_COMPLETED": 0.85,
    "VIDEO_REWATCHED": 0.85,
    "CONTENT_LIKED": 0.8,
}
_HIGH_WATCH_PERCENTAGE_SCORE = 0.7


def is_strong_engagement(event: EngagementEvent) -> bool:
    if event.event_type in NEGATIVE_EVENT_TYPES:
        return False
    if event.event_type in STRONG_EVENT_TYPES:
        return True
    return event.watch_percentage >= STRONG_WATCH_PERCENTAGE_THRESHOLD


def engagement_strength(event: EngagementEvent) -> float:
    """[0,1]. Only meaningful for an event that already passed is_strong_engagement -- callers
    must check that first; this returns 0.0 for a negative/weak event rather than raising, so
    a caller that skips the check degrades safely instead of crashing."""
    if not is_strong_engagement(event):
        return 0.0
    if event.event_type in _EVENT_ENGAGEMENT_SCORE:
        return _EVENT_ENGAGEMENT_SCORE[event.event_type]
    return _HIGH_WATCH_PERCENTAGE_SCORE
