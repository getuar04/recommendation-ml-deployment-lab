"""Local, deterministic, no-network stand-in for a real User Behavior Service. Interest
vectors are small, explicit demo constants (not derived from any real seeded interaction
history) -- this package's demo is meant to be network/DB-free, see the package docstring.
User identities are reused from scripts.seed_demo_users where a matching persona already
exists; USER_F is the one identity this package mints itself, for a persona (ART-heavy) no
existing demo user has, needed for the low-similarity exclusion scenario.
"""
from __future__ import annotations

from candidate_service.domain.models import EngagementEvent
from candidate_service.providers.demo_content_provider import content_id_for_slug
from candidate_service.providers.demo_follow_provider import (
    USER_A,
    USER_B,
    USER_C,
    USER_E,
)
from scripts.seed_demo_users import stable_uuid

USER_F = stable_uuid("candidate-service-demo-user", "user-f-art-unrelated")

# Demo/V1 interest vectors -- bounded [0,1] category affinities, long-term/recent only.
_INTEREST_VECTORS: dict[str, dict[str, float]] = {
    USER_A: {"SPORT": 0.90, "MUSIC": 0.20, "GAMING": 0.70},
    USER_B: {"SPORT": 0.80, "MUSIC": 0.25, "GAMING": 0.75},
    USER_C: {"SPORT": 0.85, "MUSIC": 0.30, "GAMING": 0.10},
    USER_E: {"SPORT": 0.92, "MUSIC": 0.10, "GAMING": 0.15},
    USER_F: {"ART": 0.85, "SPORT": 0.05, "MUSIC": 0.10},
}

BARCELONA = content_id_for_slug("sport-barcelona-ucl")
REAL_MADRID = content_id_for_slug("sport-real-madrid-laliga")
VERSTAPPEN = content_id_for_slug("sport-verstappen-monaco")
HAMILTON = content_id_for_slug("sport-hamilton-comeback")
NADAL = content_id_for_slug("sport-nadal-roland-garros")
ART_SPEEDPAINT = content_id_for_slug("art-digital-speedpaint")

_ENGAGEMENT_EVENTS: dict[str, list[EngagementEvent]] = {
    # Mutual friend: strong engagement on an unseen SPORT video, plus a NEGATIVE event on a
    # separate video that must never become a candidate (scenario E).
    USER_B: [
        EngagementEvent(user_id=USER_B, content_id=BARCELONA, event_type="VIDEO_COMPLETED", watch_percentage=96.0),
        EngagementEvent(user_id=USER_B, content_id=NADAL, event_type="CONTENT_LIKED", watch_percentage=90.0),
        EngagementEvent(user_id=USER_B, content_id=HAMILTON, event_type="CONTENT_NOT_INTERESTED", watch_percentage=3.0),
    ],
    # One-way follow: same kind of strong evidence as B, different content.
    USER_C: [
        EngagementEvent(user_id=USER_C, content_id=REAL_MADRID, event_type="CONTENT_LIKED", watch_percentage=88.0),
    ],
    # No relationship, high similarity: collaborative source.
    USER_E: [
        EngagementEvent(user_id=USER_E, content_id=VERSTAPPEN, event_type="VIDEO_COMPLETED", watch_percentage=94.0),
    ],
    # No relationship, low similarity: strong engagement exists, but similarity alone should
    # still exclude this user's content from the candidate pool (scenario D).
    USER_F: [
        EngagementEvent(user_id=USER_F, content_id=ART_SPEEDPAINT, event_type="VIDEO_COMPLETED", watch_percentage=97.0),
    ],
}

# Target user A already saw Nadal -- proves already-seen filtering (scenario F). Deliberately
# the SAME content B strongly liked, so the only thing suppressing it is the seen-state check.
_SEEN_CONTENT_IDS: dict[str, set[str]] = {
    USER_A: {NADAL},
}


class DemoUserBehaviorProvider:
    def interest_vector(self, user_id: str) -> dict[str, float]:
        return dict(_INTEREST_VECTORS.get(user_id, {}))

    def engagement_events(self, user_id: str) -> list[EngagementEvent]:
        return list(_ENGAGEMENT_EVENTS.get(user_id, []))

    def seen_content_ids(self, user_id: str) -> set[str]:
        return set(_SEEN_CONTENT_IDS.get(user_id, set()))

    def collaborative_candidate_pool(self, user_id: str) -> list[str]:
        """Demo-only convenience: the small, fixed set of OTHER users considered for
        similarity-based (COLLABORATIVE) candidate sourcing. A real implementation would get
        this short-list from an ANN/embedding-similarity search over the population, never a
        full scan -- see candidate_service.services.candidate_service's own docstring."""
        return [uid for uid in _INTEREST_VECTORS if uid != user_id]
