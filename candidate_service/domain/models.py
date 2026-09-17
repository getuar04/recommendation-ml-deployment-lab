"""Plain domain types. No pydantic, no DB, no transport -- these describe the concepts this
package reasons about, independent of how a real provider eventually fetches them."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContentItem:
    """A piece of VIDEO content a related user engaged with. Mirrors the subset of
    app.schemas.recommendation_schemas.Candidate's fields this package can actually populate --
    never the full schema (contentAgeHours/alreadySeen/creatorFollowed are request-time
    concerns the caller of candidate_service fills in, not properties of the content itself)."""
    content_id: str
    creator_id: str
    category: str
    popularity_score: float
    title: str | None = None
    hashtags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    subgenres: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FollowRelation:
    """One related user's follow relationship with the target user. `follows_target`: this
    user follows the target. `followed_by_target`: the target follows this user. Both true =
    mutual follow."""
    user_id: str
    follows_target: bool
    followed_by_target: bool

    @property
    def mutual(self) -> bool:
        return self.follows_target and self.followed_by_target


@dataclass(frozen=True)
class EngagementEvent:
    """One related user's raw interaction with one piece of content -- the same event
    vocabulary app.db.models.Interaction already uses (VIDEO_COMPLETED/CONTENT_LIKED/...),
    never a new one invented for this package."""
    user_id: str
    content_id: str
    event_type: str
    watch_percentage: float = 0.0


@dataclass(frozen=True)
class SocialCandidate:
    """One generated candidate, ready to become a Candidate(candidateSource=..., socialContext=...)
    request field -- see candidate_service.services.candidate_service.to_rms_payload."""
    content: ContentItem
    source: str  # "SOCIAL" | "COLLABORATIVE"
    source_user_id: str
    interest_similarity: float
    relationship_strength: float
    source_user_engagement: float
    mutual_follow: bool
