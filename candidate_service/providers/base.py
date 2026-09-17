"""Provider interfaces. Candidate-generation domain logic (candidate_service.services) depends
only on these Protocols, never on a concrete transport -- REST/Kafka/gRPC/an internal SDK are
all equally valid future implementations, none chosen yet. Today only the Demo* implementations
in this package exist (local, deterministic, no network); a real integration is a new class
satisfying the same Protocol, with zero change to candidate-generation logic itself.
"""
from __future__ import annotations

from typing import Protocol

from candidate_service.domain.models import ContentItem, EngagementEvent, FollowRelation


class CandidateServiceProviderError(Exception):
    """The one typed exception a provider implementation should raise for a failure the
    candidate-generation services know how to handle (network/timeout/upstream error, however
    a real transport reports it) -- concrete providers wrap their own transport-specific
    exceptions into this one before it reaches candidate_service.services. Generation degrades
    per-source-user on this (skips that user, continues with the rest) rather than either
    failing the whole request or catching bare Exception."""


class FollowRelationsProvider(Protocol):
    def related_users(self, user_id: str) -> list[FollowRelation]:
        """Every user with SOME follow relationship (either direction) to `user_id`."""
        ...


class UserBehaviorProvider(Protocol):
    def interest_vector(self, user_id: str) -> dict[str, float]:
        """Bounded [0,1] long-term/recent category-affinity profile, e.g. {"SPORT": 0.9}."""
        ...

    def engagement_events(self, user_id: str) -> list[EngagementEvent]:
        """This user's own interaction events -- source-user evidence for candidates surfaced
        from them, positive and negative alike (filtering is the caller's job)."""
        ...

    def seen_content_ids(self, user_id: str) -> set[str]:
        ...

    def collaborative_candidate_pool(self, user_id: str) -> list[str]:
        """A bounded short-list of OTHER user ids worth checking for interest similarity. A
        real implementation backs this with an ANN/embedding-similarity search over the
        population (never a full scan) -- the demo implementation returns a small fixed list."""
        ...


class ContentProvider(Protocol):
    def get(self, content_id: str) -> ContentItem | None:
        ...
