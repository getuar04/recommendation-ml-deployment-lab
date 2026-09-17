"""SOCIAL/COLLABORATIVE candidate generation. Depends only on the provider Protocols
(candidate_service.providers.base) -- swapping the demo providers for real Follow Service/User
Behavior Service-backed ones (whatever transport they eventually use) requires no change here.

Two source paths, kept separate on purpose (spec: SOCIAL is relationship-based, COLLABORATIVE
is similarity-based with no relationship required):
    SOCIAL:        every related user (any follow relationship) contributes their strong
                    engagement as candidates.
    COLLABORATIVE: every OTHER user above the similarity bar, excluding anyone who already has
                    a direct relationship (that content already reaches the target via SOCIAL).

Both paths independently filter out negative engagement and already-seen content before a
candidate is even constructed -- never RMS's job to catch what this service could have
filtered proactively (RMS's own alreadySeen/negative-feedback handling remains the real
backstop regardless, see app.ml.reranker).
"""
from __future__ import annotations

from candidate_service.domain.engagement import (
    engagement_strength,
    is_strong_engagement,
)
from candidate_service.domain.merge import merge_candidates
from candidate_service.domain.models import FollowRelation, SocialCandidate
from candidate_service.domain.relationship import relationship_strength
from candidate_service.domain.similarity import cosine_similarity
from candidate_service.providers.base import (
    CandidateServiceProviderError,
    ContentProvider,
    FollowRelationsProvider,
    UserBehaviorProvider,
)
from candidate_service.providers.demo_behavior_provider import DemoUserBehaviorProvider
from candidate_service.providers.demo_content_provider import DemoContentProvider
from candidate_service.providers.demo_follow_provider import DemoFollowRelationsProvider

# Demo/V1 bounds -- deliberately small, named, and enforced. A real deployment tunes these per
# actual traffic; the point here is that none of these loops are unbounded.
MAX_RELATED_USERS = 5
MAX_SIMILAR_USERS = 5
MAX_EVENTS_PER_SOURCE_USER = 20
MAX_SOCIAL_CANDIDATES = 20
MAX_COLLABORATIVE_CANDIDATES = 20
MIN_COLLABORATIVE_SIMILARITY = 0.5


def _social_candidates_for_relation(
    relation: FollowRelation, *, target_vector: dict[str, float], seen: set[str],
    behavior_provider: UserBehaviorProvider, content_provider: ContentProvider,
) -> list[SocialCandidate]:
    """One related user's contribution. Isolated so a single user's provider failure
    (CandidateServiceProviderError) degrades just that user, not the whole request -- see
    generate_social_candidates' try/except."""
    strength = relationship_strength(follows_target=relation.follows_target, followed_by_target=relation.followed_by_target)
    similarity = cosine_similarity(target_vector, behavior_provider.interest_vector(relation.user_id))
    events = behavior_provider.engagement_events(relation.user_id)[:MAX_EVENTS_PER_SOURCE_USER]
    candidates: list[SocialCandidate] = []
    for event in events:
        if not is_strong_engagement(event) or event.content_id in seen:
            continue
        content = content_provider.get(event.content_id)
        if content is None:
            continue
        candidates.append(SocialCandidate(
            content=content, source="SOCIAL", source_user_id=relation.user_id,
            interest_similarity=similarity, relationship_strength=strength,
            source_user_engagement=engagement_strength(event), mutual_follow=relation.mutual,
        ))
    return candidates


def generate_social_candidates(
    target_user_id: str, *, follow_provider: FollowRelationsProvider,
    behavior_provider: UserBehaviorProvider, content_provider: ContentProvider,
) -> list[SocialCandidate]:
    seen = behavior_provider.seen_content_ids(target_user_id)
    target_vector = behavior_provider.interest_vector(target_user_id)
    relations = follow_provider.related_users(target_user_id)[:MAX_RELATED_USERS]

    candidates: list[SocialCandidate] = []
    for relation in relations:
        try:
            candidates.extend(_social_candidates_for_relation(
                relation, target_vector=target_vector, seen=seen,
                behavior_provider=behavior_provider, content_provider=content_provider,
            ))
        except CandidateServiceProviderError:
            continue  # this one related user's data is unavailable -- the rest still count
    return merge_candidates(candidates)[:MAX_SOCIAL_CANDIDATES]


def _collaborative_candidates_for_user(
    other_user_id: str, *, target_vector: dict[str, float], seen: set[str],
    behavior_provider: UserBehaviorProvider, content_provider: ContentProvider,
) -> list[SocialCandidate]:
    similarity = cosine_similarity(target_vector, behavior_provider.interest_vector(other_user_id))
    if similarity < MIN_COLLABORATIVE_SIMILARITY:
        return []
    events = behavior_provider.engagement_events(other_user_id)[:MAX_EVENTS_PER_SOURCE_USER]
    candidates: list[SocialCandidate] = []
    for event in events:
        if not is_strong_engagement(event) or event.content_id in seen:
            continue
        content = content_provider.get(event.content_id)
        if content is None:
            continue
        candidates.append(SocialCandidate(
            content=content, source="COLLABORATIVE", source_user_id=other_user_id,
            interest_similarity=similarity, relationship_strength=0.0,
            source_user_engagement=engagement_strength(event), mutual_follow=False,
        ))
    return candidates


def generate_collaborative_candidates(
    target_user_id: str, *, follow_provider: FollowRelationsProvider,
    behavior_provider: UserBehaviorProvider, content_provider: ContentProvider,
) -> list[SocialCandidate]:
    seen = behavior_provider.seen_content_ids(target_user_id)
    target_vector = behavior_provider.interest_vector(target_user_id)
    related_user_ids = {relation.user_id for relation in follow_provider.related_users(target_user_id)}
    pool = behavior_provider.collaborative_candidate_pool(target_user_id)[:MAX_SIMILAR_USERS]

    candidates: list[SocialCandidate] = []
    for other_user_id in pool:
        if other_user_id in related_user_ids:
            continue  # a direct relationship already exists -- SOCIAL covers this user
        try:
            candidates.extend(_collaborative_candidates_for_user(
                other_user_id, target_vector=target_vector, seen=seen,
                behavior_provider=behavior_provider, content_provider=content_provider,
            ))
        except CandidateServiceProviderError:
            continue  # this one similar user's data is unavailable -- the rest still count
    return merge_candidates(candidates)[:MAX_COLLABORATIVE_CANDIDATES]


def generate_candidates(
    target_user_id: str, *, limit: int = 50,
    follow_provider: FollowRelationsProvider | None = None,
    behavior_provider: UserBehaviorProvider | None = None,
    content_provider: ContentProvider | None = None,
) -> list[SocialCandidate]:
    """Demo/V1 entry point. Providers default to the demo (local, deterministic, no-network)
    implementations -- pass real ones once a Follow Service/UBS integration exists."""
    if limit < 0:
        # Fail fast and explicit -- Python's `list[:negative]` slicing would otherwise return
        # almost the WHOLE list for a negative limit, the opposite of what a caller passing a
        # negative number could reasonably mean.
        raise ValueError(f"limit must be >= 0 (got {limit!r})")
    follow_provider = follow_provider or DemoFollowRelationsProvider()
    behavior_provider = behavior_provider or DemoUserBehaviorProvider()
    content_provider = content_provider or DemoContentProvider()

    social = generate_social_candidates(
        target_user_id, follow_provider=follow_provider, behavior_provider=behavior_provider, content_provider=content_provider,
    )
    collaborative = generate_collaborative_candidates(
        target_user_id, follow_provider=follow_provider, behavior_provider=behavior_provider, content_provider=content_provider,
    )
    merged = merge_candidates(social + collaborative)
    # Truncation order, not final ranking (RMS ranks) -- but which candidates get dropped when
    # the merged pool exceeds `limit` must still be deterministic and evidence-based, not
    # whatever happened to merge first. Same combined-evidence key merge_candidates already
    # uses to pick a winner within one source.
    merged.sort(key=lambda c: c.relationship_strength + c.interest_similarity + c.source_user_engagement, reverse=True)
    return merged[:limit]


def to_rms_payload(candidate: SocialCandidate, *, content_age_hours: float = 5.0) -> dict:
    """Shapes one SocialCandidate exactly like app.schemas.recommendation_schemas.Candidate's
    wire format (candidateSource/socialContext casing) -- the one contract this package must
    never diverge from. `alreadySeen`/`creatorFollowed` are always False here: this service
    already filtered out seen content (see generate_social_candidates/generate_collaborative_
    candidates), and creator-follow status is a target-user-to-creator relationship this
    package has no data for (distinct from the user-to-user follow relationships it does have)."""
    return {
        "contentId": candidate.content.content_id,
        "creatorId": candidate.content.creator_id,
        "category": candidate.content.category,
        "contentPopularityScore": candidate.content.popularity_score,
        "contentAgeHours": content_age_hours,
        "creatorFollowed": False,
        "alreadySeen": False,
        "title": candidate.content.title,
        "hashtags": candidate.content.hashtags,
        "topics": candidate.content.topics,
        "entities": candidate.content.entities,
        "subgenres": candidate.content.subgenres,
        "candidateSource": candidate.source,
        "socialContext": {
            "interestSimilarity": round(candidate.interest_similarity, 4),
            "relationshipStrength": round(candidate.relationship_strength, 4),
            "sourceUserEngagement": round(candidate.source_user_engagement, 4),
            "mutualFollow": candidate.mutual_follow,
        },
    }
