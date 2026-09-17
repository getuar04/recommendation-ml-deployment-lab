"""candidate_service: domain logic, demo providers, and end-to-end compatibility with the real
RMS /api/v1/recommendation-ml-service/recommendations endpoint. No network -- the demo providers are pure local data."""
from __future__ import annotations

import math

from candidate_service.domain.engagement import (
    engagement_strength,
    is_strong_engagement,
)
from candidate_service.domain.merge import merge_candidates
from candidate_service.domain.models import (
    ContentItem,
    EngagementEvent,
    SocialCandidate,
)
from candidate_service.domain.relationship import relationship_strength
from candidate_service.domain.similarity import cosine_similarity
from candidate_service.providers.demo_behavior_provider import (
    ART_SPEEDPAINT,
    BARCELONA,
    HAMILTON,
    NADAL,
    REAL_MADRID,
    USER_F,
    VERSTAPPEN,
)
from candidate_service.providers.demo_follow_provider import (
    USER_A,
    USER_B,
    USER_C,
    USER_E,
)
from candidate_service.services.candidate_service import (
    MIN_COLLABORATIVE_SIMILARITY,
    generate_candidates,
    generate_collaborative_candidates,
    generate_social_candidates,
    to_rms_payload,
)

# ------------------------------------------------------------------------------- similarity

def test_cosine_similarity_high_for_aligned_vectors():
    a = {"SPORT": 0.90, "MUSIC": 0.20, "GAMING": 0.70}
    b = {"SPORT": 0.80, "MUSIC": 0.25, "GAMING": 0.75}
    assert cosine_similarity(a, b) > 0.9


def test_cosine_similarity_low_for_unrelated_vectors():
    a = {"SPORT": 0.90, "MUSIC": 0.20}
    b = {"ART": 0.85, "SPORT": 0.05}
    assert cosine_similarity(a, b) < 0.2


def test_cosine_similarity_zero_vector_safe():
    assert cosine_similarity({}, {"SPORT": 0.9}) == 0.0
    assert cosine_similarity({"SPORT": 0.0}, {"SPORT": 0.0}) == 0.0


def test_cosine_similarity_bounded_and_not_nan():
    result = cosine_similarity({"SPORT": 1.0}, {"SPORT": 1.0})
    assert 0.0 <= result <= 1.0
    assert not math.isnan(result)


def test_cosine_similarity_deterministic():
    a, b = {"SPORT": 0.7, "MUSIC": 0.3}, {"SPORT": 0.6, "MUSIC": 0.4}
    assert cosine_similarity(a, b) == cosine_similarity(a, b)


# ------------------------------------------------------------------------------- relationship

def test_mutual_follow_strongest():
    mutual = relationship_strength(follows_target=True, followed_by_target=True)
    one_way = relationship_strength(follows_target=True, followed_by_target=False)
    none = relationship_strength(follows_target=False, followed_by_target=False)
    assert mutual > one_way > none == 0.0


# ------------------------------------------------------------------------------- engagement

def test_strong_engagement_event_types():
    for event_type in ("VIDEO_COMPLETED", "CONTENT_LIKED", "CONTENT_SHARED", "CONTENT_FAVORITED", "VIDEO_REWATCHED"):
        event = EngagementEvent(user_id="u", content_id="c", event_type=event_type, watch_percentage=50.0)
        assert is_strong_engagement(event), event_type


def test_high_watch_percentage_alone_is_strong():
    event = EngagementEvent(user_id="u", content_id="c", event_type="VIDEO_WATCHED", watch_percentage=95.0)
    assert is_strong_engagement(event)


def test_skip_is_never_strong():
    event = EngagementEvent(user_id="u", content_id="c", event_type="VIDEO_SKIPPED", watch_percentage=95.0)
    assert not is_strong_engagement(event)
    assert engagement_strength(event) == 0.0


def test_not_interested_is_never_strong():
    event = EngagementEvent(user_id="u", content_id="c", event_type="CONTENT_NOT_INTERESTED", watch_percentage=95.0)
    assert not is_strong_engagement(event)
    assert engagement_strength(event) == 0.0


def test_engagement_strength_bounded():
    for event_type in ("CONTENT_SHARED", "CONTENT_FAVORITED", "VIDEO_COMPLETED", "CONTENT_LIKED"):
        event = EngagementEvent(user_id="u", content_id="c", event_type=event_type, watch_percentage=90.0)
        assert 0.0 < engagement_strength(event) <= 1.0


# ------------------------------------------------------------------------------- merge/dedup

def _stub(content_id, source, source_user, *, rel=0.0, sim=0.0):
    return SocialCandidate(
        content=ContentItem(content_id=content_id, creator_id="c", category="SPORT", popularity_score=0.5),
        source=source, source_user_id=source_user, interest_similarity=sim, relationship_strength=rel,
        source_user_engagement=0.8, mutual_follow=False,
    )


def test_merge_prefers_social_over_collaborative_for_same_content():
    merged = merge_candidates([_stub("x", "COLLABORATIVE", "d", sim=0.9), _stub("x", "SOCIAL", "b", rel=0.5)])
    assert len(merged) == 1
    assert merged[0].source == "SOCIAL"


def test_merge_keeps_strongest_evidence_within_same_source():
    weak = _stub("x", "SOCIAL", "b", rel=0.5, sim=0.1)
    strong = _stub("x", "SOCIAL", "c", rel=0.9, sim=0.5)
    merged = merge_candidates([weak, strong])
    assert len(merged) == 1
    assert merged[0].source_user_id == "c"


def test_merge_is_deterministic_regardless_of_input_order():
    a = merge_candidates([_stub("x", "SOCIAL", "b", rel=0.9), _stub("x", "COLLABORATIVE", "d", sim=0.9)])
    b = merge_candidates([_stub("x", "COLLABORATIVE", "d", sim=0.9), _stub("x", "SOCIAL", "b", rel=0.9)])
    assert a[0].source == b[0].source == "SOCIAL"


# ------------------------------------------------------------------------------- generation scenarios

def test_mutual_friend_produces_social_candidate_with_high_relationship_and_similarity():
    candidates = generate_social_candidates(
        USER_A, follow_provider=_provider().follow, behavior_provider=_provider().behavior, content_provider=_provider().content,
    )
    mutual = next(c for c in candidates if c.source_user_id == USER_B)
    assert mutual.source == "SOCIAL"
    assert mutual.mutual_follow is True
    assert mutual.relationship_strength >= 0.8
    assert mutual.content.content_id == BARCELONA


def test_one_way_follow_produces_social_candidate_with_weaker_relationship():
    candidates = generate_social_candidates(
        USER_A, follow_provider=_provider().follow, behavior_provider=_provider().behavior, content_provider=_provider().content,
    )
    one_way = next(c for c in candidates if c.source_user_id == USER_C)
    assert one_way.source == "SOCIAL"
    assert one_way.mutual_follow is False
    assert one_way.content.content_id == REAL_MADRID

    mutual = next(c for c in candidates if c.source_user_id == USER_B)
    assert one_way.relationship_strength < mutual.relationship_strength


def test_similar_unconnected_user_produces_collaborative_candidate():
    candidates = generate_collaborative_candidates(
        USER_A, follow_provider=_provider().follow, behavior_provider=_provider().behavior, content_provider=_provider().content,
    )
    collaborative = next(c for c in candidates if c.source_user_id == USER_E)
    assert collaborative.source == "COLLABORATIVE"
    assert collaborative.relationship_strength == 0.0
    assert collaborative.mutual_follow is False
    assert collaborative.interest_similarity >= MIN_COLLABORATIVE_SIMILARITY
    assert collaborative.content.content_id == VERSTAPPEN


def test_low_similarity_unconnected_user_produces_no_collaborative_candidate():
    candidates = generate_collaborative_candidates(
        USER_A, follow_provider=_provider().follow, behavior_provider=_provider().behavior, content_provider=_provider().content,
    )
    assert all(c.source_user_id != USER_F for c in candidates)
    assert all(c.content.content_id != ART_SPEEDPAINT for c in candidates)


def test_negative_source_event_never_becomes_a_candidate():
    candidates = generate_candidates(USER_A)
    assert all(c.content.content_id != HAMILTON for c in candidates)


def test_already_seen_content_is_filtered_out():
    candidates = generate_candidates(USER_A)
    assert all(c.content.content_id != NADAL for c in candidates)


def test_generation_is_deterministic():
    first = generate_candidates(USER_A)
    second = generate_candidates(USER_A)
    assert [(c.source, c.source_user_id, c.content.content_id) for c in first] == \
           [(c.source, c.source_user_id, c.content.content_id) for c in second]


def test_generate_candidates_respects_limit():
    assert len(generate_candidates(USER_A, limit=1)) == 1


def test_generate_candidates_truncation_keeps_the_strongest_evidence_when_over_limit():
    """When the merged pool exceeds `limit`, which candidates survive must be deterministic
    and evidence-based (relationship_strength + interest_similarity + source_user_engagement),
    not whichever happened to merge first."""
    candidates = generate_candidates(USER_A, limit=1)
    all_candidates = generate_candidates(USER_A, limit=50)
    assert len(all_candidates) > 1, "fixture must produce more than one candidate to test truncation"
    strongest = max(
        all_candidates, key=lambda c: c.relationship_strength + c.interest_similarity + c.source_user_engagement,
    )
    assert candidates[0].content.content_id == strongest.content.content_id


# ------------------------------------------------------------------------------- RMS compatibility

def test_to_rms_payload_validates_against_the_real_candidate_schema():
    from app.schemas.recommendation_schemas import Candidate

    candidates = generate_candidates(USER_A)
    assert candidates, "fixture produced zero candidates -- test is not exercising anything"
    for candidate in candidates:
        payload = to_rms_payload(candidate)
        validated = Candidate.model_validate(payload)
        assert validated.candidate_source == candidate.source
        assert validated.social_context is not None
        assert validated.social_context.mutual_follow == candidate.mutual_follow


def test_rms_end_to_end_scores_generated_social_candidates(client, tmp_path):
    """Full request -> service -> global LogisticRegression -> social rerank -> Top-N, through
    the real, unmodified /api/v1/recommendation-ml-service/recommendations endpoint. Uses app.benchmark.runner.use_model
    to activate a real, isolated LogisticRegression fit (never the real models/ directory, see
    conftest's autouse _guard_real_model_artifacts) -- deliberately bypasses the unrestricted
    cross-family /model/train endpoint: this test proves candidate_service's payloads score
    correctly, it is not a model-selection test, so it does not need (or want) XGBRanker in
    the loop at all."""
    from app.benchmark.runner import use_model
    from app.ml.trainer import train_models
    from tests.helpers import _synthetic_labeled_frame

    trained = train_models(_synthetic_labeled_frame(), restrict_algorithm="LogisticRegression")
    candidates = generate_candidates(USER_A)
    assert candidates, "fixture produced zero candidates -- test is not exercising anything"
    payload = {"userId": USER_A, "limit": len(candidates), "candidates": [to_rms_payload(c) for c in candidates]}

    with use_model(trained["model"], "LogisticRegression", trained, tmp_path / "models"):
        response = client.post("/api/v1/recommendation-ml-service/recommendations", json=payload)
    assert response.status_code == 200, response.json()
    recs = response.json()["recommendations"]
    assert len(recs) == len(candidates)
    assert all(math.isfinite(r["score"]) for r in recs)
    assert [r["rank"] for r in recs] == list(range(1, len(recs) + 1))


class _Providers:
    def __init__(self):
        from candidate_service.providers.demo_behavior_provider import (
            DemoUserBehaviorProvider,
        )
        from candidate_service.providers.demo_content_provider import (
            DemoContentProvider,
        )
        from candidate_service.providers.demo_follow_provider import (
            DemoFollowRelationsProvider,
        )

        self.follow = DemoFollowRelationsProvider()
        self.behavior = DemoUserBehaviorProvider()
        self.content = DemoContentProvider()


def _provider() -> _Providers:
    return _Providers()
