"""Hard serving-eligibility coverage for caller-supplied ``alreadySeen`` state."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services import recommendation_service
from tests.conftest import ensure_content


def _candidate(content_id: str, *, seen: bool = False, **overrides) -> dict:
    candidate = {
        "contentId": content_id,
        "creatorId": f"creator-{content_id}",
        "category": "SPORT",
        "contentPopularityScore": 0.5,
        "contentAgeHours": 5,
        "alreadySeen": seen,
    }
    candidate.update(overrides)
    return candidate


def _profile(interaction_count: int) -> dict:
    return {
        "version": 1,
        "totalInteractionCount": interaction_count,
        "categories": [],
        "creators": [],
        "semanticAffinities": [],
    }


@pytest.fixture
def serving_stub(monkeypatch):
    scored_batch_sizes: list[int] = []
    monkeypatch.setattr(
        recommendation_service.model_cache.video_cache,
        "get",
        lambda *_args, **_kwargs: (object(), {"modelVersion": "already-seen-test-model"}),
    )

    def probabilities(_model, feature_rows):
        scored_batch_sizes.append(len(feature_rows))
        return [0.9 - index * 0.01 for index in range(len(feature_rows))]

    monkeypatch.setattr(recommendation_service, "probabilities", probabilities)
    return scored_batch_sizes


def _recommend(client, candidates: list[dict], *, interaction_count: int = 0, limit: int = 100):
    return client.post(
        "/api/v1/recommendation-ml-service/recommendations",
        json={
            "userId": f"seen-filter-user-{interaction_count}",
            "limit": limit,
            "candidates": candidates,
            "userProfile": _profile(interaction_count),
        },
    )


def test_single_seen_candidate_is_absent_and_unseen_candidate_remains(client, serving_stub):
    response = _recommend(client, [_candidate("seen", seen=True), _candidate("unseen")])
    assert response.status_code == 200
    recommendations = response.json()["recommendations"]
    assert [item["contentId"] for item in recommendations] == ["unseen"]
    assert all(item["alreadySeen"] is False for item in recommendations)
    assert serving_stub == [1]


def test_strong_seen_candidate_cannot_override_hard_filter(client, serving_stub):
    strong_seen = _candidate(
        "barcelona-seen",
        seen=True,
        contentPopularityScore=1.0,
        contentAgeHours=0,
        creatorFollowed=True,
        title="Barcelona wins dramatic Champions League match",
        hashtags=["BARCELONA", "FOOTBALL"],
        topics=["CHAMPIONS_LEAGUE"],
        entities=["BARCELONA"],
        subgenres=["FOOTBALL"],
        candidateSource="SOCIAL",
        socialContext={
            "interestSimilarity": 1.0,
            "relationshipStrength": 1.0,
            "sourceUserEngagement": 1.0,
            "mutualFollow": True,
        },
    )
    response = _recommend(client, [strong_seen, _candidate("ordinary-unseen")], interaction_count=12)
    ids = [item["contentId"] for item in response.json()["recommendations"]]
    assert ids == ["ordinary-unseen"]
    assert serving_stub == [1]


def test_multiple_seen_candidates_are_removed_and_limit_is_not_backfilled(client, serving_stub):
    candidates = [
        *[_candidate(f"seen-{index}", seen=True) for index in range(6)],
        *[_candidate(f"unseen-{index}") for index in range(4)],
    ]
    response = _recommend(client, candidates, limit=5)
    recommendations = response.json()["recommendations"]
    assert len(recommendations) == 4
    assert {item["contentId"] for item in recommendations} == {f"unseen-{index}" for index in range(4)}
    assert all(item["alreadySeen"] is False for item in recommendations)
    assert serving_stub == [4]


def test_all_seen_returns_empty_without_calling_model_scoring(client, serving_stub):
    response = _recommend(client, [_candidate("seen-a", seen=True), _candidate("seen-b", seen=True)])
    assert response.status_code == 200
    assert response.json()["recommendations"] == []
    assert serving_stub == []


@pytest.mark.parametrize(
    ("interaction_count", "expected_strategy"),
    [(0, "COLD_START"), (1, "HYBRID"), (10, "PERSONALISED_ML")],
)
def test_seen_exclusion_is_strategy_independent(
    client, serving_stub, interaction_count, expected_strategy,
):
    response = _recommend(
        client,
        [_candidate(f"seen-{expected_strategy}", seen=True), _candidate(f"unseen-{expected_strategy}")],
        interaction_count=interaction_count,
    )
    body = response.json()
    assert body["strategy"] == expected_strategy
    assert [item["contentId"] for item in body["recommendations"]] == [f"unseen-{expected_strategy}"]


def test_seen_social_candidate_is_excluded(client, serving_stub):
    social_seen = _candidate(
        "social-seen",
        seen=True,
        candidateSource="SOCIAL",
        socialContext={
            "interestSimilarity": 1.0,
            "relationshipStrength": 1.0,
            "sourceUserEngagement": 1.0,
            "mutualFollow": True,
        },
    )
    response = _recommend(client, [social_seen, _candidate("normal-unseen")])
    assert [item["contentId"] for item in response.json()["recommendations"]] == ["normal-unseen"]


@pytest.mark.parametrize("seen_copy_first", [True, False])
def test_any_seen_duplicate_makes_content_id_ineligible(client, serving_stub, seen_copy_first):
    seen_copy = _candidate("duplicate", seen=True, creatorId="seen-creator", category="SPORT")
    unseen_copy = _candidate("duplicate", seen=False, creatorId="unseen-creator", category="MUSIC")
    copies = [seen_copy, unseen_copy] if seen_copy_first else [unseen_copy, seen_copy]
    response = _recommend(client, [*copies, _candidate("eligible")])
    ids = [item["contentId"] for item in response.json()["recommendations"]]
    assert ids == ["eligible"]
    assert serving_stub == [1]


# --- CLAUDE-P1-001 regression: RMS's own resolved history (`state.seen_content_ids`, LOCAL_DB
# path via app.services.providers.user_behavior_provider._local_state) must be OR-merged into
# the hard exclusion set, not overridable by a caller-supplied `alreadySeen=false`. Everything
# above this point exercises the caller-flag side of the exclusion set (via userProfile, no DB
# rows); everything below exercises the DB-history side specifically, through the real
# POST /api/v1/recommendation-ml-service/events entrypoint -- no userProfile is supplied, so `recommend()` resolves
# history exclusively from the `interactions` table it itself wrote.

def _mark_seen_via_event(
    client, user_id: str, content_id: str, *, category: str = "SPORT", event_type: str = "VIDEO_IMPRESSION",
):
    """Stores a real interaction row via the production /api/v1/recommendation-ml-service/events entrypoint so RMS's own
    resolved history genuinely contains this content_id for this user."""
    ensure_content(client, content_id, category=category)
    response = client.post(
        "/api/v1/recommendation-ml-service/events",
        json={
            "eventId": f"seen-evt-{user_id}-{content_id}",
            "userId": user_id,
            "contentId": content_id,
            "creatorId": f"creator-{content_id}",
            "category": category,
            "eventType": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    assert response.status_code == 201, response.json()


def _recommend_local(client, user_id: str, candidates: list[dict], *, limit: int = 100):
    """No `userProfile` -> forces resolution from RMS's own `interactions` table (LOCAL_DB),
    the exact path the bug concerned."""
    return client.post(
        "/api/v1/recommendation-ml-service/recommendations",
        json={"userId": user_id, "limit": limit, "candidates": candidates},
    )


def test_db_history_seen_excludes_content_even_when_candidate_claims_unseen(client, serving_stub):
    """The core CLAUDE-P1-001 repro: RMS's own resolved history says this content was seen;
    the candidate payload claims alreadySeen=false. The hard exclusion must trust RMS's own
    history, not only the caller's copy."""
    user_id = "p1-001-user-a"
    _mark_seen_via_event(client, user_id, "db-seen-content")
    response = _recommend_local(
        client, user_id,
        [_candidate("db-seen-content", seen=False), _candidate("genuinely-unseen", seen=False)],
    )
    assert response.status_code == 200
    ids = [item["contentId"] for item in response.json()["recommendations"]]
    assert ids == ["genuinely-unseen"]
    assert serving_stub == [1]


def test_candidate_claims_seen_but_db_history_has_no_record_still_excludes(client, serving_stub):
    """Caller-supplied alreadySeen=true must still exclude even when RMS's own history has
    never heard of the content -- the OR-merge must not weaken the pre-existing candidate-side
    signal."""
    user_id = "p1-001-user-b"
    response = _recommend_local(
        client, user_id,
        [_candidate("caller-seen-only", seen=True), _candidate("unseen", seen=False)],
    )
    assert [item["contentId"] for item in response.json()["recommendations"]] == ["unseen"]


def test_both_db_and_candidate_agree_seen_excludes(client, serving_stub):
    user_id = "p1-001-user-c"
    _mark_seen_via_event(client, user_id, "both-seen")
    response = _recommend_local(
        client, user_id, [_candidate("both-seen", seen=True), _candidate("unseen", seen=False)],
    )
    assert [item["contentId"] for item in response.json()["recommendations"]] == ["unseen"]


def test_both_db_and_candidate_agree_unseen_is_eligible(client, serving_stub):
    user_id = "p1-001-user-d"
    response = _recommend_local(client, user_id, [_candidate("eligible", seen=False)])
    assert [item["contentId"] for item in response.json()["recommendations"]] == ["eligible"]
    assert serving_stub == [1]


def test_db_seen_excludes_every_duplicate_representation_even_when_all_claim_unseen(client, serving_stub):
    """The exact attack live-reproduced during the audit: DB history says seen, and EVERY
    caller-supplied duplicate representation of the same contentId claims alreadySeen=false --
    none of them may reintroduce the content."""
    user_id = "p1-001-user-e"
    _mark_seen_via_event(client, user_id, "duplicate-db-seen")
    copies = [
        _candidate("duplicate-db-seen", seen=False, creatorId="creator-x", category="SPORT"),
        _candidate("duplicate-db-seen", seen=False, creatorId="creator-y", category="MUSIC"),
    ]
    response = _recommend_local(client, user_id, [*copies, _candidate("eligible", seen=False)])
    ids = [item["contentId"] for item in response.json()["recommendations"]]
    assert ids == ["eligible"]
    assert serving_stub == [1]


@pytest.mark.parametrize("seen_copy_first", [True, False])
def test_duplicate_content_id_any_seen_source_excludes_all_copies(client, serving_stub, seen_copy_first):
    """One duplicate copy claims alreadySeen=false, the other claims true (candidate-side
    conflict), combined with DB history also disagreeing with the false copy -- proves the
    OR-merge covers every source at once, regardless of which representation is listed first."""
    user_id = f"p1-001-user-f-{seen_copy_first}"
    _mark_seen_via_event(client, user_id, "mixed-duplicate")
    false_copy = _candidate("mixed-duplicate", seen=False, creatorId="creator-x", category="SPORT")
    true_copy = _candidate("mixed-duplicate", seen=True, creatorId="creator-y", category="MUSIC")
    copies = [true_copy, false_copy] if seen_copy_first else [false_copy, true_copy]
    response = _recommend_local(client, user_id, [*copies, _candidate("eligible", seen=False)])
    ids = [item["contentId"] for item in response.json()["recommendations"]]
    assert ids == ["eligible"]


def test_db_seen_content_with_maximal_competing_signals_still_excluded(client, serving_stub):
    """A DB-seen item dressed up with the strongest possible distractor signals (max
    popularity, zero age, followed creator, rich semantic metadata, strong social evidence)
    must still be excluded -- no scoring/reranking signal can override the hard filter."""
    user_id = "p1-001-user-g"
    _mark_seen_via_event(client, user_id, "db-seen-maximal", category="SPORT")
    maximal_seen = _candidate(
        "db-seen-maximal", seen=False, contentPopularityScore=1.0, contentAgeHours=0,
        creatorFollowed=True, title="Once in a lifetime match", hashtags=["SPORT", "FINAL"],
        topics=["CHAMPIONSHIP"], entities=["TEAM"], subgenres=["FOOTBALL"], category="SPORT",
        candidateSource="SOCIAL",
        socialContext={
            "interestSimilarity": 1.0, "relationshipStrength": 1.0,
            "sourceUserEngagement": 1.0, "mutualFollow": True,
        },
    )
    response = _recommend_local(client, user_id, [maximal_seen, _candidate("ordinary-unseen", seen=False)])
    ids = [item["contentId"] for item in response.json()["recommendations"]]
    assert ids == ["ordinary-unseen"]


def test_all_candidates_db_seen_returns_empty_200_without_scoring(client, serving_stub):
    user_id = "p1-001-user-h"
    _mark_seen_via_event(client, user_id, "seen-1")
    _mark_seen_via_event(client, user_id, "seen-2")
    response = _recommend_local(
        client, user_id, [_candidate("seen-1", seen=False), _candidate("seen-2", seen=False)],
    )
    assert response.status_code == 200
    assert response.json()["recommendations"] == []
    assert serving_stub == []


def test_mixed_db_seen_and_unseen_returns_only_unseen(client, serving_stub):
    user_id = "p1-001-user-i"
    _mark_seen_via_event(client, user_id, "seen-x")
    response = _recommend_local(
        client, user_id,
        [
            _candidate("seen-x", seen=False),
            _candidate("unseen-x", seen=False),
            _candidate("unseen-y", seen=False),
        ],
    )
    ids = {item["contentId"] for item in response.json()["recommendations"]}
    assert ids == {"unseen-x", "unseen-y"}


def test_db_seen_exclusion_does_not_backfill_limit(client, serving_stub):
    """Filtering happens before the limit is applied -- a request for 5 with 6 DB-seen and 3
    unseen candidates must return exactly the 3 unseen, never padded back up toward the limit
    with seen content."""
    user_id = "p1-001-user-j"
    seen_ids = [f"seen-{index}" for index in range(6)]
    for content_id in seen_ids:
        _mark_seen_via_event(client, user_id, content_id)
    unseen = [_candidate(f"unseen-{index}", seen=False) for index in range(3)]
    seen = [_candidate(content_id, seen=False) for content_id in seen_ids]
    response = _recommend_local(client, user_id, [*seen, *unseen], limit=5)
    recommendations = response.json()["recommendations"]
    assert len(recommendations) == 3
    assert {item["contentId"] for item in recommendations} == {f"unseen-{index}" for index in range(3)}
    assert serving_stub == [3]


@pytest.mark.parametrize(
    ("extra_event_count", "expected_strategy"),
    [(0, "HYBRID"), (9, "PERSONALISED_ML")],
)
def test_db_seen_exclusion_holds_across_lifecycle_strategies(
    client, serving_stub, extra_event_count, expected_strategy,
):
    """DB-history-based exclusion must not depend on which strategy tier (Hybrid / Personalised
    ML) the user is currently in. COLD_START is not exercisable here by construction: a user
    with zero interactions has no DB-seen content to test against -- that tier's already-seen
    coverage is `test_seen_exclusion_is_strategy_independent` above, via the candidate-side
    flag with an explicit interaction_count=0 userProfile."""
    user_id = f"p1-001-user-k-{expected_strategy}"
    for index in range(extra_event_count):
        _mark_seen_via_event(client, user_id, f"filler-{index}", category="MUSIC")
    _mark_seen_via_event(client, user_id, "seen-target", category="SPORT")
    response = _recommend_local(
        client, user_id,
        [
            _candidate("seen-target", seen=False, category="SPORT"),
            _candidate("unseen-target", seen=False, category="SPORT"),
        ],
    )
    body = response.json()
    assert body["strategy"] == expected_strategy
    ids = [item["contentId"] for item in body["recommendations"]]
    assert "seen-target" not in ids
    assert "unseen-target" in ids
