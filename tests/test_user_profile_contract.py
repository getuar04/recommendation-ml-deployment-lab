"""Phase A: optional userProfile support for VIDEO recommendations.

Proves the two things the approved plan requires:
1. Parity -- a profile built from a user's real interaction history, supplied on the
   request, produces byte-identical recommendations/scores/ranks/reasons/strategy/
   interactionCount to the legacy database-backed path built from the same rows.
2. Zero reads -- when a userProfile is supplied, recommend() never touches the
   interactions table at all.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sklearn.dummy import DummyClassifier

from app.db.database import SessionLocal
from app.db.models import Interaction
from app.db.repositories import recent_interactions_for_ranking
from app.ml.dataset_builder import (
    FEATURES,
    RECENT_WINDOW,
    FeatureHistory,
    history_from_rows,
)
from app.ml.feature_builder import affinity_score
from app.ml.model_store import save


def _seed_varied_history(db, user_id):
    now = datetime.now(timezone.utc)
    rows = [
        # FOOD: mostly positive, one creator followed.
        {"event_id": "p1", "content_id": "f1", "creator_id": "chef-1", "category": "FOOD",
             "event_type": "VIDEO_COMPLETED", "watch_percentage": 95, "liked": True, "creator_followed": True},
        {"event_id": "p2", "content_id": "f2", "creator_id": "chef-1", "category": "FOOD",
             "event_type": "VIDEO_COMPLETED", "watch_percentage": 88, "liked": False, "creator_followed": False},
        {"event_id": "p3", "content_id": "f3", "creator_id": "chef-2", "category": "FOOD",
             "event_type": "VIDEO_WATCHED", "watch_percentage": 60, "liked": False, "creator_followed": False},
        # SPORT: mixed, one fast-skip (negative signal).
        {"event_id": "p4", "content_id": "s1", "creator_id": "coach-1", "category": "SPORT",
             "event_type": "VIDEO_SKIPPED", "watch_percentage": 10, "liked": False, "creator_followed": False},
        {"event_id": "p5", "content_id": "s2", "creator_id": "coach-1", "category": "SPORT",
             "event_type": "VIDEO_COMPLETED", "watch_percentage": 91, "liked": True, "creator_followed": False},
        {"event_id": "p6", "content_id": "s3", "creator_id": "coach-2", "category": "SPORT",
             "event_type": "CONTENT_NOT_INTERESTED", "watch_percentage": 5, "liked": False, "creator_followed": False},
    ]
    for i, r in enumerate(rows):
        db.add(Interaction(
            event_id=r["event_id"], user_id=user_id, content_id=r["content_id"], creator_id=r["creator_id"],
            category=r["category"], event_type=r["event_type"], watch_time_seconds=r["watch_percentage"],
            content_duration_seconds=100, watch_percentage=r["watch_percentage"], liked=r["liked"],
            shared=False, favorited=False, commented=False, creator_followed=r["creator_followed"],
            timestamp=now - timedelta(days=20) + timedelta(hours=i),
        ))
    db.commit()


def _profile_payload_from_history(history, user_id):
    """The inverse of FeatureHistory.from_profile(): reads the same internal accumulator
    state a legacy, row-built FeatureHistory holds and serializes it into the wire shape
    the userProfile contract expects. Used only to build a realistic test fixture -- not a
    production code path."""
    now = datetime.now(timezone.utc)
    categories = []
    for (uid, category), cat in history.categories.items():
        if uid != user_id:
            continue
        categories.append({
            "category": category,
            "interactionCount": cat["interactions"],
            "positiveCount": cat["positive"],
            "negativeCount": cat["negative"],
            "completedCount": cat["completed"],
            "watchCount": cat["watch_count"],
            "watchPercentageSum": cat["watch_percentage_sum"],
            "rawAffinityScore": cat["raw"],
            # The inverse of FeatureHistory.from_profile()'s "recent_raw_override": read back
            # only the recent_events deltas actually inside RECENT_WINDOW of "now" (mirrors
            # how a real UBS would compute recentRawAffinityScore -- a windowed sum using the
            # exact same weighting FeatureHistory.update() already applied).
            "recentRawAffinityScore": sum(delta for t, delta in cat["recent_events"] if now - t <= RECENT_WINDOW),
            "lastInteractionAt": cat["last"].isoformat() if cat["last"] else None,
            # app.ml.replay_saturation_policy: cat["watch"] entries are now (timestamp,
            # watch_percentage, completed, replay_weight) 4-tuples -- the 4th element has no
            # equivalent on the wire (RecentWatchEvent), matching FeatureHistory.from_profile's
            # own documented limitation that a profile-supplied history applies no saturation.
            "recentWatchEvents": [
                {"timestamp": t.isoformat(), "watchPercentage": w, "completed": bool(c)}
                for t, w, c, _weight in cat["watch"]
            ],
            # Negative-feedback feature task: the inverse of FeatureHistory.from_profile()'s
            # own explicit_negative_count/last_explicit_negative_at reconstruction.
            "explicitNegativeCount": cat["explicit_negative"],
            "lastExplicitNegativeAt": cat["last_explicit_negative"].isoformat() if cat["last_explicit_negative"] else None,
        })
    creators = []
    for (uid, creator_id), creator in history.creators.items():
        if uid != user_id:
            continue
        creators.append({"creatorId": creator_id, "interactionCount": creator["interactions"], "completedCount": creator["completed"]})
    return {
        "version": 1,
        "totalInteractionCount": history.users.get(user_id, 0),
        "followedCreatorIds": [cid for (uid, cid) in history.followed_creators if uid == user_id],
        "seenContentIds": [cid for (uid, cid) in history.seen if uid == user_id],
        "categories": categories,
        "creators": creators,
    }


def _full_metadata(**overrides):
    metadata = {
        "modelVersion": "profile-test", "modelType": "sklearn.dummy.DummyClassifier", "selectedModel": "Dummy",
        "featureNames": FEATURES, "featureDefinitions": {}, "targetDefinition": "x", "splitStrategy": "x",
        "selectionCriterion": "x", "calibration": {"applied": False}, "decisionThreshold": 0.5,
        "trainingSamples": 1, "modelSelectionSamples": 1, "calibrationSamples": 1, "thresholdTuningSamples": 1,
        "testSamples": 1, "classDistribution": {},
        "sklearnVersion": "x", "pythonVersion": "x", "randomSeed": 42, "trainingDurationSeconds": 0.1,
        "trainedAt": "now", "metrics": {}, "modelComparison": {}, "datasetSource": {},
    }
    metadata.update(overrides)
    return metadata


def _patch_model(monkeypatch, tmp_path):
    """Same DummyClassifier(strategy="prior") convention already used throughout this suite
    (tests/test_health.py, test_spec_lifecycle.py) -- it genuinely ignores X's content, so the
    HTTP-level parity check below exercises real feature construction/reason/rerank logic
    without needing a meaningfully-fitted model. The direct feature-dict comparison test
    further below is the stronger, model-independent proof."""
    import app.api.training_routes as routes
    import app.ml.model_store as store
    import app.services.recommendation_service as service
    model_path, metadata_path = tmp_path / "model.joblib", tmp_path / "metadata.json"
    for module in (routes, store, service):
        monkeypatch.setattr(module, "MODEL_PATH", model_path, raising=False)
    for module in (routes, store):
        monkeypatch.setattr(module, "METADATA_PATH", metadata_path, raising=False)
    monkeypatch.setattr(store, "MODEL_DIR", tmp_path)
    model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    save(model, _full_metadata(), model_path=model_path, metadata_path=metadata_path, model_dir=tmp_path)
    return model_path, metadata_path


_CANDIDATES = [
    {"contentId": "cand-food-1", "creatorId": "chef-1", "category": "FOOD", "contentPopularityScore": 0.6, "contentAgeHours": 3, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "cand-food-2", "creatorId": "chef-3", "category": "FOOD", "contentPopularityScore": 0.4, "contentAgeHours": 10, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "cand-sport-1", "creatorId": "coach-1", "category": "SPORT", "contentPopularityScore": 0.8, "contentAgeHours": 1, "creatorFollowed": False, "alreadySeen": False},
]


def test_from_profile_reconstructs_byte_identical_features(client):
    """The strongest, model-independent parity proof: build FeatureHistory two ways --
    from raw DB rows (legacy) and from a profile snapshot derived from those same rows
    (FeatureHistory.from_profile) -- and assert `.features()` returns identical dicts for
    several candidates spanning both a warm category/creator and a cold one."""
    user_id = "feature-parity-user"
    db = SessionLocal()
    try:
        _seed_varied_history(db, user_id)
        rows = recent_interactions_for_ranking(db, user_id, limit=500)
    finally:
        db.close()

    legacy_history = history_from_rows(rows)
    profile_payload = _profile_payload_from_history(legacy_history, user_id)

    from app.schemas.recommendation_schemas import UserProfile
    profile_history = FeatureHistory.from_profile(user_id, UserProfile(**profile_payload))

    now = datetime.now(timezone.utc)
    candidate_params = [
        {"user_id": user_id, "category": "FOOD", "creator_id": "chef-1", "content_id": "new-food-1",
             "timestamp": now, "content_popularity_score": 0.6, "content_created_at": now - timedelta(hours=3)},
        {"user_id": user_id, "category": "SPORT", "creator_id": "coach-2", "content_id": "new-sport-1",
             "timestamp": now, "content_popularity_score": 0.4, "content_created_at": now - timedelta(hours=1)},
        # A category/creator this user has no history in at all -- the cold-lookup path.
        {"user_id": user_id, "category": "MUSIC", "creator_id": "dj-1", "content_id": "new-music-1",
             "timestamp": now, "content_popularity_score": 0.5, "content_created_at": now - timedelta(hours=5)},
    ]
    for params in candidate_params:
        assert legacy_history.features(**params) == profile_history.features(**params), params


def test_supplied_profile_matches_legacy_db_path(client, monkeypatch, tmp_path):
    _patch_model(monkeypatch, tmp_path)
    user_id = "parity-user"
    db = SessionLocal()
    try:
        _seed_varied_history(db, user_id)
        rows = recent_interactions_for_ranking(db, user_id, limit=500)
        history = history_from_rows(rows)
    finally:
        db.close()
    profile_payload = _profile_payload_from_history(history, user_id)

    legacy_response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": user_id, "candidates": _CANDIDATES, "limit": 3})
    profile_response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": user_id, "candidates": _CANDIDATES, "limit": 3, "userProfile": profile_payload})

    assert legacy_response.status_code == 200 and profile_response.status_code == 200
    legacy_body, profile_body = legacy_response.json(), profile_response.json()
    assert legacy_body["strategy"] == profile_body["strategy"]
    assert legacy_body["interactionCount"] == profile_body["interactionCount"]
    assert legacy_body["recommendations"] == profile_body["recommendations"]


def test_supplied_profile_causes_zero_interaction_table_reads(client, monkeypatch, tmp_path):
    _patch_model(monkeypatch, tmp_path)
    user_id = "no-read-user"
    db = SessionLocal()
    try:
        _seed_varied_history(db, user_id)
        rows = recent_interactions_for_ranking(db, user_id, limit=500)
        history = history_from_rows(rows)
    finally:
        db.close()
    profile_payload = _profile_payload_from_history(history, user_id)

    # Dual-mode refactor: the local DB-row read now lives in
    # app.services.providers.user_behavior_provider (LOCAL_DB path) -- see that module's own
    # docstring. An explicit request.userProfile still short-circuits it entirely, unchanged.
    from app.services.providers import user_behavior_provider

    def _forbidden(*args, **kwargs):
        raise AssertionError("recent_interactions_for_ranking must not be called when userProfile is supplied")

    monkeypatch.setattr(user_behavior_provider, "recent_interactions_for_ranking", _forbidden)

    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": user_id, "candidates": _CANDIDATES, "limit": 3, "userProfile": profile_payload})
    assert response.status_code == 200


def test_legacy_flow_still_works_when_no_userProfile_is_sent(client, monkeypatch, tmp_path):
    """The plain userId+candidates request (no userProfile key at all) must keep working
    exactly as before Phase A."""
    _patch_model(monkeypatch, tmp_path)
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "legacy-only-user", "candidates": _CANDIDATES, "limit": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "COLD_START" and body["interactionCount"] == 0


def test_empty_userProfile_is_equivalent_to_a_new_user_cold_start(client, monkeypatch, tmp_path):
    _patch_model(monkeypatch, tmp_path)
    empty_profile = {"version": 1, "totalInteractionCount": 0, "followedCreatorIds": [], "seenContentIds": [], "categories": [], "creators": []}
    with_profile = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "brand-new-user", "candidates": _CANDIDATES, "limit": 3, "userProfile": empty_profile})
    without_profile = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "brand-new-user-2", "candidates": _CANDIDATES, "limit": 3})
    assert with_profile.status_code == 200 and without_profile.status_code == 200
    assert with_profile.json()["strategy"] == without_profile.json()["strategy"] == "COLD_START"
    assert with_profile.json()["interactionCount"] == without_profile.json()["interactionCount"] == 0


def test_userProfile_with_unsupported_version_is_rejected(client):
    profile = {"version": 2, "totalInteractionCount": 0}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_oversized_categories_array_is_rejected(client):
    categories = [{"category": f"CAT{i}", "interactionCount": 1, "positiveCount": 1, "negativeCount": 0,
                   "completedCount": 1, "watchCount": 1, "watchPercentageSum": 90.0, "rawAffinityScore": 4.0}
                  for i in range(201)]
    profile = {"version": 1, "totalInteractionCount": 201, "categories": categories}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_negative_count_is_rejected(client):
    profile = {"version": 1, "totalInteractionCount": -1}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def _category(**overrides):
    category = {"category": "FOOD", "interactionCount": 5, "positiveCount": 2, "negativeCount": 1,
                "completedCount": 2, "watchCount": 3, "watchPercentageSum": 90.0, "rawAffinityScore": 4.0}
    category.update(overrides)
    return category


@pytest.mark.parametrize("field,value", [
    ("positiveCount", 6), ("negativeCount", 6), ("completedCount", 6), ("watchCount", 6),
])
def test_userProfile_category_count_exceeding_interaction_count_is_rejected(client, field, value):
    profile = {"version": 1, "totalInteractionCount": 5, "categories": [_category(**{field: value})]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_category_positive_plus_negative_exceeding_interaction_count_is_rejected(client):
    profile = {"version": 1, "totalInteractionCount": 5,
               "categories": [_category(interactionCount=5, positiveCount=3, negativeCount=3)]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_category_counts_within_interaction_count_is_accepted(client):
    # No candidates and no trained-model fixture: this only proves the profile itself passes
    # validation (a 503 MODEL_NOT_TRAINED here is a pre-existing, unrelated state depending on
    # test order, not a rejection of the profile -- 422 would mean validation itself failed).
    profile = {"version": 1, "totalInteractionCount": 5, "categories": [_category()]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code in (200, 503)


def test_userProfile_creator_completed_count_exceeding_interaction_count_is_rejected(client):
    profile = {"version": 1, "totalInteractionCount": 5,
               "creators": [{"creatorId": "chef-1", "interactionCount": 2, "completedCount": 3}]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_duplicate_category_names_are_rejected(client):
    profile = {"version": 1, "totalInteractionCount": 10, "categories": [_category(category="FOOD"), _category(category="FOOD")]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_duplicate_category_names_differing_only_by_case_are_rejected(client):
    """Categories are keyed case-insensitively by FeatureHistory.from_profile (category.upper()),
    so 'food' and 'FOOD' would otherwise silently collide into one accumulator entry."""
    profile = {"version": 1, "totalInteractionCount": 10, "categories": [_category(category="food"), _category(category="FOOD")]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_duplicate_creator_ids_are_rejected(client):
    creator = {"creatorId": "chef-1", "interactionCount": 1, "completedCount": 1}
    profile = {"version": 1, "totalInteractionCount": 2, "creators": [creator, creator]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_duplicate_followed_creator_ids_are_rejected(client):
    profile = {"version": 1, "totalInteractionCount": 0, "followedCreatorIds": ["chef-1", "chef-1"]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_duplicate_seen_content_ids_are_rejected(client):
    profile = {"version": 1, "totalInteractionCount": 0, "seenContentIds": ["f1", "f1"]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


@pytest.mark.parametrize("field", ["watchPercentageSum", "rawAffinityScore", "recentRawAffinityScore"])
@pytest.mark.parametrize("value", ["Infinity", "-Infinity", "NaN"])
def test_userProfile_non_finite_category_floats_are_rejected(client, field, value):
    """httpx's own request-side JSON encoder refuses to serialize a Python float("nan")/
    float("inf") at all (raises ValueError before a request is even sent), so this bypasses
    `json=` and hand-builds the request body: json.dumps() the payload with a string sentinel
    in the target field, then substitute in the bare NaN/Infinity/-Infinity token -- the same
    non-standard-but-common extension Python's own json.dumps emits by default -- to actually
    exercise the server-side `allow_inf_nan=False` rejection."""
    import json

    sentinel = "__NONFINITE_SENTINEL__"
    profile = {"version": 1, "totalInteractionCount": 5, "categories": [_category(**{field: sentinel})]}
    payload = {"userId": "u", "candidates": [], "userProfile": profile}
    body = json.dumps(payload).replace(f'"{sentinel}"', value)
    response = client.post("/api/v1/recommendation-ml-service/recommendations", content=body.encode(), headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_userProfile_blank_category_name_is_rejected(client):
    profile = {"version": 1, "totalInteractionCount": 1, "categories": [_category(category="   ")]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


def test_userProfile_blank_creator_id_is_rejected(client):
    profile = {"version": 1, "totalInteractionCount": 1,
               "creators": [{"creatorId": "   ", "interactionCount": 1, "completedCount": 0}]}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "candidates": [], "userProfile": profile})
    assert response.status_code == 422


# --- recent_category_affinity: recentRawAffinityScore contract (Step 4/Step 8) ---

def test_from_profile_defaults_recent_category_affinity_to_neutral_when_omitted():
    """Backward compatibility: an older UBS integration that doesn't yet populate
    recentRawAffinityScore must degrade safely to a neutral recent_category_affinity, not
    crash and not silently fabricate a signal, even when the category's long-term affinity
    is strongly non-neutral."""
    from app.schemas.recommendation_schemas import UserProfile
    profile_payload = {
        "version": 1, "totalInteractionCount": 10,
        "categories": [{
            "category": "SPORT", "interactionCount": 10, "positiveCount": 8, "negativeCount": 0,
            "completedCount": 8, "watchCount": 10, "watchPercentageSum": 900.0, "rawAffinityScore": 40.0,
            # recentRawAffinityScore intentionally omitted.
        }],
    }
    history = FeatureHistory.from_profile("u", UserProfile(**profile_payload))
    now = datetime.now(timezone.utc)
    features = history.features(
        user_id="u", category="SPORT", creator_id="cr", content_id="c", timestamp=now,
        content_popularity_score=.5, content_created_at=now,
    )
    assert features["category_affinity"] == affinity_score(40.0)
    assert features["recent_category_affinity"] == 0.5


def test_from_profile_reconstructs_recent_category_affinity_from_recent_raw_affinity_score():
    """The minimum snapshot field required to reconstruct recent_category_affinity: a
    UBS-supplied recentRawAffinityScore is used verbatim (not re-derived, same convention as
    rawAffinityScore) rather than filtered from recentWatchEvents (which carry no like/share/
    skip signal of their own)."""
    from app.schemas.recommendation_schemas import UserProfile
    profile_payload = {
        "version": 1, "totalInteractionCount": 3,
        "categories": [{
            "category": "MUSIC", "interactionCount": 3, "positiveCount": 3, "negativeCount": 0,
            "completedCount": 3, "watchCount": 3, "watchPercentageSum": 285.0, "rawAffinityScore": 26.0,
            "recentRawAffinityScore": 26.0,
        }],
    }
    history = FeatureHistory.from_profile("u", UserProfile(**profile_payload))
    now = datetime.now(timezone.utc)
    features = history.features(
        user_id="u", category="MUSIC", creator_id="cr", content_id="c", timestamp=now,
        content_popularity_score=.5, content_created_at=now,
    )
    assert features["recent_category_affinity"] == affinity_score(26.0)


def test_from_profile_supports_long_term_and_recent_affinity_diverging():
    """The exact SPORT -> MUSIC contract shape: strong long-term SPORT (high rawAffinityScore,
    negative recentRawAffinityScore from a session fast-skip) alongside weak long-term MUSIC
    but a strong recent session (low rawAffinityScore, high recentRawAffinityScore) -- both
    categories must be representable simultaneously from a single UserProfile snapshot."""
    from app.schemas.recommendation_schemas import UserProfile
    profile_payload = {
        "version": 1, "totalInteractionCount": 12,
        "categories": [
            {"category": "SPORT", "interactionCount": 8, "positiveCount": 6, "negativeCount": 1,
             "completedCount": 6, "watchCount": 8, "watchPercentageSum": 720.0, "rawAffinityScore": 32.0,
             "recentRawAffinityScore": -4.0},
            {"category": "MUSIC", "interactionCount": 4, "positiveCount": 1, "negativeCount": 3,
             "completedCount": 1, "watchCount": 4, "watchPercentageSum": 140.0, "rawAffinityScore": -6.0,
             "recentRawAffinityScore": 26.0},
        ],
    }
    history = FeatureHistory.from_profile("u", UserProfile(**profile_payload))
    now = datetime.now(timezone.utc)
    sport = history.features(user_id="u", category="SPORT", creator_id="cr", content_id="cs", timestamp=now,
                              content_popularity_score=.5, content_created_at=now)
    music = history.features(user_id="u", category="MUSIC", creator_id="cr", content_id="cm", timestamp=now,
                              content_popularity_score=.5, content_created_at=now)
    assert sport["category_affinity"] > 0.7 and sport["recent_category_affinity"] < 0.5
    assert music["category_affinity"] < 0.5 and music["recent_category_affinity"] > 0.7


def test_from_profile_recentRawAffinityScore_without_recentWatchEvents_degrades_explicitly():
    """Production-review finding (Section 2): a UserProfile snapshot can legally supply
    recentRawAffinityScore (a category-level aggregate) while omitting recentWatchEvents
    (the per-event list) entirely -- UBS is not required to send both. This is the smallest
    valid contract for recent_category_affinity alone, but it CANNOT reconstruct:
      - recent_category_watch_percentage / recent_category_completion_rate (these need the
        actual per-event watch_percentage/completed values within the 30-day window, not
        just a single pre-aggregated raw score) -- both must degrade to their explicit,
        documented "no recent window data" default (0.0), never fabricated from
        recentRawAffinityScore.
      - ALL 8 session_* features (session_category_affinity, has_session_activity,
        session_average_watch_percentage, session_positive/negative_interaction_count,
        session_category_streak, last_interaction_category_match,
        session_intent_confidence) -- session reconstruction requires real per-event
        timestamps/order, which a single aggregate cannot provide. These must degrade to
        their explicit cold-start defaults, never silently inherit the recent-window signal.

    This is a real, load-bearing distributional gap versus the row-built training path
    (where recent_category_affinity and recent_category_watch_percentage/session features
    always move together, since all derive from the same real events) -- documented here so
    a UBS integration knows recentWatchEvents should be sent whenever the caller wants
    recent/session-driven reranking to be reliable, not just recent_category_affinity alone."""
    from app.schemas.recommendation_schemas import UserProfile
    profile_payload = {
        "version": 1, "totalInteractionCount": 5,
        "categories": [{
            "category": "MUSIC", "interactionCount": 5, "positiveCount": 5, "negativeCount": 0,
            "completedCount": 5, "watchCount": 5, "watchPercentageSum": 475.0, "rawAffinityScore": 40.0,
            "recentRawAffinityScore": 40.0,
            # recentWatchEvents intentionally omitted -- this is the exact contract case under test.
        }],
    }
    history = FeatureHistory.from_profile("u", UserProfile(**profile_payload))
    now = datetime.now(timezone.utc)
    features = history.features(
        user_id="u", category="MUSIC", creator_id="cr", content_id="c", timestamp=now,
        content_popularity_score=.5, content_created_at=now,
    )
    # recent_category_affinity IS correctly reconstructed from the aggregate alone.
    assert features["recent_category_affinity"] == affinity_score(40.0)
    # Everything that needs per-event data explicitly degrades -- never fabricated.
    assert features["recent_category_watch_percentage"] == 0.0
    assert features["recent_category_completion_rate"] == 0.0
    assert features["session_category_affinity"] == 0.5
    assert features["has_session_activity"] == 0
    assert features["session_average_watch_percentage"] == 0.0
    assert features["session_positive_interaction_count"] == 0.0
    assert features["session_negative_interaction_count"] == 0.0
    assert features["session_category_streak"] == 0
    assert features["last_interaction_category_match"] == 0
    assert features["session_intent_confidence"] == 0.0
