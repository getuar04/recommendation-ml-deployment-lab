"""VIDEO onboarding candidate-generation gap (Task: Sept 17 finalization -- onboarding
interests must affect candidate MEMBERSHIP, not only final reranking).

Reproduced against real Docker/PostgreSQL: a cold user (zero interactions) with a
persisted onboarding interest (e.g. SPORT) received PREFERRED_CATEGORY-sourced SPORT
candidates directly from `POST /candidates/generate`/`POST /recommendations`, using only
`userId` -- proving the fix operates at candidate retrieval, not just reranking.

Root cause (fixed 2026-09-17): `app.services.providers.video_candidate_provider.
generate_video_candidates`'s `preferred` set was derived ONLY from real Interaction history
(`build_profiles`), with no fallback to `UserContext.interests` at all.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.db.database import SessionLocal
from app.db.models import Content, Interaction
from app.schemas.recommendation_schemas import UserContext
from app.services.providers.video_candidate_provider import generate_video_candidates
from app.services.user_context_provider import persist_user_context

BASE = "/api/v1/recommendation-ml-service"
NOW = datetime.now(timezone.utc)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _full_metadata(feature_names, **overrides):
    metadata = {
        "modelVersion": "test", "modelType": "sklearn.dummy.DummyClassifier", "selectedModel": "Dummy",
        "featureNames": feature_names, "featureDefinitions": {}, "targetDefinition": "x", "splitStrategy": "x",
        "selectionCriterion": "x", "calibration": {"applied": False}, "decisionThreshold": 0.5,
        "trainingSamples": 1, "modelSelectionSamples": 1, "calibrationSamples": 1, "thresholdTuningSamples": 1,
        "testSamples": 1, "classDistribution": {},
        "sklearnVersion": "x", "pythonVersion": "x", "randomSeed": 42, "trainingDurationSeconds": 0.1,
        "trainedAt": "now", "metrics": {}, "modelComparison": {}, "datasetSource": {},
    }
    metadata.update(overrides)
    return metadata


@pytest.fixture
def trained_video_model(monkeypatch, tmp_path):
    """No VIDEO model artifact exists in a local (non-Docker) checkout -- same pattern as
    tests/test_recommendation_local_candidate_integration.py's identical fixture."""
    from sklearn.dummy import DummyClassifier

    import app.services.recommendation_service as service
    from app.ml import model_store
    from app.ml.dataset_builder import FEATURES

    model_path = tmp_path / "video.joblib"
    metadata_path = tmp_path / "video_metadata.json"
    monkeypatch.setattr(service, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_store, "MODEL_PATH", model_path)
    monkeypatch.setattr(model_store, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path)

    metadata = _full_metadata(FEATURES)
    model_store.save(DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1]), metadata,
                      model_path=model_path, metadata_path=metadata_path, model_dir=tmp_path)
    service.model_cache.video_cache.invalidate()


def _content(db, content_id, *, category, creator_id="creator-x"):
    db.add(Content(content_id=content_id, creator_id=creator_id, category=category, content_type="VIDEO",
                    popularity_score=0.5, is_active=True, created_at=NOW, updated_at=NOW))


def _seed_diverse_catalog(db):
    for i in range(20):
        _content(db, f"sport-{i}", category="SPORT", creator_id=f"sport-creator-{i}")
    for i in range(20):
        _content(db, f"gaming-{i}", category="GAMING", creator_id=f"gaming-creator-{i}")
    for i in range(20):
        _content(db, f"comedy-{i}", category="COMEDY", creator_id=f"comedy-creator-{i}")


def test_cold_user_with_sport_onboarding_gets_preferred_category_sport_candidates(db):
    _seed_diverse_catalog(db)
    persist_user_context(db, "onboarding-sport-user", UserContext(interests=["SPORT"]))
    db.commit()

    results = generate_video_candidates(db, "onboarding-sport-user", limit=50)
    preferred = [c for c in results if c.source == "PREFERRED_CATEGORY"]
    assert preferred, "onboarding interest must produce PREFERRED_CATEGORY candidate membership"
    assert all(c.candidate.category == "SPORT" for c in preferred)


def test_cold_user_with_gaming_onboarding_gets_preferred_category_gaming_candidates(db):
    _seed_diverse_catalog(db)
    persist_user_context(db, "onboarding-gaming-user", UserContext(interests=["GAMING"]))
    db.commit()

    results = generate_video_candidates(db, "onboarding-gaming-user", limit=50)
    preferred = [c for c in results if c.source == "PREFERRED_CATEGORY"]
    assert preferred
    assert all(c.candidate.category == "GAMING" for c in preferred)


def test_cold_user_with_no_onboarding_gets_no_preferred_category_no_invented_interest(db):
    _seed_diverse_catalog(db)
    db.commit()

    results = generate_video_candidates(db, "truly-cold-user-no-context", limit=50)
    preferred = [c for c in results if c.source == "PREFERRED_CATEGORY"]
    assert preferred == []
    sources = {c.source for c in results}
    assert sources <= {"TRENDING", "NEW_CONTENT", "EXPLORATION"}


def test_real_behavioral_evidence_overrides_onboarding_fallback(db):
    """Once real interaction history exists, onboarding must stop driving candidate
    membership -- real behavior is never permanently overridden by a stale onboarding
    declaration."""
    _seed_diverse_catalog(db)
    persist_user_context(db, "evidenced-user", UserContext(interests=["SPORT"]))
    db.commit()
    for i in range(3):
        db.add(Interaction(
            event_id=f"real-ev-{i}", user_id="evidenced-user", content_id=f"gaming-{i}",
            creator_id=f"gaming-creator-{i}", category="GAMING", event_type="VIDEO_COMPLETED",
            watch_percentage=100.0, timestamp=NOW,
        ))
    db.commit()

    results = generate_video_candidates(db, "evidenced-user", limit=50)
    preferred = [c for c in results if c.source == "PREFERRED_CATEGORY"]
    assert preferred, "real behavioral category profile must still populate PREFERRED_CATEGORY"
    assert all(c.candidate.category == "GAMING" for c in preferred), \
        "real GAMING behavior must win over the stale SPORT onboarding declaration"


def test_onboarding_never_hard_excludes_new_content_or_exploration(db):
    """Cold-start requirement: onboarding must never become a hidden mechanism that
    prevents NEW_CONTENT/EXPLORATION eligibility for categories outside the declared
    interest."""
    _seed_diverse_catalog(db)
    persist_user_context(db, "onboarding-sport-user-2", UserContext(interests=["SPORT"]))
    db.commit()

    results = generate_video_candidates(db, "onboarding-sport-user-2", limit=60)
    categories_seen = {c.candidate.category for c in results}
    assert "COMEDY" in categories_seen or "GAMING" in categories_seen, \
        "content outside the onboarding interest must still be reachable via NEW_CONTENT/EXPLORATION"


def test_real_http_onboarding_flow_end_to_end(client):
    """POST /users (userContext) -> POST /candidates/generate (userId only) -- the full
    public-API path, not just the internal function."""
    client.post(f"{BASE}/contents", json={"contentId": "http-sport-1", "creatorId": "http-cr-1",
                                            "contentType": "VIDEO", "category": "SPORT"})
    client.post(f"{BASE}/contents", json={"contentId": "http-comedy-1", "creatorId": "http-cr-2",
                                            "contentType": "VIDEO", "category": "COMEDY"})
    resp = client.post(f"{BASE}/users", json={"userId": "http-onboarding-user",
                                                "userContext": {"interests": ["SPORT"]}})
    assert resp.status_code == 201

    result = client.post(f"{BASE}/candidates/generate", json={"userId": "http-onboarding-user", "limit": 50})
    assert result.status_code == 200
    preferred = [c for c in result.json()["candidates"] if c["source"] == "PREFERRED_CATEGORY"]
    assert any(c["contentId"] == "http-sport-1" for c in preferred)


# --------------------------------------------------------------- localBucketSource propagation

def test_local_bucket_source_propagates_through_final_recommendations(client, db, trained_video_model):
    """candidateSource/reason observability audit: the local SOURCE_ORDER bucket label must
    survive candidate generation -> feature building -> scoring -> reranking, all the way to
    POST /recommendations' final response -- previously discarded at
    app.services.providers.candidate_provider.resolve's LOCAL_GENERATION branch."""
    client.post(f"{BASE}/contents", json={"contentId": "lbs-sport-1", "creatorId": "lbs-cr-1",
                                            "contentType": "VIDEO", "category": "SPORT"})
    client.post(f"{BASE}/users", json={"userId": "lbs-user", "userContext": {"interests": ["SPORT"]}})

    result = client.post(f"{BASE}/recommendations", json={"userId": "lbs-user", "limit": 5})
    assert result.status_code == 200
    items = result.json()["recommendations"]
    assert items, "expected at least one recommendation"
    target = next((r for r in items if r["contentId"] == "lbs-sport-1"), None)
    assert target is not None
    assert target["localBucketSource"] == "PREFERRED_CATEGORY"


def test_local_bucket_source_is_none_for_explicit_candidates(client, trained_video_model):
    """An explicit caller-supplied candidate was never sourced from a local bucket --
    localBucketSource must stay None, never fabricated."""
    result = client.post(f"{BASE}/recommendations", json={
        "userId": "explicit-cand-user", "limit": 3,
        "candidates": [{"contentId": "explicit-1", "creatorId": "cr-1", "category": "NEWS",
                         "contentPopularityScore": 0.5, "contentAgeHours": 2,
                         "creatorFollowed": False, "alreadySeen": False}],
    })
    assert result.status_code == 200
    assert result.json()["recommendations"][0]["localBucketSource"] is None
