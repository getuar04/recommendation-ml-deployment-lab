"""Additive taxonomy/version compatibility FOUNDATION, corrected for RMS ownership (Task:
correct the public Content ingestion write contract so external callers cannot forge
RMS-owned canonical semantic fields). No final taxonomy is encoded anywhere in this file --
tests deliberately use made-up/arbitrary primary_category/subcategory strings to prove no
enum/hardcoded proposed-taxonomy validation exists at the storage layer, prove that
`primaryCategory`/`subcategory`/`taxonomyVersion` are NOT part of the public `ContentCreate`
write contract (a caller sending them has zero effect on the stored/returned values -- exactly
like `categoryConfidence`/`categorySource` already behave), and prove every EXISTING VIDEO
serving-path consumer (feature builder, candidate generation, reranker/onboarding matching)
still reads only the legacy `category` column, unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sklearn.dummy import DummyClassifier

from app.db.database import SessionLocal
from app.db.models import Content
from app.ml.content_classifier import CATEGORY_TAXONOMY_VERSION, ContentClassification
from app.ml.dataset_builder import FEATURES
from app.ml.legacy_category_projection import (
    LEGACY_MODEL_UNKNOWN_CATEGORY,
    LEGACY_VIDEO_MODEL_CATEGORY_VOCABULARY,
    project_to_legacy_model_category,
    register_legacy_mapping,
)

BASE = "/api/v1/recommendation-ml-service"
FIXED_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _full_metadata(feature_names, **overrides):
    metadata = {
        "modelVersion": "taxonomy-foundation-test", "modelType": "sklearn.dummy.DummyClassifier", "selectedModel": "Dummy",
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
    import app.services.recommendation_service as service
    from app.ml import model_store

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


def _content_row(db, content_id, *, content_type="VIDEO", category="GAMING",
                  creator_id="creator-1", popularity=0.5, created_at=FIXED_NOW, **overrides):
    db.add(Content(
        content_id=content_id, creator_id=creator_id, category=category, content_type=content_type,
        popularity_score=popularity, is_active=True, created_at=created_at, updated_at=created_at,
        **overrides,
    ))


def _create(client, **overrides):
    payload = {"contentId": "c1", "creatorId": "creator-1", "contentType": "VIDEO", "category": "SPORT"}
    payload.update(overrides)
    return client.post(f"{BASE}/contents", json=payload)


# --- Part 1: old Content ingestion path is unchanged ----------------------------------------

def test_explicit_category_ingestion_unchanged_new_fields_default_none(client):
    response = _create(client, contentId="unchanged-1")
    assert response.status_code == 201
    body = response.json()
    assert body["category"] == "SPORT"
    assert body["categorySource"] is None
    assert body["categoryConfidence"] is None
    assert body["primaryCategory"] is None
    assert body["subcategory"] is None
    assert body["taxonomyVersion"] is None


def test_old_style_request_without_any_new_fields_still_works(client):
    """No caller has ever sent primaryCategory/subcategory/taxonomyVersion -- proves an
    existing/unmodified caller's exact request body still succeeds unchanged."""
    response = client.post(f"{BASE}/contents", json={
        "contentId": "old-caller-1", "creatorId": "creator-old", "contentType": "VIDEO",
        "category": "MUSIC", "popularityScore": 0.6,
    })
    assert response.status_code == 201
    assert response.json()["category"] == "MUSIC"


# --- Part 2/3: caller CANNOT control RMS-owned fields via the public write contract --------

def test_caller_supplied_primary_category_via_api_has_no_effect(client):
    """primaryCategory is not part of the public ContentCreate write contract -- a caller
    sending it gets pydantic's default "ignore unknown field" behavior, the exact same
    treatment `categoryConfidence`/`categorySource` already receive. Must NOT become the
    stored/returned value."""
    response = _create(client, contentId="forge-attempt-1", primaryCategory="SOME_FUTURE_UNAPPROVED_PRIMARY")
    assert response.status_code == 201
    body = response.json()
    assert body["primaryCategory"] is None
    fetched = client.get(f"{BASE}/contents/forge-attempt-1")
    assert fetched.json()["primaryCategory"] is None


def test_caller_supplied_subcategory_via_api_has_no_effect(client):
    response = _create(client, contentId="forge-attempt-2", subcategory="SOME_FUTURE_SUBCATEGORY")
    assert response.status_code == 201
    assert response.json()["subcategory"] is None


def test_caller_supplied_taxonomy_version_via_api_has_no_effect_on_explicit_category(client):
    """The exact false-provenance scenario from the ownership audit: a caller sends
    category=SPORT (explicit, never classified by RMS) + a claimed taxonomyVersion. RMS must
    never store the caller's claimed version -- an explicit category carries no RMS
    classification evidence, so taxonomy_version must stay NULL regardless of what the caller
    asserts."""
    response = _create(client, contentId="forge-attempt-3", taxonomyVersion="v99-caller-claimed-future")
    assert response.status_code == 201
    assert response.json()["taxonomyVersion"] is None


def test_caller_cannot_override_the_rms_stamped_taxonomy_version_for_inferred_category(client, monkeypatch):
    """Even when RMS's own classifier DOES run (category omitted), a caller-sent
    taxonomyVersion must not override RMS's own stamp -- RMS is unconditionally authoritative
    for this field whenever it has classification evidence at all."""
    from app.api import content_routes
    from app.ml.content_classifier import ContentClassification

    monkeypatch.setattr(
        content_routes, "infer_category",
        lambda *args, **kwargs: ContentClassification("SPORT", 0.9, "MODEL"),
    )
    response = client.post(f"{BASE}/contents", json={
        "contentId": "forge-attempt-4", "creatorId": "creator-1", "contentType": "VIDEO",
        "title": "anything", "taxonomyVersion": "v99-caller-claimed-future",
    })
    assert response.status_code == 201
    body = response.json()
    assert body["categorySource"] == "INFERRED"
    assert body["taxonomyVersion"] == CATEGORY_TAXONOMY_VERSION
    assert body["taxonomyVersion"] != "v99-caller-claimed-future"


# --- Part 7 (renumbered): internal/trusted storage capability is preserved ------------------

def test_internal_sqlalchemy_row_can_store_canonical_semantic_fields_directly(client, db):
    """RMS-internal code (e.g. a future trusted offline labeled-import pipeline) can still
    persist primary_category/subcategory/taxonomy_version directly through the Content model
    -- storage capability is unaffected by removing the public write path."""
    _content_row(
        db, "internal-write-1", category="SPORT",
        primary_category="SOME_FUTURE_UNAPPROVED_PRIMARY", subcategory="SOME_FUTURE_SUBCATEGORY",
        taxonomy_version="v2-hypothetical",
    )
    db.commit()
    response = client.get(f"{BASE}/contents/internal-write-1")
    assert response.status_code == 200
    body = response.json()
    assert body["primaryCategory"] == "SOME_FUTURE_UNAPPROVED_PRIMARY"
    assert body["subcategory"] == "SOME_FUTURE_SUBCATEGORY"
    assert body["taxonomyVersion"] == "v2-hypothetical"


def test_internal_storage_accepts_arbitrary_values_no_enum(client, db):
    """No enum/hardcoded proposed-taxonomy validation exists at the storage layer: a
    genuinely made-up value (not in the 18-category stress-test proposal, not in the legacy
    10) stores and reads back identically to any other string."""
    _content_row(db, "internal-arbitrary-1", primary_category="ZZZ_TOTALLY_MADE_UP_VALUE_999")
    db.commit()
    response = client.get(f"{BASE}/contents/internal-arbitrary-1")
    assert response.status_code == 200
    assert response.json()["primaryCategory"] == "ZZZ_TOTALLY_MADE_UP_VALUE_999"


def test_existing_row_with_null_semantic_fields_reads_back_cleanly(client, db):
    """A row that predates this feature (or was created without the new fields) must read
    back with primary_category/subcategory/taxonomy_version all None -- no crash, no
    fabricated default."""
    _content_row(db, "predates-feature-1")
    db.commit()
    response = client.get(f"{BASE}/contents/predates-feature-1")
    assert response.status_code == 200
    body = response.json()
    assert body["primaryCategory"] is None
    assert body["subcategory"] is None
    assert body["taxonomyVersion"] is None


# --- taxonomy_version stamping precedence ----------------------------------------------------

def test_explicit_category_leaves_taxonomy_version_none_by_default(client):
    response = _create(client, contentId="explicit-tv-1")
    assert response.status_code == 201
    assert response.json()["taxonomyVersion"] is None


def test_explicit_category_with_caller_taxonomy_version_still_stores_none(client):
    """Superseded expectation from before the ownership correction: a caller-supplied
    taxonomyVersion is no longer honored (see test_caller_supplied_taxonomy_version_via_api_
    has_no_effect_on_explicit_category above) -- kept as its own case here specifically
    alongside test_explicit_category_leaves_taxonomy_version_none_by_default to show the
    behavior is identical whether or not the caller attempts to set it."""
    response = _create(client, contentId="explicit-tv-2", taxonomyVersion="v2-hypothetical")
    assert response.status_code == 201
    assert response.json()["taxonomyVersion"] is None


def test_inferred_category_stamps_the_existing_bootstrap_taxonomy_version(client, monkeypatch):
    import app.services.content_enrichment_service as enrichment
    from app.ml.content_classifier import save_classifier, train_classifier
    from app.ml.content_classifier_data import build_dataframe

    pipeline, metadata = train_classifier(build_dataframe())
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    model_path, metadata_path = tmp / "clf.joblib", tmp / "clf_metadata.json"
    save_classifier(pipeline, metadata, model_path=model_path, metadata_path=metadata_path)
    monkeypatch.setattr(enrichment, "CONTENT_CLASSIFIER_MODEL_PATH", model_path)
    monkeypatch.setattr(enrichment, "CONTENT_CLASSIFIER_METADATA_PATH", metadata_path)
    enrichment._cache.clear()

    response = client.post(f"{BASE}/contents", json={
        "contentId": "inferred-tv-1", "creatorId": "football-creator", "contentType": "VIDEO",
        "title": "Real Madrid vs Barcelona incredible goals",
        "hashtags": ["realmadrid", "barcelona", "ucl", "football"],
    })
    assert response.status_code == 201
    body = response.json()
    assert body["categorySource"] == "INFERRED"
    assert body["taxonomyVersion"] == CATEGORY_TAXONOMY_VERSION


def test_content_classification_dataclass_defaults_taxonomy_version():
    classification = ContentClassification("SPORT", 0.9, "MODEL")
    assert classification.taxonomy_version == CATEGORY_TAXONOMY_VERSION


# --- Part 7: legacy compatibility projection abstraction ------------------------------------

def test_legacy_projection_identity_for_known_legacy_categories():
    for category in LEGACY_VIDEO_MODEL_CATEGORY_VOCABULARY:
        assert project_to_legacy_model_category(category) == category


def test_legacy_projection_unknown_fallback_for_unmapped_value():
    assert project_to_legacy_model_category("SOME_FUTURE_UNAPPROVED_PRIMARY") == LEGACY_MODEL_UNKNOWN_CATEGORY


def test_legacy_projection_unknown_fallback_for_none():
    assert project_to_legacy_model_category(None) == LEGACY_MODEL_UNKNOWN_CATEGORY


def test_legacy_projection_no_mappings_registered_by_default():
    """No unapproved mapping (e.g. a future FASHION_BEAUTY -> FASHION merge) exists yet."""
    assert project_to_legacy_model_category("FASHION_BEAUTY", taxonomy_version="v2-hypothetical") == LEGACY_MODEL_UNKNOWN_CATEGORY


def test_legacy_projection_registered_mapping_is_version_scoped():
    register_legacy_mapping("v2-test-only", {"FASHION_BEAUTY": "FASHION"})
    try:
        assert project_to_legacy_model_category("FASHION_BEAUTY", taxonomy_version="v2-test-only") == "FASHION"
        # A different (or missing) taxonomy_version must not pick up this test-only mapping.
        assert project_to_legacy_model_category("FASHION_BEAUTY") == LEGACY_MODEL_UNKNOWN_CATEGORY
        assert project_to_legacy_model_category("FASHION_BEAUTY", taxonomy_version="v3-other") == LEGACY_MODEL_UNKNOWN_CATEGORY
    finally:
        # Clean up: this module-level registry is process-global, so leaving a test-only
        # mapping registered could leak into a later test in the same process.
        from app.ml import legacy_category_projection
        legacy_category_projection._VERSIONED_MAPPINGS.pop("v2-test-only", None)


# --- Existing VIDEO serving-path consumers still use ONLY the legacy `category` column ------

def test_video_recommendation_uses_only_legacy_category_not_primary_category(client, db, trained_video_model):
    """A Content row with primary_category/subcategory set to values totally disjoint from
    its legacy category must still be selected/scored/bucketed based purely on `category` --
    proves feature building, candidate generation, and the response's own category field are
    all untouched by the new columns' presence."""
    _content_row(db, "dual-field-1", category="GAMING",
                 primary_category="SOME_FUTURE_UNAPPROVED_PRIMARY", subcategory="SOME_FUTURE_SUB",
                 taxonomy_version="v2-hypothetical")
    db.commit()
    response = client.post(f"{BASE}/recommendations", json={"userId": "dual-field-user", "limit": 10})
    assert response.status_code == 200
    recs = response.json()["recommendations"]
    assert len(recs) == 1
    assert recs[0]["contentId"] == "dual-field-1"
    assert recs[0]["category"] == "GAMING"  # legacy category, not primary_category


def test_onboarding_interest_matching_uses_legacy_category_not_primary_category(client, db, trained_video_model):
    """Reranker's explicit_interest_relevance matches Candidate.category (sourced from the
    legacy `category` column) against onboarding interests -- an onboarding interest matching
    the NEW primary_category value (but not the legacy category) must NOT get the interest
    boost, proving the reranker path is unaffected by the new columns."""
    client.post(f"{BASE}/users", json={"userId": "onboarding-user", "userContext": {"interests": ["gaming"]}})
    _content_row(db, "interest-match-1", category="MUSIC",
                 primary_category="GAMING", subcategory="MINECRAFT")
    db.commit()
    response = client.post(f"{BASE}/recommendations", json={"userId": "onboarding-user", "limit": 10})
    assert response.status_code == 200
    # Purely a smoke check that the request succeeds and the row is scored under its legacy
    # category; a dedicated reranker unit-level assertion follows below for the exact signal.
    ids = {r["contentId"] for r in response.json()["recommendations"]}
    assert "interest-match-1" in ids


def test_explicit_interest_relevance_reads_candidate_category_field_only():
    from app.ml.reranker import explicit_interest_relevance
    from app.schemas.recommendation_schemas import Candidate, UserContext

    candidate = Candidate.model_validate({
        "contentId": "x", "creatorId": "cr", "category": "MUSIC",
        "contentPopularityScore": 0.5, "contentAgeHours": 1,
    })
    context = UserContext(interests=["gaming"])
    # candidate.category="MUSIC" does not match the "gaming" interest -- relevance is 0
    # regardless of any (nonexistent-on-Candidate) primary_category concept.
    assert explicit_interest_relevance(candidate, context, cold_start=True) == 0.0


def test_candidates_generate_debug_endpoint_uses_legacy_category(client, db):
    _content_row(db, "debug-dual-1", category="FOOD", primary_category="ART_CREATIVE")
    db.commit()
    response = client.post(f"{BASE}/candidates/generate", json={"userId": "debug-user", "limit": 10})
    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["category"] == "FOOD"


# --- LIVE isolation ---------------------------------------------------------------------------

def test_live_ingestion_unaffected_caller_primary_category_has_no_effect(client):
    response = client.post(f"{BASE}/contents", json={
        "contentId": "live-taxonomy-1", "creatorId": "live-creator", "contentType": "LIVE",
        "category": "GAMING", "primaryCategory": "SOME_FUTURE_VALUE",
    })
    assert response.status_code == 201
    body = response.json()
    assert body["category"] == "GAMING"
    assert body["primaryCategory"] is None
    assert body["contentType"] == "LIVE"


def test_live_without_category_falls_back_to_creator_history_new_fields_do_not_change_that(client):
    # LIVE ingestion contract audit (superseded the old "category is required for LIVE"
    # behavior this test used to assert): a forged primaryCategory still has no effect --
    # this brand-new creator has no VIDEO history, so the real fallback is UNKNOWN, not a
    # 422 and not the forged value.
    response = client.post(f"{BASE}/contents", json={
        "contentId": "live-no-category-1", "creatorId": "live-creator-no-history", "contentType": "LIVE",
        "primaryCategory": "SOME_FUTURE_VALUE",
    })
    assert response.status_code == 201
    body = response.json()
    assert body["category"] == "UNKNOWN"
    assert body["categorySource"] == "CREATOR_HISTORY"
    assert body["primaryCategory"] is None


# --- Content classifier not on the recommendation hot path (reaffirmed) ---------------------

def test_content_classifier_still_absent_from_recommendation_hot_path(client, db, trained_video_model, monkeypatch):
    import app.services.content_enrichment_service as enrichment

    _content_row(db, "hotpath-check-1", primary_category="SOME_FUTURE_VALUE")
    db.commit()
    called = {"n": 0}

    def _boom(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("recommendations must never invoke content classification")

    monkeypatch.setattr(enrichment, "infer_category", _boom)
    response = client.post(f"{BASE}/recommendations", json={"userId": "hotpath-user", "limit": 5})
    assert response.status_code == 200
    assert called["n"] == 0
