from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from app.ml.dataset_builder import FEATURES, NUMERIC
from app.ml.ranking_snapshot import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureVectorSnapshot,
    RankedCandidateSnapshot,
    RankingSnapshotBuilder,
)


def _features() -> dict[str, str | float]:
    return {"category": "SPORT", **{name: index / 37 for index, name in enumerate(NUMERIC, 1)}}


def _kwargs() -> dict:
    return {
        "request_id": "impression-request-001",
        "ranking_timestamp": datetime(2026, 9, 1, 10, 30, tzinfo=timezone.utc),
        "artifact_metadata": {"modelVersion": "recommendation-prod-20260831095040", "featureNames": list(FEATURES)},
        "user_snapshot": {"userId": "user-1", "interactionCount": 9},
        "candidate_snapshot": {"contentId": "content-1", "creatorId": "creator-1", "category": "SPORT", "alreadySeen": False},
        "features": _features(),
        "raw_model_score": -0.12345678901234566,
        "normalized_model_score": 0.4691744315325329,
        "pre_rerank_rank": 3,
        "final_score": 0.4512345678901234,
        "final_rank": 5,
    }


def test_snapshot_contains_exact_active_feature_schema_and_all_ranking_stages():
    snapshot = RankingSnapshotBuilder.build(**_kwargs())
    assert snapshot.features.names == tuple(FEATURES) == FEATURE_NAMES
    assert len(snapshot.features.values) == 34
    assert snapshot.features.schema_version == FEATURE_SCHEMA_VERSION
    assert snapshot.raw_model_score == _kwargs()["raw_model_score"]
    assert snapshot.normalized_model_score == _kwargs()["normalized_model_score"]
    assert (snapshot.pre_rerank_rank, snapshot.final_score, snapshot.final_rank) == (3, _kwargs()["final_score"], 5)


@pytest.mark.parametrize("remove,add", [("category_affinity", None), (None, "invented_feature")])
def test_feature_schema_rejects_missing_and_extra_names(remove, add):
    features = _features()
    if remove:
        del features[remove]
    if add:
        features[add] = 1.0
    with pytest.raises(ValueError, match="feature schema mismatch"):
        FeatureVectorSnapshot.build(features)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, None, "1.0", True])
def test_non_finite_or_non_numeric_feature_values_are_rejected(value):
    features = _features()
    features[NUMERIC[0]] = value
    with pytest.raises((TypeError, ValueError), match="feature category_affinity"):
        FeatureVectorSnapshot.build(features)


def test_serialization_is_deterministic_lossless_and_round_trippable():
    snapshot = RankingSnapshotBuilder.build(**_kwargs())
    encoded = snapshot.to_json()
    restored = RankedCandidateSnapshot.from_json(encoded)
    assert restored == snapshot
    assert restored.to_json() == encoded
    assert json.loads(encoded)["features"]["values"] == list(snapshot.features.values)


def test_snapshot_is_immutable_and_does_not_alias_or_mutate_serving_inputs():
    kwargs = _kwargs()
    original_features = dict(kwargs["features"])
    original_candidate = dict(kwargs["candidate_snapshot"])
    snapshot = RankingSnapshotBuilder.build(**kwargs)
    kwargs["features"][NUMERIC[0]] = 999.0
    kwargs["candidate_snapshot"]["category"] = "MUSIC"
    assert snapshot.features.as_dict() == original_features
    assert json.loads(snapshot.candidate_snapshot_json) == original_candidate
    with pytest.raises(FrozenInstanceError):
        snapshot.final_rank = 1


def test_authoritative_request_id_and_artifact_contract_are_mandatory():
    for field, value, message in (
        ("request_id", "", "request_id"),
        ("artifact_metadata", {"modelVersion": "v", "featureNames": FEATURES[::-1]}, "artifact featureNames"),
    ):
        kwargs = _kwargs()
        kwargs[field] = value
        with pytest.raises(ValueError, match=message):
            RankingSnapshotBuilder.build(**kwargs)


def test_builder_rejects_invalid_ranks_and_candidate_feature_mismatch():
    kwargs = _kwargs()
    kwargs["pre_rerank_rank"] = 0
    with pytest.raises(ValueError, match="pre_rerank_rank"):
        RankingSnapshotBuilder.build(**kwargs)

    kwargs = _kwargs()
    kwargs["candidate_snapshot"] = {**kwargs["candidate_snapshot"], "category": "MUSIC"}
    with pytest.raises(ValueError, match="candidate category"):
        RankingSnapshotBuilder.build(**kwargs)


def test_builder_is_observational_only_for_response_fields():
    response_before = {
        "modelVersion": "recommendation-prod-20260831095040",
        "strategy": "PERSONALISED_ML",
        "recommendations": [{"contentId": "content-1", "score": 0.451235, "rank": 1, "reason": "MODEL_RELEVANCE"}],
    }
    RankingSnapshotBuilder.build(**_kwargs())
    response_after = {
        "modelVersion": "recommendation-prod-20260831095040",
        "strategy": "PERSONALISED_ML",
        "recommendations": [{"contentId": "content-1", "score": 0.451235, "rank": 1, "reason": "MODEL_RELEVANCE"}],
    }
    assert response_after == response_before
