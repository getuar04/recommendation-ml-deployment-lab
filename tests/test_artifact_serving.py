from sklearn.dummy import DummyClassifier

from app.ml.live_feature_builder import LIVE_FEATURES
from app.ml.model_store import save


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


def test_recommendations_endpoint_reports_incompatible_artifact_distinctly_from_not_trained(client, monkeypatch, tmp_path):
    """An old-schema artifact (like the presentation-verification one) must fail loud and
    distinctly from 'no model trained yet', per the artifact-compatibility contract."""
    import app.ml.model_store as store
    import app.services.recommendation_service as service

    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    monkeypatch.setattr(service, "MODEL_PATH", model_path)
    monkeypatch.setattr(store, "MODEL_PATH", model_path)
    monkeypatch.setattr(store, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(store, "MODEL_DIR", tmp_path)

    # An old-code-style metadata dict, missing the new required fields entirely.
    save(DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1]), {
        "modelVersion": "old-poc", "selectedModel": "RandomForestClassifier",
        "metrics": {"f1Score": .8}, "trainedAt": "2026-07-21T22:58:54Z",
    })

    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u1", "limit": 2, "candidates": [
        {"contentId": "new-1", "creatorId": "creator", "category": "FOOD", "contentPopularityScore": .8,
         "contentAgeHours": 2, "creatorFollowed": False, "alreadySeen": False},
    ]})
    assert response.status_code == 503
    assert response.json()["error"] == "MODEL_ARTIFACT_INVALID"


def test_recommendations_endpoint_reports_not_trained_when_no_artifact_exists(client, monkeypatch, tmp_path):
    import app.services.recommendation_service as service
    monkeypatch.setattr(service, "MODEL_PATH", tmp_path / "missing.joblib")
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u1", "limit": 2, "candidates": [
        {"contentId": "new-1", "creatorId": "creator", "category": "FOOD", "contentPopularityScore": .8,
         "contentAgeHours": 2, "creatorFollowed": False, "alreadySeen": False},
    ]})
    assert response.status_code == 503
    assert response.json()["error"] == "MODEL_NOT_TRAINED"


def _patch_live_paths(monkeypatch, tmp_path):
    import app.ml.model_store as store
    import app.services.live_recommendation_service as service
    model_path = tmp_path / "live.joblib"
    metadata_path = tmp_path / "live_metadata.json"
    monkeypatch.setattr(service, "LIVE_MODEL_PATH", model_path)
    monkeypatch.setattr(service, "LIVE_METADATA_PATH", metadata_path)
    monkeypatch.setattr(store, "MODEL_DIR", tmp_path)
    return model_path, metadata_path


def test_live_recommendations_empty_candidates_with_valid_artifact_returns_200(client, monkeypatch, tmp_path):
    model_path, metadata_path = _patch_live_paths(monkeypatch, tmp_path)
    save(DummyClassifier().fit([[0], [1]], [0, 1]), _full_metadata(LIVE_FEATURES), model_path=model_path, metadata_path=metadata_path)
    response = client.post("/api/v1/recommendation-ml-service/recommendations/live", json={"userId": "u", "limit": 5, "candidates": []})
    assert response.status_code == 200 and response.json()["recommendations"] == []


def test_live_recommendations_empty_candidates_with_incompatible_artifact_is_reported(client, monkeypatch, tmp_path):
    model_path, metadata_path = _patch_live_paths(monkeypatch, tmp_path)
    save(DummyClassifier().fit([[0], [1]], [0, 1]), {"modelVersion": "old"}, model_path=model_path, metadata_path=metadata_path)
    response = client.post("/api/v1/recommendation-ml-service/recommendations/live", json={"userId": "u", "limit": 5, "candidates": []})
    assert response.status_code == 503 and response.json()["error"] == "LIVE_MODEL_ARTIFACT_INVALID"


def test_live_recommendations_empty_candidates_with_missing_metadata_is_not_trained(client, monkeypatch, tmp_path):
    _patch_live_paths(monkeypatch, tmp_path)
    response = client.post("/api/v1/recommendation-ml-service/recommendations/live", json={"userId": "u", "limit": 5, "candidates": []})
    assert response.status_code == 503 and response.json()["error"] == "LIVE_MODEL_NOT_TRAINED"


def test_live_recommendations_empty_candidates_with_corrupted_metadata_is_reported(client, monkeypatch, tmp_path):
    model_path, metadata_path = _patch_live_paths(monkeypatch, tmp_path)
    save(DummyClassifier().fit([[0], [1]], [0, 1]), _full_metadata(LIVE_FEATURES), model_path=model_path, metadata_path=metadata_path)
    metadata_path.write_text("{not valid json", encoding="utf-8")
    response = client.post("/api/v1/recommendation-ml-service/recommendations/live", json={"userId": "u", "limit": 5, "candidates": []})
    assert response.status_code == 503 and response.json()["error"] == "LIVE_MODEL_ARTIFACT_INVALID"


def test_live_recommendations_empty_candidates_with_checksum_mismatch_is_reported(client, monkeypatch, tmp_path):
    model_path, metadata_path = _patch_live_paths(monkeypatch, tmp_path)
    save(DummyClassifier().fit([[0], [1]], [0, 1]), _full_metadata(LIVE_FEATURES), model_path=model_path, metadata_path=metadata_path)
    model_path.write_bytes(b"corrupted-not-a-real-joblib-file")
    response = client.post("/api/v1/recommendation-ml-service/recommendations/live", json={"userId": "u", "limit": 5, "candidates": []})
    assert response.status_code == 503 and response.json()["error"] == "LIVE_MODEL_ARTIFACT_INVALID"
