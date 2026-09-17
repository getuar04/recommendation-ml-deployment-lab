"""Group D: candidate -> active -> previous artifact promotion lifecycle."""
import pytest
from sklearn.dummy import DummyClassifier

from app.ml import artifact_lifecycle, model_store
from app.ml.dataset_builder import FEATURES


def full_model_metadata(**overrides):
    """Return complete schema-compatible metadata for artifact lifecycle tests."""
    metadata = {
        "modelVersion": "test",
        "modelType": "sklearn.dummy.DummyClassifier",
        "selectedModel": "Dummy",
        "featureNames": FEATURES,
        "featureDefinitions": {},
        "targetDefinition": "x",
        "splitStrategy": "x",
        "selectionCriterion": "x",
        "calibration": {"applied": False},
        "decisionThreshold": 0.5,
        "trainingSamples": 1,
        "modelSelectionSamples": 1,
        "calibrationSamples": 1,
        "thresholdTuningSamples": 1,
        "testSamples": 1,
        "classDistribution": {},
        "sklearnVersion": "x",
        "pythonVersion": "x",
        "randomSeed": 42,
        "trainingDurationSeconds": 0.1,
        "trainedAt": "now",
        "metrics": {},
        "modelComparison": {},
        "datasetSource": {},
    }
    metadata.update(overrides)
    return metadata



def _paths(tmp_path):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    return model_path, metadata_path, artifact_lifecycle.sibling_paths(model_path, metadata_path)


def _fit_dummy():
    return DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])


def _promote(model_path, metadata_path, paths, version, **overrides):
    model_store.save(_fit_dummy(), full_model_metadata(modelVersion=version, **overrides),
                      model_path=paths["candidate_model"], metadata_path=paths["candidate_metadata"],
                      model_dir=paths["candidate_model"].parent)
    return artifact_lifecycle.promote(
        candidate_model_path=paths["candidate_model"], candidate_metadata_path=paths["candidate_metadata"],
        active_model_path=model_path, active_metadata_path=metadata_path,
        previous_model_path=paths["previous_model"], previous_metadata_path=paths["previous_metadata"],
        feature_names=FEATURES,
    )


def test_candidate_is_saved_but_not_served_before_promotion(tmp_path):
    model_path, _metadata_path, paths = _paths(tmp_path)
    model_store.save(_fit_dummy(), full_model_metadata(modelVersion="cand-1"),
                      model_path=paths["candidate_model"], metadata_path=paths["candidate_metadata"], model_dir=tmp_path)
    assert paths["candidate_model"].exists()
    assert not model_path.exists()  # active path untouched until promote() runs


def test_promotion_moves_candidate_to_active(tmp_path):
    model_path, metadata_path, paths = _paths(tmp_path)
    promoted = _promote(model_path, metadata_path, paths, "cand-1")
    assert model_path.exists() and metadata_path.exists()
    assert not paths["candidate_model"].exists() and not paths["candidate_metadata"].exists()
    assert promoted["modelVersion"] == "cand-1" and "promotedAt" in promoted
    _, metadata = model_store.load_validated(FEATURES, model_path=model_path, metadata_path=metadata_path)
    assert metadata["modelVersion"] == "cand-1"


def test_second_promotion_retires_first_to_previous(tmp_path):
    model_path, metadata_path, paths = _paths(tmp_path)
    _promote(model_path, metadata_path, paths, "v1")
    _promote(model_path, metadata_path, paths, "v2")
    assert paths["previous_model"].exists() and paths["previous_metadata"].exists()
    _, previous_metadata = model_store.load_validated(
        FEATURES, model_path=paths["previous_model"], metadata_path=paths["previous_metadata"])
    assert previous_metadata["modelVersion"] == "v1"
    _, active_metadata = model_store.load_validated(FEATURES, model_path=model_path, metadata_path=metadata_path)
    assert active_metadata["modelVersion"] == "v2"


def test_corrupted_candidate_is_rejected_and_active_untouched(tmp_path):
    model_path, metadata_path, paths = _paths(tmp_path)
    _promote(model_path, metadata_path, paths, "good-1")  # establish a valid active artifact
    model_store.save(_fit_dummy(), full_model_metadata(modelVersion="bad-candidate"),
                      model_path=paths["candidate_model"], metadata_path=paths["candidate_metadata"], model_dir=tmp_path)
    paths["candidate_model"].write_bytes(b"not a real joblib file")  # corrupt after the checksum was recorded

    with pytest.raises(artifact_lifecycle.PromotionValidationError):
        artifact_lifecycle.promote(
            candidate_model_path=paths["candidate_model"], candidate_metadata_path=paths["candidate_metadata"],
            active_model_path=model_path, active_metadata_path=metadata_path,
            previous_model_path=paths["previous_model"], previous_metadata_path=paths["previous_metadata"],
            feature_names=FEATURES,
        )
    _, active_metadata = model_store.load_validated(FEATURES, model_path=model_path, metadata_path=metadata_path)
    assert active_metadata["modelVersion"] == "good-1"  # untouched
    assert not paths["previous_model"].exists()  # promotion never got that far


def test_incompatible_candidate_is_rejected(tmp_path):
    model_path, metadata_path, paths = _paths(tmp_path)
    model_store.save(_fit_dummy(), full_model_metadata(featureNames=["totally", "different"]),
                      model_path=paths["candidate_model"], metadata_path=paths["candidate_metadata"], model_dir=tmp_path)
    with pytest.raises(artifact_lifecycle.PromotionValidationError):
        artifact_lifecycle.promote(
            candidate_model_path=paths["candidate_model"], candidate_metadata_path=paths["candidate_metadata"],
            active_model_path=model_path, active_metadata_path=metadata_path,
            previous_model_path=paths["previous_model"], previous_metadata_path=paths["previous_metadata"],
            feature_names=FEATURES,
        )
    assert not model_path.exists()


def test_promoting_a_missing_candidate_raises_not_found(tmp_path):
    model_path, metadata_path, paths = _paths(tmp_path)
    with pytest.raises(model_store.ArtifactNotFoundError):
        artifact_lifecycle.promote(
            candidate_model_path=paths["candidate_model"], candidate_metadata_path=paths["candidate_metadata"],
            active_model_path=model_path, active_metadata_path=metadata_path,
            previous_model_path=paths["previous_model"], previous_metadata_path=paths["previous_metadata"],
            feature_names=FEATURES,
        )


def test_rollback_restores_previous(tmp_path):
    model_path, metadata_path, paths = _paths(tmp_path)
    _promote(model_path, metadata_path, paths, "v1")
    _promote(model_path, metadata_path, paths, "v2")
    rolled_back = artifact_lifecycle.rollback(
        active_model_path=model_path, active_metadata_path=metadata_path,
        previous_model_path=paths["previous_model"], previous_metadata_path=paths["previous_metadata"],
        feature_names=FEATURES,
    )
    assert rolled_back["modelVersion"] == "v1" and "rolledBackAt" in rolled_back
    _, active_metadata = model_store.load_validated(FEATURES, model_path=model_path, metadata_path=metadata_path)
    assert active_metadata["modelVersion"] == "v1"


def test_rollback_without_a_previous_artifact_raises_not_found(tmp_path):
    model_path, metadata_path, paths = _paths(tmp_path)
    _promote(model_path, metadata_path, paths, "only-active")
    with pytest.raises(model_store.ArtifactNotFoundError):
        artifact_lifecycle.rollback(
            active_model_path=model_path, active_metadata_path=metadata_path,
            previous_model_path=paths["previous_model"], previous_metadata_path=paths["previous_metadata"],
            feature_names=FEATURES,
        )


def test_rollback_endpoint_returns_404_without_a_previous_model(client, monkeypatch, tmp_path):
    # Isolate from the repo's real models/ directory: a checkout that legitimately has a
    # promoted+rolled-over real VIDEO artifact (previous_model sibling present) must not make
    # this "no previous model" case falsely pass by actually rolling back a real artifact.
    import app.api.training_routes as routes
    model_path = tmp_path / "recommendation_model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    monkeypatch.setattr(routes, "MODEL_PATH", model_path, raising=False)
    monkeypatch.setattr(routes, "METADATA_PATH", metadata_path, raising=False)
    response = client.post("/api/v1/recommendation-ml-service/model/rollback")
    assert response.status_code == 404 and response.json()["error"] == "NO_PREVIOUS_MODEL"


def test_live_rollback_endpoint_returns_404_without_a_previous_model(client, monkeypatch, tmp_path):
    # Same isolation as above, for the LIVE rollback endpoint and the real LIVE artifact.
    import app.api.training_routes as routes
    model_path = tmp_path / "live_recommendation_model.joblib"
    metadata_path = tmp_path / "live_model_metadata.json"
    monkeypatch.setattr(routes, "LIVE_MODEL_PATH", model_path, raising=False)
    monkeypatch.setattr(routes, "LIVE_METADATA_PATH", metadata_path, raising=False)
    response = client.post("/api/v1/recommendation-ml-service/model/rollback/live")
    assert response.status_code == 404 and response.json()["error"] == "NO_PREVIOUS_LIVE_MODEL"

