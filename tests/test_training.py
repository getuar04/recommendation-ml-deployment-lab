import pytest

from app.db.database import SessionLocal
from app.ml.split_lifecycle import InsufficientLifecycleDataError
from app.services import training_service
from app.services.training_service import InsufficientData, train


def test_insufficient_data():
    db=SessionLocal()
    try:
        try: train(db); assert False
        except InsufficientData: pass
    finally: db.close()


def test_insufficient_lifecycle_data_from_cross_family_selection_is_wrapped_as_insufficient_data(monkeypatch):
    """train_and_select_cross_family()/train_models() raising InsufficientLifecycleDataError
    (no chronological split attempt could produce valid, class-balanced splits for this
    dataset) must be translated into this service's own InsufficientData, not leak the
    split-lifecycle-specific exception type to callers -- distinct from the too-few-rows
    InsufficientData raised earlier in train() (test_insufficient_data above)."""
    from tests.helpers import _synthetic_labeled_frame

    monkeypatch.setattr(training_service, "build_dataset", lambda rows, content_by_id: _synthetic_labeled_frame())

    def _boom(df):
        raise InsufficientLifecycleDataError("no valid chronological split for this dataset")

    monkeypatch.setattr(training_service, "train_and_select_cross_family", _boom)
    db = SessionLocal()
    try:
        with pytest.raises(InsufficientData, match="no valid chronological split"):
            train(db)
    finally:
        db.close()

def test_train_job_status_not_found(client):
    response=client.get("/api/v1/recommendation-ml-service/model/train/jobs/does-not-exist")
    assert response.status_code==404 and response.json()["error"]=="JOB_NOT_FOUND"

def test_train_endpoint_is_async_and_reports_job_failure(client):
    response=client.post("/api/v1/recommendation-ml-service/model/train")
    assert response.status_code==202
    body=response.json()
    assert body["status"]=="PENDING" and "jobId" in body
    job=client.get(f"/api/v1/recommendation-ml-service/model/train/jobs/{body['jobId']}").json()
    assert job["status"]=="FAILED" and job["error"]["error"]=="INSUFFICIENT_TRAINING_DATA"


def test_train_async_alias_behaves_identically_to_train(client):
    """Spec §31: a separately-named async endpoint, same job/lock/background-task path as
    POST /model/train -- no training logic duplicated."""
    response=client.post("/api/v1/recommendation-ml-service/model/train/async")
    assert response.status_code==202
    body=response.json()
    assert body["status"]=="PENDING" and "jobId" in body
    job=client.get(f"/api/v1/recommendation-ml-service/model/train/jobs/{body['jobId']}").json()
    assert job["status"]=="FAILED" and job["error"]["error"]=="INSUFFICIENT_TRAINING_DATA"


def test_train_async_alias_shares_the_same_training_lock_as_train(client):
    """The two routes must not be able to run conflicting VIDEO training jobs concurrently
    -- they share one lock, exactly like two calls to the same endpoint would."""
    from app.db.database import SessionLocal
    from app.services import training_lock
    db=SessionLocal()
    try:
        training_lock.acquire(db,"VIDEO","held-by-test")
        response=client.post("/api/v1/recommendation-ml-service/model/train/async")
        assert response.status_code==409
        assert response.json()["error"]=="TRAINING_ALREADY_RUNNING"
    finally:
        training_lock.release(db,"VIDEO","held-by-test")
        db.close()


def test_model_versions_are_null_before_any_training(client,monkeypatch,tmp_path):
    import app.api.training_routes as routes
    monkeypatch.setattr(routes,"MODEL_PATH",tmp_path/"model.joblib")
    monkeypatch.setattr(routes,"METADATA_PATH",tmp_path/"model.json")
    versions=client.get("/api/v1/recommendation-ml-service/model/versions").json()
    assert versions=={"active":None,"previous":None,"candidate":None}


def test_model_versions_reflects_active_artifact_after_promotion(client,monkeypatch,tmp_path):
    from sklearn.dummy import DummyClassifier

    import app.api.training_routes as routes
    from app.ml.dataset_builder import FEATURES
    from app.ml.model_store import save
    model_path,metadata_path=tmp_path/"model.joblib",tmp_path/"model.json"
    monkeypatch.setattr(routes,"MODEL_PATH",model_path)
    monkeypatch.setattr(routes,"METADATA_PATH",metadata_path)
    metadata={"modelVersion":"v-versions-test","selectedModel":"Dummy","featureNames":FEATURES,
              "featureDefinitions":{},"targetDefinition":"x","splitStrategy":"x","selectionCriterion":"x",
              "calibration":{"applied":False},"decisionThreshold":.5,"trainingSamples":1,
              "modelSelectionSamples":1,"calibrationSamples":1,"thresholdTuningSamples":1,"testSamples":1,
              "classDistribution":{},"metrics":{"f1Score":.8},"sklearnVersion":"x","pythonVersion":"x",
              "randomSeed":42,"trainingDurationSeconds":.1,"trainedAt":"now","modelComparison":{},"datasetSource":{}}
    save(DummyClassifier(strategy="prior").fit([[0],[1]],[0,1]),metadata,
         model_path=model_path,metadata_path=metadata_path,model_dir=tmp_path)
    versions=client.get("/api/v1/recommendation-ml-service/model/versions").json()
    assert versions["active"]["modelVersion"]=="v-versions-test"
    assert versions["active"]["selectedModel"]=="Dummy"
    assert versions["previous"] is None and versions["candidate"] is None
