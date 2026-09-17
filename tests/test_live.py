from datetime import datetime, timezone
from types import SimpleNamespace

from app.ml.live_feature_builder import (
    LIVE_FEATURES,
    derive_live_affinities,
    live_feature_row,
    live_target,
)
from app.schemas.live_schemas import LiveCandidate
from tests.conftest import ensure_content


def raw(stream_id="live-1",**changes):
    value={"streamId":stream_id,"creatorId":"creator-1","category":"GAMING","status":"ACTIVE","currentViewerCount":850,"viewerGrowthRate":.18,"liveAgeMinutes":12,"creatorFollowed":False,"previousLiveInteractions":3,"previousLiveWatchTime":180,"region":"eu-central-1","language":"sq","regionMatch":True,"languageMatch":True,"alreadyJoined":False}
    value.update(changes);return value

def test_live_targets_are_separate_and_neutral_is_excluded():
    assert live_target(SimpleNamespace(joined=True,watch_time_seconds=61,liked=False,shared=False,commented=False,gift_sent=False,creator_followed=False,impression=True))==1
    assert live_target(SimpleNamespace(joined=False,watch_time_seconds=0,liked=False,shared=False,commented=False,gift_sent=False,creator_followed=False,impression=True))==0
    assert live_target(SimpleNamespace(joined=True,watch_time_seconds=30,liked=False,shared=False,commented=False,gift_sent=False,creator_followed=False,impression=True)) is None

def test_live_training_prediction_feature_parity():
    row=live_feature_row(LiveCandidate.model_validate(raw()),scoring_time=datetime.now(timezone.utc))
    assert list(row)==LIVE_FEATURES

def test_live_affinity_derivation_shared_by_training_and_serving():
    import app.ml.live_synthetic_data as synthetic_data
    import app.services.live_recommendation_service as service
    assert synthetic_data.derive_live_affinities is derive_live_affinities
    assert service.derive_live_affinities is derive_live_affinities
    assert derive_live_affinities(0,0)["live_category_affinity"]==0.5
    assert derive_live_affinities(0,0)["creator_affinity"]==0.5
    assert derive_live_affinities(20,0)["live_category_affinity"]==1.0
    assert derive_live_affinities(3,180)["average_live_watch_time_for_category"]==60
    assert derive_live_affinities(3,180)["recent_live_category_activity"]==3

def test_live_candidate_generation_filters_deduplicates_and_sources(client):
    streams=[raw("followed",creatorFollowed=True),raw("preferred",previousLiveInteractions=8),raw("ended",status="ENDED"),raw("followed",creatorFollowed=True),raw("new",liveAgeMinutes=5,currentViewerCount=5,viewerGrowthRate=0)]
    response=client.post("/api/v1/recommendation-ml-service/candidates/generate/live",json={"userId":"cold-user","limit":10,"streams":streams})
    assert response.status_code==200;items=response.json()["candidates"]
    assert all(x["status"]=="ACTIVE" for x in items)
    assert len({x["streamId"] for x in items})==len(items)
    assert "ended" not in {x["streamId"] for x in items}
    assert "FOLLOWED_CREATOR" in {x["source"] for x in items}
    assert {x["source"] for x in items}&{"PREFERRED_LIVE_CATEGORY","NEW_LIVE","EXPLORATION"}

def test_live_model_not_trained_and_empty_candidates(client,monkeypatch,tmp_path):
    import app.services.live_recommendation_service as service
    monkeypatch.setattr(service,"LIVE_MODEL_PATH",tmp_path/"missing.joblib");monkeypatch.setattr(service,"LIVE_METADATA_PATH",tmp_path/"missing.json")
    response=client.post("/api/v1/recommendation-ml-service/recommendations/live",json={"userId":"u","limit":5,"candidates":[]})
    assert response.status_code==503 and response.json()["error"]=="LIVE_MODEL_NOT_TRAINED"

def test_live_model_artifact_busy_maps_to_503_busy(client, monkeypatch):
    """recommend_live's ArtifactBusyError branch: the artifact kept changing (a retrain is in
    progress) and a stable read could not be obtained -- distinct from NOT_TRAINED/
    INCOMPATIBLE/CORRUPTED, and transient rather than a hard failure."""
    import app.services.live_recommendation_service as service
    from app.ml import model_store

    def _boom(*args, **kwargs):
        raise model_store.ArtifactBusyError("simulated: LIVE artifact kept changing during read")

    monkeypatch.setattr(service.model_cache.live_cache, "get", _boom)
    response = client.post("/api/v1/recommendation-ml-service/recommendations/live", json={"userId": "u", "limit": 5, "candidates": []})
    assert response.status_code == 503
    assert response.json()["reason"] == "BUSY"


def test_live_training_recommendation_sorting_penalty_and_diversity(client,monkeypatch,tmp_path):
    import app.api.training_routes as routes
    import app.ml.live_trainer as trainer
    import app.services.live_recommendation_service as service
    model=tmp_path/"live.joblib";metadata=tmp_path/"live.json"
    monkeypatch.setattr(trainer,"MODEL_DIR",tmp_path);monkeypatch.setattr(trainer,"LIVE_MODEL_PATH",model);monkeypatch.setattr(trainer,"LIVE_METADATA_PATH",metadata)
    monkeypatch.setattr(service,"LIVE_MODEL_PATH",model);monkeypatch.setattr(service,"LIVE_METADATA_PATH",metadata)
    monkeypatch.setattr(routes,"LIVE_MODEL_PATH",model);monkeypatch.setattr(routes,"LIVE_METADATA_PATH",metadata)
    trained=client.post("/api/v1/recommendation-ml-service/model/train/live");assert trained.status_code==202
    job=client.get(f"/api/v1/recommendation-ml-service/model/train/jobs/{trained.json()['jobId']}").json()
    assert job["status"]=="SUCCEEDED" and job["result"]["selectedModel"] in {"LogisticRegression","RandomForestClassifier"}
    assert job["result"]["calibration"]["method"]=="sigmoid" and job["result"]["calibration"]["applied"] is True
    metrics=client.get("/api/v1/recommendation-ml-service/model/metrics/live");assert metrics.status_code==200 and "f1Score" in metrics.json()["metrics"]
    streams=[raw(f"s{i}",creatorId=f"c{i%3}",category="GAMING" if i<4 else ("MUSIC" if i<6 else "SPORT"),alreadyJoined=(i==0),currentViewerCount=900-i*30) for i in range(8)]
    response=client.post("/api/v1/recommendation-ml-service/recommendations/live",json={"userId":"u","limit":8,"candidates":streams+[raw("inactive",status="ENDED")]})
    assert response.status_code==200;items=response.json()["recommendations"]
    assert all(items[i]["score"]>=items[i+1]["score"] for i in range(len(items)-1))
    assert "inactive" not in {x["streamId"] for x in items}
    # Under-fill guard (LIVE candidate-generation verification task): this scenario has only 3
    # distinct creators and MAX_CREATOR_IN_TOP=2 (app.ml.live_reranker), so honoring the
    # creator/category diversity caps for every one of the 8 eligible candidates is
    # mathematically impossible (3 creators x 2 = 6 < 8) -- app.ml.live_reranker.live_rerank
    # deliberately relaxes those caps rather than silently returning fewer than the 8 eligible,
    # still-active candidates actually available. Diversity-holds-when-achievable is covered
    # directly by tests/test_live_reranker.py's own dedicated unit tests instead.
    assert len(items)==8
    assert items[0]["streamId"]!="s0" or len(items)==1

def test_live_event_does_not_require_duration(client):
    ensure_content(client,"live-1",content_type="LIVE")
    response=client.post("/api/v1/recommendation-ml-service/events",json={"eventId":"live-event","userId":"u","contentId":"live-1","creatorId":"c","category":"gaming","eventType":"LIVE_WATCHED","liveWatchTimeSeconds":75,"timestamp":datetime.now(timezone.utc).isoformat()})
    assert response.status_code==201 and response.json()["watchPercentage"] is None
