"""Spec §59: one deterministic end-to-end scenario across the whole workflow —
users → contents → interactions → real training → READY artifact → strategy-labelled
recommendations for both an active and a brand-new user. No mocked model: the real
trainer runs on the real synthetic generator's data."""
import time

from scripts.generate_synthetic_data import generate
from tests.conftest import SYNTHETIC_DATA_REFERENCE_TIMESTAMP


def test_end_to_end_workflow(client, monkeypatch, tmp_path):
    # Train into tmp_path, never into the repo's models/ directory: other tests assert the
    # repo's legacy artifact stays legacy, and a real E2E training run must not clobber it.
    import app.api.training_routes as routes
    import app.ml.model_store as store
    import app.services.recommendation_service as service
    from app.services import training_service
    model_path = tmp_path / "recommendation_model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    for module in (routes, store, service):
        monkeypatch.setattr(module, "MODEL_PATH", model_path, raising=False)
    for module in (routes, store):
        monkeypatch.setattr(module, "METADATA_PATH", metadata_path, raising=False)
    monkeypatch.setattr(store, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(training_service, "MODEL_PATH", model_path, raising=False)
    monkeypatch.setattr(training_service, "METADATA_PATH", metadata_path, raising=False)
    # 1. Create a brand-new user: cold start, no fabricated interests.
    assert client.post("/api/v1/recommendation-ml-service/users", json={"userId": "e2e-new-user"}).status_code == 201
    assert client.get("/api/v1/recommendation-ml-service/users/e2e-new-user/behaviour-profile").json()["status"] == "COLD_START"

    # 2-4. Content + positive/negative interactions persisted (real generator, fixed seed,
    # fixed reference timestamp -- see tests/conftest.py's SYNTHETIC_DATA_REFERENCE_TIMESTAMP).
    generate(reference_timestamp=SYNTHETIC_DATA_REFERENCE_TIMESTAMP)

    # 5-7. Train the real model through the API job and wait for completion.
    job_id = client.post("/api/v1/recommendation-ml-service/model/train").json()["jobId"]
    for _ in range(300):
        status = client.get(f"/api/v1/recommendation-ml-service/model/train/jobs/{job_id}").json()
        if status["status"] in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.2)
    assert status["status"] == "SUCCEEDED", status

    # 8-9. Valid artifact exists and model status is READY. Cross-family production selection
    # (XGBRanker promotion task) means the winner is not necessarily a classifier -- assert
    # family-appropriate metrics rather than hardcoding classifier-only keys, mirroring
    # app.ml.artifact_metadata's own family-conditional metadata contract.
    model_status = client.get("/api/v1/recommendation-ml-service/model/status").json()
    assert model_status["status"] == "READY" and model_status["modelVersion"]
    summary = client.get("/api/v1/recommendation-ml-service/model/metrics/summary").json()
    if summary["modelFamily"] == "ranker":
        assert "criticalPassRate" in summary["metrics"]
        assert "prAuc" not in summary["metrics"] and "f1Score" not in summary["metrics"]
    else:
        assert "f1Score" in summary["metrics"] and "prAuc" in summary["metrics"]

    candidates = [
        {"contentId": f"e2e-video-{i}", "creatorId": f"creator-{i % 3}", "category": category,
         "contentPopularityScore": popularity, "contentAgeHours": 5,
         "creatorFollowed": i == 0, "alreadySeen": False}
        for i, (category, popularity) in enumerate(
            [("FOOD", .9), ("SPORT", .7), ("MUSIC", .6), ("FOOD", .5), ("TRAVEL", .8)])
    ]

    # 10-11. An active user receives concrete content IDs from the real model.
    active = client.post("/api/v1/recommendation-ml-service/recommendations",
                         json={"userId": "user-1", "limit": 3, "candidates": candidates}).json()
    assert active["strategy"] == "PERSONALISED_ML" and active["interactionCount"] > 0
    assert active["recommendations"] and all("contentId" in r for r in active["recommendations"])

    # 12-13. The brand-new user gets a COLD_START-labelled response, never fake personalisation.
    cold = client.post("/api/v1/recommendation-ml-service/recommendations",
                       json={"userId": "e2e-new-user", "limit": 3, "candidates": candidates}).json()
    assert cold["strategy"] == "COLD_START" and cold["interactionCount"] == 0
    assert cold["recommendations"], "cold-start must still return content, not an empty feed"

    # 14-15. Model survives a cache reset (in-process restart analogue) and still serves.
    from app.ml import model_cache
    model_cache.video_cache.invalidate()
    again = client.post("/api/v1/recommendation-ml-service/recommendations",
                        json={"userId": "user-1", "limit": 3, "candidates": candidates})
    assert again.status_code == 200 and again.json()["recommendations"]
