"""Semantic metadata (title/hashtags/topics/entities/subgenres) on content ingestion
(POST /contents) and candidate generation (POST /candidates/generate) -- VIDEO only."""
from datetime import datetime, timedelta, timezone

from app.db.database import SessionLocal
from app.db.models import Content


def test_content_create_accepts_and_normalizes_semantic_metadata(client):
    response = client.post("/api/v1/recommendation-ml-service/contents", json={
        "contentId": "sem-1", "creatorId": "cr-1", "category": "SPORT",
        "title": "  Messi Master Class  ",
        "hashtags": ["#Messi", "messi", "Football"],
        "entities": ["Lionel Messi"],
    })
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Messi Master Class"
    assert body["hashtags"] == ["MESSI", "FOOTBALL"]  # deduped, order-preserving, normalized
    assert body["entities"] == ["LIONEL_MESSI"]
    assert body["topics"] == []
    assert body["subgenres"] == []


def test_content_create_without_semantic_fields_defaults_to_empty(client):
    response = client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": "sem-2", "creatorId": "cr-1", "category": "SPORT"})
    assert response.status_code == 201
    body = response.json()
    assert body["title"] is None
    assert body["hashtags"] == body["topics"] == body["entities"] == body["subgenres"] == []


def test_content_get_reads_back_semantic_metadata(client):
    client.post("/api/v1/recommendation-ml-service/contents", json={
        "contentId": "sem-3", "creatorId": "cr-1", "category": "MUSIC", "hashtags": ["Metallica"],
    })
    response = client.get("/api/v1/recommendation-ml-service/contents/sem-3")
    assert response.status_code == 200
    assert response.json()["hashtags"] == ["METALLICA"]


def test_content_create_rejects_non_list_hashtags(client):
    response = client.post("/api/v1/recommendation-ml-service/contents", json={
        "contentId": "sem-4", "creatorId": "cr-1", "category": "SPORT", "hashtags": "messi",
    })
    assert response.status_code == 422


def test_content_create_rejects_too_many_tokens(client):
    response = client.post("/api/v1/recommendation-ml-service/contents", json={
        "contentId": "sem-5", "creatorId": "cr-1", "category": "SPORT",
        "hashtags": [f"tag{i}" for i in range(20)],
    })
    assert response.status_code == 422


def _seed_content_with_semantics(count=3):
    db = SessionLocal()
    now = datetime.now(timezone.utc)
    try:
        for i in range(count):
            db.add(Content(
                content_id=f"gen-sem-{i}", creator_id=f"cr{i}", category="SPORT", popularity_score=0.5,
                created_at=now - timedelta(hours=i), title="Messi Highlights",
                hashtags_json='["FOOTBALL"]', entities_json='["MESSI"]',
            ))
        db.commit()
    finally:
        db.close()


def test_candidate_generation_passes_through_semantic_metadata(client):
    _seed_content_with_semantics()
    response = client.post("/api/v1/recommendation-ml-service/candidates/generate", json={"userId": "u1", "limit": 10})
    assert response.status_code == 200
    candidates = response.json()["candidates"]
    assert candidates
    for candidate in candidates:
        assert candidate["title"] == "Messi Highlights"
        assert candidate["hashtags"] == ["FOOTBALL"]
        assert candidate["entities"] == ["MESSI"]
        assert candidate["topics"] == []
        assert candidate["subgenres"] == []


def test_candidate_generation_semantic_fields_default_empty_for_untagged_content(client):
    db = SessionLocal()
    try:
        db.add(Content(content_id="untagged-1", creator_id="cr-1", category="FOOD",
                        popularity_score=0.5, created_at=datetime.now(timezone.utc)))
        db.commit()
    finally:
        db.close()
    response = client.post("/api/v1/recommendation-ml-service/candidates/generate", json={"userId": "u1", "limit": 10})
    assert response.status_code == 200
    candidate = next(c for c in response.json()["candidates"] if c["contentId"] == "untagged-1")
    assert candidate["title"] is None
    assert candidate["hashtags"] == []
