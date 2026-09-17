"""Group A/B: content lifecycle -- contentType/duration persistence, isActive, integrity."""
from datetime import datetime, timezone

from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.models import Content


def test_content_type_and_duration_persist_for_video(client):
    response = client.post("/api/v1/recommendation-ml-service/contents", json={
        "contentId": "v-1", "creatorId": "cr-1", "contentType": "VIDEO",
        "category": "food", "durationSeconds": 120, "popularityScore": .5,
    })
    assert response.status_code == 201
    body = response.json()
    assert body["contentType"] == "VIDEO" and body["durationSeconds"] == 120
    assert body["isActive"] is True and body["updatedAt"] is not None

    fetched = client.get("/api/v1/recommendation-ml-service/contents/v-1").json()
    assert fetched["contentType"] == "VIDEO" and fetched["durationSeconds"] == 120
    assert fetched["isActive"] is True


def test_live_content_duration_may_be_null(client):
    response = client.post("/api/v1/recommendation-ml-service/contents", json={
        "contentId": "l-1", "creatorId": "cr-1", "contentType": "LIVE", "category": "gaming",
    })
    assert response.status_code == 201
    body = response.json()
    assert body["contentType"] == "LIVE" and body["durationSeconds"] is None


def test_content_before_creator_projection_remains_allowed(client):
    """Eventual-consistency audit (Task: local-existence blocking validation): content
    ingestion has no local User/creator existence check today -- confirmed unchanged by this
    task's interaction-side fix. `creatorId` is never queried against the local users table
    at content-creation time, so a creator whose own user.registered projection hasn't
    arrived yet must not block their content from being created."""
    response = client.post("/api/v1/recommendation-ml-service/contents", json={
        "contentId": "content-before-creator-1", "creatorId": "creator-never-registered-locally",
        "contentType": "VIDEO", "category": "food",
    })
    assert response.status_code == 201
    assert response.json()["creatorId"] == "creator-never-registered-locally"


def test_inactive_content_rejects_new_interactions(client):
    client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": "inactive-1", "creatorId": "cr-1",
                                           "contentType": "VIDEO", "category": "food"})
    db = SessionLocal()
    try:
        row = db.scalar(select(Content).where(Content.content_id == "inactive-1"))
        row.is_active = False
        db.commit()
    finally:
        db.close()

    response = client.post("/api/v1/recommendation-ml-service/events", json={
        "eventId": "e-inactive", "userId": "u", "contentId": "inactive-1", "creatorId": "cr-1",
        "category": "FOOD", "eventType": "VIDEO_WATCHED", "watchTimeSeconds": 30,
        "contentDurationSeconds": 60, "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert response.status_code == 409 and response.json()["error"] == "CONTENT_INACTIVE"


def test_duplicate_event_for_now_inactive_content_still_returns_the_original_idempotently(client):
    """Idempotency must not regress: a duplicate delivery of an already-accepted event
    returns the stored result even if the content was deactivated afterward -- content
    validation only gates whether a *new* interaction is accepted."""
    client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": "going-inactive", "creatorId": "cr-1",
                                           "contentType": "VIDEO", "category": "food"})
    payload = {"eventId": "e-before-deactivation", "userId": "u", "contentId": "going-inactive",
               "creatorId": "cr-1", "category": "FOOD", "eventType": "VIDEO_WATCHED",
               "watchTimeSeconds": 30, "contentDurationSeconds": 60,
               "timestamp": datetime.now(timezone.utc).isoformat()}
    first = client.post("/api/v1/recommendation-ml-service/events", json=payload)
    assert first.status_code == 201

    db = SessionLocal()
    try:
        row = db.scalar(select(Content).where(Content.content_id == "going-inactive"))
        row.is_active = False
        db.commit()
    finally:
        db.close()

    duplicate = client.post("/api/v1/recommendation-ml-service/events", json=payload)
    assert duplicate.status_code == 201 and duplicate.json()["stored"] is False


def test_patch_content_deactivates_and_reactivates(client):
    client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": "patchable-1", "creatorId": "cr-1",
                                           "contentType": "VIDEO", "category": "food"})
    deactivated = client.patch("/api/v1/recommendation-ml-service/contents/patchable-1", json={"isActive": False})
    assert deactivated.status_code == 200 and deactivated.json()["isActive"] is False

    fetched = client.get("/api/v1/recommendation-ml-service/contents/patchable-1").json()
    assert fetched["isActive"] is False

    reactivated = client.patch("/api/v1/recommendation-ml-service/contents/patchable-1", json={"isActive": True})
    assert reactivated.status_code == 200 and reactivated.json()["isActive"] is True


def test_patch_content_not_found(client):
    response = client.patch("/api/v1/recommendation-ml-service/contents/ghost-content", json={"isActive": False})
    assert response.status_code == 404 and response.json()["error"] == "CONTENT_NOT_FOUND"


def test_deactivated_content_blocks_new_interactions_via_public_api(client):
    """End-to-end through the real PATCH endpoint (not a direct DB write), unlike the
    older tests above that predate this endpoint existing."""
    client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": "patch-then-event", "creatorId": "cr-1",
                                           "contentType": "VIDEO", "category": "food"})
    assert client.patch("/api/v1/recommendation-ml-service/contents/patch-then-event", json={"isActive": False}).status_code == 200
    response = client.post("/api/v1/recommendation-ml-service/events", json={
        "eventId": "e-patch-then-event", "userId": "u", "contentId": "patch-then-event", "creatorId": "cr-1",
        "category": "FOOD", "eventType": "VIDEO_WATCHED", "watchTimeSeconds": 30,
        "contentDurationSeconds": 60, "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    assert response.status_code == 409 and response.json()["error"] == "CONTENT_INACTIVE"


def test_deactivated_content_excluded_from_candidate_generation(client):
    """Spec §24: candidate generation must return active content only."""
    client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": "cand-active", "creatorId": "cr-1",
                                           "contentType": "VIDEO", "category": "food", "popularityScore": .9})
    client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": "cand-inactive", "creatorId": "cr-1",
                                           "contentType": "VIDEO", "category": "food", "popularityScore": .9})
    client.patch("/api/v1/recommendation-ml-service/contents/cand-inactive", json={"isActive": False})

    result = client.post("/api/v1/recommendation-ml-service/candidates/generate", json={"userId": "cand-user", "limit": 50})
    assert result.status_code == 200
    ids = {c["contentId"] for c in result.json()["candidates"]}
    assert "cand-active" in ids
    assert "cand-inactive" not in ids


def test_future_timestamp_is_rejected(client):
    client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": "future-c", "creatorId": "cr-1",
                                           "contentType": "VIDEO", "category": "food"})
    from datetime import timedelta
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    response = client.post("/api/v1/recommendation-ml-service/events", json={
        "eventId": "e-future", "userId": "u", "contentId": "future-c", "creatorId": "cr-1",
        "category": "FOOD", "eventType": "VIDEO_WATCHED", "watchTimeSeconds": 30,
        "contentDurationSeconds": 60, "timestamp": future,
    })
    assert response.status_code == 422


def test_blank_content_id_after_strip_is_rejected(client):
    response = client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": "   ", "creatorId": "cr-1", "category": "SPORT"})
    assert response.status_code == 422


def test_corrupt_token_json_column_decodes_to_an_empty_list_not_a_crash():
    """Content.hashtags/topics/entities/subgenres all decode their *_json column via the same
    _decode_token_list() helper -- corrupt/non-JSON text (e.g. a hand-edited row, or a future
    migration bug) must degrade to [] like an absent column already does, never raise."""
    from app.db.models import _decode_token_list

    assert _decode_token_list("not valid json{{{") == []
    assert _decode_token_list('{"not": "a list"}') == []
    assert _decode_token_list(None) == []


def test_naive_created_at_is_accepted_and_normalized_at_write_time(client):
    """Exercises `created.replace(tzinfo=timezone.utc)` for a naive createdAt -- the
    normalized value is what gets persisted (SQLite itself does not round-trip tzinfo through
    storage/read-back, so the response body's exact ISO suffix isn't asserted here; the point
    is that the naive-input code path runs without error)."""
    response = client.post("/api/v1/recommendation-ml-service/contents", json={
        "contentId": "naive-ts-content", "creatorId": "cr-1", "category": "SPORT",
        "createdAt": "2026-01-01T00:00:00",  # no timezone offset
    })
    assert response.status_code == 201
    assert response.json()["createdAt"] is not None
