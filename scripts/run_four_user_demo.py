"""Four-user live presentation demo, run in-process against the REAL production code paths
(FastAPI TestClient -- same convention as scripts/demo_production_recommendation_trace.py and
scripts/run_two_tower_shadow_comparison.py), never a live server, never the real repository
DB/models directory.

Reuses the EXISTING deterministic demo fixtures (scripts.seed_demo_users) -- same 4 uuid5
user ids, same 40-item shared candidate pool -- rather than inventing a second demo dataset.
Ingests everything through the real POST /contents / POST /events endpoints (not a DB shortcut),
trains ONE isolated DEMO/LOCAL model via the real production trainer (never promoted, never the
real models/ artifact -- clearly labelled below), then exercises: candidate generation, User A
(long-term SPORT) vs User B (long-term MUSIC) on the SAME candidate pool, User C's live SPORT->
MUSIC shift (before/events/after, same modelVersion), a small fast-skip movement, User D's
semantic (not creator) personalization, a NOT_INTERESTED live suppression demo, a search-intent
demo, and a seen-content demo. Writes the exact JSON request bodies used to docs/ as it goes, so
the same calls are reproducible from Postman/curl afterward.

RECOMMENDATION_DATA_MODE stays LOCAL throughout (the default) -- no UBS/Candidate Service/Kafka
involved; this demo proves the existing local/demo flow end-to-end.

Usage: python -m scripts.run_four_user_demo
"""
from __future__ import annotations

import json
import os
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite://")

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("MODEL_ARTIFACT_ROOT", tempfile.mkdtemp(prefix="fouruser_demo_models_"))

from fastapi.testclient import TestClient

from app.api import training_routes
from app.db.database import Base, SessionLocal, engine
from app.db.repositories import interactions as get_interactions
from app.main import app
from app.ml import model_store
from app.ml.dataset_builder import FEATURE_DEFINITIONS, FEATURES
from app.ml.feature_builder import TARGET_DEFINITION
from app.ml.trainer import train_models
from scripts.generate_synthetic_data import (
    generate as generate_bulk_synthetic_data,
)
from scripts.seed_demo_users import (
    build_all_users,
    build_shared_candidate_pool,
    build_user_c_shift_events,
    stable_uuid,
)

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
REFERENCE_TIMESTAMP = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _hr(title: str = "") -> None:
    print()
    print("=" * 88)
    if title:
        print(title)
        print("=" * 88)


def _write_json(name: str, payload) -> None:
    path = DOCS_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote docs/{name}")


# --------------------------------------------------------------------------------------
# Step 1: isolated DEMO/LOCAL model -- trained ONCE via the real production trainer, never
# promoted, never the real repository models/ artifact. modelVersion is prefixed
# "fouruser-demo-" so it can never be confused with a real "recommendation-prod-*" version.
# --------------------------------------------------------------------------------------
def train_isolated_demo_model(db) -> dict:
    from sqlalchemy import select

    from app.db.models import Content

    rows = get_interactions(db)
    content_by_id = {c.content_id: c for c in db.scalars(select(Content)).all()}
    from app.ml.dataset_builder import build_dataset
    df = build_dataset(rows, content_by_id)
    result = train_models(df)
    model = result.pop("model")
    now = datetime.now(timezone.utc)
    metadata = {
        "modelVersion": f"fouruser-demo-{now:%Y%m%d%H%M%S}",
        "modelType": f"{type(model).__module__}.{type(model).__qualname__}",
        "selectedModel": result["selectedModel"], "featureNames": FEATURES,
        "featureDefinitions": FEATURE_DEFINITIONS, "targetDefinition": TARGET_DEFINITION,
        "splitStrategy": result["splitLifecycleDescription"], "selectionCriterion": result["selectionCriterion"],
        "calibration": {"applied": True, "method": "sigmoid"}, "decisionThreshold": result["decisionThreshold"],
        "trainingSamples": result["splitSizes"]["train"], "modelSelectionSamples": result["splitSizes"]["modelSelection"],
        "calibrationSamples": result["splitSizes"]["calibration"], "thresholdTuningSamples": result["splitSizes"]["thresholdTuning"],
        "testSamples": result["splitSizes"]["test"], "classDistribution": result.get("classDistribution", {}),
        "sklearnVersion": "n/a", "pythonVersion": "n/a", "randomSeed": 42, "trainingDurationSeconds": 0.0,
        "trainedAt": now.isoformat(), "metrics": result.get("metrics", {}), "modelComparison": result.get("modelComparison", {}),
        "datasetSource": {"synthetic": False, "totalRowCount": len(df)},
    }
    model_store.save(model, metadata, model_path=training_routes.MODEL_PATH, metadata_path=training_routes.METADATA_PATH)
    return metadata


# --------------------------------------------------------------------------------------
# Step 2: seed the 4 users + shared candidate pool through the REAL ingestion endpoints.
# --------------------------------------------------------------------------------------
def seed_via_client(client: TestClient, user_id: str, builder) -> None:
    for event in builder.events:
        content_payload = {
            "contentId": event["content_id"], "creatorId": event["creator_id"], "contentType": "VIDEO",
            "category": event["category"], "durationSeconds": event["content_duration_seconds"], "popularityScore": 0.5,
        }
        for field in ("title", "hashtags", "topics", "entities", "subgenres"):
            if field in event:
                content_payload[field] = event[field]
        client.post("/api/v1/recommendation-ml-service/contents", json=content_payload)
        response = client.post("/api/v1/recommendation-ml-service/events", json={
            "eventId": event["event_id"], "userId": event["user_id"], "contentId": event["content_id"],
            "creatorId": event["creator_id"], "category": event["category"], "eventType": event["event_type"],
            "watchTimeSeconds": event["watch_time_seconds"], "contentDurationSeconds": event["content_duration_seconds"],
            "liked": event["liked"], "shared": event["shared"], "favorited": event["favorited"],
            "commented": event["commented"], "creatorFollowed": event["creator_followed"],
            "timestamp": event["timestamp"],
        })
        assert response.status_code == 201, response.text


def seed_candidate_pool_via_client(client: TestClient, pool: list[dict]) -> None:
    for entry in pool:
        payload = {"contentId": entry["contentId"], "creatorId": entry["creatorId"], "contentType": "VIDEO",
                   "category": entry["category"], "durationSeconds": 100.0, "popularityScore": entry["popularityScore"]}
        for field in ("title", "hashtags", "topics", "entities", "subgenres"):
            if field in entry:
                payload[field] = entry[field]
        resp = client.post("/api/v1/recommendation-ml-service/contents", json=payload)
        assert resp.status_code in (201, 409), resp.text


def candidate_request_body(entry: dict) -> dict:
    return {
        "contentId": entry["contentId"], "creatorId": entry["creatorId"], "category": entry["category"],
        "contentPopularityScore": entry["popularityScore"], "contentAgeHours": entry["ageHours"],
        "creatorFollowed": False, "alreadySeen": False, "title": entry["title"],
        "hashtags": entry["hashtags"], "topics": entry["topics"], "entities": entry["entities"], "subgenres": entry["subgenres"],
    }


def recommend_body(user_id: str, pool: list[dict], *, limit: int = 40, search_intent: dict | None = None) -> dict:
    body = {"userId": user_id, "limit": limit, "candidates": [candidate_request_body(e) for e in pool]}
    if search_intent is not None:
        body["searchIntent"] = search_intent
    return body


def print_topn(body: dict, n: int = 10) -> None:
    print(f"  {'rank':>4s} {'category':10s} {'score':>7s}  reason  |  title")
    for r in body["recommendations"][:n]:
        print(f"  {r['rank']:>4d} {r['category']:10s} {r['score']:>7.4f}  {r['reason']:<24s} {r.get('title') or ''}")


def category_distribution(body: dict, n: int = 10) -> dict:
    from collections import Counter
    return dict(Counter(r["category"] for r in body["recommendations"][:n]))


def rank_of(body: dict, content_id: str) -> int | None:
    return next((r["rank"] for r in body["recommendations"] if r["contentId"] == content_id), None)


def score_of(body: dict, content_id: str) -> float | None:
    return next((r["score"] for r in body["recommendations"] if r["contentId"] == content_id), None)


def print_score_deltas(label_a: str, resp_a: dict, label_b: str, resp_b: dict, pool: list[dict], categories: tuple[str, ...]) -> None:
    """Exact per-item score comparison for every candidate in the given categories --
    precise, not diluted by Top-10 slot/diversity-rerank noise the way a category-count
    comparison is. This is the direct evidence for "same model, same candidates, different
    user features -> different scores". Prints a per-category average delta summary first
    (the headline number for a live presentation), then the full per-item breakdown."""
    deltas_by_category: dict[str, list[float]] = {c: [] for c in categories}
    rows = []
    for entry in pool:
        if entry["category"] not in categories:
            continue
        sa = score_of(resp_a, entry["contentId"])
        sb = score_of(resp_b, entry["contentId"])
        if sa is None or sb is None:
            continue
        delta = sb - sa
        deltas_by_category[entry["category"]].append(delta)
        rows.append((entry["title"], entry["category"], sa, sb, delta))

    print(f"  average score delta ({label_b} - {label_a}) by category:")
    for category in categories:
        values = deltas_by_category.get(category) or []
        if values:
            print(f"    {category:10s} mean_delta={sum(values) / len(values):+.4f}  (n={len(values)})")
    print(f"\n  {'title':52s} {'category':8s} {label_a:>10s} {label_b:>10s} {'delta':>9s}")
    for title, category, sa, sb, delta in rows:
        print(f"  {title[:52]:52s} {category:8s} {sa:>10.4f} {sb:>10.4f} {delta:>+9.4f}")


def main() -> None:
    Base.metadata.create_all(engine)
    client = TestClient(app)

    _hr("SETUP: isolated DEMO/LOCAL model + 4 deterministic UUID demo users")
    now = REFERENCE_TIMESTAMP
    # The DEMO/LOCAL model is trained on the SAME large, deterministic synthetic bulk
    # population (scripts.generate_synthetic_data, ~12000+ interactions across 100 general
    # users + the established behavioral cohorts) already used and verified throughout this
    # project's own core-model work -- not on the 4 demo users' own ~120 rows alone, which is
    # far too small for a classifier to learn robust category/session/semantic behavior from
    # (confirmed by direct experiment: training on the demo rows alone produced a model
    # dominated by content_popularity_score noise, not real personalization). The 4 demo
    # users' own real interaction history is layered on top of this same isolated DB via the
    # real POST /events endpoint below, then scored by this well-trained shared model at
    # request time -- exactly the "one global model, per-user features" architecture this
    # demo exists to show.
    generate_bulk_synthetic_data(reference_timestamp=now)
    print("  seeded large synthetic bulk population (scripts.generate_synthetic_data) for real model training")
    users = build_all_users(now)
    for label, (user_id, builder) in users.items():
        seed_via_client(client, user_id, builder)
        print(f"  User {label}: {user_id}  ({len(builder.events)} interactions seeded)")

    pool = build_shared_candidate_pool()
    seed_candidate_pool_via_client(client, pool)
    print(f"  shared candidate pool: {len(pool)} items across {len({e['category'] for e in pool})} categories, seeded as real Content rows")

    db = SessionLocal()
    try:
        metadata = train_isolated_demo_model(db)
    finally:
        db.close()
    print(f"  trained DEMO/LOCAL model (isolated artifact, never promoted): modelVersion={metadata['modelVersion']} "
          f"selectedModel={metadata['selectedModel']}")

    # Training is done -- the bulk synthetic catalog's own content ids (scripts.
    # generate_synthetic_data's "video-1".."video-300", never UUIDs -- that generator's ids
    # are internal training fixtures, not audience-facing) must not leak into candidate
    # generation or any other demo-facing endpoint, which the CRITICAL ID REQUIREMENT above
    # demands be UUIDs throughout. Deactivating them (not deleting -- their interaction
    # history, already used for training, is untouched) scopes every subsequent
    # candidate-generation/content-lookup call to exactly this demo's 40-item UUID catalog.
    # A direct ORM update, not a new endpoint -- PATCH /contents/{id} one-at-a-time for 300
    # rows would be pure overhead for a same-process isolated demo DB.
    from app.db.models import Content as _Content
    db = SessionLocal()
    try:
        deactivated = db.query(_Content).filter(_Content.content_id.like("video-%")).update(
            {"is_active": False}, synchronize_session=False,
        )
        db.commit()
    finally:
        db.close()
    print(f"  deactivated {deactivated} bulk synthetic training-only content rows (non-UUID ids) -- "
          f"candidate generation now scoped to the {len(pool)}-item UUID demo catalog only")

    a_id, _ = users["A"]
    b_id, _ = users["B"]
    c_id, _ = users["C"]
    d_id, _ = users["D"]

    # ---------------------------------------------------------------------------------- health
    _hr("ENDPOINT VALIDATION")
    health = client.get("/api/v1/recommendation-ml-service/health").json()
    print(f"  GET /health -> status={health['status']} dependencies={health['dependencies']} "
          f"(RECOMMENDATION_DATA_MODE=LOCAL: no ubs/candidateService/kafka keys expected)")

    # ---------------------------------------------------------------------------- candidate gen
    _hr("CANDIDATE GENERATION (POST /api/v1/recommendation-ml-service/candidates/generate)")
    gen_results = {}
    for label, uid in (("A", a_id), ("B", b_id), ("C", c_id), ("D", d_id)):
        req = {"userId": uid, "limit": 30}
        _write_json(f"candidate-generate-user-{label.lower()}.json", req)
        resp = client.post("/api/v1/recommendation-ml-service/candidates/generate", json=req)
        assert resp.status_code == 200, resp.text
        gen_body = resp.json()
        cids = [c["contentId"] for c in gen_body["candidates"]]
        for cid in cids:
            uuid.UUID(cid)  # raises if not a real UUID
        assert len(cids) == len(set(cids)), "duplicate candidate content IDs"
        gen_results[label] = gen_body
        print(f"  User {label}: {len(gen_body['candidates'])} generated candidates, all UUID contentIds, no duplicates")

    # ---------------------------------------------------------------------------------- A vs B
    _hr("USER A -- long-term SPORT (shared candidate pool)")
    body_a = recommend_body(a_id, pool)
    _write_json("user-a-recommend.json", body_a)
    resp_a = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_a)
    assert resp_a.status_code == 200, resp_a.text
    resp_a = resp_a.json()
    print(f"  strategy={resp_a['strategy']} interactionCount={resp_a['interactionCount']} modelVersion={resp_a['modelVersion']}")
    print_topn(resp_a)
    dist_a = category_distribution(resp_a)
    print(f"  Top-10 category distribution: {dist_a}")

    _hr("USER B -- long-term MUSIC (SAME candidate pool)")
    body_b = recommend_body(b_id, pool)
    _write_json("user-b-recommend.json", body_b)
    resp_b = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_b)
    assert resp_b.status_code == 200, resp_b.text
    resp_b = resp_b.json()
    print(f"  strategy={resp_b['strategy']} interactionCount={resp_b['interactionCount']} modelVersion={resp_b['modelVersion']}")
    print_topn(resp_b)
    dist_b = category_distribution(resp_b)
    print(f"  Top-10 category distribution: {dist_b}")

    _hr("A vs B -- SAME model, SAME candidates, DIFFERENT ranking")
    print(f"  User A top category: {max(dist_a, key=lambda k: dist_a[k])}  |  User B top category: {max(dist_b, key=lambda k: dist_b[k])}")
    print(f"  SPORT share (Top-10) -> A: {dist_a.get('SPORT', 0)}/10   B: {dist_b.get('SPORT', 0)}/10")
    print(f"  MUSIC share (Top-10) -> A: {dist_a.get('MUSIC', 0)}/10   B: {dist_b.get('MUSIC', 0)}/10")
    print("\n  exact per-candidate score comparison (precise -- not diluted by Top-10/diversity slotting):")
    print_score_deltas("scoreA", resp_a, "scoreB", resp_b, pool, ("SPORT", "MUSIC"))

    # ---------------------------------------------------------------------------------- User C
    _hr("USER C BEFORE -- long-term SPORT, no recent MUSIC shift yet")
    body_c_before = recommend_body(c_id, pool)
    _write_json("user-c-before.json", body_c_before)
    resp_c_before = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_c_before)
    assert resp_c_before.status_code == 200, resp_c_before.text
    resp_c_before = resp_c_before.json()
    print(f"  strategy={resp_c_before['strategy']} modelVersion={resp_c_before['modelVersion']}")
    print_topn(resp_c_before)
    dist_c_before = category_distribution(resp_c_before)
    print(f"  Top-10 category distribution BEFORE: {dist_c_before}")

    _hr("USER C -- live SPORT -> MUSIC shift events (POST /api/v1/recommendation-ml-service/events)")
    _, shift_builder = build_user_c_shift_events(datetime.now(timezone.utc))
    event_name_map = [
        ("user-c-music-watch-1.json", 0), ("user-c-music-like.json", 1), ("user-c-music-share.json", 2),
    ]
    for name, idx in event_name_map:
        ev = shift_builder.events[idx]
        _write_json(name, {
            "eventId": ev["event_id"], "userId": ev["user_id"], "contentId": ev["content_id"], "creatorId": ev["creator_id"],
            "category": ev["category"], "eventType": ev["event_type"], "watchTimeSeconds": ev["watch_time_seconds"],
            "contentDurationSeconds": ev["content_duration_seconds"], "liked": ev["liked"], "shared": ev["shared"],
            "favorited": ev["favorited"], "commented": ev["commented"], "creatorFollowed": ev["creator_followed"],
            "timestamp": ev["timestamp"],
        })
    skip_events = [ev for ev in shift_builder.events if ev["event_type"] == "VIDEO_SKIPPED"]
    for i, ev in enumerate(skip_events[:2], start=1):
        _write_json(f"user-c-sport-fast-skip-{i}.json", {
            "eventId": ev["event_id"], "userId": ev["user_id"], "contentId": ev["content_id"], "creatorId": ev["creator_id"],
            "category": ev["category"], "eventType": ev["event_type"], "watchTimeSeconds": ev["watch_time_seconds"],
            "contentDurationSeconds": ev["content_duration_seconds"], "liked": False, "shared": False,
            "favorited": False, "commented": False, "creatorFollowed": False, "timestamp": ev["timestamp"],
        })
    for event in shift_builder.events:
        content_payload = {"contentId": event["content_id"], "creatorId": event["creator_id"], "contentType": "VIDEO",
                           "category": event["category"], "durationSeconds": event["content_duration_seconds"], "popularityScore": 0.5}
        client.post("/api/v1/recommendation-ml-service/contents", json=content_payload)
        resp = client.post("/api/v1/recommendation-ml-service/events", json={
            "eventId": event["event_id"], "userId": event["user_id"], "contentId": event["content_id"],
            "creatorId": event["creator_id"], "category": event["category"], "eventType": event["event_type"],
            "watchTimeSeconds": event["watch_time_seconds"], "contentDurationSeconds": event["content_duration_seconds"],
            "liked": event["liked"], "shared": event["shared"], "favorited": event["favorited"],
            "commented": event["commented"], "creatorFollowed": event["creator_followed"], "timestamp": event["timestamp"],
        })
        assert resp.status_code == 201, resp.text
    print(f"  sent {len(shift_builder.events)} real events via POST /api/v1/recommendation-ml-service/events "
          f"(7 MUSIC watch/like/share/favorite, 2 SPORT fast-skips, 1 SPORT CONTENT_NOT_INTERESTED)")

    _hr("USER C AFTER -- SAME candidate pool, no retraining")
    body_c_after = recommend_body(c_id, pool)
    _write_json("user-c-after.json", body_c_after)
    resp_c_after = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_c_after)
    assert resp_c_after.status_code == 200, resp_c_after.text
    resp_c_after = resp_c_after.json()
    print(f"  strategy={resp_c_after['strategy']} modelVersion={resp_c_after['modelVersion']}")
    print_topn(resp_c_after)
    dist_c_after = category_distribution(resp_c_after)
    print(f"  Top-10 category distribution AFTER: {dist_c_after}")
    print(f"\n  MUSIC share (Top-10) -> BEFORE: {dist_c_before.get('MUSIC', 0)}/10   AFTER: {dist_c_after.get('MUSIC', 0)}/10")
    print(f"  SPORT share (Top-10) -> BEFORE: {dist_c_before.get('SPORT', 0)}/10   AFTER: {dist_c_after.get('SPORT', 0)}/10")
    print("\n  exact per-candidate score comparison (precise -- not diluted by Top-10/diversity slotting):")
    print_score_deltas("before", resp_c_before, "after", resp_c_after, pool, ("SPORT", "MUSIC"))
    print(f"\n  modelVersion unchanged: {resp_c_before['modelVersion'] == resp_c_after['modelVersion']} "
          f"({resp_c_before['modelVersion']})")
    print("  MODEL_RETRAINED=false  (no POST /model/train call was made anywhere in this demo)")

    _hr("USER C -- small controlled fast-skip movement")
    sport_probe = next(e for e in pool if e["category"] == "SPORT")
    before_score = score_of(resp_c_after, sport_probe["contentId"])
    before_rank = rank_of(resp_c_after, sport_probe["contentId"])
    print(f"  probe candidate: {sport_probe['title']} ({sport_probe['contentId']})")
    print(f"  score/rank before fast-skip: score={before_score} rank={before_rank}")
    fs_event_1 = {"eventId": stable_uuid("demo2-event", f"{c_id}:fastskip-probe-1"), "userId": c_id,
                  "contentId": sport_probe["contentId"], "creatorId": sport_probe["creatorId"], "category": "SPORT",
                  "eventType": "VIDEO_SKIPPED", "watchTimeSeconds": 5.0, "contentDurationSeconds": 100.0,
                  "liked": False, "shared": False, "favorited": False, "commented": False, "creatorFollowed": False,
                  "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()}
    fs_event_2 = {**fs_event_1, "eventId": stable_uuid("demo2-event", f"{c_id}:fastskip-probe-2"),
                  "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()}
    for ev in (fs_event_1, fs_event_2):
        resp = client.post("/api/v1/recommendation-ml-service/events", json={
            "eventId": ev["eventId"], "userId": ev["userId"], "contentId": ev["contentId"], "creatorId": ev["creatorId"],
            "category": ev["category"], "eventType": ev["eventType"], "watchTimeSeconds": ev["watchTimeSeconds"],
            "contentDurationSeconds": ev["contentDurationSeconds"], "liked": ev["liked"], "shared": ev["shared"],
            "favorited": ev["favorited"], "commented": ev["commented"], "creatorFollowed": ev["creatorFollowed"],
            "timestamp": ev["timestamp"],
        })
        assert resp.status_code == 201, resp.text
    resp_c_fastskip = client.post("/api/v1/recommendation-ml-service/recommendations", json=recommend_body(c_id, pool)).json()
    after_score = score_of(resp_c_fastskip, sport_probe["contentId"])
    after_rank = rank_of(resp_c_fastskip, sport_probe["contentId"])
    print(f"  score/rank after 2 fast-skips: score={after_score} rank={after_rank}")
    print(f"  moved in expected negative direction: {(after_score or 0) <= (before_score or 0) + 1e-6}")
    print("  (not presented as a guaranteed strict skip2->skip3->skip4 monotonic sequence -- known limitation, see report)")

    # ---------------------------------------------------------------------------------- User D
    _hr("USER D -- semantic personalization (GAMING: preferred BATTLE_ROYALE > neutral > rejected RACING_SIM)")
    d_candidates = [
        {"contentId": stable_uuid("demo2-d-cand", "preferred"), "creatorId": stable_uuid("demo2-creator", f"{d_id}:GAMING:preferred"),
         "category": "GAMING", "contentPopularityScore": 0.5, "contentAgeHours": 24, "creatorFollowed": False, "alreadySeen": False,
         "title": "New battle royale gameplay highlights", "hashtags": ["BATTLE_ROYALE"], "topics": ["BATTLE_ROYALE"],
         "entities": ["BATTLE_ROYALE"], "subgenres": ["BATTLE_ROYALE"]},
        {"contentId": stable_uuid("demo2-d-cand", "neutral"), "creatorId": stable_uuid("demo2-creator", "neutral-gaming-creator"),
         "category": "GAMING", "contentPopularityScore": 0.5, "contentAgeHours": 24, "creatorFollowed": False, "alreadySeen": False,
         "title": "General gaming roundup this week", "hashtags": [], "topics": [], "entities": [], "subgenres": []},
        {"contentId": stable_uuid("demo2-d-cand", "rejected"), "creatorId": stable_uuid("demo2-creator", f"{d_id}:GAMING:weak"),
         "category": "GAMING", "contentPopularityScore": 0.5, "contentAgeHours": 24, "creatorFollowed": False, "alreadySeen": False,
         "title": "New racing sim gameplay highlights", "hashtags": ["RACING_SIM"], "topics": ["RACING_SIM"],
         "entities": ["RACING_SIM"], "subgenres": ["RACING_SIM"]},
    ]
    body_d = {"userId": d_id, "limit": 3, "candidates": d_candidates}
    _write_json("user-d-recommend.json", body_d)
    resp_d = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_d)
    assert resp_d.status_code == 200, resp_d.text
    resp_d = resp_d.json()
    print_topn(resp_d, n=3)
    pref_score = score_of(resp_d, str(d_candidates[0]["contentId"]))
    neutral_score = score_of(resp_d, str(d_candidates[1]["contentId"]))
    reject_score = score_of(resp_d, str(d_candidates[2]["contentId"]))
    print(f"  preferred={pref_score}  neutral={neutral_score}  rejected={reject_score}")
    print(f"  ordering preferred > neutral > rejected: {(pref_score or 0) > (neutral_score or 0) > (reject_score or 0)}")

    # ---------------------------------------------------------------------------- NOT_INTERESTED
    _hr("NOT_INTERESTED live suppression demo (separate deterministic user, TENNIS theme)")
    ni_user = stable_uuid("demo2-user", "user-e-not-interested-demo")
    nadal = next(e for e in pool if e["contentId"] == stable_uuid("demo2-candidate-content", "sport-nadal-roland-garros"))
    djokovic = next(e for e in pool if e["contentId"] == stable_uuid("demo2-candidate-content", "sport-djokovic-practice"))
    unrelated = next(e for e in pool if e["category"] == "MUSIC")
    ni_candidates = [nadal, djokovic, unrelated]
    ni_pool_body = [candidate_request_body(e) for e in ni_candidates]
    # Give the demo user a small, EQUAL positive baseline on Nadal/Djokovic (same watch%,
    # same liked flag) so the exact-vs-semantic-neighbor comparison below isn't confounded by
    # an unequal starting point -- only the later NOT_INTERESTED event (on Nadal only) should
    # differ between the two. `category` matches each candidate's REAL category (SPORT for
    # both tennis items, MUSIC for the unrelated one) -- not hardcoded.
    for i, wp in enumerate([88, 88, 85]):
        ev_id = stable_uuid("demo2-event", f"{ni_user}:ni-base-{i}")
        seed_ev = {"eventId": ev_id, "userId": ni_user, "contentId": ni_candidates[i]["contentId"],
                   "creatorId": ni_candidates[i]["creatorId"], "category": ni_candidates[i]["category"],
                   "eventType": "VIDEO_WATCHED", "watchTimeSeconds": wp, "contentDurationSeconds": 100.0,
                   "liked": True, "shared": False, "favorited": False, "commented": False, "creatorFollowed": False,
                   "timestamp": (datetime.now(timezone.utc) - timedelta(days=5 + i)).isoformat()}
        resp = client.post("/api/v1/recommendation-ml-service/events", json=seed_ev)
        assert resp.status_code == 201, resp.text
    body_ni_before = {"userId": ni_user, "limit": 3, "candidates": ni_pool_body}
    _write_json("not-interested-before.json", body_ni_before)
    resp_ni_before = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_ni_before).json()
    nadal_before = score_of(resp_ni_before, nadal["contentId"])
    djokovic_before = score_of(resp_ni_before, djokovic["contentId"])
    unrelated_before = score_of(resp_ni_before, unrelated["contentId"])
    print(f"  BEFORE -- Nadal(exact)={nadal_before}  Djokovic(semantic neighbor, TENNIS)={djokovic_before}  "
          f"{unrelated['title']}(unrelated, MUSIC)={unrelated_before}")

    ni_event = {"eventId": stable_uuid("demo2-event", f"{ni_user}:not-interested-nadal"), "userId": ni_user,
                "contentId": nadal["contentId"], "creatorId": nadal["creatorId"], "category": "SPORT",
                "eventType": "CONTENT_NOT_INTERESTED", "watchTimeSeconds": 4.0, "contentDurationSeconds": 100.0,
                "liked": False, "shared": False, "favorited": False, "commented": False, "creatorFollowed": False,
                "timestamp": datetime.now(timezone.utc).isoformat()}
    _write_json("not-interested-event.json", ni_event)
    resp = client.post("/api/v1/recommendation-ml-service/events", json=ni_event)
    assert resp.status_code == 201, resp.text

    body_ni_after = {"userId": ni_user, "limit": 3, "candidates": ni_pool_body}
    _write_json("not-interested-after.json", body_ni_after)
    resp_ni_after = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_ni_after).json()
    nadal_after = score_of(resp_ni_after, nadal["contentId"])
    djokovic_after = score_of(resp_ni_after, djokovic["contentId"])
    unrelated_after = score_of(resp_ni_after, unrelated["contentId"])
    print(f"  AFTER  -- Nadal(exact)={nadal_after}  Djokovic(semantic neighbor)={djokovic_after}  "
          f"{unrelated['title']}(unrelated)={unrelated_after}")
    nadal_drop = (nadal_before or 0) - (nadal_after or 0)
    djokovic_drop = (djokovic_before or 0) - (djokovic_after or 0)
    unrelated_drop = (unrelated_before or 0) - (unrelated_after or 0)
    print(f"  drop   -- Nadal={nadal_drop:.4f}  Djokovic={djokovic_drop:.4f}  unrelated={unrelated_drop:.4f}")
    print(f"  exact > semantic-neighbor > unrelated suppression: {nadal_drop >= djokovic_drop >= unrelated_drop - 1e-6}")

    # Progressive suppression: two more related NOT_INTERESTED events.
    for i, target in enumerate([djokovic, nadal], start=2):
        ev = {"eventId": stable_uuid("demo2-event", f"{ni_user}:not-interested-{i}"), "userId": ni_user,
              "contentId": target["contentId"], "creatorId": target["creatorId"], "category": "SPORT",
              "eventType": "CONTENT_NOT_INTERESTED", "watchTimeSeconds": 4.0, "contentDurationSeconds": 100.0,
              "liked": False, "shared": False, "favorited": False, "commented": False, "creatorFollowed": False,
              "timestamp": (datetime.now(timezone.utc) + timedelta(seconds=i)).isoformat()}
        _write_json(f"demo-not-interested-{i}.json", ev)
        resp = client.post("/api/v1/recommendation-ml-service/events", json=ev)
        assert resp.status_code == 201, resp.text
    _write_json("demo-not-interested-1.json", ni_event)
    resp_ni_progressive = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_ni_after).json()
    print(f"  after 2 more related NOT_INTERESTED events, Nadal score: {score_of(resp_ni_progressive, nadal['contentId'])} "
          f"(progressive, not manually edited -- normal event ingestion only)")

    # ---------------------------------------------------------------------------------- Search
    _hr("SEARCH-INTENT demo (User A: long-term SPORT + temporary 'music concert' search)")
    body_search_before = recommend_body(a_id, pool)
    _write_json("search-before.json", body_search_before)
    resp_search_before = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_search_before).json()
    dist_before = category_distribution(resp_search_before)
    print(f"  BEFORE (no search): top category={max(dist_before, key=lambda k: dist_before[k])}  distribution={dist_before}")

    search_intent = {"query": "music concert", "topics": ["Live Performance"], "entities": [], "subgenres": [], "confidence": 1.0}
    body_search_active = recommend_body(a_id, pool, search_intent=search_intent)
    _write_json("search-active.json", body_search_active)
    resp_search_active = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_search_active).json()
    dist_active = category_distribution(resp_search_active)
    coldplay_id = stable_uuid("demo2-candidate-content", "music-coldplay-tour")
    print(f"  ACTIVE (searchIntent='music concert'): top category={max(dist_active, key=lambda k: dist_active[k])}  distribution={dist_active}")
    print(f"  Coldplay tour rank BEFORE={rank_of(resp_search_before, coldplay_id)}  ACTIVE={rank_of(resp_search_active, coldplay_id)}")

    body_search_after = recommend_body(a_id, pool)
    _write_json("search-after.json", body_search_after)
    resp_search_after = client.post("/api/v1/recommendation-ml-service/recommendations", json=body_search_after).json()
    dist_after = category_distribution(resp_search_after)
    print(f"  AFTER (search cleared, request omits searchIntent): top category={max(dist_after, key=lambda k: dist_after[k])}  "
          f"distribution={dist_after}")
    print(f"  long-term profile restored (AFTER == BEFORE distribution): {dist_after == dist_before}")

    # ---------------------------------------------------------------------------- seen-content
    _hr("SEEN-CONTENT demo (User A)")
    seen_probe = next(e for e in pool if e["category"] == "TRAVEL")
    seen_event = {"eventId": stable_uuid("demo2-event", f"{a_id}:seen-probe"), "userId": a_id,
                  "contentId": seen_probe["contentId"], "creatorId": seen_probe["creatorId"], "category": "TRAVEL",
                  "eventType": "VIDEO_WATCHED", "watchTimeSeconds": 40.0, "contentDurationSeconds": 100.0,
                  "liked": False, "shared": False, "favorited": False, "commented": False, "creatorFollowed": False,
                  "timestamp": datetime.now(timezone.utc).isoformat()}
    before_seen_resp = client.post("/api/v1/recommendation-ml-service/recommendations", json=recommend_body(a_id, pool)).json()
    score_before_seen = score_of(before_seen_resp, seen_probe["contentId"])
    resp = client.post("/api/v1/recommendation-ml-service/events", json=seen_event)
    assert resp.status_code == 201, resp.text
    after_seen_resp = client.post("/api/v1/recommendation-ml-service/recommendations", json=recommend_body(a_id, pool)).json()
    score_after_seen = score_of(after_seen_resp, seen_probe["contentId"])
    other_travel = next(e for e in pool if e["category"] == "TRAVEL" and e is not seen_probe)
    other_score = score_of(after_seen_resp, other_travel["contentId"])
    print(f"  probe: {seen_probe['title']}  score before watch={score_before_seen}  after watch={score_after_seen}")
    print(f"  demoted after being seen: {(score_after_seen or 0) <= (score_before_seen or 0) + 1e-6}")
    print(f"  unrelated-but-unseen TRAVEL item unaffected/still eligible: {other_score is not None}")

    # ---------------------------------------------------------------------------- cold start
    _hr("OPTIONAL: cold-start demo user (zero history)")
    cold_id = stable_uuid("demo2-user", "user-cold-start-optional")
    resp_cold = client.post("/api/v1/recommendation-ml-service/recommendations", json=recommend_body(cold_id, pool[:10])).json()
    print(f"  userId={cold_id}  strategy={resp_cold['strategy']}  interactionCount={resp_cold['interactionCount']}  "
          f"recommendations={len(resp_cold['recommendations'])}")

    # ---------------------------------------------------------------------------- summary
    _hr("SUMMARY")
    print(f"  modelVersion (constant throughout): {metadata['modelVersion']}")
    print("  MODEL_RETRAINED=false")
    print(f"  RECOMMENDATION_DATA_MODE={os.environ.get('RECOMMENDATION_DATA_MODE', 'LOCAL (default)')}")
    print("\nDONE")


if __name__ == "__main__":
    main()
