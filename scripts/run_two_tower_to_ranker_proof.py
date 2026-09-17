"""Phase 3 Step 10 (optional) -- internal proof only, NOT a production endpoint.

user -> Two-Tower Top-K retrieval -> existing ranker scores them -> existing reranker -> Top-N.

Two separate, independent, purely in-memory models are trained here, both against the same
throwaway offline SQLite dataset this whole PoC uses:
  1. the Two-Tower retrieval model (app.ml.two_tower) -- narrows the full catalog to a
     candidate shortlist per user.
  2. a real RandomForest/LogisticRegression ranker, trained via the EXACT SAME
     app.ml.dataset_builder.build_dataset + app.ml.trainer.train_models the production ranker
     uses -- but never saved via app.ml.model_store, never touching MODEL_PATH/models/ at all,
     so this can never collide with, overwrite, or promote over the real active model
     (recommendation-prod-...). It exists only as a local Python object for this one proof run.

This does NOT modify, retrain, or promote the production model in any way, and does NOT expose
a new HTTP endpoint.
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone

os.environ["DATABASE_URL"] = "sqlite://"

import numpy as np

from app.ml.dataset_builder import FeatureHistory, build_dataset
from app.ml.predictor import probabilities
from app.ml.reranker import explanation, rerank
from app.ml.trainer import train_models
from app.ml.two_tower.dataset import build_examples, chronological_split
from app.ml.two_tower.features import (
    build_content_vector,
    build_user_vector,
    content_vector_dim,
    user_vector_dim,
)
from app.ml.two_tower.retrieval import (
    build_content_index,
    embed_user,
    retrieve_top_k,
)
from app.ml.two_tower.trainer import train_two_tower
from scripts.run_two_tower_poc import (
    _load_offline_dataset,
    _pick_demo_users,
)

TOP_K_RETRIEVAL = 30
TOP_N_FINAL = 10


class _CandidateLike:
    """Minimal stand-in for app.schemas.recommendation_schemas.Candidate -- only the
    attributes app.ml.reranker.rerank()/explanation() actually read."""

    def __init__(self, content):
        self.content_id = content.content_id
        self.creator_id = content.creator_id
        self.category = content.category
        self.content_popularity_score = float(content.popularity_score or 0.5)
        self.content_age_hours = 24.0
        self.creator_followed = False
        self.already_seen = False
        self.title = content.title
        self.hashtags = content.hashtags or []
        self.topics = content.topics or []
        self.entities = content.entities or []
        self.subgenres = content.subgenres or []
        self.language = None
        self.regions = []
        self.candidate_source = None
        self.social_context = None


def main():
    print("=== Phase 3 Step 10 (optional): Two-Tower retrieval -> existing ranker -> reranker ===")
    count = 2000
    rows, content_by_id = _load_offline_dataset(count)
    categories = sorted({(c.category or "").upper() for c in content_by_id.values()})

    print("\n--- Training the Two-Tower retrieval model (offline, isolated) ---")
    tt_examples = build_examples(rows, content_by_id, categories)
    tt_splits = chronological_split(tt_examples, ratios={"train": 0.8, "eval": 0.2})
    tt_result = train_two_tower(
        tt_splits["train"], user_input_dim=user_vector_dim(categories),
        content_input_dim=content_vector_dim(categories), epochs=60,
    )
    all_content_ids = list(content_by_id.keys())
    content_vectors = np.stack([build_content_vector(content_by_id[cid], categories) for cid in all_content_ids])
    tt_index = build_content_index(tt_result.model, all_content_ids, content_vectors)

    print("\n--- Training the EXISTING ranker (RandomForest/LogisticRegression) on the same "
          "offline dataset -- in-memory only, never saved to models/, never touches the "
          "production artifact ---")
    ranker_df = build_dataset(rows, content_by_id)
    ranker_result = train_models(ranker_df)
    ranker_model = ranker_result["model"]
    print(f"Ranker selected: {ranker_result['selectedModel']} eligibleSelection={ranker_result['eligibleSelection']}")

    demo_users = _pick_demo_users(rows, content_by_id, categories)
    user_id = demo_users["B_music_heavy"]
    print(f"\n--- Proof run for demo user B_music_heavy ({user_id}) ---")

    user_rows = sorted([r for r in rows if r.user_id == user_id], key=lambda r: r.timestamp)
    history = FeatureHistory()
    for r in user_rows:
        history.update(r, content=content_by_id.get(r.content_id))
    user_vec = build_user_vector(history, user_id, categories)
    user_embedding = embed_user(tt_result.model, user_vec)
    seen = {r.content_id for r in user_rows}
    retrieved = retrieve_top_k(user_embedding, tt_index, TOP_K_RETRIEVAL, exclude_content_ids=seen)
    retrieved_ids = [cid for cid, _score in retrieved]
    print(f"Two-Tower retrieved {len(retrieved_ids)} candidates "
          f"(categories: {dict(Counter(content_by_id[cid].category for cid in retrieved_ids))})")

    now = datetime.now(timezone.utc)
    scored = []
    for cid in retrieved_ids:
        content = content_by_id[cid]
        candidate = _CandidateLike(content)
        features = history.features(
            user_id=user_id, category=candidate.category, creator_id=candidate.creator_id,
            content_id=candidate.content_id, timestamp=now,
            content_popularity_score=candidate.content_popularity_score,
            content_created_at=now, creator_followed=False, already_seen=False,
            hashtags=candidate.hashtags, topics=candidate.topics, entities=candidate.entities,
            subgenres=candidate.subgenres, title=candidate.title,
        )
        model_score = float(probabilities(ranker_model, [features])[0])
        scored.append({
            "candidate": candidate, "features": features, "model_score": model_score,
            "reason": explanation(features, candidate, cold_start=False),
        })
    scored.sort(key=lambda item: item["model_score"], reverse=True)
    chosen = rerank(scored, TOP_N_FINAL)

    print(f"\nFinal Top-{TOP_N_FINAL} (Two-Tower retrieval -> existing ranker -> existing reranker):")
    for rank, item in enumerate(chosen, 1):
        c = item["candidate"]
        print(f"  {rank:2d}. {c.category:8s} score={item['adjusted_score']:.4f} reason={item['reason']:24s} content_id={c.content_id}")


if __name__ == "__main__":
    main()
