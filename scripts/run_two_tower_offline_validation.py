"""Phase 3.3 -- fast end-to-end offline validation + guarded-integration readiness check.

Read-only w.r.t. any real data source: this checkout has no real/anonymized interaction data,
no persisted local Postgres data for this project's schema (checked directly against the local
Postgres server -- only unrelated databases exist there: app_db/2af/rockempire_db, not
recommendation_ml), and no production-like fixture/log files beyond hand-authored Postman demo
bodies in docs/ (not representative interaction history). The best available source is
therefore the SAME deterministic synthetic generator (scripts.generate_synthetic_data) already
used and relied on throughout this project's own test suite and Phase 3.1/3.2 -- never
fabricated ad hoc for this report. Everything below runs against a throwaway in-memory SQLite
database (never the real repository DB) and, for the RandomForest side, an isolated temp
model-artifact directory (never the real repository models/ directory).
"""
from __future__ import annotations

import bisect
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from time import perf_counter

os.environ["DATABASE_URL"] = "sqlite://"
os.environ.setdefault("MODEL_ARTIFACT_ROOT", tempfile.mkdtemp(prefix="two_tower_phase33_rf_"))

import numpy as np

from app.db.database import Base, make_engine
from app.db.models import Content, Interaction
from app.ml.dataset_builder import FeatureHistory
from app.ml.predictor import probabilities
from app.ml.reranker import explanation, rerank
from app.ml.trainer import RANDOM_SEED
from app.ml.two_tower.artifact_store import save_two_tower
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
from app.ml.two_tower.trainer import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    train_two_tower,
)
from app.services.two_tower_shadow_service import (
    content_to_candidate,
    retrieve_and_rank_with_two_tower,
)
from scripts.run_two_tower_poc import _load_offline_dataset
from scripts.run_two_tower_shadow_comparison import (
    _train_isolated_demo_ranker,
)

COUNT = 4000  # matches Phase 3.1's own validated setup, for directly comparable numbers
MAX_K = 20
SHADOW_SAMPLE_SIZE = 20
SHADOW_RETRIEVAL_K = 30
SHADOW_TOP_N = 10


def _clone_row(row, model_cls):
    fields = {c.name: getattr(row, c.name) for c in row.__table__.columns if c.name != "id"}
    return model_cls(**fields)


def build_pointintime_shadow_db(content_by_id, user_rows_before_cutoff):
    """Fresh, isolated, throwaway in-memory DB containing the full content catalog plus ONLY
    one user's interactions strictly BEFORE their held-out event -- lets the EXISTING Phase 3.2
    shadow service (retrieve_and_rank_with_two_tower) be reused, unmodified, for a genuine
    point-in-time check without ever touching the real DB or any shared evaluation state."""
    from sqlalchemy.orm import sessionmaker

    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = Session()
    for content in content_by_id.values():
        db.add(_clone_row(content, Content))
    for row in user_rows_before_cutoff:
        db.add(_clone_row(row, Interaction))
    db.commit()
    return db


def measure_latency(tt_model, rf_model, content_by_id, categories, rows, sample_user_id):
    user_rows = [r for r in rows if r.user_id == sample_user_id]
    history = FeatureHistory()
    for r in sorted(user_rows, key=lambda r: r.timestamp):
        history.update(r, content=content_by_id.get(r.content_id))

    t0 = perf_counter()
    user_vec = build_user_vector(history, sample_user_id, categories)
    user_embedding = embed_user(tt_model, user_vec)
    t1 = perf_counter()

    all_content_ids = list(content_by_id.keys())
    content_vectors = np.stack([build_content_vector(content_by_id[cid], categories) for cid in all_content_ids])
    index = build_content_index(tt_model, all_content_ids, content_vectors)
    seen = {r.content_id for r in user_rows}
    retrieved = retrieve_top_k(user_embedding, index, SHADOW_RETRIEVAL_K, exclude_content_ids=seen)
    t2 = perf_counter()

    now = datetime.now(timezone.utc)
    candidates = [c for cid, _s in retrieved if (c := content_to_candidate(content_by_id[cid], now)) is not None]
    feature_rows = [
        history.features(
            user_id=sample_user_id, category=c.category, creator_id=c.creator_id, content_id=c.content_id,
            timestamp=now, content_popularity_score=c.content_popularity_score,
            content_created_at=now - timedelta(hours=c.content_age_hours),
            creator_followed=c.creator_followed, already_seen=c.already_seen,
            hashtags=c.hashtags, topics=c.topics, entities=c.entities, subgenres=c.subgenres, title=c.title,
        )
        for c in candidates
    ]
    scores = probabilities(rf_model, feature_rows)
    t3 = perf_counter()

    scored = [
        {"candidate": c, "features": f, "model_score": float(s), "reason": explanation(f, c, False)}
        for c, f, s in zip(candidates, feature_rows, scores)
    ]
    scored.sort(key=lambda item: item["model_score"], reverse=True)
    rerank(scored, SHADOW_TOP_N, cold_start=False)
    t4 = perf_counter()

    return {
        "embed_ms": (t1 - t0) * 1000, "retrieve_ms": (t2 - t1) * 1000,
        "rf_score_ms": (t3 - t2) * 1000, "rerank_ms": (t4 - t3) * 1000, "total_ms": (t4 - t0) * 1000,
        "catalog_size": len(all_content_ids), "candidates_scored": len(candidates),
    }


def main():
    print("=== Phase 3.3: fast offline validation + guarded-integration readiness ===")
    print(f"Data source: deterministic synthetic (scripts.generate_synthetic_data), count={COUNT} "
          f"-- see module docstring for why real/Postgres/fixture sources were checked and are unavailable.")

    rows, content_by_id = _load_offline_dataset(COUNT)
    categories = sorted({(c.category or "").upper() for c in content_by_id.values()})
    distinct_users = {r.user_id for r in rows}
    print(f"\ndataset: interactions={len(rows)} contents={len(content_by_id)} "
          f"distinct_users={len(distinct_users)} categories={len(categories)}")

    # ---------------------------------------------------------------- chronological evaluation
    examples = build_examples(rows, content_by_id, categories)
    splits = chronological_split(examples, ratios={"train": 0.8, "eval": 0.2})
    train_examples, eval_examples = splits["train"], splits["eval"]
    positives = [ex for ex in eval_examples if ex.label == 1]
    leak_free = max(e.timestamp for e in train_examples) < min(e.timestamp for e in eval_examples)
    print(f"\nchronological split: train={len(train_examples)} eval={len(eval_examples)} "
          f"positive_eval_queries={len(positives)} train.max_ts<eval.min_ts={leak_free} "
          f"(per-example user_vector is built from history strictly BEFORE that example's own "
          f"event -- see app.ml.two_tower.dataset.build_examples -- so no future leakage either "
          f"within a split or across the train/eval cutoff)")

    user_dim = user_vector_dim(categories)
    content_dim = content_vector_dim(categories)
    t_train0 = perf_counter()
    result = train_two_tower(train_examples, user_input_dim=user_dim, content_input_dim=content_dim, epochs=DEFAULT_EPOCHS)
    t_train1 = perf_counter()
    print(f"trained Two-Tower (EXP4 config, unchanged): epochs={DEFAULT_EPOCHS} "
          f"loss {result.epoch_losses[0]:.4f} -> {result.epoch_losses[-1]:.4f} "
          f"(offline training time: {t_train1 - t_train0:.1f}s, not part of the latency measurement below)")

    all_content_ids = list(content_by_id.keys())
    content_vectors = np.stack([build_content_vector(content_by_id[cid], categories) for cid in all_content_ids])
    index = build_content_index(result.model, all_content_ids, content_vectors)

    # -------------------------------------------------------------------- retrieval metrics
    user_row_ts = defaultdict(list)
    for r in sorted(rows, key=lambda r: r.timestamp):
        user_row_ts[r.user_id].append(r.timestamp)

    hits = {5: 0, 10: 0, 20: 0}
    reciprocal_ranks = []
    ndcg_sum = 0.0
    retrieved_counter = Counter()
    per_query_prior_count = []
    per_query_rank = []
    for ex in positives:
        user_embedding = embed_user(result.model, ex.user_vector)
        ranked = retrieve_top_k(user_embedding, index, MAX_K)
        ranked_ids = [cid for cid, _s in ranked]
        retrieved_counter.update(ranked_ids)
        rank = ranked_ids.index(ex.content_id) + 1 if ex.content_id in ranked_ids else None
        for k in (5, 10, 20):
            if rank is not None and rank <= k:
                hits[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if rank is not None and rank <= 10:
            ndcg_sum += 1.0 / np.log2(rank + 1)
        prior_count = bisect.bisect_left(user_row_ts[ex.user_id], ex.timestamp)
        per_query_prior_count.append(prior_count)
        per_query_rank.append(rank)

    n = len(positives)
    recall = {k: hits[k] / n for k in (5, 10, 20)}
    mrr = float(np.mean(reciprocal_ranks))
    ndcg10 = ndcg_sum / n
    print(f"\nRETRIEVAL METRICS (n={n} positive eval queries, full catalog of {len(all_content_ids)} items):")
    print(f"  Recall@5={recall[5]:.4f}  Recall@10={recall[10]:.4f}  Recall@20={recall[20]:.4f}")
    print(f"  MRR={mrr:.4f}  NDCG@10={ndcg10:.4f}")

    unique_retrieved = len(retrieved_counter)
    coverage = unique_retrieved / len(all_content_ids)
    print(f"\nCOVERAGE: {unique_retrieved}/{len(all_content_ids)} unique content items ever appeared "
          f"in a Top-{MAX_K} across all {n} queries (coverage={coverage:.3f})")
    print("  top repeatedly retrieved items (content_id: appearances / n queries, category):")
    for cid, cnt in retrieved_counter.most_common(7):
        cat = content_by_id[cid].category
        print(f"    {cid}: {cnt}/{n} ({cnt/n:.1%})  category={cat}")
    attractor_like = [cid for cid, cnt in retrieved_counter.items() if cnt / n > 0.5]
    print(f"  universal-attractor check (>50% of ALL queries' Top-{MAX_K}): "
          f"{len(attractor_like)} item(s) flagged" + (f" -> {attractor_like}" if attractor_like else " -> none"))

    # -------------------------------------------------------------------- cohort breakdown
    counts_sorted = sorted(per_query_prior_count)
    p33 = counts_sorted[len(counts_sorted) // 3] if counts_sorted else 0
    p66 = counts_sorted[2 * len(counts_sorted) // 3] if counts_sorted else 0
    print(f"\nprior-interaction-count distribution at query time: p33={p33} p66={p66} "
          f"(cohort thresholds derived from this run's own data, not hardcoded)")

    def cohort_of(count):
        if count <= p33:
            return "sparse"
        if count <= p66:
            return "medium"
        return "high"

    cohort_stats = defaultdict(lambda: {"n": 0, "hit10": 0, "hit20": 0})
    for prior_count, rank in zip(per_query_prior_count, per_query_rank):
        c = cohort_stats[cohort_of(prior_count)]
        c["n"] += 1
        if rank is not None and rank <= 10:
            c["hit10"] += 1
        if rank is not None and rank <= 20:
            c["hit20"] += 1
    print("USER-COHORT QUALITY:")
    for cohort in ("sparse", "medium", "high"):
        c = cohort_stats[cohort]
        if c["n"]:
            print(f"  {cohort:6s}: n={c['n']:4d}  Recall@10={c['hit10']/c['n']:.4f}  Recall@20={c['hit20']/c['n']:.4f}")
        else:
            print(f"  {cohort:6s}: n=0 (no queries in this bucket)")

    # -------------------------------------------------------------------- shadow pipeline
    from pathlib import Path

    tmp_dir = tempfile.mkdtemp(prefix="two_tower_phase33_artifact_")
    two_tower_artifact_path = Path(tmp_dir) / "two_tower_model.pt"
    save_two_tower(
        result.model, categories=categories, hidden_dim=DEFAULT_HIDDEN_DIM, embedding_dim=DEFAULT_EMBEDDING_DIM,
        trained_at=datetime.now(timezone.utc).isoformat(), random_seed=RANDOM_SEED, epochs=DEFAULT_EPOCHS,
        path=two_tower_artifact_path,
    )

    from app.db.database import SessionLocal
    from app.ml import model_store

    eval_db = SessionLocal()
    rf_metadata = _train_isolated_demo_ranker(eval_db)
    eval_db.close()
    rf_model = model_store.load_model(model_path=model_store.MODEL_PATH)
    print(f"\nisolated RandomForest artifact for this run: modelVersion={rf_metadata['modelVersion']} "
          f"selectedModel={rf_metadata['selectedModel']}")

    sample = positives[:SHADOW_SAMPLE_SIZE]
    retrieved_hits = 0
    retained_hits = 0
    final_ranks = []
    rows_by_user = defaultdict(list)
    for r in rows:
        rows_by_user[r.user_id].append(r)

    for ex in sample:
        before_cutoff_rows = [r for r in rows_by_user[ex.user_id] if r.timestamp < ex.timestamp]

        # Retrieval-only check (pre-ranker): pure in-memory Two-Tower retrieval against the
        # SAME point-in-time history, using the content index already built above -- no DB
        # needed for this half of the check.
        history = FeatureHistory()
        for r in sorted(before_cutoff_rows, key=lambda r: r.timestamp):
            history.update(r, content=content_by_id.get(r.content_id))
        user_vec = build_user_vector(history, ex.user_id, categories)
        user_embedding = embed_user(result.model, user_vec)
        seen = {r.content_id for r in before_cutoff_rows}
        retrieval_only = retrieve_top_k(user_embedding, index, SHADOW_RETRIEVAL_K, exclude_content_ids=seen)
        was_retrieved = ex.content_id in [cid for cid, _s in retrieval_only]
        if was_retrieved:
            retrieved_hits += 1

        # Full shadow pipeline (retrieve -> EXISTING ranker -> EXISTING reranker), via a fresh,
        # isolated point-in-time DB so the existing, unmodified Phase 3.2 service can be reused.
        shadow_db = build_pointintime_shadow_db(content_by_id, before_cutoff_rows)
        try:
            shadow_result = retrieve_and_rank_with_two_tower(
                shadow_db, ex.user_id, limit=SHADOW_TOP_N, retrieval_k=SHADOW_RETRIEVAL_K,
                artifact_path=two_tower_artifact_path,
            )
        finally:
            shadow_db.close()
        final_ids = [rec["contentId"] for rec in shadow_result["recommendations"]]
        if ex.content_id in final_ids:
            retained_hits += 1
            final_ranks.append(final_ids.index(ex.content_id) + 1)

    lost_by_ranking = (retrieved_hits - retained_hits) / retrieved_hits if retrieved_hits else float("nan")
    print(f"\nSHADOW PIPELINE (Two-Tower retrieve -> existing RandomForest -> existing reranker), "
          f"sample of {len(sample)} held-out positive queries, retrieval_k={SHADOW_RETRIEVAL_K}, top_n={SHADOW_TOP_N}:")
    print(f"  held-out positive retrieved in Top-{SHADOW_RETRIEVAL_K} (pre-ranker): {retrieved_hits}/{len(sample)} "
          f"({retrieved_hits/len(sample):.1%})")
    print(f"  held-out positive retained in final Top-{SHADOW_TOP_N} (post-ranker/reranker): {retained_hits}/{len(sample)} "
          f"({retained_hits/len(sample):.1%})")
    print(f"  of items retrieved pre-ranker, lost by downstream ranking/reranking: {lost_by_ranking:.1%}")
    if final_ranks:
        print(f"  final rank when retained: mean={np.mean(final_ranks):.2f} "
              f"median={np.median(final_ranks):.1f} best={min(final_ranks)} worst={max(final_ranks)}")

    # -------------------------------------------------------------------- latency
    sample_user = sample[0].user_id if sample else next(iter(distinct_users))
    latency = measure_latency(result.model, rf_model, content_by_id, categories, rows, sample_user)
    print(f"\nLATENCY (single user, catalog={latency['catalog_size']} items, "
          f"{latency['candidates_scored']} candidates scored, CPU, no optimization):")
    print(f"  user vector + embed:  {latency['embed_ms']:.1f} ms")
    print(f"  retrieval (Top-{SHADOW_RETRIEVAL_K}):     {latency['retrieve_ms']:.1f} ms")
    print(f"  RandomForest scoring: {latency['rf_score_ms']:.1f} ms")
    print(f"  reranking:             {latency['rerank_ms']:.1f} ms")
    print(f"  TOTAL:                 {latency['total_ms']:.1f} ms")


if __name__ == "__main__":
    main()
