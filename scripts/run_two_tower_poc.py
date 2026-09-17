"""Phase 3 -- Two-Tower retrieval PoC, offline only.

Runs entirely against a throwaway in-memory SQLite database populated by the existing,
deterministic `scripts/generate_synthetic_data.generate()` (same generator the ranker's own
tests/experiments already rely on) -- NOT the live/demo Docker Postgres database, which this
script never connects to and never touches. Safe to (re)run any number of times; it never
writes to the real `models/` directory, the real Postgres database, or any production
artifact -- everything it produces lives only in this process's memory (and, if `--dump-json`
is given, one local diagnostics file).

This does NOT replace, retrain, or promote the existing RandomForest ranking model in any way.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

# Must happen BEFORE any `app.*` import -- mirrors tests/conftest.py's exact pattern so this
# script never touches the real Postgres DATABASE_URL (app.core.config resolves it at import
# time). An isolated, throwaway, in-memory database, discarded when the process exits.
os.environ["DATABASE_URL"] = "sqlite://"

from sqlalchemy import select

from app.db.database import Base, SessionLocal, engine
from app.db.models import Content
from app.db.repositories import interactions
from app.ml.trainer import RANDOM_SEED
from app.ml.two_tower.dataset import build_examples, chronological_split
from app.ml.two_tower.evaluation import evaluate_recall_at_k
from app.ml.two_tower.features import content_vector_dim, user_vector_dim
from app.ml.two_tower.retrieval import (
    build_content_index,
    embed_user,
    retrieve_top_k,
)
from app.ml.two_tower.trainer import train_two_tower
from scripts import generate_synthetic_data

REFERENCE_TIMESTAMP = datetime(2026, 8, 1, tzinfo=timezone.utc)  # fixed -> bit-for-bit reproducible dataset
TOP_K_DEMO = 10


def _load_offline_dataset(count: int):
    Base.metadata.create_all(engine)
    generate_synthetic_data.generate(count=count, reference_timestamp=REFERENCE_TIMESTAMP)
    db = SessionLocal()
    try:
        rows = interactions(db)
        content_by_id = {item.content_id: item for item in db.scalars(select(Content)).all()}
    finally:
        db.close()
    return rows, content_by_id


def _pick_demo_users(rows, content_by_id, categories):
    """Empirically selects 3 real, already-generated personas (never hardcoded rankings):
    the user with the strongest empirical SPORT share, the strongest MUSIC share, and the
    clearest category-shift user (highest share of a category among their OWN first half of
    history that drops the most in their own second half) -- all derived from the actual
    generated data, not asserted in advance."""
    by_user: dict[str, list] = {}
    for row in rows:
        by_user.setdefault(row.user_id, []).append(row)
    for user_rows in by_user.values():
        user_rows.sort(key=lambda r: r.timestamp)

    def category_share(user_rows, category):
        return sum(1 for r in user_rows if r.category == category) / len(user_rows)

    sport_user = max(by_user, key=lambda u: category_share(by_user[u], "SPORT"))
    music_user = max(by_user, key=lambda u: category_share(by_user[u], "MUSIC"))

    def shift_score(user_rows):
        if len(user_rows) < 10:
            return -1.0
        mid = len(user_rows) // 2
        first, second = user_rows[:mid], user_rows[mid:]
        first_top = Counter(r.category for r in first).most_common(1)[0]
        first_cat, first_count = first_top
        first_share = first_count / len(first)
        second_share = sum(1 for r in second if r.category == first_cat) / len(second)
        return first_share - second_share  # large positive = strong early preference that faded

    shift_user = max(by_user, key=lambda u: shift_score(by_user[u]))
    return {"A_sport_heavy": sport_user, "B_music_heavy": music_user, "C_shift": shift_user}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4000, help="Synthetic bulk-interaction count (cohort rows are always included in full).")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--dump-json", default=None, help="Optional: write the final report as JSON to this path.")
    parser.add_argument("--save-artifact", default=None, nargs="?", const="__default__",
                         help="Optional: export the trained model via app.ml.two_tower.artifact_store.save_two_tower "
                              "(Phase 3.2 shadow integration). Bare flag uses the default artifact path; "
                              "a value overrides it. Training/runtime stay separate -- this only happens here, "
                              "never on a request-serving path.")
    args = parser.parse_args()

    print(f"=== Phase 3 Two-Tower PoC (offline, deterministic synthetic data, seed={RANDOM_SEED}) ===")
    rows, content_by_id = _load_offline_dataset(args.count)
    categories = sorted({(c.category or "").upper() for c in content_by_id.values()})
    print(f"Loaded {len(rows)} interactions, {len(content_by_id)} content items, {len(categories)} categories: {categories}")

    examples = build_examples(rows, content_by_id, categories)
    label_counts = Counter(ex.label for ex in examples)
    print(f"Built {len(examples)} point-in-time labeled training pairs (positive={label_counts[1]}, negative={label_counts[0]})")

    splits = chronological_split(examples, ratios={"train": 0.8, "eval": 0.2})
    train_examples, eval_examples = splits["train"], splits["eval"]
    print(f"Chronological split: train={len(train_examples)} eval={len(eval_examples)} "
          f"(train.max_ts < eval.min_ts: "
          f"{max(e.timestamp for e in train_examples) < min(e.timestamp for e in eval_examples)})")

    user_dim = user_vector_dim(categories)
    content_dim = content_vector_dim(categories)
    print(f"User vector dim={user_dim}, Content vector dim={content_dim}, embedding dim=32")

    result = train_two_tower(train_examples, user_input_dim=user_dim, content_input_dim=content_dim, epochs=args.epochs)
    print(f"Training loss: epoch 1={result.epoch_losses[0]:.4f} -> epoch {len(result.epoch_losses)}={result.epoch_losses[-1]:.4f}")

    if args.save_artifact is not None:
        from pathlib import Path

        from app.ml.trainer import RANDOM_SEED as _SEED
        from app.ml.two_tower.artifact_store import save_two_tower
        from app.ml.two_tower.trainer import DEFAULT_EMBEDDING_DIM, DEFAULT_HIDDEN_DIM

        artifact_path = None if args.save_artifact == "__default__" else Path(args.save_artifact)
        saved = save_two_tower(
            result.model, categories=categories, hidden_dim=DEFAULT_HIDDEN_DIM, embedding_dim=DEFAULT_EMBEDDING_DIM,
            trained_at=datetime.now(timezone.utc).isoformat(), random_seed=_SEED, epochs=args.epochs, path=artifact_path,
        )
        from app.ml.two_tower.artifact_store import TWO_TOWER_ARTIFACT_PATH
        print(f"Saved Two-Tower artifact -> {artifact_path or TWO_TOWER_ARTIFACT_PATH} "
              f"(categories={len(saved['categories'])}, userInputDim={saved['userInputDim']}, "
              f"contentInputDim={saved['contentInputDim']})")

    all_content_ids = list(content_by_id.keys())
    import numpy as np

    from app.ml.two_tower.features import build_content_vector
    content_vectors = np.stack([build_content_vector(content_by_id[cid], categories) for cid in all_content_ids])
    index = build_content_index(result.model, all_content_ids, content_vectors)

    metrics = evaluate_recall_at_k(result.model, eval_examples, index, k_values=[5, 10, 20])
    print(f"Retrieval evaluation over {metrics.queries_evaluated} positive eval queries, full catalog of {len(all_content_ids)} items:")
    for k in metrics.k_values:
        print(f"  Recall@{k} = {metrics.recall_at_k[k]:.4f}")

    demo_users = _pick_demo_users(rows, content_by_id, categories)
    print("\n=== Demo retrieval scenarios ===")
    demo_report = {}
    for label, user_id in demo_users.items():
        # Full up-to-date history (not point-in-time-truncated) -- this is "retrieve for this
        # user right now", not a training example.
        from app.ml.dataset_builder import FeatureHistory
        history = FeatureHistory()
        for row in sorted([r for r in rows if r.user_id == user_id], key=lambda r: r.timestamp):
            history.update(row, content=content_by_id.get(row.content_id))
        from app.ml.two_tower.features import build_user_vector
        user_vec = build_user_vector(history, user_id, categories)
        user_embedding = embed_user(result.model, user_vec)
        seen = {row.content_id for row in rows if row.user_id == user_id}
        top_k = retrieve_top_k(user_embedding, index, TOP_K_DEMO, exclude_content_ids=seen)
        cat_counts = Counter(content_by_id[cid].category for cid, _score in top_k)
        print(f"\n{label} ({user_id}): top category in Top-{TOP_K_DEMO} = {cat_counts.most_common(1)}")
        for cid, score in top_k[:5]:
            c = content_by_id[cid]
            print(f"    {c.category:8s} sim={score:.4f} content_id={cid} creator={c.creator_id}")
        demo_report[label] = {
            "user_id": user_id, "top_k": [(cid, round(score, 4)) for cid, score in top_k],
            "category_counts_in_top_k": dict(cat_counts),
        }

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump({
                "recall_at_k": metrics.recall_at_k, "queries_evaluated": metrics.queries_evaluated,
                "train_size": len(train_examples), "eval_size": len(eval_examples),
                "demo_users": demo_report,
            }, f, indent=2)
        print(f"\nwrote {args.dump_json}")


if __name__ == "__main__":
    main()
