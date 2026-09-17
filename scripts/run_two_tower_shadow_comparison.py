"""Phase 3.2 -- shadow comparison: current candidate/ranking path vs. Two-Tower retrieval ->
the SAME existing ranker/reranker, for the SAME users. This is pipeline verification, not a
claim that Two-Tower is "better" -- three demo personas are not a benchmark (Step 6).

Offline only, throwaway in-memory SQLite (same convention as scripts/run_two_tower_poc.py).
Trains its own Two-Tower artifact for this run (the offline step) and saves it via
app.ml.two_tower.artifact_store, then loads it back for the shadow retrieval call -- runtime
retrieval never trains.

RandomForest artifact: this dev checkout has no production model artifact on disk at all (no
models/recommendation_model.joblib), so there is nothing to load for Path A/B to share.
MODEL_ARTIFACT_ROOT is pointed at an isolated temp directory (set BEFORE any `app.*` import,
mirroring this file's own DATABASE_URL override), and a model is fit with the SAME
`app.ml.trainer.train_models()` production training code (unmodified), then persisted with
the SAME `app.ml.model_store.save()` production persistence code (unmodified) to that isolated
path only. Deliberately does NOT call `app.services.training_service.train()` (that function's
"promote a new active model" orchestration and mandatory eligibility gating is production
model-promotion policy, out of scope here and explicitly disallowed by this task) -- so nothing
here is a "retrain the production RandomForest" or "promote a new production ranker" action in
either the literal (real models/ directory) or policy (promotion flow) sense. The saved
metadata's modelVersion is prefixed "two-tower-shadow-demo-", never "recommendation-prod-",
so it can never be confused with a real promoted artifact. This exists solely so both paths
below can go through the SAME app.ml.model_cache/model_store LOADING code production uses,
proving the pipeline mechanics rather than skipping that step. Never calls the public HTTP
endpoint -- it calls app.services.recommendation_service.recommend() directly, in-process,
exactly as the HTTP layer would, so Path A below is genuinely "today's ranking/reranking code,"
not a reimplementation, just scored against this run's local demo artifact instead of a real
production one (which does not exist in this checkout).
"""
from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ["DATABASE_URL"] = "sqlite://"
os.environ.setdefault("MODEL_ARTIFACT_ROOT", tempfile.mkdtemp(prefix="two_tower_shadow_rf_"))

from sqlalchemy import select

from app.core.config import RECOMMENDATION_MAX_CANDIDATES
from app.db.database import SessionLocal
from app.db.models import Content
from app.db.repositories import interactions
from app.ml import model_store
from app.ml.dataset_builder import (
    FEATURE_DEFINITIONS,
    FEATURES,
    build_dataset,
)
from app.ml.feature_builder import TARGET_DEFINITION
from app.ml.trainer import CALIBRATION_METHOD, RANDOM_SEED, train_models
from app.ml.two_tower.artifact_store import save_two_tower
from app.ml.two_tower.dataset import build_examples, chronological_split
from app.ml.two_tower.features import content_vector_dim, user_vector_dim
from app.ml.two_tower.trainer import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HIDDEN_DIM,
    train_two_tower,
)
from app.schemas.recommendation_schemas import RecommendationRequest
from app.services.recommendation_service import recommend
from app.services.two_tower_shadow_service import (
    content_to_candidate,
    retrieve_and_rank_with_two_tower,
)
from scripts.run_two_tower_poc import (
    _load_offline_dataset,
    _pick_demo_users,
)

TOP_N = 10


def _train_isolated_demo_ranker(db) -> dict:
    """Fits + persists a RandomForest/LogisticRegression artifact via the SAME
    app.ml.trainer.train_models()/app.ml.model_store.save() production code, to the isolated
    MODEL_ARTIFACT_ROOT set above only -- see module docstring for why this deliberately does
    NOT call app.services.training_service.train() (production promotion policy, out of scope
    and disallowed here). modelVersion is prefixed "two-tower-shadow-demo-", never
    "recommendation-prod-"."""
    from time import perf_counter

    started = perf_counter()
    rows = interactions(db)
    content_by_id = {item.content_id: item for item in db.scalars(select(Content)).all()}
    df = build_dataset(rows, content_by_id)
    result = train_models(df)  # same production training code; no eligibility-gated promotion
    model = result.pop("model")
    now = datetime.now(timezone.utc)
    metadata = {
        "modelVersion": f"two-tower-shadow-demo-{now:%Y%m%d%H%M%S}",
        "modelType": f"{type(model).__module__}.{type(model).__qualname__}",
        "selectedModel": result["selectedModel"],
        "featureNames": FEATURES,
        "featureDefinitions": FEATURE_DEFINITIONS,
        "targetDefinition": TARGET_DEFINITION,
        "splitStrategy": result["splitLifecycleDescription"],
        "selectionCriterion": result["selectionCriterion"],
        "calibration": {"applied": True, "method": CALIBRATION_METHOD},
        "decisionThreshold": result["decisionThreshold"],
        "trainingSamples": result["splitSizes"]["train"],
        "modelSelectionSamples": result["splitSizes"]["modelSelection"],
        "calibrationSamples": result["splitSizes"]["calibration"],
        "thresholdTuningSamples": result["splitSizes"]["thresholdTuning"],
        "testSamples": result["splitSizes"]["test"],
        "classDistribution": result["classDistribution"],
        "sklearnVersion": model_store.SKLEARN_VERSION,
        "pythonVersion": model_store.PYTHON_VERSION,
        "randomSeed": RANDOM_SEED,
        "trainingDurationSeconds": round(perf_counter() - started, 3),
        "trainedAt": now.isoformat(),
        "metrics": result["metrics"],
        "modelComparison": result["modelComparison"],
        "datasetSource": {
            "type": "database", "synthetic": True, "syntheticRowCount": len(rows), "totalRowCount": len(rows),
            "note": "Phase 3.2 shadow-comparison demo artifact -- synthetic data only, isolated "
                    "temp path, never promoted, never touches the real production artifact.",
        },
        "eligibleSelection": result["eligibleSelection"],  # informational only; not gated on here
    }
    return model_store.save(model, metadata, model_path=model_store.MODEL_PATH, metadata_path=model_store.METADATA_PATH,
                             model_dir=model_store.MODEL_DIR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--artifact-path", default=None,
                         help="Where to save/load the Two-Tower artifact for this run "
                              "(defaults to app.ml.two_tower.artifact_store's default path).")
    args = parser.parse_args()
    artifact_path = Path(args.artifact_path) if args.artifact_path else None

    print("=== Phase 3.2 shadow comparison: current path vs. Two-Tower -> existing ranker/reranker ===")
    print(f"(isolated RandomForest artifact dir for this run: {os.environ['MODEL_ARTIFACT_ROOT']} -- "
          f"never the real repository models/ directory)")
    rows, content_by_id = _load_offline_dataset(args.count)
    categories = sorted({(c.category or "").upper() for c in content_by_id.values()})

    # --- OFFLINE training step (Step 4: training/runtime are strictly separate) ---
    examples = build_examples(rows, content_by_id, categories)
    splits = chronological_split(examples, ratios={"train": 0.8, "eval": 0.2})
    result = train_two_tower(
        splits["train"], user_input_dim=user_vector_dim(categories),
        content_input_dim=content_vector_dim(categories), epochs=args.epochs,
    )
    saved = save_two_tower(
        result.model, categories=categories, hidden_dim=DEFAULT_HIDDEN_DIM, embedding_dim=DEFAULT_EMBEDDING_DIM,
        trained_at=datetime.now(timezone.utc).isoformat(), random_seed=RANDOM_SEED, epochs=args.epochs,
        path=artifact_path,
    )
    print(f"Trained + saved Two-Tower artifact for this run (epochs={args.epochs}, "
          f"userInputDim={saved['userInputDim']}, contentInputDim={saved['contentInputDim']}).")

    demo_users = _pick_demo_users(rows, content_by_id, categories)
    db = SessionLocal()
    try:
        # Isolated RandomForest/LogisticRegression artifact for THIS run only (see module
        # docstring) -- production train_models()/model_store.save() code, called unmodified,
        # writing only to the temp MODEL_ARTIFACT_ROOT set above. Never promotion, never real.
        rf_metadata = _train_isolated_demo_ranker(db)
        print(f"Trained isolated RandomForest artifact for this run: modelVersion={rf_metadata['modelVersion']} "
              f"selectedModel={rf_metadata['selectedModel']}")

        now = datetime.now(timezone.utc)
        full_catalog = db.scalars(select(Content).where(Content.is_active.is_(True))).all()
        # RecommendationRequest.candidates is bounded (RECOMMENDATION_MAX_CANDIDATES) -- the
        # same real-world constraint a Candidate-Service-fed request would hit; capped
        # deterministically (sorted by content_id) rather than silently truncated arbitrarily.
        all_candidates = sorted(
            (c for c in (content_to_candidate(item, now) for item in full_catalog) if c is not None),
            key=lambda c: c.content_id,
        )[:RECOMMENDATION_MAX_CANDIDATES]

        for label, user_id in demo_users.items():
            print(f"\n--- {label} ({user_id}) ---")

            # PATH A: current production candidate/ranking path -- recommend(), called exactly
            # as the HTTP layer calls it, UNMODIFIED. This is "today's behavior," not a
            # reimplementation, and does not involve Two-Tower at all.
            request = RecommendationRequest(userId=user_id, limit=TOP_N, candidates=all_candidates)
            current_result = recommend(db, request)
            current_cats = Counter(r["category"] for r in current_result["recommendations"])
            print(f"A. current path   Top-{TOP_N} categories: {dict(current_cats)}  modelVersion={current_result['modelVersion']}")
            for r in current_result["recommendations"][:5]:
                print(f"    {r['rank']:2d}. {r['category']:8s} score={r['score']:.4f} reason={r['reason']:22s} content_id={r['contentId']}")

            # PATH B: Two-Tower retrieval -> the SAME existing ranker/reranker.
            shadow_result = retrieve_and_rank_with_two_tower(db, user_id, limit=TOP_N, artifact_path=artifact_path)
            shadow_cats = Counter(r["category"] for r in shadow_result["recommendations"])
            print(f"B. Two-Tower path Top-{TOP_N} categories: {dict(shadow_cats)}  "
                  f"retrieved={shadow_result['twoTowerRetrievedCount']}  modelVersion={shadow_result['modelVersion']}")
            for r in shadow_result["recommendations"][:5]:
                print(f"    {r['rank']:2d}. {r['category']:8s} score={r['score']:.4f} reason={r['reason']:22s} content_id={r['contentId']}")

            assert current_result["modelVersion"] == shadow_result["modelVersion"], (
                "both paths must score with the SAME active RandomForest modelVersion"
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
