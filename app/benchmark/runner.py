"""Benchmark execution.

Seeds a scenario's history into a database as real `Content`/`Interaction` rows (the exact
tables `app.services.providers.user_behavior_provider`'s LOCAL_DB path reads), scores the
scenario's candidates through the REAL, unmodified `app.services.recommendation_service.
recommend()` -- reusing `app.experiments.comparative_scoring.score_with_raw_capture` to
capture the raw, pre-reranking model probability in transit -- and evaluates both the raw
and the final reranked ranking against the scenario's constraints and graded relevance labels.

To benchmark a SPECIFIC algorithm (not just whatever happens to be the currently-active
production model), `use_model()` temporarily repoints the exact module-level attributes
`recommend()` reads (mirroring this project's own established test convention, e.g.
tests/test_step11_rf_candidate_regression.py's `_isolate_model_paths`) to an isolated
artifact, then restores them -- the real active artifact is never touched.
"""
from __future__ import annotations

import contextlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import sessionmaker

from app.benchmark.constraints import classify_reranker_effect, evaluate_constraints
from app.benchmark.metrics import compute_ranking_metrics, diversity_diagnostics
from app.benchmark.scenario_types import BenchmarkScenario, ScenarioResult, StageResult
from app.benchmark.scenarios import (
    all_named_scenarios,
    dominant_category_scenario,
    not_interested_localization_scenarios,
    temporal_shift_scenarios,
)
from app.db.database import Base, make_engine
from app.db.models import Content, Interaction
from app.experiments.comparative_scoring import score_with_raw_capture
from app.ml import model_cache, model_store
from app.ml.algorithm_registry import ALL_ALGORITHM_NAMES, unavailable_algorithms
from app.ml.dataset_builder import FEATURE_DEFINITIONS, FEATURES
from app.ml.feature_builder import TARGET_DEFINITION
from app.ml.trainer import CALIBRATION_METHOD, RANDOM_SEED, train_models
from app.schemas.recommendation_schemas import (
    Candidate,
    RecommendationRequest,
    SocialContext,
)

# Fixed, never wall-clock: every scenario's `HistoryEvent.when` is relative to this anchor,
# so the same scenario seeds byte-identical rows regardless of when the benchmark actually runs.
REFERENCE_TIMESTAMP = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


def seed_scenario(db: Any, scenario: BenchmarkScenario, *, reference_timestamp: datetime = REFERENCE_TIMESTAMP) -> None:
    """Writes `scenario.history` into `db` as real Content + Interaction rows."""
    contents: dict[str, Content] = {}
    rows: list[Interaction] = []
    for event in scenario.history:
        if event.content_id not in contents:
            created_at = reference_timestamp + event.when - timedelta(hours=1)
            contents[event.content_id] = Content(
                content_id=event.content_id, creator_id=event.creator_id, category=event.category,
                content_type="VIDEO", popularity_score=0.5, is_active=True,
                created_at=created_at, updated_at=created_at, title=event.title,
                hashtags_json=json.dumps(event.hashtags) if event.hashtags else None,
                topics_json=json.dumps(event.topics) if event.topics else None,
                entities_json=json.dumps(event.entities) if event.entities else None,
                subgenres_json=json.dumps(event.subgenres) if event.subgenres else None,
            )
        rows.append(Interaction(
            event_id=f"bench-{scenario.scenario_id}-{event.content_id}-{len(rows)}",
            user_id=scenario.user_id, content_id=event.content_id, creator_id=event.creator_id,
            category=event.category, event_type=event.event_type,
            watch_time_seconds=event.watch_percentage, content_duration_seconds=100.0,
            watch_percentage=event.watch_percentage, liked=event.liked, shared=event.shared,
            favorited=event.favorited, creator_followed=event.creator_followed,
            timestamp=reference_timestamp + event.when,
        ))
    db.add_all(list(contents.values()))
    db.flush()
    db.add_all(rows)
    db.commit()


def build_request(scenario: BenchmarkScenario) -> RecommendationRequest:
    candidates = [
        Candidate(
            contentId=c.content_id, creatorId=c.creator_id, category=c.category,
            contentPopularityScore=c.content_popularity_score, contentAgeHours=c.content_age_hours,
            creatorFollowed=c.creator_followed, alreadySeen=c.already_seen,
            title=c.title, hashtags=c.hashtags, topics=c.topics, entities=c.entities, subgenres=c.subgenres,
            candidateSource=c.candidate_source,
            socialContext=SocialContext.model_validate(c.social_context) if c.social_context is not None else None,
            language=None,  # matches the field default (mypy sees aliased optional fields as required -- see Candidate)
            localBucketSource=None,
        )
        for c in scenario.candidates
    ]
    return RecommendationRequest(userId=scenario.user_id, limit=scenario.effective_limit, candidates=candidates,
                                  userProfile=None, searchIntent=None, userContext=None)


def _benchmark_artifact_metadata(model: Any, algorithm_name: str, training_result: dict[str, Any]) -> dict[str, Any]:
    """The minimal valid `app.ml.model_store.REQUIRED_METADATA_FIELDS` set, assembled from a
    real `app.ml.trainer.train_models(..., restrict_algorithm=algorithm_name)` result --
    deliberately NOT a full copy of `app.services.training_service.train()`'s metadata
    construction, since this artifact is never promoted/served for real (see `use_model`)."""
    return {
        "schemaVersion": model_store.SCHEMA_VERSION,
        "modelVersion": f"benchmark-{algorithm_name}",
        "modelType": f"{type(model).__module__}.{type(model).__qualname__}",
        "selectedModel": algorithm_name,
        "featureNames": FEATURES,
        "featureDefinitions": FEATURE_DEFINITIONS,
        "targetDefinition": TARGET_DEFINITION,
        "splitStrategy": training_result["splitLifecycleDescription"],
        "selectionCriterion": "benchmark (algorithm fixed by caller, not selected)",
        "calibration": {"applied": True, "method": CALIBRATION_METHOD},
        "decisionThreshold": training_result["decisionThreshold"],
        "trainingSamples": training_result["splitSizes"]["train"],
        "modelSelectionSamples": training_result["splitSizes"]["modelSelection"],
        "calibrationSamples": training_result["splitSizes"]["calibration"],
        "thresholdTuningSamples": training_result["splitSizes"]["thresholdTuning"],
        "testSamples": training_result["splitSizes"]["test"],
        "classDistribution": training_result["classDistribution"],
        "sklearnVersion": model_store.SKLEARN_VERSION,
        "pythonVersion": model_store.PYTHON_VERSION,
        "randomSeed": RANDOM_SEED,
        "trainingDurationSeconds": 0.0,
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "metrics": training_result["metrics"],
        "modelComparison": training_result["modelComparison"],
        "datasetSource": {"type": "benchmark-training", "synthetic": True},
    }


@contextlib.contextmanager
def use_model(model: Any, algorithm_name: str, training_result: dict[str, Any], tmp_dir: Path):
    """Temporarily makes `model` the active VIDEO model `recommendation_service.recommend()`
    serves. Patches the exact bound module attributes that module and `model_store` already
    read (not `app.core.config`'s, which neither rebinds from at call time) -- the same
    monkeypatch convention this project's tests already use. Restores both, and invalidates
    the cache again, on the way out (even on error)."""
    import app.services.recommendation_service as service

    tmp_dir.mkdir(parents=True, exist_ok=True)
    model_path = tmp_dir / f"{algorithm_name}-model.joblib"
    metadata_path = tmp_dir / f"{algorithm_name}-metadata.json"
    model_store.save(
        model, _benchmark_artifact_metadata(model, algorithm_name, training_result),
        model_path=model_path, metadata_path=metadata_path, model_dir=tmp_dir,
    )
    original_model_path = service.MODEL_PATH
    original_metadata_path = model_store.METADATA_PATH
    service.MODEL_PATH = model_path
    model_store.METADATA_PATH = metadata_path
    model_cache.video_cache.invalidate()
    try:
        yield
    finally:
        service.MODEL_PATH = original_model_path
        model_store.METADATA_PATH = original_metadata_path
        model_cache.video_cache.invalidate()


def run_scenario(db: Any, scenario: BenchmarkScenario, *, algorithm: str) -> ScenarioResult:
    """Seeds `scenario`, scores it through the real service, and evaluates both stages. The
    caller is responsible for having already put the algorithm under test into place (see
    `use_model`) and for providing an isolated `db` (a scenario's history must never bleed
    into another scenario's or another algorithm's run)."""
    seed_scenario(db, scenario)
    request = build_request(scenario)
    response, raw_scores = score_with_raw_capture(db, request)

    relevance_by_id = {c.content_id: c.relevance for c in scenario.candidates}
    category_by_id = {c.content_id: c.category for c in scenario.candidates}
    creator_by_id = {c.content_id: c.creator_id for c in scenario.candidates}

    raw_ranking = sorted(raw_scores, key=lambda content_id: raw_scores[content_id], reverse=True)
    raw_constraint_results = evaluate_constraints(scenario.constraints, raw_scores, stage="raw")
    raw_stage = StageResult(
        ranking=raw_ranking, scores_by_content_id=dict(raw_scores),
        metrics={
            **compute_ranking_metrics([relevance_by_id[cid] for cid in raw_ranking]),
            **diversity_diagnostics([category_by_id[cid] for cid in raw_ranking], [creator_by_id[cid] for cid in raw_ranking]),
        },
        constraint_results=raw_constraint_results,
    )

    final_ranking = [item["contentId"] for item in response["recommendations"]]
    final_scores = {item["contentId"]: item["score"] for item in response["recommendations"]}
    final_constraint_results = evaluate_constraints(scenario.constraints, final_scores, stage="reranked")
    final_stage = StageResult(
        ranking=final_ranking, scores_by_content_id=final_scores,
        metrics={
            **compute_ranking_metrics([relevance_by_id[cid] for cid in final_ranking]),
            **diversity_diagnostics([category_by_id[cid] for cid in final_ranking], [creator_by_id[cid] for cid in final_ranking]),
        },
        constraint_results=final_constraint_results,
    )

    improved, degraded, neutral = classify_reranker_effect(raw_constraint_results, final_constraint_results)
    return ScenarioResult(
        scenario_id=scenario.scenario_id, difficulty=scenario.difficulty, algorithm=algorithm,
        raw=raw_stage, reranked=final_stage,
        reranker_improved=improved, reranker_degraded=degraded, reranker_neutral=neutral,
    )


@contextlib.contextmanager
def isolated_session():
    """A fresh, empty, in-memory database for exactly one scenario run. Two scenarios (or
    two states of the SAME paired scenario, e.g. temporal T0/T1) must never share a database
    -- several benchmark scenarios reuse the same user_id/content_ids on purpose (the paired
    temporal-shift and NOT_INTERESTED-localization scenarios), which would silently
    accumulate history from both halves into one user if they shared a connection."""
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def run_scenario_isolated(scenario: BenchmarkScenario, *, algorithm: str) -> ScenarioResult:
    """`run_scenario` against a fresh, private database -- see `isolated_session`. Requires
    the algorithm under test to already be in place (see `use_model`)."""
    with isolated_session() as db:
        return run_scenario(db, scenario, algorithm=algorithm)


def train_all_algorithms(
    training_df: Any, *, algorithms: list[str] | None = None, random_seed: int = RANDOM_SEED,
) -> dict[str, dict[str, Any]]:
    """Trains every requested algorithm (default: every available candidate in
    app.ml.algorithm_registry) on the SAME dataset/split via the real
    app.ml.trainer.train_models(restrict_algorithm=...) -- one fair, shared experiment, never
    a separately-shuffled dataset per algorithm. Returns {name: train_models() result},
    including the calibrated model itself under result["model"].

    `random_seed` only controls each estimator's own internal randomness (e.g. RandomForest's
    tree bootstrap) -- scripts/generate_synthetic_data.py's SEED is a fixed module constant,
    deliberately not re-parameterized here (that generator is shared, heavily-tuned production
    training data; see its own module docstring)."""
    names = algorithms or [name for name in ALL_ALGORITHM_NAMES if name not in unavailable_algorithms()]
    return {name: train_models(training_df, restrict_algorithm=name, random_seed=random_seed) for name in names}


def run_full_matrix(
    training_results: dict[str, dict[str, Any]], *, tmp_dir: Path,
) -> dict[str, Any]:
    """The complete benchmark matrix for every algorithm in `training_results`: EASY/MEDIUM/
    HARD/ADVERSARIAL scenario results, the dominant-category diversity scenario, and the
    temporal-shift / NOT_INTERESTED-localization paired comparisons.
    """
    named_scenarios = all_named_scenarios()
    dominant_scenario = dominant_category_scenario()
    temporal_t0, temporal_t1 = temporal_shift_scenarios()
    localization_a, localization_b = not_interested_localization_scenarios()

    matrix: dict[str, Any] = {}
    for algorithm, training_result in training_results.items():
        model = training_result["model"]
        with use_model(model, algorithm, training_result, tmp_dir / algorithm):
            scenario_results = [
                run_scenario_isolated(scenario, algorithm=algorithm) for scenario in named_scenarios
            ]
            dominant_result = run_scenario_isolated(dominant_scenario, algorithm=algorithm)
            t0_result = run_scenario_isolated(temporal_t0, algorithm=algorithm)
            t1_result = run_scenario_isolated(temporal_t1, algorithm=algorithm)
            localization_a_result = run_scenario_isolated(localization_a, algorithm=algorithm)
            localization_b_result = run_scenario_isolated(localization_b, algorithm=algorithm)

        matrix[algorithm] = {
            "scenarios": {result.scenario_id: result for result in scenario_results},
            "dominantCategory": dominant_result,
            "temporal": {"t0": t0_result, "t1": t1_result},
            "localization": {"sameSubtheme": localization_a_result, "diverseSubthemes": localization_b_result},
        }
    return matrix
