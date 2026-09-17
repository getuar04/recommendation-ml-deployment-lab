"""LIVE recommendation model training.

Mirrors the VIDEO trainer's split-lifecycle/selection/calibration/threshold/baseline/
artifact approach exactly (see `app.ml.trainer`, `app.ml.split_lifecycle`), but LIVE and
VIDEO targets and feature sets stay strictly separate (`app.ml.live_feature_builder` vs.
`app.ml.dataset_builder`).

There is currently no real LIVE behavior log, so `train_live_model()` defaults to a
deterministic synthetic dataset (`app.ml.live_synthetic_data`). The trainer also accepts
an injected DataFrame -- e.g. a future repository-backed dataset -- and always records in
metadata (`datasetSource`) whether the data used was synthetic, so synthetic metrics can
never be mistaken for production evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core.config import LIVE_METADATA_PATH, LIVE_MODEL_PATH, MODEL_DIR
from app.ml import artifact_lifecycle, model_cache, model_store
from app.ml.baselines import evaluate_baselines
from app.ml.drift_baseline import build_baseline
from app.ml.evaluator import evaluate, select_threshold
from app.ml.live_feature_builder import (
    LIVE_CATEGORICAL,
    LIVE_FEATURE_DEFINITIONS,
    LIVE_FEATURES,
    LIVE_NUMERIC,
)
from app.ml.live_synthetic_data import (
    DEFAULT_SEED,
    DEFAULT_SIZE,
    generate_synthetic_live_dataset,
)
from app.ml.pipeline_builder import build_classifier_pipeline
from app.ml.split_lifecycle import (
    CANONICAL_SPLIT_NAMES,
    InsufficientLifecycleDataError,
    resolve_split_lifecycle,
)
from app.ml.training_utils import class_distribution, permutation_importance_report

CALIBRATION_METHOD = "sigmoid"
RANDOM_SEED = 42
SELECTION_CRITERION = "prAuc"
THRESHOLD_OBJECTIVE = "f1"
LIVE_TARGET_DEFINITION = (
    "positive (1): joined and watched >=60s, or like/share/comment/gift/follow; "
    "negative (0): impression without join, or joined and left before 10s without a positive action; "
    "neutral (excluded from training): everything else."
)
VIEWER_RANK_COLUMN = "_viewer_count_rank"
LIVE_BASELINE_SCORE_COLUMNS = {"categoryAffinityOnly": "live_category_affinity", "viewerCountRankOnly": VIEWER_RANK_COLUMN}


class InsufficientLiveData(Exception):
    pass


def _build_pipeline(model: Any, *, scale_numeric: bool) -> Pipeline:
    return build_classifier_pipeline(model, categorical=LIVE_CATEGORICAL, numeric=LIVE_NUMERIC, scale_numeric=scale_numeric)


def _candidate_models(random_seed: int) -> dict[str, Pipeline]:
    return {
        "LogisticRegression": _build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed), scale_numeric=True,
        ),
        "RandomForestClassifier": _build_pipeline(
            RandomForestClassifier(n_estimators=80, class_weight="balanced", random_state=random_seed, n_jobs=1),
            scale_numeric=False,
        ),
    }


def _with_viewer_rank(df: pd.DataFrame) -> pd.DataFrame:
    if "current_viewer_count" not in df.columns:
        return df
    df = df.copy()
    df[VIEWER_RANK_COLUMN] = df["current_viewer_count"].rank(pct=True)
    return df


def train_live_model(
    dataset: pd.DataFrame | None = None,
    *,
    random_seed: int = RANDOM_SEED,
    dataset_provenance: Literal["synthetic", "injected", "real"] | None = None,
) -> dict[str, Any]:
    """Train the LIVE model with explicit data provenance.

    An internally generated dataset is always synthetic. A caller-provided DataFrame remains
    "injected" by default for backward compatibility, but a trusted caller may explicitly pass
    ``dataset_provenance="synthetic"`` (generated through ``app.ml.live_synthetic_data``) or
    ``dataset_provenance="real"`` (built from real, session-reconstructed LIVE interactions via
    ``app.ml.live_dataset_builder.build_live_dataset`` -- see
    ``app.services.live_training_service.train_live``, the only production caller of the real
    path). Provenance is never guessed from data values.
    """
    started = perf_counter()
    generated_internally = dataset is None
    if generated_internally and dataset_provenance in ("injected", "real"):
        raise ValueError(f"An internally generated LIVE dataset cannot be marked as {dataset_provenance}.")
    provenance = dataset_provenance or ("synthetic" if generated_internally else "injected")
    is_synthetic = provenance == "synthetic"
    if dataset is None:
        dataset = generate_synthetic_live_dataset(DEFAULT_SIZE, seed=DEFAULT_SEED)
    if len(dataset) < 100 or dataset.target.nunique() < 2:
        raise InsufficientLiveData(
            f"At least 100 labeled LIVE interactions containing positive and negative samples "
            f"are required; found {len(dataset)} labeled {provenance} rows."
        )

    try:
        lifecycle = resolve_split_lifecycle(dataset)
    except InsufficientLifecycleDataError as exc:
        raise InsufficientLiveData(str(exc)) from exc
    splits = lifecycle["splits"]
    train, model_selection, calibration, threshold_tuning, test = (splits[name] for name in CANONICAL_SPLIT_NAMES)

    candidates = _candidate_models(random_seed)
    comparison: dict[str, Any] = {}
    for name, model in candidates.items():
        model.fit(train[LIVE_FEATURES], train.target)
        selection_probability = model.predict_proba(model_selection[LIVE_FEATURES])[:, 1]
        comparison[name] = evaluate(
            model_selection.target.to_numpy(), selection_probability, model_selection.candidate_group.to_numpy(),
        )
    selected_name = max(comparison, key=lambda name: comparison[name][SELECTION_CRITERION])
    selected_model = candidates[selected_name]

    calibrated = CalibratedClassifierCV(FrozenEstimator(selected_model), method=CALIBRATION_METHOD)
    calibrated.fit(calibration[LIVE_FEATURES], calibration.target)

    threshold_tuning_probability = calibrated.predict_proba(threshold_tuning[LIVE_FEATURES])[:, 1]
    threshold = select_threshold(
        threshold_tuning.target.to_numpy(), threshold_tuning_probability, objective=THRESHOLD_OBJECTIVE,
    )

    test_probability = calibrated.predict_proba(test[LIVE_FEATURES])[:, 1]
    test_metrics = evaluate(
        test.target.to_numpy(), test_probability, test.candidate_group.to_numpy(), threshold=threshold, bootstrap=True,
    )

    baseline_metrics = evaluate_baselines(
        train.target.to_numpy(), _with_viewer_rank(threshold_tuning), _with_viewer_rank(test),
        threshold_objective=THRESHOLD_OBJECTIVE, score_columns=LIVE_BASELINE_SCORE_COLUMNS,
    )
    importance = permutation_importance_report(calibrated, test[LIVE_FEATURES], test.target, random_seed, feature_names=LIVE_FEATURES)
    # Drift monitoring baseline built from the exact `train` split rows just fit above.
    # provenance/is_synthetic below carries through to metadata["datasetSource"] as usual --
    # a synthetic LIVE baseline is never presented as production evidence (see
    # driftBaseline.trainingDataProvenance below and README "Drift monitoring").
    drift_baseline = build_baseline(train, model_type="LIVE", numeric_features=LIVE_NUMERIC, categorical_features=LIVE_CATEGORICAL)
    # Additive, LIVE-only field: lets a drift API consumer see synthetic-vs-real provenance
    # directly on the baseline itself, without needing to separately cross-reference
    # metadata.datasetSource -- the same is_synthetic signal computed above for that field.
    drift_baseline["trainingDataProvenance"] = {"type": provenance, "synthetic": is_synthetic}

    physical = lifecycle["physicalSplits"]
    role_mapping = lifecycle["roleMapping"]
    now = datetime.now(timezone.utc).isoformat()
    version = f"live-recommendation-prod-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    metadata = {
        "modelVersion": version,
        "modelType": f"{type(calibrated).__module__}.{type(calibrated).__qualname__}",
        "selectedModel": selected_name,
        "featureNames": LIVE_FEATURES,
        "featureDefinitions": LIVE_FEATURE_DEFINITIONS,
        "featureImportance": importance,
        "targetDefinition": LIVE_TARGET_DEFINITION,
        "splitStrategy": lifecycle["description"],
        "splitLifecycleUsedFallback": lifecycle["usedFallback"],
        "splitRoleMapping": role_mapping,
        "splitRatios": lifecycle["ratios"],
        "selectionCriterion": f"Highest modelSelection-split {SELECTION_CRITERION} (threshold-independent).",
        "calibration": {
            "applied": True,
            "method": CALIBRATION_METHOD,
            "calibratedOn": "calibration split" if role_mapping["calibration"] == "calibration" else
                             f"{role_mapping['calibration']} split (fallback: no dedicated calibration split; see splitStrategy)",
            "usedDedicatedCalibrationSplit": role_mapping["calibration"] == "calibration",
        },
        "decisionThreshold": threshold,
        "thresholdObjective": THRESHOLD_OBJECTIVE,
        "thresholdSelectedOn": "thresholdTuning split" if role_mapping["thresholdTuning"] == "thresholdTuning" else
                                f"{role_mapping['thresholdTuning']} split (fallback: no dedicated threshold-tuning split; see splitStrategy)",
        "trainingSamples": len(train),
        "modelSelectionSamples": len(model_selection),
        "calibrationSamples": len(calibration),
        "thresholdTuningSamples": len(threshold_tuning),
        "testSamples": len(test),
        "physicalSplitSizes": {name: len(part) for name, part in physical.items()},
        "classDistribution": {name: class_distribution(part.target) for name, part in physical.items()},
        "trainingDataTimeRange": {
            "start": dataset.timestamp.min().isoformat() if len(dataset) else None,
            "end": dataset.timestamp.max().isoformat() if len(dataset) else None,
        },
        "sklearnVersion": model_store.SKLEARN_VERSION,
        "pythonVersion": model_store.PYTHON_VERSION,
        "randomSeed": random_seed,
        "trainingDurationSeconds": round(perf_counter() - started, 3),
        "trainedAt": now,
        "metrics": test_metrics,
        "modelComparison": comparison,
        "rankingEvaluationDiagnostics": test_metrics["groupDiagnostics"],
        "baselines": baseline_metrics,
        "driftBaseline": drift_baseline,
        "datasetSource": {
            "type": provenance,
            "synthetic": is_synthetic,
            "totalRowCount": len(dataset),
            # uniqueUsers/uniqueContents/labelDistribution are only ever populated for a
            # dataset that actually carries user_id/content_id columns -- currently just the
            # "real" path (app.ml.live_dataset_builder.build_live_dataset). The synthetic
            # generator has never modeled distinct users/streams, so these stay None for it
            # rather than reporting a misleadingly precise-looking number for data that was
            # never meant to represent one.
            "uniqueUsers": int(dataset["user_id"].nunique()) if "user_id" in dataset.columns else None,
            "uniqueContents": int(dataset["content_id"].nunique()) if "content_id" in dataset.columns else None,
            "labelDistribution": class_distribution(dataset.target) if len(dataset) else {"positive": 0, "negative": 0},
            "note": {
                "synthetic": (
                    "Generated deterministically by app.ml.live_synthetic_data (no real LIVE "
                    "behavior log exists yet); these metrics describe pipeline correctness, not "
                    "real user preference, and must not be presented as production evidence."
                ),
                "real": (
                    "Built from real, session-reconstructed LIVE interactions stored in this "
                    "service's own database (app.ml.live_dataset_builder) -- reflects actually "
                    "observed user behavior, subject to this deployment's current real-data volume."
                ),
                "injected": "Caller-provided dataset (not generated by this trainer).",
            }[provenance],
        },
        "neutralRowsExcluded": True,
    }
    # Group D: same candidate -> validate -> promote flow as VIDEO (app.services.training_service),
    # kept separate per-model-type (spec §16). LIVE_MODEL_PATH/LIVE_METADATA_PATH are read as
    # module globals so a monkeypatched active path is respected.
    paths = artifact_lifecycle.sibling_paths(LIVE_MODEL_PATH, LIVE_METADATA_PATH)
    model_store.save(calibrated, metadata, model_path=paths["candidate_model"],
                      metadata_path=paths["candidate_metadata"], model_dir=MODEL_DIR)
    promoted_metadata = artifact_lifecycle.promote(
        candidate_model_path=paths["candidate_model"], candidate_metadata_path=paths["candidate_metadata"],
        active_model_path=LIVE_MODEL_PATH, active_metadata_path=LIVE_METADATA_PATH,
        previous_model_path=paths["previous_model"], previous_metadata_path=paths["previous_metadata"],
        feature_names=LIVE_FEATURES, categorical_features=LIVE_CATEGORICAL,
    )
    model_cache.live_cache.invalidate()
    return promoted_metadata
