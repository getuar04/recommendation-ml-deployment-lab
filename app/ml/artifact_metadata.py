"""Assembles the versioned artifact-metadata contract (app.ml.model_store.
COMMON/CLASSIFIER/RANKER_REQUIRED_METADATA_FIELDS) from a training result -- family-conditional,
so a classifier winner gets classifier-shaped metadata (calibration/decisionThreshold/
splitStrategy/...) and a ranker winner gets ranker-shaped metadata (objective/scoreSemantics/
normalizationStrategy/rankingGroupVersion/...), never a mix of the two and never a fabricated
placeholder for a field that genuinely does not apply to that family (XGBRanker promotion task).

Extracted out of app.services.training_service so that module stays orchestration-only (build
dataset -> train -> assemble metadata -> save -> promote), matching this project's Clean
Architecture convention of one small, named responsibility per module.
"""
from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any

from app.ml import model_store
from app.ml.dataset_builder import FEATURE_DEFINITIONS, FEATURES
from app.ml.feature_builder import TARGET_DEFINITION
from app.ml.ranking_groups import (
    RANKING_GROUP_VERSION as SYNTHETIC_RANKING_GROUP_VERSION,
)
from app.ml.trainer import CALIBRATION_METHOD, RANDOM_SEED
from app.ml.training_data_source import TrainingDataSource

SYNTHETIC_EVENT_PREFIX = "syn-"


def dataset_source(rows: list[Any]) -> dict[str, Any]:
    """Describes the flat classifier interaction dataset built for this training run --
    shared by both metadata builders below: even when a ranker wins, the SAME run also trained
    and evaluated classifier candidates on this dataset (see `modelComparison`/
    `candidateEvaluations` in the persisted metadata)."""
    synthetic_count = sum(1 for row in rows if row.event_id.startswith(SYNTHETIC_EVENT_PREFIX))
    return {
        "type": "database",
        "synthetic": len(rows) > 0 and synthetic_count == len(rows),
        "syntheticRowCount": synthetic_count,
        "totalRowCount": len(rows),
        "note": (
            "Rows with event_id prefix 'syn-' come from scripts/generate_synthetic_data.py; "
            "a nonzero syntheticRowCount means at least part of this training run is not "
            "backed by real user behavior and metrics must not be presented as production evidence."
        ),
    }


def _cross_family_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Fields only present under `app.ml.trainer.train_and_select_cross_family` -- absent
    (None) for `train_models`/`train_and_select`, where cross-family comparison never ran.
    Shared verbatim by both family-specific builders below."""
    return {
        "modelSelectionVersion": result.get("modelSelectionVersion"),
        "eligibilitySeverity": result.get("eligibilitySeverity"),
        "candidateEvaluations": result.get("candidateEvaluations"),
        "eligibilityDecisions": result.get("eligibilityDecisions"),
        "qualityScores": result.get("qualityScores"),
        "crossFamilyQualityScores": result.get("crossFamilyQualityScores"),
        "candidatePool": result.get("candidatePool"),
        "benchmarkDifficulties": result.get("benchmarkDifficulties"),
        "unavailableAlgorithms": result.get("unavailableAlgorithms"),
        "unavailableProductionRankers": result.get("unavailableProductionRankers"),
        "productionRankerTrainingError": result.get("productionRankerTrainingError"),
        "productionRankerNamesConsidered": result.get("productionRankerNamesConsidered"),
    }


def build_classifier_metadata(
    result: dict[str, Any], *, model: Any, version: str, now: str, started: float, rows: list[Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Classifier-shaped metadata -- unchanged in content/order from before the XGBRanker
    promotion task, for `train_models` (TRAINING_ALGORITHM_LOCK) and any classifier winner of
    `train_and_select`/`train_and_select_cross_family` alike."""
    role_mapping = result["splitRoleMapping"]
    return {
        "modelVersion": version,
        "modelType": f"{type(model).__module__}.{type(model).__qualname__}",
        "modelFamily": "classifier",
        "objective": "binary_classification",
        "scoreSemantics": "probability",
        "selectedModel": result["selectedModel"],
        "featureNames": FEATURES,
        "featureDefinitions": FEATURE_DEFINITIONS,
        "featureImportance": result["featureImportance"],
        "targetDefinition": TARGET_DEFINITION,
        "splitStrategy": result["splitLifecycleDescription"],
        "splitLifecycleUsedFallback": result["splitLifecycleUsedFallback"],
        "splitRoleMapping": role_mapping,
        "selectionCriterion": result["selectionCriterion"],
        "selectionDetails": result["selectionDetails"],
        **_cross_family_fields(result),
        "calibration": {
            "applied": True,
            "method": CALIBRATION_METHOD,
            "calibratedOn": "calibration split" if result["calibrationIsDedicatedSplit"] else
                             f"{role_mapping['calibration']} split (fallback: no dedicated calibration split; see splitStrategy)",
            "usedDedicatedCalibrationSplit": result["calibrationIsDedicatedSplit"],
        },
        "decisionThreshold": result["decisionThreshold"],
        "thresholdObjective": result["thresholdObjective"],
        "thresholdSelectedOn": (
            "thresholdTuning split, using calibrated probabilities (never the test split)."
            if result["thresholdTuningIsDedicatedSplit"] else
            f"{role_mapping['thresholdTuning']} split (fallback: no dedicated threshold-tuning split; see splitStrategy), "
            "using calibrated probabilities (never the test split)."
        ),
        "trainingSamples": result["splitSizes"]["train"],
        "modelSelectionSamples": result["splitSizes"]["modelSelection"],
        "calibrationSamples": result["splitSizes"]["calibration"],
        "thresholdTuningSamples": result["splitSizes"]["thresholdTuning"],
        "testSamples": result["splitSizes"]["test"],
        "physicalSplitSizes": result["physicalSplitSizes"],
        "classDistribution": result["classDistribution"],
        "trainingDataTimeRange": result["trainingDataTimeRange"],
        "eligibility": result["eligibility"],
        "eligibleSelection": result["eligibleSelection"],
        "sklearnVersion": model_store.SKLEARN_VERSION,
        "pythonVersion": model_store.PYTHON_VERSION,
        "randomSeed": RANDOM_SEED,
        "trainingDurationSeconds": round(perf_counter() - started, 3),
        "trainedAt": now,
        "metrics": result["metrics"],
        "modelComparison": result["modelComparison"],
        "rankingEvaluationDiagnostics": result["metrics"]["groupDiagnostics"],
        "rerankEvaluation": result["rerankEvaluation"],
        "baselines": result["baselines"],
        "driftBaseline": result["driftBaseline"],
        "datasetSource": dataset_source(rows),
        "neutralRowsExcluded": True,
        "datasetDiagnostics": diagnostics,
    }


def build_ranker_metadata(
    result: dict[str, Any], *, model: Any, version: str, now: str, started: float, rows: list[Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Ranker-shaped metadata (XGBRanker promotion task): honest ranking-native fields only --
    see app.ml.model_store.RANKER_REQUIRED_METADATA_FIELDS for the strict contract this must
    satisfy, and app.ml.trainer._ranker_tail for where most of these `result` fields come from.
    Deliberately omits every classifier-only field (calibration method, decisionThreshold-based
    concepts, split-lifecycle sample counts, class distribution) rather than filling them with
    placeholder/zero values."""
    return {
        "modelVersion": version,
        "modelType": f"{type(model).__module__}.{type(model).__qualname__}",
        "modelFamily": "ranker",
        "objective": result["objective"],
        "scoreSemantics": result["scoreSemantics"],
        "normalizationStrategy": result["normalizationStrategy"],
        "normalizationScale": result["normalizationScale"],
        "rankingGroupVersion": result["rankingGroupVersion"],
        "groupParamName": result["groupParamName"],
        "trainGroups": result["trainGroups"],
        "scaleGroups": result["scaleGroups"],
        "testGroups": result["testGroups"],
        "selectedModel": result["selectedModel"],
        "featureNames": FEATURES,
        "splitStrategy": result["splitStrategy"],
        "selectionCriterion": result["selectionCriterion"],
        "selectionDetails": result["selectionDetails"],
        **_cross_family_fields(result),
        "calibration": result["calibration"],
        "decisionThreshold": result["decisionThreshold"],
        "thresholdObjective": result["thresholdObjective"],
        "eligibility": result["eligibility"],
        "eligibleSelection": result["eligibleSelection"],
        "sklearnVersion": model_store.SKLEARN_VERSION,
        "pythonVersion": model_store.PYTHON_VERSION,
        "randomSeed": RANDOM_SEED,
        "trainingDurationSeconds": round(perf_counter() - started, 3),
        "trainedAt": now,
        "metrics": result["metrics"],
        "modelComparison": result["modelComparison"],
        "rerankEvaluation": result["rerankEvaluation"],
        "baselines": result["baselines"],
        "featureImportance": result["featureImportance"],
        "driftBaseline": result["driftBaseline"],
        "trainingDataTimeRange": result["trainingDataTimeRange"],
        # Kept the SAME shape as the classifier metadata's `datasetSource` (type/synthetic/
        # syntheticRowCount/totalRowCount/note) -- callers that only care "how much of this
        # run's data was synthetic" (e.g. app.experiments.runner's report) must not need a
        # family-specific branch just to read it. The ranker's OWN, entirely different training
        # data (app.ml.ranking_groups, never the flat interaction dataset) gets its own key
        # below instead of overloading this one with an incompatible shape.
        "datasetSource": dataset_source(rows),
        "rankerTrainingDataSource": {
            "type": "syntheticRankingGroups",
            "rankingGroupVersion": result["rankingGroupVersion"],
            "trainGroups": result["trainGroups"], "scaleGroups": result["scaleGroups"], "testGroups": result["testGroups"],
            "note": "This ranker was actually TRAINED on app.ml.ranking_groups synthetic ranking "
                    "groups, NOT on the flat classifier interaction dataset described by "
                    "`datasetSource` above (which still reflects the flat dataset every classifier "
                    "candidate in this same run was trained/compared on -- see modelComparison/"
                    "candidateEvaluations).",
        },
        "neutralRowsExcluded": True,
        "datasetDiagnostics": diagnostics,
    }


def build_metadata(
    result: dict[str, Any], *, model: Any, version: str, now: str, started: float, rows: list[Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Dispatches to the family-appropriate builder. Absence of `modelFamily` (the
    TRAINING_ALGORITHM_LOCK/`train_models` path, which predates the ranker concept entirely)
    means classifier, mirroring app.ml.model_store._model_family_of's own convention."""
    family = result.get("modelFamily") or "classifier"
    builder = build_ranker_metadata if family == "ranker" else build_classifier_metadata
    return builder(result, model=model, version=version, now=now, started=started, rows=rows, diagnostics=diagnostics)


def build_real_ranker_metadata(
    *, model: Any, version: str, now: str, training_duration_seconds: float,
    objective: str, score_semantics: str, normalization_strategy: str, normalization_scale: float,
    ranking_group_contract_version: str, train_groups: int, test_groups: int,
    feature_names: Sequence[str] = FEATURES,
) -> dict[str, Any]:
    """Future real-observed-data ranker metadata builder (RMS real-data training foundation
    task) -- NOT wired into any current training entrypoint, save path, or promotion path; no
    real training exists yet to call this. Mirrors `build_ranker_metadata`'s ranker-shaped field
    set, extended with explicit `trainingDataSource` provenance.

    Hard-rejects `ranking_group_contract_version` reuse of the synthetic generator's own
    `app.ml.ranking_groups.RANKING_GROUP_VERSION` ("v3-multisignal" today) -- a real-observed
    artifact must be recorded under its own, distinct group-contract version so it can never be
    silently equivalenced with a synthetic run (see the RMS training-path reconciliation
    finding: "CURRENT TRAINING PATH EQUIVALENCE NOT PROVEN").
    """
    if not ranking_group_contract_version or not ranking_group_contract_version.strip():
        raise ValueError("ranking_group_contract_version must be a non-blank, explicit real-data group contract version")
    if ranking_group_contract_version == SYNTHETIC_RANKING_GROUP_VERSION:
        raise ValueError(
            f"ranking_group_contract_version must not reuse the synthetic generator's own "
            f"version ({SYNTHETIC_RANKING_GROUP_VERSION!r}) -- real-observed training data must "
            "be recorded under its own distinct group contract version"
        )
    return {
        "modelVersion": version,
        "modelType": f"{type(model).__module__}.{type(model).__qualname__}",
        "modelFamily": "ranker",
        "trainingDataSource": TrainingDataSource.REAL_OBSERVED.value,
        "objective": objective,
        "scoreSemantics": score_semantics,
        "normalizationStrategy": normalization_strategy,
        "normalizationScale": normalization_scale,
        "rankingGroupVersion": ranking_group_contract_version,
        "trainGroups": train_groups,
        "testGroups": test_groups,
        "featureNames": list(feature_names),
        "sklearnVersion": model_store.SKLEARN_VERSION,
        "pythonVersion": model_store.PYTHON_VERSION,
        "trainingDurationSeconds": round(training_duration_seconds, 3),
        "trainedAt": now,
    }
