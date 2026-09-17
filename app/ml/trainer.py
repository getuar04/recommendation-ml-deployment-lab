"""VIDEO recommendation model training: split lifecycle, model comparison, calibration, thresholding.

See `app.ml.split_lifecycle` for the exact train -> modelSelection -> calibration ->
thresholdTuning -> test chronological lifecycle (and its documented small-data
fallbacks). Data usage only ever moves forward in time; test is evaluated exactly once.

Model selection compares behaviorally-eligible candidates (app.ml.eligibility) using a
ranking-first weighted selection score (app.ml.model_selection: NDCG@10/Precision@10/PR-AUC/
Recall@10) rather than F1 at an arbitrary, uncalibrated 0.5 threshold. F1 (and every other
`evaluate()` metric) remains available as a diagnostic in `modelComparison`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from app.core.config import (
    TRAINING_ALGORITHM_LOCK,
    TRAINING_ENABLE_PRODUCTION_RANKERS,
    TRAINING_USE_SAMPLE_WEIGHTS,
)
from app.core.logging import logger
from app.ml import candidate_pool, eligibility_policy, gate_severity, quality_scorer
from app.ml.algorithm_registry import build_candidate_pipelines, unavailable_algorithms
from app.ml.baselines import evaluate_baselines
from app.ml.candidate_evaluation import SELECTION_DIFFICULTIES, evaluate_candidate
from app.ml.dataset_builder import CATEGORICAL, FEATURES, NUMERIC
from app.ml.drift_baseline import build_baseline
from app.ml.eligibility import evaluate_eligibility
from app.ml.evaluator import evaluate, select_threshold
from app.ml.model_selection import select_winner, select_winner_v2
from app.ml.pipeline_builder import sample_weight_fit_params
from app.ml.quality_scorer import quality_score_cross_family
from app.ml.ranker_registry import PRODUCTION_RANKER_NAMES, unavailable_rankers
from app.ml.ranker_trainer import train_ranking_challengers
from app.ml.ranking_groups import (
    PRODUCTION_GROUP_SEED,
    PRODUCTION_REPLICAS_PER_ARCHETYPE,
)
from app.ml.replay_saturation_policy import compute_replay_weights
from app.ml.reranking_eval import evaluate_reranking
from app.ml.sample_weight_policy import SAMPLE_WEIGHT_COLUMN, compute_sample_weights
from app.ml.split_lifecycle import CANONICAL_SPLIT_NAMES, resolve_split_lifecycle
from app.ml.training_utils import class_distribution, permutation_importance_report

CALIBRATION_METHOD = "sigmoid"
RANDOM_SEED = 42
SELECTION_CRITERION = "selectionScore"
THRESHOLD_OBJECTIVE = "f1"

# Task 5: bumped whenever the eligibility/selection POLICY itself changes shape (which gates
# are HARD/SOFT, how qualityScore is weighted, what counts as an eligible candidate) -- not on
# every training run. Lets a saved artifact's metadata be compared against a future policy
# change ("this model was selected under v1/gate-only selection" vs. "under v2/layered
# selection") without a full migration framework; see `train_and_select`'s returned metadata.
MODEL_SELECTION_VERSION = "v2"

# XGBRanker promotion task: bumped separately from MODEL_SELECTION_VERSION since cross-family
# selection is a materially different policy (app.ml.candidate_pool.select_cross_family_winner,
# quality_score_cross_family) layered on top of, not replacing, Task 5's classifier-only v2
# eligibility/quality policy -- `train_and_select` (classifier-only) keeps returning "v2"
# unchanged; only `train_and_select_cross_family`'s result carries this version.
CROSS_FAMILY_SELECTION_VERSION = "crossFamily-v1"


def _candidate_models(random_seed: int, *, only: str | None = None) -> dict[str, Pipeline]:
    """`only`, when given, restricts the candidate pool to exactly that one named model --
    used to prove/enforce "same algorithm, different dataset" comparisons (spec: comparative
    experiments) without disabling automatic model selection for normal training, which keeps
    comparing every enabled, available candidate in app.ml.algorithm_registry by default
    (`only=None`)."""
    return build_candidate_pipelines(random_seed, categorical=CATEGORICAL, numeric=NUMERIC, only=only)


def _fit_candidates(
    candidates: dict[str, Pipeline], train: pd.DataFrame, model_selection: pd.DataFrame, train_weights: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fits every candidate on `train` and scores it on `model_selection`; returns
    (comparison, eligibility) keyed by algorithm name. Shared by `train_models` (single- or
    multi-candidate, old selection policy) and `train_and_select` (multi-candidate, Task 5
    layered selection) so the fit/raw-evaluation step is never duplicated between them."""
    comparison: dict[str, Any] = {}
    eligibility: dict[str, Any] = {}
    for name, model in candidates.items():
        fit_started = perf_counter()
        model.fit(train[FEATURES], train.target, **sample_weight_fit_params(train_weights))
        fit_duration = perf_counter() - fit_started
        selection_probability = model.predict_proba(model_selection[FEATURES])[:, 1]
        comparison[name] = evaluate(
            model_selection.target.to_numpy(), selection_probability, model_selection.candidate_group.to_numpy(),
        )
        comparison[name]["trainingDurationSeconds"] = round(fit_duration, 3)
        eligibility[name] = evaluate_eligibility(model)
    return comparison, eligibility


def train_models(
    df: pd.DataFrame, *, random_seed: int = RANDOM_SEED, restrict_algorithm: str | None = TRAINING_ALGORITHM_LOCK,
    use_sample_weights: bool = TRAINING_USE_SAMPLE_WEIGHTS,
) -> dict[str, Any]:
    # app.ml.sample_weight_policy: a per-row training-importance column, computed once here
    # (before any split) from each row's OWN already-happened event fields -- never from the
    # split it lands in or another row -- then carried through resolve_split_lifecycle/
    # chronological_group_split exactly like "candidate_group"/"timestamp" already are. The
    # binary `target` itself is completely unchanged; see that module for the full rationale.
    #
    # app.ml.replay_saturation_policy.compute_replay_weights: a SEPARATE factor, multiplied in
    # here rather than inside sample_weight_policy itself -- that module's per-row-only
    # contract has no visibility into any other row, so it cannot see "is this the 8th
    # passively-replayed row for this exact (user, content) pair" on its own. This chronological
    # preprocessing pass reads only each row's own already-happened event_type/user_id/
    # content_id/timestamp (never a future row, never the split a row lands in -- see that
    # function's own docstring), so this stays point-in-time safe by the same construction
    # compute_sample_weights already relies on. Both factors are 1.0 for every row unless a
    # (user, content) pair has actually been passively replayed 3+ times, so this is a no-op
    # multiplication for the overwhelming majority of any real dataset.
    df = df.copy()
    df[SAMPLE_WEIGHT_COLUMN] = (
        compute_sample_weights(df) * compute_replay_weights(df) if use_sample_weights else 1.0
    )

    lifecycle = resolve_split_lifecycle(df)
    splits = lifecycle["splits"]
    train, model_selection, calibration, threshold_tuning, test = (splits[name] for name in CANONICAL_SPLIT_NAMES)

    # Two-stage selection (finalization spec Step 3, extended): every candidate is first fit
    # and scored exactly as before, then run through app.ml.eligibility's generic behavioral
    # gates (long-term/recent/session/negative/notInterested/semantic/creator/alreadySeen/
    # coldStart/subthemeRejectionLocalization -- no named entities, production-safe). A model
    # failing even one mandatory gate is never selected while ANY eligible candidate exists,
    # regardless of selection score/PR-AUC/ROC-AUC/F1 -- this is what stops an aggregate-
    # metric-only winner (e.g. RandomForest on this project's real data, which has higher
    # PR-AUC but fails the "recent" gate) from being silently promoted. If NO candidate is
    # eligible, selection falls back to the metric-only winner (so training/calibration/
    # diagnostics can still complete) but `eligibleSelection=False` is returned so the caller
    # (app.services.training_service) can refuse to promote it -- "no eligible candidate"
    # must never silently become "promote the best of a bad set".
    candidates = _candidate_models(random_seed, only=restrict_algorithm)
    train_weights = train[SAMPLE_WEIGHT_COLUMN].to_numpy() if use_sample_weights else None
    comparison, eligibility = _fit_candidates(candidates, train, model_selection, train_weights)

    eligible_names = [name for name, report in eligibility.items() if report["eligible"]]
    eligible_selection = bool(eligible_names)
    selection_pool = eligible_names if eligible_selection else list(comparison)
    # app.ml.model_selection.select_winner: a ranking-first weighted blend (NDCG@10/
    # Precision@10/PR-AUC/Recall@10), never plain PR-AUC/accuracy alone -- see that module for
    # the exact weights and the fully deterministic tie-break chain.
    selection = select_winner(selection_pool, comparison)
    selected_name = selection["selected"]
    selected_model = candidates[selected_name]

    # Calibration only ever moves forward from the modelSelection split; it is never fit
    # on the threshold-tuning or test splits. The underlying model is frozen (FrozenEstimator
    # never refits it), so sample_weight here only influences the sigmoid calibration curve
    # itself -- the same per-row training-importance policy applied consistently, never
    # touching threshold-tuning/test.
    calibrated = CalibratedClassifierCV(FrozenEstimator(selected_model), method=CALIBRATION_METHOD)
    calibration_weight = calibration[SAMPLE_WEIGHT_COLUMN].to_numpy() if use_sample_weights else None
    calibrated.fit(calibration[FEATURES], calibration.target, sample_weight=calibration_weight)

    tail = _finalize(
        df=df, calibrated=calibrated, train=train, threshold_tuning=threshold_tuning, test=test,
        lifecycle=lifecycle, use_sample_weights=use_sample_weights, random_seed=random_seed,
    )
    return {
        "model": calibrated,
        # Diagnostic-only (app.benchmark.diagnostics.calibration_ranking_comparison): the
        # SAME fitted pipeline `calibrated` above wraps via FrozenEstimator, exposed here so
        # a caller can compare pre-/post-calibration ranking without refitting anything.
        # Never used by training/selection/serving itself -- purely an already-computed
        # object being returned alongside the calibrated one.
        "uncalibratedModel": selected_model,
        "selectedModel": selected_name,
        "selectionCriterion": SELECTION_CRITERION,
        "selectionDetails": selection,
        "modelComparison": comparison,
        "eligibility": eligibility,
        "eligibleSelection": eligible_selection,
        "unavailableAlgorithms": unavailable_algorithms(),
        "modelSelectionVersion": "v1",
        **tail,
    }


def _finalize(
    *, df: pd.DataFrame, calibrated: Any, train: pd.DataFrame, threshold_tuning: pd.DataFrame, test: pd.DataFrame,
    lifecycle: dict[str, Any], use_sample_weights: bool, random_seed: int,
) -> dict[str, Any]:
    """Everything downstream of "a winner has been chosen and calibrated" that both
    `train_models` and `train_and_select` need identically: threshold selection, test-split
    evaluation, baselines, rerank diagnostics, feature importance, the drift baseline, and
    every split-bookkeeping field the artifact metadata (app.services.training_service)
    reads. Factored out so the two selection policies never duplicate this tail."""
    threshold_tuning_probability = calibrated.predict_proba(threshold_tuning[FEATURES])[:, 1]
    threshold = select_threshold(
        threshold_tuning.target.to_numpy(), threshold_tuning_probability, objective=THRESHOLD_OBJECTIVE,
    )

    test_probability = calibrated.predict_proba(test[FEATURES])[:, 1]
    test_metrics = evaluate(
        test.target.to_numpy(), test_probability, test.candidate_group.to_numpy(), threshold=threshold, bootstrap=True,
    )

    baseline_metrics = evaluate_baselines(train.target.to_numpy(), threshold_tuning, test, threshold_objective=THRESHOLD_OBJECTIVE)
    rerank_diagnostics = evaluate_reranking(test, test_probability, k=10)
    importance = permutation_importance_report(calibrated, test[FEATURES], test.target, random_seed, feature_names=FEATURES)
    # Drift monitoring baseline (app.ml.drift_baseline / app.ml.drift_detector): built from
    # the exact `train` split rows just fit above -- the literal training feature
    # distribution, not model_selection/calibration/threshold_tuning/test. Named
    # "driftBaseline" (singular) to stay unambiguous next to "baselines" (plural) above,
    # which is an unrelated concept (majority-class/popularity/category-affinity ranking
    # baselines from app.ml.baselines).
    drift_baseline = build_baseline(train, model_type="VIDEO", numeric_features=NUMERIC, categorical_features=CATEGORICAL)

    physical = lifecycle["physicalSplits"]
    role_mapping = lifecycle["roleMapping"]
    splits = lifecycle["splits"]
    return {
        "metrics": test_metrics,
        "decisionThreshold": threshold,
        "thresholdObjective": THRESHOLD_OBJECTIVE,
        "thresholdTuningIsDedicatedSplit": role_mapping["thresholdTuning"] == "thresholdTuning",
        "calibrationIsDedicatedSplit": role_mapping["calibration"] == "calibration",
        "splitLifecycleDescription": lifecycle["description"],
        "splitLifecycleUsedFallback": lifecycle["usedFallback"],
        "splitRoleMapping": role_mapping,
        "splitRatios": lifecycle["ratios"],
        "physicalSplitSizes": {name: len(part) for name, part in physical.items()},
        "splitSizes": {name: len(splits[name]) for name in CANONICAL_SPLIT_NAMES},
        "classDistribution": {name: class_distribution(part.target) for name, part in physical.items()},
        "baselines": baseline_metrics,
        "rerankEvaluation": rerank_diagnostics,
        "featureImportance": importance,
        "driftBaseline": drift_baseline,
        "trainingDataTimeRange": {
            "start": df.timestamp.min().isoformat() if len(df) else None,
            "end": df.timestamp.max().isoformat() if len(df) else None,
        },
    }


def train_and_select(
    df: pd.DataFrame, *, random_seed: int = RANDOM_SEED, use_sample_weights: bool = TRAINING_USE_SAMPLE_WEIGHTS,
    benchmark_difficulties: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Task 5: layered, multi-candidate model selection -- PHASE 1 eligibility (HARD
    ModelBehavior gates + HARD RerankerPolicy check + minimum END_TO_END critical-constraint
    rate; app.ml.eligibility_policy) then PHASE 2 quality ranking among eligible candidates
    (app.ml.quality_scorer, end-to-end reranked NDCG-first). Used whenever more than one
    algorithm is actually being compared; `app.services.training_service.train` calls this
    instead of `train_models` unless TRAINING_ALGORITHM_LOCK restricts training to exactly one
    algorithm, in which case `train_models` alone (no benchmark dependency, selection is
    trivial with one candidate) is used, matching Task 5 spec section 21.

    Every candidate is fit, then calibrated (each on its OWN calibration-split fit, mirroring
    exactly what the eventual winner receives -- so RerankerPolicy/END_TO_END are measured
    against the SAME kind of model that will actually be served, not an uncalibrated proxy;
    Task 4 already showed calibration causes zero raw-ranking inversions, but the reranker
    blends ABSOLUTE probabilities with fixed multiplicative business rules, so a per-candidate
    recalibration is not guaranteed to leave the FINAL reranked ordering identical). Only the
    winner then goes through threshold-tuning/test-split evaluation (`_finalize`) -- the test
    split is still touched exactly once, for the winner alone.
    """
    df = df.copy()
    # See train_models' identical line above for why compute_replay_weights is multiplied in
    # here too (app.ml.replay_saturation_policy) -- same rationale, same point-in-time safety.
    df[SAMPLE_WEIGHT_COLUMN] = (
        compute_sample_weights(df) * compute_replay_weights(df) if use_sample_weights else 1.0
    )
    difficulties = benchmark_difficulties or SELECTION_DIFFICULTIES
    built = _build_classifier_pool(df, random_seed=random_seed, use_sample_weights=use_sample_weights, difficulties=difficulties)
    candidates = built["candidates"]

    selection = select_winner_v2(list(candidates), built["eligibility_decisions"], built["quality_scores"], built["comparison"])
    selected_name = selection["selected"]
    selected_model_calibrated = built["calibrated_candidates"][selected_name]

    tail = _finalize(
        df=df, calibrated=selected_model_calibrated, train=built["train"], threshold_tuning=built["threshold_tuning"],
        test=built["test"], lifecycle=built["lifecycle"], use_sample_weights=use_sample_weights, random_seed=random_seed,
    )
    return {
        "model": selected_model_calibrated,
        "uncalibratedModel": candidates[selected_name],
        "selectedModel": selected_name,
        "selectionCriterion": "layeredSelection (Task 5): PHASE 1 eligibility (HARD ModelBehavior "
                               "gates + HARD RerankerPolicy + minimum END_TO_END critical-constraint "
                               "rate) then PHASE 2 qualityScore ranking among eligible candidates.",
        "selectionDetails": selection,
        "modelComparison": built["comparison"],
        "eligibility": built["eligibility"],
        "eligibilitySeverity": built["severity"],
        "candidateEvaluations": built["candidate_evaluations"],
        "eligibilityDecisions": built["eligibility_decisions"],
        "qualityScores": built["quality_scores"],
        "eligibleSelection": selection["eligibleSelection"],
        "unavailableAlgorithms": unavailable_algorithms(),
        "modelSelectionVersion": MODEL_SELECTION_VERSION,
        "benchmarkDifficulties": list(difficulties),
        **tail,
    }


def _build_classifier_pool(
    df: pd.DataFrame, *, random_seed: int, use_sample_weights: bool, difficulties: tuple[str, ...],
) -> dict[str, Any]:
    """Fits every enabled classifier candidate on `train`, calibrates each independently on
    `calibration`, and runs each through the Task 5 layered eligibility/quality pipeline.
    Extracted so `train_and_select` (classifier-only selection) and
    `train_and_select_cross_family` (XGBRanker promotion task: classifier + production-ranker
    selection) never duplicate this fit/calibrate/evaluate step. `df` must already carry
    SAMPLE_WEIGHT_COLUMN (see both callers)."""
    lifecycle = resolve_split_lifecycle(df)
    splits = lifecycle["splits"]
    train, model_selection, calibration, threshold_tuning, test = (splits[name] for name in CANONICAL_SPLIT_NAMES)

    candidates = _candidate_models(random_seed, only=None)
    train_weights = train[SAMPLE_WEIGHT_COLUMN].to_numpy() if use_sample_weights else None
    comparison, eligibility = _fit_candidates(candidates, train, model_selection, train_weights)

    calibration_weight = calibration[SAMPLE_WEIGHT_COLUMN].to_numpy() if use_sample_weights else None
    calibrated_candidates: dict[str, Any] = {}
    for name, model in candidates.items():
        candidate_calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=CALIBRATION_METHOD)
        candidate_calibrated.fit(calibration[FEATURES], calibration.target, sample_weight=calibration_weight)
        calibrated_candidates[name] = candidate_calibrated

    severity = {name: gate_severity.severity_report_from_eligibility(report) for name, report in eligibility.items()}

    candidate_evaluations: dict[str, Any] = {}
    physical = lifecycle["physicalSplits"]
    split_sizes = {name: len(splits[name]) for name in CANONICAL_SPLIT_NAMES}
    class_distribution_by_split = {name: class_distribution(part.target) for name, part in physical.items()}
    with tempfile.TemporaryDirectory(prefix="train_and_select_") as tmp:
        tmp_dir = Path(tmp)
        for name, candidate_calibrated in calibrated_candidates.items():
            training_result_stub = {
                "splitLifecycleDescription": lifecycle["description"],
                "decisionThreshold": 0.5,
                "splitSizes": split_sizes,
                "classDistribution": class_distribution_by_split,
                "metrics": comparison[name],
                "modelComparison": comparison,
            }
            candidate_evaluations[name] = evaluate_candidate(
                name, candidate_calibrated, training_result_stub, tmp_dir, difficulties=difficulties,
            )

    eligibility_decisions = {
        name: eligibility_policy.decide(severity[name], candidate_evaluations[name]) for name in candidates
    }
    quality_scores = {
        name: quality_scorer.quality_score(comparison[name], candidate_evaluations[name]) for name in candidates
    }
    return {
        "lifecycle": lifecycle, "train": train, "threshold_tuning": threshold_tuning, "test": test,
        "candidates": candidates, "calibrated_candidates": calibrated_candidates,
        "comparison": comparison, "eligibility": eligibility, "severity": severity,
        "candidate_evaluations": candidate_evaluations, "eligibility_decisions": eligibility_decisions,
        "quality_scores": quality_scores,
    }


def _ranker_tail(
    winner: candidate_pool.TrainedModelCandidate, ranker_result: dict[str, Any], *,
    df: pd.DataFrame, flat_train: pd.DataFrame,
) -> dict[str, Any]:
    """The ranker-family analogue of `_finalize`'s tail: honest ranking-native fields only --
    no calibration, no decision threshold, no classifier test-split evaluate() metrics.
    `metrics` is built entirely from the SAME independent end-to-end benchmark evidence already
    computed during selection (app.ml.candidate_evaluation) -- there is no second, ranker-only
    test split to evaluate against (a ranker trains on app.ml.ranking_groups, never the flat
    classifier `df`).

    `driftBaseline`/`trainingDataTimeRange`, unlike the fields above, are NOT about the winning
    model at all -- they describe the flat interaction data's own feature distribution/time
    range (app.ml.drift_baseline.build_baseline), which every candidate in this run (classifier
    or ranker) was evaluated against identically. Production drift monitoring compares live
    traffic features against this baseline regardless of which model is currently serving, so
    it stays meaningful -- and is computed from `flat_train`/`df` (never from the ranker's own
    ranking-groups training data, which has no real-world time range at all) -- even when a
    ranker wins."""
    name = winner.name
    candidate_eval = ranker_result["candidateEvaluations"][name]
    by_difficulty = candidate_eval["endToEnd"]["byDifficulty"]
    eligibility_decision = ranker_result["eligibilityDecisions"][name]
    metrics = {
        "modelFamily": "ranker",
        "note": "Ranking-native metrics from the independent end-to-end benchmark "
                "(app.ml.candidate_evaluation), not a held-out classifier test split. "
                "Classifier-only metrics (PR-AUC/ROC-AUC/F1/logLoss/accuracy) do not apply to a "
                "ranking-native model and are intentionally omitted rather than fabricated as 0.",
        "ndcgByDifficulty": {d: stats["finalNdcgAt10"] for d, stats in by_difficulty.items()},
        "rawNdcgByDifficulty": {d: stats["rawNdcgAt10"] for d, stats in by_difficulty.items()},
        "criticalPassRate": eligibility_decision["criticalPassRate"],
        "criticalPassed": eligibility_decision["criticalPassed"],
        "criticalTotal": eligibility_decision["criticalTotal"],
        "unseenBeatsSeenPassed": candidate_eval["rerankerPolicy"]["unseenBeatsSeenPassed"],
    }
    return {
        "metrics": metrics,
        "decisionThreshold": None,
        "thresholdObjective": None,
        "calibration": {"applied": False},
        "splitStrategy": (
            f"Ranking-group challenger split (rankingGroupVersion={ranker_result['rankingGroupVersion']}): "
            f"{ranker_result['trainGroups']} train / {ranker_result['scaleGroups']} scale-normalization / "
            f"{ranker_result['testGroups']} held-out query groups, group-preserving, chronological. "
            "Entirely independent of the flat classifier training/modelSelection/calibration/"
            "thresholdTuning/test split lifecycle."
        ),
        "rankingGroupVersion": ranker_result["rankingGroupVersion"],
        "trainGroups": ranker_result["trainGroups"],
        "scaleGroups": ranker_result["scaleGroups"],
        "testGroups": ranker_result["testGroups"],
        "objective": winner.objective,
        "scoreSemantics": winner.score_semantics,
        "normalizationStrategy": winner.metadata.get("normalizationStrategy"),
        "normalizationScale": winner.metadata.get("normalizationScale"),
        "groupParamName": winner.metadata.get("groupParamName"),
        "rerankEvaluation": None,
        "baselines": None,
        "featureImportance": None,
        "driftBaseline": build_baseline(flat_train, model_type="VIDEO", numeric_features=NUMERIC, categorical_features=CATEGORICAL),
        "trainingDataTimeRange": {
            "start": df.timestamp.min().isoformat() if len(df) else None,
            "end": df.timestamp.max().isoformat() if len(df) else None,
        },
    }


def train_and_select_cross_family(
    df: pd.DataFrame, *, random_seed: int = RANDOM_SEED, use_sample_weights: bool = TRAINING_USE_SAMPLE_WEIGHTS,
    benchmark_difficulties: tuple[str, ...] | None = None,
    include_production_rankers: bool = TRAINING_ENABLE_PRODUCTION_RANKERS,
    ranker_replicas_per_archetype: int = PRODUCTION_REPLICAS_PER_ARCHETYPE,
    ranker_group_seed: int = PRODUCTION_GROUP_SEED,
) -> dict[str, Any]:
    """XGBRanker promotion task: production model selection across BOTH families -- every
    enabled classifier candidate (trained on the flat point-in-time dataset, exactly as
    `train_and_select` already does via `_build_classifier_pool`) plus every
    app.ml.ranker_registry.PRODUCTION_RANKER_NAMES ranker (trained on an entirely separate
    synthetic ranking-groups dataset, app.ml.ranking_groups -- never the flat classifier
    dataset; see app.ml.ranker_trainer.train_ranking_challengers). Both training methodologies
    stay exactly as they already are; only the SELECTION layer becomes shared
    (app.ml.candidate_pool).

    Selection uses `quality_score_cross_family` -- a pure function of each candidate's own
    independent end-to-end benchmark result (app.ml.candidate_evaluation), identically defined
    and computed for classifier and ranker candidates alike, so neither family's own
    training-time metrics (PR-AUC/precision/recall for a classifier; nothing analogous for a
    ranker) can tilt the comparison. Eligibility (app.ml.eligibility_policy, the unchanged Task
    5 policy) gates both families identically first -- an ineligible candidate can never win
    regardless of its cross-family score.

    Ranker training failure (e.g. a broken optional native extension discovered only at fit
    time) is caught and logged, never allowed to abort classifier evaluation -- production
    selection must still be able to complete with classifiers alone; see
    `productionRankerTrainingError` in the returned dict."""
    df = df.copy()
    # See train_models' identical line above for why compute_replay_weights is multiplied in
    # here too (app.ml.replay_saturation_policy) -- same rationale, same point-in-time safety.
    df[SAMPLE_WEIGHT_COLUMN] = (
        compute_sample_weights(df) * compute_replay_weights(df) if use_sample_weights else 1.0
    )
    difficulties = benchmark_difficulties or SELECTION_DIFFICULTIES

    built = _build_classifier_pool(df, random_seed=random_seed, use_sample_weights=use_sample_weights, difficulties=difficulties)
    pool: dict[str, candidate_pool.TrainedModelCandidate] = {}
    for name in built["candidates"]:
        pool[name] = candidate_pool.TrainedModelCandidate(
            name=name, model=built["calibrated_candidates"][name], model_family="classifier",
            objective="binary_classification", score_semantics="probability",
            eligible=built["eligibility_decisions"][name]["eligible"],
            eligibility=built["eligibility_decisions"][name],
            cross_family_quality=quality_score_cross_family(built["candidate_evaluations"][name]),
            native_quality=built["quality_scores"][name],
            metrics=built["comparison"][name],
            training_duration_seconds=float(built["comparison"][name].get("trainingDurationSeconds") or 0.0),
        )

    ranker_result: dict[str, Any] | None = None
    ranker_error: str | None = None
    if include_production_rankers and PRODUCTION_RANKER_NAMES:
        try:
            ranker_result = train_ranking_challengers(
                random_seed=random_seed, replicas_per_archetype=ranker_replicas_per_archetype,
                group_seed=ranker_group_seed, benchmark_difficulties=difficulties, only=PRODUCTION_RANKER_NAMES,
            )
        except Exception as exc:  # noqa: BLE001 -- a broken optional ranker must never abort classifier selection
            logger.exception("Production ranker training failed; continuing with classifier candidates only.")
            ranker_error = str(exc)
        else:
            for name, scorer in ranker_result["scorers"].items():
                meta = ranker_result["metadata"][name]
                pool[name] = candidate_pool.TrainedModelCandidate(
                    name=name, model=scorer, model_family="ranker", objective=meta["objective"],
                    score_semantics=meta["scoreSemantics"],
                    eligible=ranker_result["eligibilityDecisions"][name]["eligible"],
                    eligibility=ranker_result["eligibilityDecisions"][name],
                    cross_family_quality=quality_score_cross_family(ranker_result["candidateEvaluations"][name]),
                    native_quality=ranker_result["qualityScores"][name],
                    metrics=None,
                    training_duration_seconds=0.0,
                    metadata=meta,
                )

    selection = candidate_pool.select_cross_family_winner(pool)
    selected_name = selection["selected"]
    winner = pool[selected_name]

    if winner.model_family == "classifier":
        tail = _finalize(
            df=df, calibrated=winner.model, train=built["train"], threshold_tuning=built["threshold_tuning"],
            test=built["test"], lifecycle=built["lifecycle"], use_sample_weights=use_sample_weights, random_seed=random_seed,
        )
    else:
        assert ranker_result is not None  # a ranker can only win if ranker training actually ran and succeeded
        tail = _ranker_tail(winner, ranker_result, df=df, flat_train=built["train"])

    return {
        "model": winner.model,
        "selectedModel": selected_name,
        "modelFamily": winner.model_family,
        "selectionCriterion": "crossFamilySelection (XGBRanker promotion task): PHASE 1 eligibility "
                               "(unchanged Task 5 policy, applied identically to classifiers and "
                               "production rankers) then PHASE 2 ranking by quality_score_cross_family "
                               "(app.ml.quality_scorer) -- the SAME formula, using only independent "
                               "end-to-end benchmark evidence, for every candidate regardless of family.",
        "selectionDetails": selection,
        "eligibleSelection": selection["eligibleSelection"],
        "candidatePool": {
            name: {
                "modelFamily": c.model_family, "eligible": c.eligible,
                "crossFamilyScore": c.cross_family_quality["score"],
                "nativeQualityScore": c.native_quality["score"],
            }
            for name, c in pool.items()
        },
        "modelComparison": built["comparison"],
        "eligibility": {
            **built["eligibility"], **(ranker_result["eligibility"] if ranker_result else {}),
        },
        "eligibilitySeverity": {
            **built["severity"], **(ranker_result["eligibilitySeverity"] if ranker_result else {}),
        },
        "candidateEvaluations": {
            **built["candidate_evaluations"], **(ranker_result["candidateEvaluations"] if ranker_result else {}),
        },
        "eligibilityDecisions": {name: c.eligibility for name, c in pool.items()},
        "qualityScores": {name: c.native_quality for name, c in pool.items()},
        "crossFamilyQualityScores": {name: c.cross_family_quality for name, c in pool.items()},
        "unavailableAlgorithms": unavailable_algorithms(),
        "unavailableProductionRankers": unavailable_rankers() if include_production_rankers else {},
        "productionRankerTrainingError": ranker_error,
        "productionRankerNamesConsidered": list(PRODUCTION_RANKER_NAMES) if include_production_rankers else [],
        "modelSelectionVersion": CROSS_FAMILY_SELECTION_VERSION,
        "benchmarkDifficulties": list(difficulties),
        **tail,
    }
