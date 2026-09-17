"""Ranking-native challenger training (Task 7) -- CHALLENGER ONLY, never wired into
app.ml.trainer.train_and_select's production classifier selection (spec section 23: rankers
stay outside the dynamic candidate registry until fully verified and explicitly enabled).

Fits each available ranker (app.ml.ranker_registry) on the synthetic ranking groups
(app.ml.ranking_groups), wraps it in app.ml.ranker_adapter.RankerScorer so it speaks the exact
same `predict_proba` contract every classifier already does, then runs it through the UNCHANGED
Task 5 layered-evaluation machinery (app.ml.gate_severity/app.ml.eligibility_policy/
app.ml.quality_scorer/app.ml.candidate_evaluation) -- no gate, benchmark, or selection logic is
duplicated or modified for rankers.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from app.ml import eligibility_policy, gate_severity, quality_scorer
from app.ml.candidate_evaluation import SELECTION_DIFFICULTIES, evaluate_candidate
from app.ml.dataset_builder import CATEGORICAL, NUMERIC
from app.ml.eligibility import evaluate_eligibility
from app.ml.pipeline_builder import build_classifier_pipeline, ranker_fit_params
from app.ml.ranker_adapter import RankerScorer, compute_normalization_scale
from app.ml.ranker_registry import build_candidate_rankers, unavailable_rankers
from app.ml.ranking_groups import (
    PRODUCTION_GROUP_SEED,
    PRODUCTION_REPLICAS_PER_ARCHETYPE,
    RANKING_GROUP_VERSION,
    build_ranking_groups,
    group_sizes,
    per_group_weights,
)

# Chronological, GROUP-preserving split (never splits one query_id's rows across sets) --
# mirrors app.ml.split_lifecycle's "data only ever moves forward in time" discipline at a
# smaller scale appropriate for a challenger-only experiment (no threshold-tuning split: a
# ranker is never binarized/thresholded, see spec section 9).
TRAIN_GROUP_FRACTION = 0.70
SCALE_GROUP_FRACTION = 0.15  # the calibration-analogous split RankerScorer's normalization scale is computed from.


def _group_preserving_split(groups_df: Any) -> tuple[Any, Any, Any]:
    query_order = groups_df.drop_duplicates("query_id").sort_values("timestamp")["query_id"].tolist()
    n = len(query_order)
    train_end = int(n * TRAIN_GROUP_FRACTION)
    scale_end = train_end + max(1, int(n * SCALE_GROUP_FRACTION))
    train_ids = set(query_order[:train_end])
    scale_ids = set(query_order[train_end:scale_end])
    test_ids = set(query_order[scale_end:])
    train = groups_df[groups_df.query_id.isin(train_ids)].reset_index(drop=True)
    scale_split = groups_df[groups_df.query_id.isin(scale_ids)].reset_index(drop=True)
    test = groups_df[groups_df.query_id.isin(test_ids)].reset_index(drop=True)
    return train, scale_split, test


def _training_result_stub(algorithm: str, train_rows: int, test_rows: int) -> dict[str, Any]:
    """The minimal fields app.benchmark.runner.use_model's artifact metadata reads -- see
    app.ml.candidate_evaluation.evaluate_candidate's own docstring: this benchmark artifact is
    never promoted/served for real, so it does not need to be a full production metadata dict."""
    return {
        "splitLifecycleDescription": f"Ranking-group challenger split (rankingGroupVersion={RANKING_GROUP_VERSION}): "
                                      f"{int(TRAIN_GROUP_FRACTION * 100)}% train / {int(SCALE_GROUP_FRACTION * 100)}% "
                                      "scale-normalization / remainder held out, group-preserving, chronological.",
        "decisionThreshold": 0.5,
        "splitSizes": {"train": train_rows, "modelSelection": 0, "calibration": 0, "thresholdTuning": 0, "test": test_rows},
        "classDistribution": {},
        "metrics": {},
        "modelComparison": {},
    }


def train_ranking_challengers(
    *, random_seed: int = 42, replicas_per_archetype: int = PRODUCTION_REPLICAS_PER_ARCHETYPE,
    group_seed: int = PRODUCTION_GROUP_SEED,
    benchmark_difficulties: tuple[str, ...] = SELECTION_DIFFICULTIES, only: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Trains every available ranker challenger on one shared set of synthetic ranking groups,
    evaluates each through the unchanged Task 5 layered pipeline, and returns a per-algorithm
    result dict shaped like enough of app.ml.trainer.train_and_select's own per-candidate
    fields (`eligibility`, `eligibilitySeverity`, `candidateEvaluations`, `eligibilityDecisions`,
    `qualityScores`) that a caller can report classifiers and rankers side by side without a
    second reporting path. Never touches app.ml.trainer/app.ml.algorithm_registry -- this is a
    fully separate, additive training entrypoint.

    `only`, when given, restricts training to just those ranker names (still subject to the
    same enabled/available checks; see app.ml.ranker_registry.build_candidate_rankers) --
    used by app.ml.trainer.train_and_select_cross_family to train only
    app.ml.ranker_registry.PRODUCTION_RANKER_NAMES rather than every challenger.
    """
    groups_df = build_ranking_groups(replicas_per_archetype=replicas_per_archetype, seed=group_seed)
    train, scale_split, test = _group_preserving_split(groups_df)

    rankers = build_candidate_rankers(random_seed, only=only)
    feature_columns = CATEGORICAL + list(NUMERIC)

    eligibility: dict[str, Any] = {}
    eligibility_severity: dict[str, Any] = {}
    candidate_evaluations: dict[str, Any] = {}
    eligibility_decisions: dict[str, Any] = {}
    quality_scores: dict[str, Any] = {}
    scorers: dict[str, RankerScorer] = {}
    metadata: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="ranker_challenger_") as tmp:
        tmp_dir = Path(tmp)
        for name, (estimator, group_param_name, weight_granularity) in rankers.items():
            pipeline = build_classifier_pipeline(estimator, categorical=CATEGORICAL, numeric=list(NUMERIC), scale_numeric=False)
            group_values = group_sizes(train) if group_param_name == "group" else train["query_id"].to_numpy()
            weights = (
                train["sample_weight"].to_numpy() if weight_granularity == "per_row"
                else per_group_weights(train)
            )
            fit_params = ranker_fit_params(group_param_name=group_param_name, group_values=group_values, sample_weight=weights)
            pipeline.fit(train[feature_columns], train["relevance"], **fit_params)

            scale_raw_scores = pipeline.predict(scale_split[feature_columns])
            scale = compute_normalization_scale(scale_raw_scores)
            scorer = RankerScorer(pipeline, scale=scale, algorithm_name=name)
            scorers[name] = scorer

            eligibility[name] = evaluate_eligibility(scorer)
            eligibility_severity[name] = gate_severity.severity_report_from_eligibility(eligibility[name])

            training_result_stub = _training_result_stub(name, len(train), len(test))
            candidate_evaluations[name] = evaluate_candidate(
                name, scorer, training_result_stub, tmp_dir, difficulties=benchmark_difficulties,
            )
            eligibility_decisions[name] = eligibility_policy.decide(eligibility_severity[name], candidate_evaluations[name])
            quality_scores[name] = quality_scorer.quality_score_for_ranker(candidate_evaluations[name])

            metadata[name] = {
                "algorithm": name,
                "modelFamily": "ranker",
                "objective": type(estimator).__name__,
                "featureSchema": feature_columns,
                "scoreSemantics": scorer.score_semantics,
                "normalizationStrategy": scorer.normalization_strategy,
                "normalizationScale": scale,
                "rankingGroupVersion": RANKING_GROUP_VERSION,
                "groupParamName": group_param_name,
            }

    return {
        "scorers": scorers,
        "eligibility": eligibility,
        "eligibilitySeverity": eligibility_severity,
        "candidateEvaluations": candidate_evaluations,
        "eligibilityDecisions": eligibility_decisions,
        "qualityScores": quality_scores,
        "metadata": metadata,
        "unavailableRankers": unavailable_rankers(),
        "rankingGroupVersion": RANKING_GROUP_VERSION,
        "trainGroups": train.query_id.nunique(),
        "scaleGroups": scale_split.query_id.nunique(),
        "testGroups": test.query_id.nunique(),
    }
