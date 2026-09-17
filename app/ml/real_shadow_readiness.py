"""Real shadow readiness gate: summarizes whether a real-data challenger's evidence
(`app.ml.real_shadow_evaluation.RealShadowEvaluationResult`) is complete and consistent enough
to present to a human/lead for review.

NOT a promotion gate. `READY_FOR_REVIEW` means exactly "safe to present to a human/lead for
evaluation" -- it never means "safe to promote automatically", and no status this module can
return authorizes a save, promote, rollback, retrain, or any active-model change. This module
performs no I/O, no training, no scoring, no model construction: it only reads the fields
already present on an already-produced `RealShadowEvaluationResult` (plus the caller-supplied
`shadow_status`/`human_confirmed_sufficient_volume` that result now carries -- see
`app.ml.real_shadow_evaluation`'s own recent extension) and an explicit caller acknowledgement.

WHY EACH INPUT FIELD IS TRUSTED, NOT RE-DERIVED (evidence, not invention):
  - `training_data_source` / `shadow_status`: both are STRUCTURAL guarantees of the pipeline
    that produces a `RealShadowEvaluationResult` at all (`fit_real_ranker`'s own return type
    can only ever declare `TrainingDataSource.REAL_OBSERVED`/`ShadowTrainingStatus.SHADOW_ONLY`
    -- see `app.ml.real_ranker_trainer.RealRankerShadowTrainingResult.__post_init__`) -- checked
    here explicitly anyway, never assumed, in case a future caller constructs one differently.
  - `human_confirmed_sufficient_volume` / `group_isolation_verified`: also structural guarantees
    once a `RealShadowEvaluationResult` exists at all (`prepare_real_ranker_training_input`
    raises `RealDataVolumeNotConfirmed` before construction when false; `chronological_group_
    split` cannot produce an overlapping partition -- see `app.ml.real_shadow_evaluation`'s own
    `assert`) -- re-checked here for the same defense-in-depth reason.
  - baseline availability / evaluation completion / no-regression: read directly from
    `evaluation.status` (`BASELINE_UNAVAILABLE`/`INSUFFICIENT_DATA`/`FAIL`/`PASS`) and
    `evaluation.candidate_metrics`/`baseline_metrics` -- never re-computed, never re-scored.

ACCEPTANCE-POLICY ACKNOWLEDGEMENT IS A REQUIRED, EXPLICIT CALLER INPUT, not a passive note:
mirrors this codebase's own established pattern for exactly this kind of irreducible human
judgment call (`human_confirmed_sufficient_volume` -- see `app.ml.real_data_readiness`'s own
module docstring: "never inferred from a hardcoded... threshold"). `acknowledge_no_promotion_
policy` must be explicitly `True`, or the gate returns `ACCEPTANCE_POLICY_MISSING` regardless of
how strong the underlying evidence is -- READY_FOR_REVIEW can never be reached silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.ml.real_data_shadow import ShadowTrainingStatus
from app.ml.real_shadow_evaluation import (
    RealShadowEvaluationResult,
    RealShadowEvaluationStatus,
)
from app.ml.training_data_source import TrainingDataSource

__all__ = [
    "RealShadowReadinessResult",
    "RealShadowReadinessStatus",
    "assess_real_shadow_readiness",
]

ACCEPTANCE_POLICY_NOTE = (
    "No authoritative promotion/acceptance policy exists in this repository for real-data "
    "shadow candidates (app.ml.quality_scorer, app.ml.eligibility_policy, and app.benchmark.* "
    "were searched -- none define a candidate-vs-persisted-baseline regression/promotion "
    "threshold; see app.ml.real_shadow_evaluation's own module docstring for that search). "
    "READY_FOR_REVIEW means 'safe to present to a human/lead for evaluation', never 'safe to "
    "promote automatically' -- no status this module returns authorizes promotion, saving, "
    "rollback, retraining, or any active-model change."
)


class RealShadowReadinessStatus(str, Enum):
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    BASELINE_UNAVAILABLE = "BASELINE_UNAVAILABLE"
    ACCEPTANCE_POLICY_MISSING = "ACCEPTANCE_POLICY_MISSING"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    NOT_REAL_OBSERVED = "NOT_REAL_OBSERVED"
    NOT_SHADOW_ONLY = "NOT_SHADOW_ONLY"


@dataclass(frozen=True, slots=True)
class RealShadowReadinessResult:
    status: RealShadowReadinessStatus
    reason: str
    algorithm: str
    training_data_source_confirmed: bool
    shadow_only_confirmed: bool
    human_confirmed_sufficient_volume: bool
    group_isolation_verified: bool
    baseline_available: bool
    candidate_evaluation_completed: bool
    no_metric_regression: bool
    acceptance_policy_acknowledged: bool
    acceptance_policy_note: str
    # The full underlying evidence this assessment was computed from, always attached so a
    # human reviewer (or a future audit) never has to trust a summary without the source.
    evaluation: RealShadowEvaluationResult


def assess_real_shadow_readiness(
    evaluation: RealShadowEvaluationResult, *, acknowledge_no_promotion_policy: bool,
) -> RealShadowReadinessResult:
    """Pure function: reads `evaluation`'s already-computed fields plus one explicit caller
    acknowledgement, returns a structured readiness assessment. Performs no I/O, training,
    scoring, saving, or promotion of any kind."""
    training_data_source_confirmed = evaluation.training_data_source is TrainingDataSource.REAL_OBSERVED
    shadow_only_confirmed = evaluation.shadow_status is ShadowTrainingStatus.SHADOW_ONLY
    human_confirmed_sufficient_volume = evaluation.human_confirmed_sufficient_volume
    group_isolation_verified = evaluation.group_isolation_verified
    baseline_available = (
        evaluation.status not in (RealShadowEvaluationStatus.BASELINE_UNAVAILABLE, RealShadowEvaluationStatus.INSUFFICIENT_DATA)
        and evaluation.baseline_metrics is not None
    )
    candidate_evaluation_completed = evaluation.candidate_metrics is not None
    no_metric_regression = evaluation.status is RealShadowEvaluationStatus.PASS

    if not training_data_source_confirmed:
        return RealShadowReadinessResult(
            status=RealShadowReadinessStatus.NOT_REAL_OBSERVED,
            reason=f"evaluation.training_data_source={evaluation.training_data_source!r} is not REAL_OBSERVED",
            algorithm=evaluation.algorithm,
            training_data_source_confirmed=training_data_source_confirmed, shadow_only_confirmed=shadow_only_confirmed,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            group_isolation_verified=group_isolation_verified, baseline_available=baseline_available,
            candidate_evaluation_completed=candidate_evaluation_completed, no_metric_regression=no_metric_regression,
            acceptance_policy_acknowledged=acknowledge_no_promotion_policy, acceptance_policy_note=ACCEPTANCE_POLICY_NOTE,
            evaluation=evaluation,
        )

    if not shadow_only_confirmed:
        return RealShadowReadinessResult(
            status=RealShadowReadinessStatus.NOT_SHADOW_ONLY,
            reason=f"evaluation.shadow_status={evaluation.shadow_status!r} is not SHADOW_ONLY",
            algorithm=evaluation.algorithm,
            training_data_source_confirmed=training_data_source_confirmed, shadow_only_confirmed=shadow_only_confirmed,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            group_isolation_verified=group_isolation_verified, baseline_available=baseline_available,
            candidate_evaluation_completed=candidate_evaluation_completed, no_metric_regression=no_metric_regression,
            acceptance_policy_acknowledged=acknowledge_no_promotion_policy, acceptance_policy_note=ACCEPTANCE_POLICY_NOTE,
            evaluation=evaluation,
        )

    if not human_confirmed_sufficient_volume or not group_isolation_verified:
        return RealShadowReadinessResult(
            status=RealShadowReadinessStatus.INSUFFICIENT_DATA,
            reason=(
                "human_confirmed_sufficient_volume is False" if not human_confirmed_sufficient_volume
                else "group_isolation_verified is False"
            ),
            algorithm=evaluation.algorithm,
            training_data_source_confirmed=training_data_source_confirmed, shadow_only_confirmed=shadow_only_confirmed,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            group_isolation_verified=group_isolation_verified, baseline_available=baseline_available,
            candidate_evaluation_completed=candidate_evaluation_completed, no_metric_regression=no_metric_regression,
            acceptance_policy_acknowledged=acknowledge_no_promotion_policy, acceptance_policy_note=ACCEPTANCE_POLICY_NOTE,
            evaluation=evaluation,
        )

    if evaluation.status is RealShadowEvaluationStatus.INSUFFICIENT_DATA:
        return RealShadowReadinessResult(
            status=RealShadowReadinessStatus.INSUFFICIENT_DATA, reason=evaluation.reason,
            algorithm=evaluation.algorithm,
            training_data_source_confirmed=training_data_source_confirmed, shadow_only_confirmed=shadow_only_confirmed,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            group_isolation_verified=group_isolation_verified, baseline_available=baseline_available,
            candidate_evaluation_completed=candidate_evaluation_completed, no_metric_regression=no_metric_regression,
            acceptance_policy_acknowledged=acknowledge_no_promotion_policy, acceptance_policy_note=ACCEPTANCE_POLICY_NOTE,
            evaluation=evaluation,
        )

    if evaluation.status is RealShadowEvaluationStatus.BASELINE_UNAVAILABLE:
        return RealShadowReadinessResult(
            status=RealShadowReadinessStatus.BASELINE_UNAVAILABLE, reason=evaluation.reason,
            algorithm=evaluation.algorithm,
            training_data_source_confirmed=training_data_source_confirmed, shadow_only_confirmed=shadow_only_confirmed,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            group_isolation_verified=group_isolation_verified, baseline_available=baseline_available,
            candidate_evaluation_completed=candidate_evaluation_completed, no_metric_regression=no_metric_regression,
            acceptance_policy_acknowledged=acknowledge_no_promotion_policy, acceptance_policy_note=ACCEPTANCE_POLICY_NOTE,
            evaluation=evaluation,
        )

    if evaluation.status is RealShadowEvaluationStatus.FAIL:
        return RealShadowReadinessResult(
            status=RealShadowReadinessStatus.EVALUATION_FAILED, reason=evaluation.reason,
            algorithm=evaluation.algorithm,
            training_data_source_confirmed=training_data_source_confirmed, shadow_only_confirmed=shadow_only_confirmed,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            group_isolation_verified=group_isolation_verified, baseline_available=baseline_available,
            candidate_evaluation_completed=candidate_evaluation_completed, no_metric_regression=no_metric_regression,
            acceptance_policy_acknowledged=acknowledge_no_promotion_policy, acceptance_policy_note=ACCEPTANCE_POLICY_NOTE,
            evaluation=evaluation,
        )

    if not acknowledge_no_promotion_policy:
        return RealShadowReadinessResult(
            status=RealShadowReadinessStatus.ACCEPTANCE_POLICY_MISSING,
            reason=(
                "acknowledge_no_promotion_policy=False -- every evidence check passed, but this "
                "gate refuses to reach READY_FOR_REVIEW without the caller explicitly "
                "acknowledging that no authoritative promotion policy exists"
            ),
            algorithm=evaluation.algorithm,
            training_data_source_confirmed=training_data_source_confirmed, shadow_only_confirmed=shadow_only_confirmed,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            group_isolation_verified=group_isolation_verified, baseline_available=baseline_available,
            candidate_evaluation_completed=candidate_evaluation_completed, no_metric_regression=no_metric_regression,
            acceptance_policy_acknowledged=acknowledge_no_promotion_policy, acceptance_policy_note=ACCEPTANCE_POLICY_NOTE,
            evaluation=evaluation,
        )

    # evaluation.status is PASS here (the only remaining RealShadowEvaluationStatus value) and
    # every other gate above already passed.
    return RealShadowReadinessResult(
        status=RealShadowReadinessStatus.READY_FOR_REVIEW,
        reason="all readiness checks passed -- safe to present to a human/lead for evaluation (not a promotion decision)",
        algorithm=evaluation.algorithm,
        training_data_source_confirmed=training_data_source_confirmed, shadow_only_confirmed=shadow_only_confirmed,
        human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
        group_isolation_verified=group_isolation_verified, baseline_available=baseline_available,
        candidate_evaluation_completed=candidate_evaluation_completed, no_metric_regression=no_metric_regression,
        acceptance_policy_acknowledged=acknowledge_no_promotion_policy, acceptance_policy_note=ACCEPTANCE_POLICY_NOTE,
        evaluation=evaluation,
    )
