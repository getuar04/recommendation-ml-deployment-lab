"""Pure readiness report for a collection of real ranking decisions.

Deliberately NOT a "do we have enough data" gate -- see `sufficient_volume_for_model_training`
below, which this module can never set to True by itself. This only measures facts; deciding
whether those facts represent enough real-world evidence to train on is left to a human/product
decision informed by this report (see the real-data training research report, Section M, for
what those facts should be weighed against).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.ml.real_ranking_decision import (
    ExposureState,
    ObservationState,
    RealRankingDecision,
)
from app.ml.real_ranking_group_validation import GroupValidationOutcome, validate_group


@dataclass(frozen=True, slots=True)
class RealDataReadinessReport:
    ranking_decision_count: int
    total_candidate_count: int

    trainable_group_count: int
    excluded_group_count: int  # VALID_NOT_TRAINABLE -- structurally fine, diagnostics only

    ranked_candidate_count: int  # exposure_state == RANKED (never served)
    served_candidate_count: int  # exposure_state == SERVED (served, not confirmed impressed)
    impressed_candidate_count: int  # exposure_state == IMPRESSED

    resolved_candidate_count: int
    not_yet_resolved_candidate_count: int
    censored_candidate_count: int

    candidates_per_qid: tuple[int, ...]
    resolved_labels_per_qid: tuple[int, ...]

    technically_valid_for_dataset_building: bool

    # Deliberately always None: whether the above facts represent ENOUGH real-world evidence to
    # train on is a human/product decision informed by this report, never fabricated from a
    # hardcoded row-count threshold. See app.ml.real_ranker_trainer's
    # `human_confirmed_sufficient_volume` parameter for where that decision is actually supplied.
    sufficient_volume_for_model_training: None = None


def evaluate_readiness(decisions: Sequence[RealRankingDecision]) -> RealDataReadinessReport:
    trainable_group_count = 0
    excluded_group_count = 0
    ranked_candidate_count = 0
    served_candidate_count = 0
    impressed_candidate_count = 0
    resolved_candidate_count = 0
    not_yet_resolved_candidate_count = 0
    censored_candidate_count = 0
    candidates_per_qid: list[int] = []
    resolved_labels_per_qid: list[int] = []

    for decision in decisions:
        result = validate_group(decision)
        if result.outcome is GroupValidationOutcome.VALID_TRAINABLE:
            trainable_group_count += 1
        else:
            excluded_group_count += 1

        candidates_per_qid.append(len(decision.candidates))
        resolved_labels_per_qid.append(result.resolved_candidate_count)

        for candidate in decision.candidates:
            if candidate.exposure_state is ExposureState.RANKED:
                ranked_candidate_count += 1
            elif candidate.exposure_state is ExposureState.SERVED:
                served_candidate_count += 1
            else:
                impressed_candidate_count += 1

            if candidate.observation_state is ObservationState.RESOLVED:
                resolved_candidate_count += 1
            elif candidate.observation_state is ObservationState.NOT_YET_RESOLVED:
                not_yet_resolved_candidate_count += 1
            elif candidate.observation_state is ObservationState.CENSORED:
                censored_candidate_count += 1

    total_candidate_count = ranked_candidate_count + served_candidate_count + impressed_candidate_count

    return RealDataReadinessReport(
        ranking_decision_count=len(decisions),
        total_candidate_count=total_candidate_count,
        trainable_group_count=trainable_group_count,
        excluded_group_count=excluded_group_count,
        ranked_candidate_count=ranked_candidate_count,
        served_candidate_count=served_candidate_count,
        impressed_candidate_count=impressed_candidate_count,
        resolved_candidate_count=resolved_candidate_count,
        not_yet_resolved_candidate_count=not_yet_resolved_candidate_count,
        censored_candidate_count=censored_candidate_count,
        candidates_per_qid=tuple(candidates_per_qid),
        resolved_labels_per_qid=tuple(resolved_labels_per_qid),
        technically_valid_for_dataset_building=trainable_group_count > 0,
    )
