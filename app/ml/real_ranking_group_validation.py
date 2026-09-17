"""Pure trainability validator for a `RealRankingDecision`.

Structural hard-reject conditions (blank/mismatched qid, duplicate content, non-finite/
schema-mismatched features, impossible timestamps) are already enforced -- impossible to bypass
-- at `RealRankingDecision`/`RealCandidateObservation`/`RankedCandidateSnapshot` construction
time (see `app.ml.real_ranking_decision` and `app.ml.ranking_snapshot`). A `RealRankingDecision`
object that exists at all has therefore already passed every one of those checks; this module
only answers the question construction cannot: does this structurally-valid group actually carry
enough RESOLVED, differentiated signal to be useful to an XGBRanker fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.ml.real_ranking_decision import ObservationState, RealRankingDecision


class GroupValidationOutcome(str, Enum):
    VALID_TRAINABLE = "VALID_TRAINABLE"
    # Structurally fine (already guaranteed by construction), but no usable within-group
    # ordering signal -- kept for diagnostics, never fed to a fit.
    VALID_NOT_TRAINABLE = "VALID_NOT_TRAINABLE"


@dataclass(frozen=True, slots=True)
class GroupValidationResult:
    outcome: GroupValidationOutcome
    reasons: tuple[str, ...]
    candidate_count: int
    resolved_candidate_count: int
    distinct_resolved_relevance_count: int


def validate_group(decision: RealRankingDecision) -> GroupValidationResult:
    resolved = [c for c in decision.candidates if c.observation_state is ObservationState.RESOLVED]
    distinct_relevances = {c.resolved_relevance for c in resolved}

    if len(decision.candidates) < 2:
        return GroupValidationResult(
            GroupValidationOutcome.VALID_NOT_TRAINABLE,
            (f"group has {len(decision.candidates)} candidate(s); at least 2 are required for within-group ordering",),
            len(decision.candidates), len(resolved), len(distinct_relevances),
        )
    if len(resolved) == 0:
        return GroupValidationResult(
            GroupValidationOutcome.VALID_NOT_TRAINABLE,
            ("zero resolved candidates -- nothing is known yet",),
            len(decision.candidates), len(resolved), len(distinct_relevances),
        )
    if len(resolved) == 1:
        return GroupValidationResult(
            GroupValidationOutcome.VALID_NOT_TRAINABLE,
            ("only one resolved candidate; remaining candidates are unresolved/not observed and provide no ordering signal",),
            len(decision.candidates), len(resolved), len(distinct_relevances),
        )
    if len(distinct_relevances) < 2:
        return GroupValidationResult(
            GroupValidationOutcome.VALID_NOT_TRAINABLE,
            ("all resolved candidates share identical relevance -- no within-group ordering signal",),
            len(decision.candidates), len(resolved), len(distinct_relevances),
        )
    return GroupValidationResult(
        GroupValidationOutcome.VALID_TRAINABLE, (),
        len(decision.candidates), len(resolved), len(distinct_relevances),
    )
