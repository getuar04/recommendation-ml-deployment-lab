"""Real-data sample-weight policy: the `RealSampleWeightPolicy` port, plus one concrete,
deliberately coarse implementation (`RelevanceTierSampleWeightPolicy`).

Deliberately NOT `app.ml.sample_weight_policy` (the synthetic policy): its constants were swept
and empirically validated against real trained candidates and `app.ml.ranking_groups`' synthetic
archetypes (see that module's own docstring, and `_row_weight`'s comments there) -- a real-data
ranker must never silently inherit those synthetic-tuned NUMBERS just because they happen to
exist. Nothing below reuses a single numeric constant from that module.

ARCHITECTURAL LIMIT THIS IMPLEMENTATION IS DELIBERATELY SHAPED AROUND (read before changing the
weight table): `app.ml.sample_weight_policy._row_weight` -- the synthetic reference this task
was told to inspect for structural ideas, not values -- computes its weight from a training
ROW's raw fields: `event_type`, `event_watch_percentage`, `event_liked`, `event_shared`,
`event_favorited`, `event_creator_followed`. Those raw fields are what let it tell an EXPLICIT
CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED rejection apart from an ambiguous implicit fast-skip
(both otherwise `target==0`), and a share/favorite apart from a creator-follow (both otherwise
"positive"). `app.ml.real_ranking_decision.RealCandidateObservation` -- the ONLY thing this
policy is allowed to read (see POLICY REQUIREMENTS) -- retains none of that: outcome resolution
(`app.ml.real_outcome_resolution.resolve_observation`) collapses every raw event field into a
single `resolved_relevance` int in [0, 4] and then discards the rest. Concretely, from
`resolved_relevance` alone this policy CANNOT tell apart:
  - relevance 0: an explicit CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED rejection vs. an
    ambiguous implicit fast-skip -- two outcomes this codebase's own synthetic policy and
    `app.ml.dataset_builder`'s feature-engineering weights (CONTENT_NOT_INTERESTED_CATEGORY_
    PENALTY=8 vs. a fast-skip's own -4) already treat as meaningfully different confidence.
  - relevance 3: liked vs. a near-full completion.
  - relevance 4: shared vs. favorited vs. a creator-follow.
This is a REAL, provable gap (verified directly against `_row_weight`'s own required inputs
above), not a hypothetical one -- and per this task's own instruction, inventing a distinction
the data does not support would misrepresent what was actually observed. The implementation
below therefore intentionally does NOT attempt any of the four distinctions above: it grades
confidence ONLY on `resolved_relevance`'s own ordinal tier, which IS reliable, always-present,
non-inferred evidence (see `_grade_relevance` semantics documented in
`app.ml.real_outcome_resolution`). Closing the finer-grained gaps above would need a genuinely
small, additive extension to `RealCandidateObservation` (e.g. an optional
`dominant_evidence: str` field recording which discrete category -- EXPLICIT_REJECTION /
FAST_SKIP / SHARE_OR_FAVORITE_OR_FOLLOW / LIKE / COMPLETION / STRONG_WATCH / WEAK_WATCH --
`resolve_observation` actually observed, alongside `resolved_relevance`, never replacing it).
That extension is NOT implemented here -- it touches a different module's frozen contract and
needs its own explicit sign-off; this docstring is that proposal, not the implementation.

NOT EMPIRICALLY VALIDATED (unlike the synthetic policy): no real trained candidate, real
eligibility-gate regression suite, or real distribution exists yet to sweep these weights
against -- see `app.ml.real_data_shadow`'s own module docstring for the same "nothing here has
been proven against anything real yet" posture. The three weights below are therefore a modest,
round-number, monotonic STARTING POINT grounded only in `resolved_relevance`'s own already-
documented qualitative tiers ("negative/unwanted" / "weak" / "moderate" / "strong" /
"exceptional" -- see `app.ml.real_outcome_resolution`'s docstring), not a guess at precision the
codebase's synthetic sweep-and-measure process (see `app.ml.sample_weight_policy`'s own
docstring) would require before treating a value as final. A future real-data eligibility sweep
replaces these, exactly like the synthetic policy's own history.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from app.ml.real_ranking_decision import ExposureState, ObservationState

if TYPE_CHECKING:
    from app.ml.real_ranking_decision import (
        RealCandidateObservation,
        RealRankingDecision,
    )

__all__ = [
    "RELEVANCE_TIER_SAMPLE_WEIGHT_POLICY",
    "RealSampleWeightPolicy",
    "RelevanceTierSampleWeightPolicy",
    "apply_weight",
    "apply_weight_to_decision",
]


class RealSampleWeightPolicy(Protocol):
    """Contract a real-data weight policy must implement."""

    def weight_for(self, observation: RealCandidateObservation) -> float: ...


# Deliberately three values, not five: relevance 0 and 1 share BASE_WEIGHT because relevance 0
# is itself an unresolvable mix of a highly-decisive explicit rejection and a weakly-decisive
# implicit fast-skip (see module docstring) -- assuming the more dramatic (explicit-rejection)
# interpretation just because a similarly-elevated weight WOULD be justified if we could confirm
# it would be exactly the invented-information problem this task warns against. Not elevating it
# above the weak-positive baseline is the conservative, evidence-honest choice. Relevance 2 is
# its own single-cause tier ("solid watch, no stronger signal") and gets a modest step up;
# relevance 3-4 (liked/completion/share/favorite/follow) are uniformly strong, explicit,
# decisive signals regardless of which specific one occurred, and share the top tier.
BASE_WEIGHT = 1.0
MODERATE_CONFIDENCE_WEIGHT = 1.25
HIGH_CONFIDENCE_WEIGHT = 1.5

_HIGH_CONFIDENCE_RELEVANCE = frozenset({3, 4})


class RelevanceTierSampleWeightPolicy:
    """Concrete `RealSampleWeightPolicy`: weight is a function of `resolved_relevance`'s
    ordinal tier only -- see module docstring for exactly what this can and cannot distinguish,
    and why. Stateless; safe to share via the `RELEVANCE_TIER_SAMPLE_WEIGHT_POLICY` singleton
    below."""

    def weight_for(self, observation: RealCandidateObservation) -> float:
        if observation.observation_state is not ObservationState.RESOLVED:
            raise ValueError(
                f"weight_for requires a RESOLVED observation, got observation_state="
                f"{observation.observation_state.value!r} -- an unresolved/censored/never-"
                "observed candidate has no outcome to weight"
            )
        if observation.exposure_state is not ExposureState.IMPRESSED:
            raise ValueError(
                f"weight_for requires an IMPRESSED observation, got exposure_state="
                f"{observation.exposure_state.value!r}"
            )
        # RealCandidateObservation.__post_init__ already guarantees resolved_relevance is a
        # non-None int in [0, 4] whenever observation_state is RESOLVED -- see
        # app.ml.real_ranking_decision. Not re-validated here, matching
        # app.ml.real_ranking_dataset_builder's identical use of this same guarantee.
        relevance = observation.resolved_relevance
        assert relevance is not None
        if relevance in _HIGH_CONFIDENCE_RELEVANCE:
            return HIGH_CONFIDENCE_WEIGHT
        if relevance == 2:
            return MODERATE_CONFIDENCE_WEIGHT
        return BASE_WEIGHT


RELEVANCE_TIER_SAMPLE_WEIGHT_POLICY = RelevanceTierSampleWeightPolicy()


def apply_weight(
    observation: RealCandidateObservation, policy: RealSampleWeightPolicy = RELEVANCE_TIER_SAMPLE_WEIGHT_POLICY,
) -> RealCandidateObservation:
    """The ONE place a real `RealCandidateObservation.resolved_weight` is ever set from a
    policy -- the single authoritative integration point between outcome resolution
    (`app.ml.real_outcome_resolution`, which always leaves `resolved_weight=None`) and dataset
    construction (`app.ml.real_ranking_dataset_builder`/`app.ml.real_ranker_trainer`, neither of
    which computes a weight themselves; see their own docstrings). Computes nothing itself --
    `policy.weight_for(observation)` is the only calculation; this function only threads its
    result through.

    `RealCandidateObservation` is frozen (`app.ml.real_ranking_decision`) -- this never mutates
    `observation`. It returns a NEW instance via `dataclasses.replace`; every field except
    `resolved_weight` (snapshot, exposure_state, observation_state, observed_at,
    resolved_relevance) is carried over byte-identical.

    A non-RESOLVED observation (NOT_OBSERVED/NOT_YET_RESOLVED/CENSORED) is returned UNCHANGED,
    not an error: there is nothing to weight, and `policy.weight_for` already enforces the
    RESOLVED+IMPRESSED precondition for the one case that IS weighable. This pass-through lets a
    caller apply weighting uniformly across a `RealRankingDecision`'s full candidate list (a
    natural mix of resolved and unresolved candidates -- see `apply_weight_to_decision` below)
    without pre-filtering by state.

    Refuses to silently overwrite an observation that already carries a `resolved_weight`
    (raises `ValueError`) -- an accidental double-application, or an accidental collision with a
    caller-supplied custom weight, must never be silent. A genuine re-weight requires the caller
    to `dataclasses.replace(observation, resolved_weight=None)` first, making the intent explicit.
    """
    if observation.observation_state is not ObservationState.RESOLVED:
        return observation
    if observation.resolved_weight is not None:
        raise ValueError(
            "observation already has a resolved_weight -- apply_weight refuses to silently "
            "overwrite an existing value; replace(observation, resolved_weight=None) first if "
            "re-weighting is genuinely intended"
        )
    weight = policy.weight_for(observation)
    return replace(observation, resolved_weight=weight)


def apply_weight_to_decision(
    decision: RealRankingDecision, policy: RealSampleWeightPolicy = RELEVANCE_TIER_SAMPLE_WEIGHT_POLICY,
) -> RealRankingDecision:
    """Batch convenience: `apply_weight` mapped across every candidate in `decision`, returning
    a NEW `RealRankingDecision` (also frozen -- see `app.ml.real_ranking_decision`). No separate
    weighting logic lives here; `apply_weight` remains the only place a weight is computed."""
    return replace(decision, candidates=tuple(apply_weight(candidate, policy) for candidate in decision.candidates))
