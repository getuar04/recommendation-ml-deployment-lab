"""Domain types for a future REAL (not synthetic) XGBRanker ranking decision.

Composes the existing, already-immutable `app.ml.ranking_snapshot.RankedCandidateSnapshot` (the
frozen T0 feature/score trace `app.services.recommendation_service.recommend()` can already
build in production -- see `app.ml.ranking_snapshot_capture`) rather than duplicating a second
feature-snapshot concept. This module adds only what real-data training needs on top of that
existing foundation: an authoritative group identity, and an explicit exposure/observation state
machine that makes it structurally impossible to construct a candidate that would silently read
as a negative label before its outcome is actually known.

Nothing here is wired into serving or into any current training entrypoint. Nothing here invents
a real Feed/Candidate/ETS/UBS field name -- `ranking_group_id`/`observed_at` are RMS-internal
placeholders for identities/timestamps a future confirmed contract will supply.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from app.ml.ranking_snapshot import RankedCandidateSnapshot

# Reuses the same 0-4 graded-relevance convention already established for XGBRanker groups in
# this codebase (see app.ml.ranking_groups.relevance_grade's docstring) -- not a new invented
# scale. Defined locally (not imported from app.ml.ranking_groups) so this real-data domain
# module has zero import dependency on the synthetic generator: the isolation this task requires
# must hold at the module-graph level, not just by convention.
MIN_RESOLVED_RELEVANCE = 0
MAX_RESOLVED_RELEVANCE = 4


class ExposureState(str, Enum):
    """The FURTHEST exposure stage a candidate is known to have reached -- mutually exclusive
    and exhaustive by construction (never "ranked AND served" as two separate flags), so a
    diagnostic reading ranked_candidate_count/served_candidate_count/impressed_candidate_count
    never double-counts one candidate under two states."""

    RANKED = "RANKED"  # RMS scored it; nothing further is known (never served, or unknown)
    SERVED = "SERVED"  # included in what was served; not confirmed impressed
    IMPRESSED = "IMPRESSED"  # confirmed client-side exposure


class ObservationState(str, Enum):
    """Whether -- and how -- an outcome has been resolved for an IMPRESSED candidate.

    RANKED/SERVED candidates are never impressed, so they can only ever be NOT_OBSERVED (see
    RealCandidateObservation's invariants below) -- there is no code path that lets "never
    impressed" collapse into "resolved negative"."""

    NOT_OBSERVED = "NOT_OBSERVED"  # ranked/served, never impressed -- permanently no signal
    NOT_YET_RESOLVED = "NOT_YET_RESOLVED"  # impressed; outcome/observation window still open
    CENSORED = "CENSORED"  # impressed; a future observation-window policy closed this without a decisive outcome
    RESOLVED = "RESOLVED"  # impressed; a decisive relevance outcome is attached


def _parse_aware_timestamp(value: str, *, field: str) -> datetime:
    """Mirrors app.ml.ranking_snapshot.RankingSnapshotBuilder.build's own timestamp parsing
    convention exactly (ISO-8601, timezone-aware required) -- one definition of "valid ranking
    timestamp" for this codebase, not a second, independently-written parser."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed


@dataclass(frozen=True, slots=True)
class RealCandidateObservation:
    """One candidate's complete exposure/outcome trace within a real ranking decision.

    `snapshot` is the frozen T0 record (see module docstring) -- this class never recomputes or
    re-derives anything from it, only attaches what happened AFTER T0. `resolved_relevance`/
    `resolved_weight` are supplied by a future, out-of-scope resolution layer (mapping raw
    outcome events into a relevance grade is a product decision this task deliberately does not
    make -- see the real-data training research report's label-semantics section); this type
    only enforces that such a value can never appear except on a genuinely RESOLVED, IMPRESSED
    candidate.
    """

    snapshot: RankedCandidateSnapshot
    exposure_state: ExposureState
    observation_state: ObservationState
    observed_at: str | None = None
    resolved_relevance: int | None = None
    resolved_weight: float | None = None

    def __post_init__(self) -> None:
        if self.exposure_state in (ExposureState.RANKED, ExposureState.SERVED):
            if self.observation_state is not ObservationState.NOT_OBSERVED:
                raise ValueError(
                    f"exposure_state={self.exposure_state.value} (never impressed) must have "
                    "observation_state=NOT_OBSERVED -- a candidate that was never confirmed "
                    "impressed must never carry a resolved/censored/pending outcome"
                )
            if self.observed_at is not None or self.resolved_relevance is not None or self.resolved_weight is not None:
                raise ValueError(
                    f"exposure_state={self.exposure_state.value} must not carry observed_at/"
                    "resolved_relevance/resolved_weight -- nothing was ever observed"
                )
            return

        # IMPRESSED from here on.
        if self.observation_state is ObservationState.NOT_OBSERVED:
            raise ValueError("exposure_state=IMPRESSED cannot have observation_state=NOT_OBSERVED -- it WAS observed")
        if not self.observed_at or not self.observed_at.strip():
            raise ValueError("an IMPRESSED candidate must carry a non-blank observed_at timestamp")
        observed = _parse_aware_timestamp(self.observed_at, field="observed_at")
        ranking_ts = _parse_aware_timestamp(self.snapshot.ranking_timestamp, field="snapshot.ranking_timestamp")
        if observed < ranking_ts:
            raise ValueError(
                f"observed_at ({self.observed_at}) precedes this candidate's own ranking_timestamp "
                f"({self.snapshot.ranking_timestamp}) -- impossible temporal ordering"
            )

        if self.observation_state in (ObservationState.NOT_YET_RESOLVED, ObservationState.CENSORED):
            if self.resolved_relevance is not None or self.resolved_weight is not None:
                raise ValueError(
                    f"observation_state={self.observation_state.value} must not carry a "
                    "resolved_relevance/resolved_weight -- an unresolved or censored candidate "
                    "must never be silently labeled"
                )
            return

        # RESOLVED from here on.
        if self.resolved_relevance is None:
            raise ValueError("observation_state=RESOLVED requires a resolved_relevance")
        if isinstance(self.resolved_relevance, bool) or not isinstance(self.resolved_relevance, int):
            raise TypeError("resolved_relevance must be an int")
        if not (MIN_RESOLVED_RELEVANCE <= self.resolved_relevance <= MAX_RESOLVED_RELEVANCE):
            raise ValueError(
                f"resolved_relevance must be within [{MIN_RESOLVED_RELEVANCE}, {MAX_RESOLVED_RELEVANCE}], "
                f"got {self.resolved_relevance}"
            )
        if self.resolved_weight is not None:
            if isinstance(self.resolved_weight, bool) or not isinstance(self.resolved_weight, (int, float)):
                raise TypeError("resolved_weight must be numeric")
            if not (self.resolved_weight > 0):
                raise ValueError("resolved_weight must be a positive, finite number")


@dataclass(frozen=True, slots=True)
class RealRankingDecision:
    """One real ranking decision: an authoritative group identity, a shared point-in-time
    ranking timestamp, and the candidates RMS scored together at that instant.

    `ranking_group_id` is NEVER auto-generated -- there is no default, and a blank value is
    rejected. Supplying it is the caller's (a future integration layer's) responsibility once an
    authoritative identity contract is confirmed; see the RMS cross-service contract package.
    """

    ranking_group_id: str
    ranking_timestamp: str
    candidates: tuple[RealCandidateObservation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ranking_group_id, str) or not self.ranking_group_id.strip():
            raise ValueError("ranking_group_id (qid) must be a non-blank, explicitly supplied string")
        _parse_aware_timestamp(self.ranking_timestamp, field="ranking_timestamp")

        content_ids = [candidate.snapshot.content_id for candidate in self.candidates]
        duplicates = {content_id for content_id in content_ids if content_ids.count(content_id) > 1}
        if duplicates:
            raise ValueError(
                f"duplicate content_id(s) {sorted(duplicates)} within ranking_group_id={self.ranking_group_id!r} "
                "-- one candidate row per (qid, content) is required"
            )

        for candidate in self.candidates:
            if candidate.snapshot.request_id != self.ranking_group_id:
                raise ValueError(
                    f"candidate content_id={candidate.snapshot.content_id!r} belongs to snapshot "
                    f"request_id={candidate.snapshot.request_id!r}, not this decision's "
                    f"ranking_group_id={self.ranking_group_id!r} -- candidates from separate "
                    "ranking decisions must never share one group"
                )
            if candidate.snapshot.ranking_timestamp != self.ranking_timestamp:
                raise ValueError(
                    f"candidate content_id={candidate.snapshot.content_id!r} has snapshot "
                    f"ranking_timestamp={candidate.snapshot.ranking_timestamp!r}, which differs "
                    f"from this decision's ranking_timestamp={self.ranking_timestamp!r} -- every "
                    "candidate in one ranking decision must share the same point-in-time state"
                )
