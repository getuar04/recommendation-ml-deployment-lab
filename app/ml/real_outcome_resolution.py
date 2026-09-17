"""Pure T0 -> T1+ outcome resolution: turns real interaction events arriving AFTER a
`RankedCandidateSnapshot`'s ranking_timestamp into a valid `RealCandidateObservation`
(see `app.ml.real_ranking_decision`).

No DB access, no repository access, no HTTP, no Kafka, no model loading, no training, no
global mutable state -- this module only reads plain objects the caller already has in hand
and returns a new, already-self-validated `RealCandidateObservation`. Not wired into serving,
`/training`, or any persistence path; see the RMS real-data gap analysis for where this sits
in the still-incomplete real-data pipeline.

TEMPORAL RULE (T0/T1): `snapshot.ranking_timestamp` is T0. An event strictly BEFORE T0 must
never influence the resolved outcome -- filtered out entirely, never merely down-weighted.
The T0-equality boundary (`event.timestamp == snapshot.ranking_timestamp`) is treated as
IN-WINDOW (eligible), not excluded -- this mirrors, not invents, the boundary
`RealCandidateObservation.__post_init__` itself already enforces: it rejects `observed_at`
strictly LESS THAN `ranking_timestamp` ("impossible temporal ordering"), and therefore already
accepts `observed_at == ranking_timestamp` as valid. Using a stricter (`>`) filter here would
silently disagree with that existing, load-bearing invariant.

IMPRESSION IS EVIDENCE-DERIVED, NOT ASSUMED: this codebase's event model has no event type
distinct from engagement itself that means "the user saw this" -- a watch/like/share/comment/
rejection event is itself the only available proof of impression. So: zero eligible events
means there is no evidence this candidate was ever confirmed seen, and this resolver honestly
reports `ExposureState.SERVED` / `ObservationState.NOT_OBSERVED` (not `RANKED`: every current
`RankedCandidateSnapshot` construction path -- `app.ml.ranking_snapshot_capture.
build_ranking_decision_snapshots` -- builds snapshots only from the final SERVED slate, i.e.
`chosen`, never the raw pre-rerank list) rather than fabricating a neutral/negative label or an
`ObservationState.NOT_YET_RESOLVED` with no real timestamp to justify it. At least one eligible
event resolves the candidate as `IMPRESSED`/`RESOLVED` in the same call -- this module never
produces `ObservationState.NOT_YET_RESOLVED` or `CENSORED`: both require a real-world "outcome
window is still open" / "a window-closing policy decided to give up" fact this pure function,
given only a fixed event list, cannot possess. A future caller with an actual open-window
concept (e.g. re-checking a candidate 10 minutes vs. 10 days after T0) owns that distinction;
this module only ever sees "the events I have so far," never "time has now passed."

OUTCOME SEMANTICS: reuses this codebase's existing real-interaction conventions rather than
inventing a second recommendation-label system:
  - `app.ml.feature_builder.EXPLICIT_NEGATIVE_EVENT_TYPES`/`interaction_signals` (the exact
    real-Interaction-row reader `target_for`/`app.ml.ranking_groups` already use) for
    liked/shared/favorited/creator_followed and explicit-rejection detection.
  - The 0-4 graded scale mirrors `app.ml.ranking_groups.relevance_grade`'s exact semantics
    (explicit rejection / fast-skip -> 0; shared/favorited/creator_followed -> 4; liked or a
    completed watch -> 3; a solid watch -> 2; anything else eligible -> 1) -- deliberately NOT
    imported from there: `app.ml.ranking_groups` is the SYNTHETIC archetype generator, and
    every real_* module in this package carries zero import dependency on it by design (see
    `app.ml.real_ranking_decision`'s own module docstring, e.g. its locally-redefined
    MIN/MAX_RESOLVED_RELEVANCE). This module's small local `_grade_relevance` helper is that
    same isolation precedent applied to relevance grading specifically.
  - "completed" reuses `app.ml.feature_builder.build_profiles`'s own completion-detection
    convention (`event_type == "VIDEO_COMPLETED" or watch_percentage >= threshold`), not a
    watch_percentage-only check -- a real VIDEO_COMPLETED event resolves as a completion even
    when watch_percentage is absent or was recorded low for that specific event.
  - Explicit negative feedback (CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED) DOMINATES: if any
    eligible event is an explicit rejection, the resolved relevance is 0 regardless of any
    other eligible event's watch percentage or positive flags -- the same precedence
    `app.ml.feature_builder.target_for` and `app.ml.dataset_builder`'s negative-feedback
    session fix already establish elsewhere in this codebase.
  - A missing `watch_percentage` on an event (e.g. a like/comment/rejection with no associated
    watch session) is never treated as an implicit 0% fast-skip -- mirrors `target_for`'s own
    `wp is not None` guard (not `app.ml.dataset_builder`'s session-scoped `watch_percentage or
    0` shortcut, which exists for a different, RecentWatchEvent-parity reason that does not
    apply here).

AGGREGATION / DETERMINISM: multiple eligible events for the same candidate are combined
order-independently -- explicit-negative and each positive flag are an `any(...)` (OR) across
all eligible events, watch_percentage is a `max(...)` across all eligible events that reported
one, and `observed_at` is the LATEST eligible event's timestamp (the moment the complete,
final-answer picture -- including any later event that could still flip the outcome, e.g.
"watch, then like, then not_interested" -- became fully knowable). `any`/`max` are both
commutative, so the result never depends on the order `events` was supplied in.

SAMPLE WEIGHT: `resolved_weight` is always `None` here, on purpose. Computing one is
`app.ml.real_sample_weight_policy`'s job -- today an intentionally unimplemented `Protocol` (no
real-data weight policy has been written or empirically validated yet). This module must not
quietly invent one; `RealCandidateObservation`/`app.ml.real_ranking_dataset_builder` already
accept `resolved_weight=None` as a first-class, valid state.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from app.ml.feature_builder import (
    COMPLETION_WATCH_PERCENTAGE_THRESHOLD,
    EXPLICIT_NEGATIVE_EVENT_TYPES,
    FAST_SKIP_WATCH_PERCENTAGE_THRESHOLD,
    interaction_signals,
)
from app.ml.ranking_snapshot import RankedCandidateSnapshot
from app.ml.real_ranking_decision import (
    ExposureState,
    ObservationState,
    RealCandidateObservation,
)

__all__ = ["RealOutcomeEvent", "resolve_observation"]

# Mirrors the literal `wp >= 70` threshold already used identically by
# `app.ml.feature_builder.target_for` and `app.ml.ranking_groups.relevance_grade` for the same
# "moderate positive, no stronger signal" tier -- named locally (that literal is never a shared
# importable constant in either source) so this module's own tier boundary is self-documenting.
MODERATE_ENGAGEMENT_WATCH_PERCENTAGE_THRESHOLD = 70


class RealOutcomeEvent(Protocol):
    """Duck-typed shape this resolver reads from a real interaction/event object -- matches
    `app.db.models.Interaction`'s own field names exactly (never a second, differently-named
    event schema), but declared as a structural `Protocol` rather than imported from
    `app.db.models` so this module stays DB-free per its own pure-function contract."""

    user_id: str
    content_id: str
    event_type: str
    timestamp: datetime
    watch_percentage: float | None
    liked: bool
    shared: bool
    favorited: bool
    commented: bool
    creator_followed: bool


def _is_eligible(event: RealOutcomeEvent, *, user_id: str, content_id: str, ranking_timestamp: datetime) -> bool:
    """Only an event for the SAME (user_id, content_id) at-or-after T0 may resolve this
    candidate's outcome -- see module docstring for why T0-equality is included, not excluded."""
    if event.user_id != user_id or event.content_id != content_id:
        return False
    ts = event.timestamp
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError(f"event.timestamp for content_id={event.content_id!r} must be timezone-aware")
    return ts >= ranking_timestamp


def _grade_relevance(
    *, explicit_negative: bool, max_watch_percentage: float | None,
    liked: bool, shared: bool, favorited: bool, creator_followed: bool, completed: bool,
) -> int:
    """Mirrors `app.ml.ranking_groups.relevance_grade`'s exact 0-4 semantics -- see module
    docstring for why this is a small, locally-defined helper rather than an import."""
    if explicit_negative:
        return 0
    fast_skip = (
        max_watch_percentage is not None
        and max_watch_percentage < FAST_SKIP_WATCH_PERCENTAGE_THRESHOLD
        and not (liked or shared or favorited or creator_followed)
    )
    if fast_skip:
        return 0
    if shared or favorited or creator_followed:
        return 4
    if liked or completed:
        return 3
    if max_watch_percentage is not None and max_watch_percentage >= MODERATE_ENGAGEMENT_WATCH_PERCENTAGE_THRESHOLD:
        return 2
    return 1


def resolve_observation(
    snapshot: RankedCandidateSnapshot, events: Sequence[RealOutcomeEvent],
) -> RealCandidateObservation:
    """Resolves `events` against `snapshot` into one valid `RealCandidateObservation`.

    Filters `events` to those eligible (same user_id/content_id, timestamp >= T0), then either:
      - no eligible events -> `SERVED`/`NOT_OBSERVED` (no evidence of impression; see module
        docstring's "IMPRESSION IS EVIDENCE-DERIVED" section)
      - at least one eligible event -> `IMPRESSED`/`RESOLVED`, with `resolved_relevance` graded
        from the aggregate of every eligible event (order-independent; see module docstring)
        and `observed_at` set to the latest eligible event's timestamp. `resolved_weight` is
        always `None` (see module docstring's "SAMPLE WEIGHT" section).

    The returned `RealCandidateObservation` re-validates itself at construction (see
    `app.ml.real_ranking_decision.RealCandidateObservation.__post_init__`) -- this function
    never bypasses those invariants.
    """
    ranking_timestamp = datetime.fromisoformat(snapshot.ranking_timestamp)
    eligible = [
        event for event in events
        if _is_eligible(event, user_id=snapshot.user_id, content_id=snapshot.content_id, ranking_timestamp=ranking_timestamp)
    ]
    if not eligible:
        return RealCandidateObservation(
            snapshot=snapshot, exposure_state=ExposureState.SERVED, observation_state=ObservationState.NOT_OBSERVED,
        )

    signals = [interaction_signals(event) for event in eligible]
    liked = any(s.liked for s in signals)
    shared = any(s.shared for s in signals)
    favorited = any(s.favorited for s in signals)
    # `commented` is read for completeness (interaction_signals reports it) but deliberately
    # never influences relevance below -- matches target_for/relevance_grade/row_weight, none
    # of which treat a comment alone as a positive or negative signal.
    creator_followed = any(s.creator_followed for s in signals)
    explicit_negative = any(event.event_type in EXPLICIT_NEGATIVE_EVENT_TYPES for event in eligible)
    completed = any(
        event.event_type == "VIDEO_COMPLETED" or (event.watch_percentage or 0) >= COMPLETION_WATCH_PERCENTAGE_THRESHOLD
        for event in eligible
    )
    observed_watch_percentages = [event.watch_percentage for event in eligible if event.watch_percentage is not None]
    max_watch_percentage = max(observed_watch_percentages) if observed_watch_percentages else None

    relevance = _grade_relevance(
        explicit_negative=explicit_negative, max_watch_percentage=max_watch_percentage,
        liked=liked, shared=shared, favorited=favorited, creator_followed=creator_followed, completed=completed,
    )
    observed_at = max(event.timestamp for event in eligible)

    return RealCandidateObservation(
        snapshot=snapshot, exposure_state=ExposureState.IMPRESSED, observation_state=ObservationState.RESOLVED,
        observed_at=observed_at.isoformat(), resolved_relevance=relevance, resolved_weight=None,
    )
