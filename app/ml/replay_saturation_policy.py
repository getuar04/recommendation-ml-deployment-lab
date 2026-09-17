"""Central replay/exposure saturation policy (VIDEO only -- LIVE has its own separate
feature-building pipeline, app.ml.live_feature_builder, not touched here).

Product intent: repeated PASSIVE consumption/exposure of the SAME (user_id, content_id) pair
may only contribute a diminishing, then zero, amount of ADDITIONAL recommendation influence --
occurrence #1-2 full, #3-5 diminishing, #6+ zero. An EXPLICIT action (like/share/favorite/
comment/follow, or an explicit CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED rejection) always
retains full, unsaturated influence regardless of how many times its content has already been
passively replayed -- a user deliberately liking/sharing/following/rejecting something is a
qualitatively different, deliberate signal from a video simply playing (or replaying) again.

This module owns the ONE curve/threshold table and the ONE "is this row saturation-controlled"
classification -- every consumer below shares it rather than each defining/duplicating its
own magic numbers:

- `app.ml.dataset_builder.FeatureHistory.update()` -- protects serving-time AND training-row
  behavioral aggregation (category/creator/semantic/session accumulators) from replay
  contamination, since both paths call this exact same function (train/serve parity).
- `app.services.providers.user_behavior_provider._local_state` (via `effective_interaction_count`)
  -- protects COLD_START/HYBRID/PERSONALISED_ML strategy maturity from being reachable via N
  passive replays of ONE content.
- `app.ml.trainer` (via `compute_replay_weights`) -- protects the training SAMPLE WEIGHT
  (app.ml.sample_weight_policy, deliberately left untouched: its pure per-row contract cannot
  see other rows) from repeated-content rows each drawing full-weight gradient influence. This
  is a separate, chronological PREPROCESSING layer over that module's output, not a change to
  it -- see `compute_replay_weights`'s own docstring for the point-in-time-safety argument.
- `app.services.cohort_aggregation_service` -- protects a cohort's `preference_score` from one
  user's repeated single-content replay disproportionately swaying OTHER users' regional/age
  cohort recommendations.

Raw events/interactions are NEVER dropped, rejected, hidden, or reinterpreted by this policy
-- it only scales each occurrence's CONTRIBUTION to derived recommendation state/training
weight/cohort evidence. Seen state, persistence, provenance, and analytics are computed
elsewhere (app.services.event_service, FeatureHistory.seen) and are completely unaffected.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Occurrence #1-2: full influence. #3-5: diminishing. #6+ (anything not listed): zero
# additional influence. These five numbers are the ONLY place this curve is defined anywhere
# in the codebase -- not final/tuned, an initial product-specified curve (see callers).
_REPEAT_INFLUENCE_WEIGHTS: dict[int, float] = {1: 1.0, 2: 1.0, 3: 0.6, 4: 0.3, 5: 0.1}


def repeat_influence_weight(occurrence_number: int) -> float:
    """[0,1] influence multiplier for the Nth (1-indexed) passive, saturation-controlled
    occurrence of the SAME (user, content) pair. `occurrence_number <= 1` is treated as the
    first occurrence (defensive default; callers are expected to pass >= 1, counting only
    saturation-controlled occurrences -- see `is_saturation_controlled_row`)."""
    if occurrence_number <= 1:
        return _REPEAT_INFLUENCE_WEIGHTS[1]
    return _REPEAT_INFLUENCE_WEIGHTS.get(occurrence_number, 0.0)


# Passive consumption/exposure event types -- the ONLY event types a row's occurrence count
# can ever advance for, and the only ones whose OWN contribution is ever saturated. Every
# explicit-action event type (CONTENT_LIKED/SHARED/FAVORITED/COMMENTED/CREATOR_FOLLOWED) and
# the explicit-rejection types (CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED -- already handled
# by their own, separately-tuned penalty formula in FeatureHistory.update, never touched here)
# are deliberately excluded: a row of one of those types always keeps full, unsaturated
# influence, and never consumes/advances a content's passive-occurrence tally either.
PASSIVE_SATURATION_EVENT_TYPES = frozenset({
    "VIDEO_IMPRESSION", "VIDEO_STARTED", "VIDEO_WATCHED", "VIDEO_COMPLETED",
    "VIDEO_REWATCHED", "VIDEO_SKIPPED",
})


def is_saturation_controlled_row(
    *, event_type: str, liked: bool, shared: bool, favorited: bool, commented: bool, creator_followed: bool,
) -> bool:
    """True only for a passive event type carrying NO explicit-intent flag. An explicit action
    riding along on a passive event type (e.g. a VIDEO_WATCHED row that also sets liked=True)
    still always retains full, unsaturated influence -- explicit intent is exempt regardless
    of which event_type carries it."""
    if event_type not in PASSIVE_SATURATION_EVENT_TYPES:
        return False
    return not any((liked, shared, favorited, commented, creator_followed))


def is_saturation_controlled_interaction(row: Any) -> bool:
    """Same policy as `is_saturation_controlled_row`, adapted to a real interaction/ORM row
    (app.db.models.Interaction, or an equivalent raw-row-like object) via
    app.ml.feature_builder.interaction_signals -- the canonical reader of a row's own explicit-
    intent flags, reused rather than re-implemented. Used by FeatureHistory.update()."""
    from app.ml.feature_builder import (
        interaction_signals,  # local import: avoids a module-load-order cycle with dataset_builder
    )
    signals = interaction_signals(row)
    event_type = getattr(row, "event_type", "")
    event_type = getattr(event_type, "value", event_type)
    return is_saturation_controlled_row(
        event_type=event_type, liked=signals.liked, shared=signals.shared,
        favorited=signals.favorited, commented=signals.commented, creator_followed=signals.creator_followed,
    )


def effective_interaction_count(rows: Any) -> float:
    """Sum of each row's replay-influence weight: 1.0 for every explicit-action/non-passive
    row, and the same per-(user, content) occurrence-based saturation curve for passive rows
    -- used in place of a raw row count for strategy-maturity evidence (COLD_START/HYBRID/
    PERSONALISED_ML), so N passive repeats of ONE content cannot, by themselves, mature a
    user's strategy the way N distinct/legitimate interactions do.

    `rows` MUST already be in chronological (oldest-first) order -- occurrence numbering is
    order-dependent (the temporally-FIRST occurrence of a content must get full weight, not
    the most recent one). Callers already maintain this order (see
    app.db.repositories.recent_interactions_for_ranking)."""
    occurrence: dict[tuple[Any, Any], int] = {}
    total = 0.0
    for row in rows:
        if is_saturation_controlled_interaction(row):
            key = (row.user_id, row.content_id)
            occurrence[key] = occurrence.get(key, 0) + 1
            total += repeat_influence_weight(occurrence[key])
        else:
            total += 1.0
    return total


def compute_replay_weights(df: pd.DataFrame) -> np.ndarray:
    """One replay-influence weight per row of `df` (app.ml.dataset_builder.build_dataset's
    output: carries user_id/content_id/timestamp plus TRAINING_METADATA_COLUMNS). Meant to be
    multiplied into app.ml.sample_weight_policy.compute_sample_weights' own per-row result at
    the training call site (app.ml.trainer) -- a separate factor, not a replacement.

    Chronological and point-in-time safe by construction: sorts a POSITIONAL copy of the
    relevant columns by (user_id, content_id, timestamp) (stable sort, so ties keep `df`'s own
    original relative order) and assigns each saturation-controlled row its 1-indexed
    occurrence rank among only that (user, content) pair's OWN prior rows in that sorted
    order -- a row's weight can only ever depend on rows with an earlier-or-equal timestamp for
    the exact same (user, content) pair, never on any later row, and never on any other user's
    or content's rows at all. Non-saturation-controlled rows (explicit actions) always get 1.0
    and are never counted toward any content's occurrence tally.

    Known imprecision (documented, not a new risk): `df`'s TRAINING_METADATA_COLUMNS carries
    no `commented` field, so a training row whose event_type is a passive type but which also
    happens to carry a comment cannot be distinguished from a pure passive row here -- it is
    conservatively treated as saturation-controlled (the same or MORE saturation than the
    fully-precise app.ml.dataset_builder.FeatureHistory.update() path applies for the
    equivalent live row, never less protection)."""
    if len(df) == 0:
        return np.array([], dtype=float)
    positions = np.arange(len(df))
    controlled = np.fromiter(
        (
            is_saturation_controlled_row(
                event_type=row.event_type, liked=bool(row.event_liked), shared=bool(row.event_shared),
                favorited=bool(row.event_favorited), commented=False, creator_followed=bool(row.event_creator_followed),
            )
            for row in df.itertuples()
        ),
        dtype=bool, count=len(df),
    )
    order = pd.DataFrame({
        "_pos": positions, "user_id": df["user_id"].to_numpy(), "content_id": df["content_id"].to_numpy(),
        "timestamp": df["timestamp"].to_numpy(), "_controlled": controlled,
    })
    order = order.sort_values(["user_id", "content_id", "timestamp"], kind="stable")

    weights = np.ones(len(df), dtype=float)
    occurrence: dict[tuple[Any, Any], int] = {}
    for pos, user_id, content_id, is_controlled in zip(
        order["_pos"].to_numpy(), order["user_id"].to_numpy(), order["content_id"].to_numpy(), order["_controlled"].to_numpy(),
    ):
        if not is_controlled:
            continue
        key = (user_id, content_id)
        occurrence[key] = occurrence.get(key, 0) + 1
        weights[pos] = repeat_influence_weight(occurrence[key])
    return weights
