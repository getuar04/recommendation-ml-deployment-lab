"""LIVE ML scoring plus deliberately light, documented business reranking.

Reranking/explanation logic lives in the pure, I/O-free `app.ml.live_reranker` module,
mirroring the VIDEO flow's `app.ml.reranker` split.

History source precedence (foundation-work addition): this service's own stored, real LIVE
history (`app.services.providers.live_history_provider`) is authoritative over the
`previous_live_interactions`/`previous_live_watch_time` fields a caller supplies on each
candidate -- a caller that predates real local history simply keeps working exactly as
before (see `_derived_affinities` below). Dynamic, request-time-only candidate fields
(`live_age_minutes`, `region`/`language`(-Match), `already_joined`) remain exactly as the
caller supplied them for EXPLICIT candidates -- this service still has no authoritative local
source for those (see `app.ml.live_dataset_builder`'s `UNAVAILABLE_FROM_HISTORY_FEATURES`).
`current_viewer_count`/`viewer_growth_rate` are the ONE exception (dynamic-signal foundation):
for LOCALLY-GENERATED candidates they are already real
(`app.services.providers.live_dynamic_state_provider`, set at candidate-generation time in
`app.services.providers.live_candidate_provider._to_candidate`) -- this module reads them
unchanged off the candidate exactly as it always has, it just happens to now be reading a real
value instead of a placeholder for that one source. An EXPLICIT candidate's own claimed
current_viewer_count/viewer_growth_rate is still never second-guessed, unchanged.

Candidate source precedence (local-candidate-foundation addition): `request.candidates`,
when the caller supplies any, is used unchanged and takes precedence -- byte-identical to
before this module gained a local candidate source. An empty/omitted `request.candidates`
falls back to this service's own local LIVE candidate pool
(`app.services.providers.live_candidate_provider.load_active_live_candidates`), never an
HTTP self-call to `POST /candidates/generate/live` -- the same service-layer function that
endpoint itself calls for its own local-fallback path.

Recent-engagement/LIVE_NOT_INTERESTED (dynamic-signal foundation, Parts 8/13): applied as a
bounded RERANKING-only adjustment (`app.ml.live_reranker.live_adjusted_score`), never a new
model feature (would require retraining the pre-existing, unretrained active LIVE model) and
never a hard exclusion. Computed fresh here for whatever the final `valid` candidate list
turns out to be, regardless of candidate source -- a stream_id with no real local Interaction
history (most explicit external candidates in practice) legitimately evaluates to "no recent
engagement, not previously rejected", never a fabricated non-zero guess.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import (
    LIVE_METADATA_PATH,
    LIVE_MODEL_PATH,
    RECOMMENDATION_MAX_CANDIDATES,
)
from app.core.logging import logger
from app.ml import model_cache, model_store
from app.ml.live_dataset_builder import LiveHistory, live_feature_snapshot
from app.ml.live_feature_builder import (
    LIVE_FEATURES,
    LIVE_NUMERIC,
    derive_live_affinities,
    live_feature_row,
)
from app.ml.live_reranker import live_adjusted_score, live_explanation, live_rerank
from app.ml.predictor import sanitize_numeric
from app.services.providers.live_candidate_provider import load_active_live_candidates
from app.services.providers.live_dynamic_state_provider import (
    NEUTRAL_DYNAMIC_STATE,
    compute_dynamic_state,
)
from app.services.providers.live_history_provider import load_live_history_for_user
from app.services.providers.live_personalization_provider import (
    not_interested_stream_ids,
    video_bootstrap_affinities,
    video_evidence_count,
)
from app.services.recommendation_service import strategy_for

__all__ = ["LiveModelArtifactInvalid", "LiveModelNotTrained", "recommend_live"]


class LiveModelNotTrained(Exception):
    pass


# How many distinct stream_ids this user has ever sent LIVE_NOT_INTERESTED for to remember
# for the bounded, stream-specific reranking suppression (see module docstring, Part 13).
_NOT_INTERESTED_LOOKUP_LIMIT = 50


class LiveModelArtifactInvalid(Exception):
    """The LIVE model artifact exists but is corrupted, incompatible, or transiently
    unstable (retraining in progress); `reason` is one of "INCOMPATIBLE" / "CORRUPTED" / "BUSY"."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def _candidate_video_bootstrap(video_bootstrap: dict[str, dict[str, float]], candidate) -> dict[str, float]:
    """Narrows the per-request `video_bootstrap_affinities()` result down to this one
    candidate's category/creator -- see `app.ml.live_dataset_builder.live_feature_snapshot`'s
    own docstring for exactly when a value here gets used (only in place of a flat neutral
    default, never overriding real LIVE evidence)."""
    result: dict[str, float] = {}
    category_key = (candidate.category or "").upper()
    if category_key in video_bootstrap.get("category", {}):
        result["live_category_affinity"] = video_bootstrap["category"][category_key]
    if candidate.creator_id in video_bootstrap.get("creator", {}):
        result["creator_affinity"] = video_bootstrap["creator"][candidate.creator_id]
    return result


def _derived_affinities(candidate, real_history: LiveHistory | None, *, user_id: str, at,
                         video_bootstrap: dict[str, dict[str, float]]):
    """Real stored LIVE history, when this user has any, is authoritative for this candidate's
    category/creator-affinity-derived features; otherwise falls back to the pre-existing
    caller-supplied-scalar derivation (`derive_live_affinities`), unchanged. Either way, the
    VIDEO cross-format bootstrap (`video_bootstrap`, see `_candidate_video_bootstrap` above)
    substitutes for `live_category_affinity`/`creator_affinity`'s neutral default ONLY when
    this candidate's category/creator has zero real LIVE-specific evidence -- never
    `previous_live_interaction_count`/`previous_live_watch_time`/`creator_followed`, which stay
    exactly as they were before this bootstrap existed (see module docstring)."""
    candidate_bootstrap = _candidate_video_bootstrap(video_bootstrap, candidate)
    if real_history is not None:
        return live_feature_snapshot(
            real_history, user_id=user_id, category=candidate.category, creator_id=candidate.creator_id, at=at,
            video_bootstrap=candidate_bootstrap,
        )
    derived = derive_live_affinities(candidate.previous_live_interactions, candidate.previous_live_watch_time)
    category_affinity = derived["live_category_affinity"]
    creator_affinity = derived["creator_affinity"]
    if candidate.previous_live_interactions == 0:
        category_affinity = candidate_bootstrap.get("live_category_affinity", category_affinity)
        creator_affinity = candidate_bootstrap.get("creator_affinity", creator_affinity)
    return {
        "live_category_affinity": category_affinity, "creator_affinity": creator_affinity,
        "average_live_watch_time_for_category": derived["average_live_watch_time_for_category"],
        "recent_live_category_activity": derived["recent_live_category_activity"],
        "previous_live_interaction_count": candidate.previous_live_interactions,
        "previous_live_watch_time": candidate.previous_live_watch_time,
        "creator_followed": candidate.creator_followed,
    }


def recommend_live(request, db: Session):
    # Validated consistently regardless of whether there turn out to be any active
    # candidates to score: an incompatible/corrupted artifact must be reported the same
    # way for an empty request as for a normal one, not masked by a cheaper, permissive read.
    try:
        model, metadata = model_cache.live_cache.get(LIVE_MODEL_PATH, LIVE_METADATA_PATH, expected_features=LIVE_FEATURES)
    except model_store.ArtifactNotFoundError as exc:
        raise LiveModelNotTrained() from exc
    except model_store.ArtifactIncompatibleError as exc:
        raise LiveModelArtifactInvalid(str(exc), reason="INCOMPATIBLE") from exc
    except model_store.ArtifactCorruptedError as exc:
        raise LiveModelArtifactInvalid(str(exc), reason="CORRUPTED") from exc
    except model_store.ArtifactBusyError as exc:
        raise LiveModelArtifactInvalid(str(exc), reason="BUSY") from exc

    # Candidate source precedence: explicit request.candidates first; an empty/omitted list
    # falls back to the local LIVE candidate pool (see module docstring).
    candidate_pool = request.candidates or load_active_live_candidates(
        db, request.user_id, limit=RECOMMENDATION_MAX_CANDIDATES,
    )
    valid = []
    ids: set[str] = set()
    for candidate in candidate_pool:
        if candidate.status.value != "ACTIVE" or candidate.stream_id in ids:
            continue
        ids.add(candidate.stream_id)
        valid.append(candidate)

    now = datetime.now(timezone.utc)
    real_history = load_live_history_for_user(db, request.user_id)
    # Lifecycle/strategy observability (LIVE lifecycle audit): reuses
    # app.services.recommendation_service.strategy_for's existing COLD_START/HYBRID/
    # PERSONALISED_ML thresholds verbatim -- never a new LIVE-specific rule. LIVE-native
    # evidence (real reconstructed LIVE sessions) is authoritative when this user has any;
    # otherwise falls back to real VIDEO interaction evidence -- the same cross-format
    # bootstrap this module now uses for actual feature values (see video_bootstrap below),
    # so a user this service can genuinely personalize for is never mislabeled COLD_START
    # just because they have not yet joined a LIVE stream.
    live_evidence_count = real_history.user_session_count(request.user_id) if real_history is not None else 0
    if live_evidence_count == 0:
        live_evidence_count = video_evidence_count(db, request.user_id)
    strategy = strategy_for(live_evidence_count)

    if not valid:
        return {
            "userId": request.user_id, "modelVersion": metadata["modelVersion"],
            "strategy": strategy, "interactionCount": live_evidence_count, "recommendations": [],
        }

    # Cross-format LIVE-relevance bootstrap (architecture requirement): computed once per
    # request, never per candidate -- see video_bootstrap_affinities' own docstring.
    video_bootstrap = video_bootstrap_affinities(db, request.user_id)
    features = []
    for candidate in valid:
        derived = _derived_affinities(candidate, real_history, user_id=request.user_id, at=now,
                                       video_bootstrap=video_bootstrap)
        features.append(live_feature_row(
            candidate, category_affinity=derived["live_category_affinity"], creator_affinity=derived["creator_affinity"],
            average_category_watch_time=derived["average_live_watch_time_for_category"],
            recent_category_activity=derived["recent_live_category_activity"], scoring_time=now,
            previous_interaction_count=derived["previous_live_interaction_count"],
            previous_watch_time=derived["previous_live_watch_time"], creator_followed=derived["creator_followed"],
        ))
    frame = pd.DataFrame(sanitize_numeric(features, LIVE_NUMERIC))[LIVE_FEATURES]
    scores = model.predict_proba(frame)[:, 1]

    dynamic_state = compute_dynamic_state(db, [candidate.stream_id for candidate in valid], now=now)
    not_interested = not_interested_stream_ids(db, request.user_id, limit=_NOT_INTERESTED_LOOKUP_LIMIT)
    ranked = [
        (
            live_adjusted_score(
                float(score), already_joined=candidate.already_joined,
                recent_likes=dynamic_state.get(candidate.stream_id, NEUTRAL_DYNAMIC_STATE).recent_likes,
                recent_gifts=dynamic_state.get(candidate.stream_id, NEUTRAL_DYNAMIC_STATE).recent_gifts,
                not_interested_for_stream=candidate.stream_id in not_interested,
            ),
            candidate, live_explanation(row, candidate),
        )
        for candidate, row, score in zip(valid, features, scores)
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = live_rerank(ranked, request.limit)
    # DEBUG, not INFO: fires on every successful request (hot path) -- mirrors the same
    # reduction applied to app.services.recommendation_service's identical per-request log.
    logger.debug("recommendation completed userId=%s domain=LIVE strategy=%s modelVersion=%s count=%d",
                 request.user_id, strategy, metadata["modelVersion"], len(chosen))
    return {
        "userId": request.user_id,
        "modelVersion": metadata["modelVersion"],
        "strategy": strategy,
        "interactionCount": live_evidence_count,
        "recommendations": [
            {"streamId": candidate.stream_id, "category": candidate.category, "score": round(score, 6), "rank": rank, "reason": reason}
            for rank, (score, candidate, reason) in enumerate(chosen, 1)
        ],
    }
