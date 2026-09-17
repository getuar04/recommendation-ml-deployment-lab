"""User-behavior data-adapter boundary (dual-mode integration).

Normalizes three possible sources into the SAME `UserBehaviorState` that
`app.services.recommendation_service.recommend()` already builds today -- this module
changes only WHERE that state comes from, never how it is scored/reranked downstream:

1. `REQUEST_PROFILE` -- the caller already supplied `request.userProfile` (Phase A path,
   unchanged, works regardless of RECOMMENDATION_DATA_MODE): the production shape a real Feed
   Service already uses.
2. `UBS` -- RECOMMENDATION_DATA_MODE=REAL, no userProfile supplied: fetched from a real User
   Behavior Service and validated against the exact same `UserProfile` contract as (1) (see
   README "userProfile: optional production-style request path"), so a UBS response and a
   caller-supplied userProfile are handled by the identical code path from here on
   (`FeatureHistory.from_profile`) -- no second, parallel normalization.
3. `LOCAL_DB` -- the pre-existing legacy/demo path: this project's own `interactions` table,
   bounded to `RECOMMENDATION_HISTORY_MAX_INTERACTIONS` rows. Used whenever (1) and (2) don't
   apply, and as REAL mode's own fallback when UBS is unavailable and
   `UBS_FALLBACK_TO_LOCAL` is true (default) -- this is what keeps REAL mode from ever being
   less resilient than LOCAL mode for a demo/partially-configured deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import (
    RECOMMENDATION_DATA_MODE,
    RECOMMENDATION_HISTORY_MAX_INTERACTIONS,
    UBS_BASE_URL,
    UBS_BEHAVIOR_PROFILE_PATH,
    UBS_FALLBACK_TO_LOCAL,
    UBS_TIMEOUT_MS,
)
from app.core.logging import logger
from app.db.models import Content
from app.db.repositories import recent_interactions_for_ranking
from app.ml.dataset_builder import FeatureHistory, history_from_rows
from app.ml.feature_builder import is_video_hard_seen_event
from app.ml.replay_saturation_policy import effective_interaction_count
from app.schemas.recommendation_schemas import UserProfile
from app.services.service_clients import UpstreamServiceError, call_json

__all__ = ["UserBehaviorSourceUnavailable", "UserBehaviorState", "resolve"]

# An explicit CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED is a rejection signal, not proof the
# user actually consumed the content -- app.ml.dataset_builder/app.ml.reranker already treat it
# as a gradual score-suppression signal (CONTENT_NOT_INTERESTED_CATEGORY_PENALTY et al.), never
# a hard removal. It must therefore stay out of `seen_content_ids` below (the hard
# serving-eligibility exclusion set, recommendation_service.recommend()'s `already_seen_content_ids`)
# or a rejected-but-barely-viewed item would vanish from serving entirely instead of merely
# ranking lower -- collapsing the tested NOT_INTERESTED gradual-suppression contract into a
# hard block it was never meant to be. Delegates to the canonical `is_video_hard_seen_event`
# policy (app.ml.feature_builder) rather than re-declaring its own event-type set, so this
# authoritative definition can never drift from video_candidate_provider's (Task: unify VIDEO
# seen-state semantics -- see that module's own `seen` computation for the other consumer).


@dataclass(frozen=True)
class UserBehaviorState:
    """Everything `recommend()` needs about a user's behavior history, regardless of source.

    `interaction_count` is the RAW, unweighted count -- unchanged meaning, still exposed
    byte-identical in `RecommendationResponse.interactionCount` (a public, pre-existing
    contract field several existing tests assert exact integer values against; never
    silently redefined). `effective_interaction_count` (app.ml.replay_saturation_policy) is a
    SEPARATE, internal-only evidence signal used exclusively to select COLD_START/HYBRID/
    PERSONALISED_ML strategy maturity (see app.services.recommendation_service.recommend) --
    it discounts N passive replays of the SAME content the way N distinct/legitimate
    interactions should not be discounted. Defaults to `interaction_count` (no saturation
    applied) for the two profile-supplied sources (REQUEST_PROFILE/UBS), which carry no
    per-content history to derive an occurrence count from -- a documented limitation of
    that path, not a behavior regression (those callers never see saturation today either)."""
    history: FeatureHistory
    interaction_count: int
    cold_start: bool
    seen_content_ids: set[str] = field(default_factory=set)
    source: str = "LOCAL_DB"  # "REQUEST_PROFILE" | "LOCAL_DB" | "UBS"
    effective_interaction_count: float | None = None

    def evidence_count(self) -> float:
        """The count to use for strategy-maturity evidence (COLD_START/HYBRID/
        PERSONALISED_ML) -- `effective_interaction_count` when explicitly resolved, else
        `interaction_count` itself unchanged (a caller/test that constructs UserBehaviorState
        directly without this field gets exactly the old raw-count behavior; `None`, not
        `0.0`, is the "not specified" sentinel so a genuinely-computed zero effective count --
        e.g. a user whose only history is 6+ replays of ONE content -- is never confused with
        "not provided")."""
        return self.effective_interaction_count if self.effective_interaction_count is not None else float(self.interaction_count)


class UserBehaviorSourceUnavailable(Exception):
    """Raised only when REAL mode's UBS call fails AND UBS_FALLBACK_TO_LOCAL is false --
    a clear, observable dependency failure rather than a silent/degraded response."""


def _state_from_profile(user_id: str, profile: UserProfile, *, source: str) -> UserBehaviorState:
    history = FeatureHistory.from_profile(user_id, profile)
    interaction_count = profile.total_interaction_count
    return UserBehaviorState(
        history=history, interaction_count=interaction_count,
        cold_start=interaction_count == 0, seen_content_ids=set(profile.seen_content_ids), source=source,
        effective_interaction_count=float(interaction_count),
    )


def _local_state(db: Session, user_id: str) -> UserBehaviorState:
    """Same query/bound as the pre-existing legacy/demo DB-row path, now scoped to VIDEO only
    (see the VIDEO-only filter below) before anything derives from it."""
    raw_rows = recent_interactions_for_ranking(db, user_id, limit=RECOMMENDATION_HISTORY_MAX_INTERACTIONS)
    raw_content_ids = {row.content_id for row in raw_rows}
    # Fetched for the RAW (pre-filter) set -- same single query the pre-existing code already
    # ran here (it always covered every raw interacted content_id), just needed one step
    # earlier so content_type is known before filtering.
    content_by_id = (
        {item.content_id: item for item in db.scalars(select(Content).where(Content.content_id.in_(raw_content_ids))).all()}
        if raw_content_ids else {}
    )
    # VIDEO-only local history: `recent_interactions_for_ranking` (shared repository, left
    # unmodified -- other callers are unaffected) has no VIDEO/LIVE discriminator of its own;
    # event ingestion (app.services.event_service.store_event) accepts LIVE_*/domain-ambiguous
    # event types into this same table with no cross-check against the referenced content's own
    # type. A row PROVABLY LIVE (its Content row exists and is explicitly content_type=="LIVE")
    # is excluded here, in this adapter only -- interaction_count/seen_content_ids/history below
    # all derive from this filtered set. A row whose Content is missing/unavailable is left in
    # unchanged (mirrors app.services.training_service's own same-shaped fix): "unavailable" is
    # not evidence of being LIVE, and history_from_rows already tolerates a missing Content row.
    rows = [row for row in raw_rows if getattr(content_by_id.get(row.content_id), "content_type", "VIDEO") != "LIVE"]
    # Hard serving-eligibility exclusion (canonical `is_video_hard_seen_event` policy, see the
    # module-level comment above) is narrower than "ever interacted with": a content_id whose
    # only row(s) are NOT_INTERESTED-type stays eligible for scoring/suppression; one with any
    # genuine view/engagement row is still seen.
    seen_content_ids = {row.content_id for row in rows if is_video_hard_seen_event(row.event_type)}
    history = history_from_rows(rows, content_by_id)
    # Replay/exposure saturation (app.ml.replay_saturation_policy): `rows` is already in
    # chronological (oldest-first) order (recent_interactions_for_ranking's own contract),
    # exactly what effective_interaction_count requires. `cold_start` is deliberately still
    # derived from raw row presence (`not rows`), not the effective count -- cold_start gates
    # onboarding-interest/regional-cohort reranking, a separate concern from strategy
    # maturity; a user with ANY real row is not a genuine cold start regardless of replay.
    return UserBehaviorState(
        history=history, interaction_count=len(rows), cold_start=not rows,
        seen_content_ids=seen_content_ids, source="LOCAL_DB",
        effective_interaction_count=effective_interaction_count(rows),
    )


def _fetch_from_ubs(user_id: str) -> UserBehaviorState:
    """Calls the real User Behavior Service. Never raises a raw httpx/pydantic exception --
    always UpstreamServiceError (network/HTTP) or a ValidationError, both translated to
    UserBehaviorSourceUnavailable by the caller below.

    KNOWN GAP -- confirmed, not assumed: a real-contract search of every sibling repository in
    the local Soft Dome workspace (follow-service, ranking-service, interaction-count-service,
    viewer-count-service, faq-service) found NO "User Behavior Service" repository, OpenAPI
    doc, or Postman collection anywhere. This targets a documented-shape endpoint
    (`GET {UBS_BASE_URL}{UBS_BEHAVIOR_PROFILE_PATH}`, path configurable -- see
    app.core.config.UBS_BEHAVIOR_PROFILE_PATH) that is this project's own reasonable
    placeholder against the ALREADY-established `UserProfile` contract
    (app.schemas.recommendation_schemas.UserProfile) -- not a verified external API. A real
    UBS deployment's actual path/verb/field names can be pointed at by adjusting
    UBS_BEHAVIOR_PROFILE_PATH (and, if its response omits fields this schema requires, by
    making those optional/neutral in UserProfile) once that contract exists and is
    discoverable -- no code change needed for a path-only difference."""
    assert UBS_BASE_URL is not None  # only called once resolve() has already checked this
    try:
        path = UBS_BEHAVIOR_PROFILE_PATH.format(userId=user_id)
    except (KeyError, IndexError, ValueError) as exc:
        # A misconfigured UBS_BEHAVIOR_PROFILE_PATH (e.g. missing/misnamed placeholder) must
        # degrade exactly like any other UBS failure -- not bypass resolve()'s
        # except (UpstreamServiceError, ValidationError) fallback with an unrelated raw
        # str.format() error and break the documented UBS_FALLBACK_TO_LOCAL guarantee.
        raise UpstreamServiceError("UBS", "INVALID_PATH_CONFIG", "UBS_BEHAVIOR_PROFILE_PATH is misconfigured") from exc
    body = call_json(
        service="UBS", base_url=UBS_BASE_URL,
        path=path,
        timeout_ms=UBS_TIMEOUT_MS,
    )
    profile = UserProfile.model_validate(body)
    return _state_from_profile(user_id, profile, source="UBS")


def resolve(db: Session, request) -> UserBehaviorState:
    """Priority: an explicit `request.userProfile` always wins outright (unchanged from
    before this module existed, and correct regardless of RECOMMENDATION_DATA_MODE -- a
    caller that already did the UBS call itself, e.g. a real Feed Service, must not be
    second-guessed). Otherwise: REAL mode tries UBS; LOCAL mode (and REAL mode with UBS
    unavailable/unconfigured, when fallback is enabled) uses the local interactions table."""
    if request.user_profile is not None:
        return _state_from_profile(request.user_id, request.user_profile, source="REQUEST_PROFILE")

    if RECOMMENDATION_DATA_MODE == "REAL":
        if not UBS_BASE_URL:
            logger.info("UBS not configured userId=%s fallbackToLocal=%s", request.user_id, UBS_FALLBACK_TO_LOCAL)
            if not UBS_FALLBACK_TO_LOCAL:
                raise UserBehaviorSourceUnavailable("UBS_BASE_URL is not configured")
        else:
            try:
                return _fetch_from_ubs(request.user_id)
            except (UpstreamServiceError, ValidationError) as exc:
                logger.info("UBS unavailable userId=%s reason=%s fallbackToLocal=%s",
                            request.user_id, str(exc)[:200], UBS_FALLBACK_TO_LOCAL)
                if not UBS_FALLBACK_TO_LOCAL:
                    raise UserBehaviorSourceUnavailable(str(exc)) from exc

    return _local_state(db, request.user_id)
