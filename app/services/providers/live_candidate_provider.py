"""Local LIVE candidate sourcing (event-driven-architecture foundation work): lets LIVE
serving/candidate-generation read this service's own local Content projection instead of
requiring the caller to supply a full `streams[]`/`candidates[]` list on every request.

Local projection boundary (see the event-driven architecture audit): a `Content` row with
`content_type == "LIVE"` and `is_active == True` is this service's current, only
authoritative signal that a LIVE stream exists and is currently recommendation-eligible.
This is a STREAM-LIFECYCLE concept, deliberately independent of viewer SESSION state
(`app.ml.live_session_builder`) -- a viewer disconnecting/reconnecting, or never sending
LIVE_LEFT, must never decide whether the stream itself is active; only this Content row's
`is_active` flag does. A future real Content/LIVE Service Kafka consumer can create/
update/deactivate these same rows (mirroring how `app.services.event_service.store_event`
already lets `app.services.kafka_behavior_consumer` populate `interactions` without any
serving code caring about the source) -- no code in this module or its callers needs to
change when that consumer exists; this module IS the local projection boundary.

Dynamic, request-time-only signals this service has never persisted anywhere
(`currentViewerCount`, `viewerGrowthRate`, `region`, `language`) are given documented
neutral placeholders here, matching `app.ml.live_dataset_builder.
UNAVAILABLE_FROM_HISTORY_FEATURES`'s established convention -- never fabricated/random
values. `liveAgeMinutes` is derived from `Content.created_at`, a reasonable proxy for
stream-start time under this project's current content-creation flow (`is_active` defaults
`True` at creation -- see `app.db.models.Content`) but NOT a guaranteed "stream started at"
timestamp in general; this is a documented limitation, not a claim of precision. Current
viewer count/growth rate are deliberately NOT derived from LIVE_JOINED/LIVE_LEFT session
data: doing so would require a "how long before an open, un-LEFT session is considered
stale" policy that does not exist anywhere in this project's configuration today, and
inventing one here would contradict `app.ml.live_session_builder`'s own established
position that an open-ended (un-LEFT) segment's duration is unknowable, not a proxy for
"still watching".

Candidate MEMBERSHIP personalization (LIVE candidate-personalization foundation): this used
to be pure generic retrieval ("newest active LIVE, `user_id` unused except for logging" --
one query, `ORDER BY created_at DESC LIMIT limit`). It is now four SEPARATE, independently-
filtered queries run in priority order -- FOLLOWED_CREATOR, then RECENT_CREATOR_INTERACTION
(Ylli's cross-format requirement: recent, positive VIDEO engagement with a creator makes that
creator's LIVE candidate-worthy even with zero LIVE history), then PREFERRED_CATEGORY
(onboarding interests + real LIVE/VIDEO category history, see
`app.services.providers.live_personalization_provider`), then a generic newest-active
EXPLORATION backfill -- each excluding content already selected by a higher-priority bucket
and capped at however many slots remain. Querying each bucket directly (never "fetch the
newest N overall, then filter in Python") is deliberate: a followed creator's older LIVE
stream must still be found even when it is not among the newest streams platform-wide, which
a single recency-limited fetch could never surface no matter how the results were re-sorted
afterward. Every eligible candidate is still reachable via the EXPLORATION backfill, so this
can never under-fill; it only changes which candidates are included FIRST once eligible
content exceeds `limit`. `previousLiveInteractions`/`previousLiveWatchTime`/`creatorFollowed`
on the returned candidates now reflect this user's REAL local history (via
`app.services.providers.live_history_provider.load_live_history_for_user`) instead of always
defaulting -- ranking already applied real history via
`app.services.live_recommendation_service._derived_affinities` regardless, but
`app.ml.live_reranker.live_explanation`'s FOLLOWED_CREATOR heuristic reads `candidate.
creator_followed` DIRECTLY, so a locally-generated candidate previously could never produce
that explanation even for a genuinely followed creator -- this also fixes that.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.db.models import Content
from app.ml.live_dataset_builder import LiveHistory
from app.ml.live_reranker import TRENDING_GROWTH_RATE_THRESHOLD
from app.schemas.live_schemas import LiveCandidate, LiveStatus
from app.services.providers import live_personalization_provider
from app.services.providers.live_dynamic_state_provider import (
    NEUTRAL_DYNAMIC_STATE,
    LiveDynamicState,
    compute_dynamic_state,
)
from app.services.providers.live_history_provider import load_live_history_for_user
from app.services.user_context_provider import load_user_context

# Documented neutral placeholders for signals this service still has no authoritative local
# source for (see module docstring) -- never fabricated/random values. currentViewerCount/
# viewerGrowthRate USED to be in this list too; they are now real
# (app.services.providers.live_dynamic_state_provider), see `_to_candidate` below.
_UNAVAILABLE_REGION = "unknown"
_UNAVAILABLE_LANGUAGE = "unknown"
_MAX_LIVE_AGE_MINUTES = 1440.0  # matches LiveCandidate.live_age_minutes' own upper bound

# How many extra active-LIVE candidates (beyond however many slots remain) to consider when
# evaluating the TRENDING_LIVE bucket -- dynamic state can only be checked for content that
# was actually fetched, so this bucket looks at a broader pool than just `_remaining()` slots
# before filtering down to genuinely trending ones. Bounded, not unbounded.
_TRENDING_EVALUATION_POOL_CAP = 200

# Bucketed-retrieval tuning: how many distinct followed/recently-interacted creators or
# preferred categories feed each bucket's SQL filter. Bounded, not unbounded.
_SIGNAL_LOOKUP_LIMIT = 10


def _live_age_minutes(created_at: datetime, *, now: datetime) -> float:
    created = created_at if created_at.tzinfo is not None else created_at.replace(tzinfo=timezone.utc)
    age_minutes = (now - created).total_seconds() / 60
    return max(0.0, min(_MAX_LIVE_AGE_MINUTES, age_minutes))


def _to_candidate(
    content: Content, *, now: datetime, live_history: LiveHistory | None, user_id: str, followed_creators: set[str],
    dynamic_state: LiveDynamicState,
) -> LiveCandidate:
    # Real local history, when this user has any, sets creatorFollowed/previousLiveInteractions/
    # previousLiveWatchTime honestly instead of always defaulting -- see module docstring for
    # why this matters even though ranking already applies real history independently.
    if live_history is not None:
        snapshot = live_history.snapshot(user_id=user_id, category=content.category, creator_id=content.creator_id, at=now)
        previous_interactions = int(snapshot["previous_creator_live_interaction_count"])
        previous_watch_time = snapshot["previous_creator_live_watch_time"]
        live_history_followed = bool(snapshot["creator_followed"])
    else:
        previous_interactions = 0
        previous_watch_time = 0.0
        live_history_followed = False
    creator_followed = live_history_followed or content.creator_id in followed_creators

    # model_validate (not direct kwargs) matches this project's existing convention for
    # constructing a Candidate/LiveCandidate from a plain mapping (see tests/test_live.py's
    # `LiveCandidate.model_validate(raw())`) and lets every field this function does not set
    # (region/languageMatch, alreadyJoined) keep using the caller-supplied contract's own
    # defaults, without repeating them.
    return LiveCandidate.model_validate({
        "streamId": content.content_id, "creatorId": content.creator_id, "category": content.category,
        "status": LiveStatus.ACTIVE,
        # Real, query-derived dynamic state (app.services.providers.live_dynamic_state_provider)
        # -- no longer a hardcoded neutral placeholder. See that module's own docstring for
        # exactly what "current"/"growth" mean and their honesty limitations.
        "currentViewerCount": dynamic_state.current_viewer_count, "viewerGrowthRate": dynamic_state.viewer_growth_rate,
        "liveAgeMinutes": _live_age_minutes(content.created_at, now=now),
        "region": _UNAVAILABLE_REGION, "language": _UNAVAILABLE_LANGUAGE,
        "creatorFollowed": creator_followed,
        "previousLiveInteractions": previous_interactions, "previousLiveWatchTime": previous_watch_time,
    })


def _query_active_live(db: Session, *, extra_filter=None, exclude_ids: set[str], limit: int) -> list[Content]:
    """One bucket's SQL query: active LIVE content, optionally narrowed by `extra_filter`,
    excluding anything already selected by a higher-priority bucket. Each bucket queries the
    DB directly by its own filter (never "take the newest N, then filter in Python") -- that
    is what lets an older but genuinely relevant stream (e.g. a followed creator's LIVE that
    isn't among the newest overall) still be found instead of being truncated away before
    personalization ever gets a chance to see it."""
    if limit <= 0:
        return []
    stmt = select(Content).where(Content.content_type == "LIVE", Content.is_active.is_(True))
    if extra_filter is not None:
        stmt = stmt.where(extra_filter)
    if exclude_ids:
        stmt = stmt.where(Content.content_id.notin_(exclude_ids))
    stmt = stmt.order_by(Content.created_at.desc(), Content.content_id.asc()).limit(limit)
    return list(db.scalars(stmt).all())


def load_active_live_candidates(db: Session, user_id: str, *, limit: int) -> list[LiveCandidate]:
    """Recommendation-eligible LIVE streams from local Postgres state: `Content` rows with
    `content_type == "LIVE"` and `is_active == True`. Never raises. `Content.content_id` is
    unique at the DB level, so this can never return duplicate stream ids.

    Candidate SOURCING blends personalized bucketing with per-user history (see module
    docstring): up to five SEPARATE, independently-filtered queries run in priority order --
    FOLLOWED_CREATOR, RECENT_CREATOR_INTERACTION, PREFERRED_CATEGORY, TRENDING_LIVE (real
    dynamic-state evidence, see app.services.providers.live_dynamic_state_provider -- only
    added once that module existed; a neutral-placeholder growth rate could never legitimately
    justify a candidate SOURCE), then a generic newest-active EXPLORATION backfill -- each
    excluding content already selected by a higher-priority bucket, each capped at however
    many slots remain. Every eligible candidate is still reachable via the EXPLORATION
    backfill, so this can never under-fill relative to the old generic-only behavior; it only
    changes which candidates are included FIRST once eligible content exceeds `limit`.
    """
    now = datetime.now(timezone.utc)

    live_history = load_live_history_for_user(db, user_id)
    user_context = load_user_context(db, user_id)
    followed_creators = set(live_personalization_provider.followed_creator_ids(db, user_id, limit=_SIGNAL_LOOKUP_LIMIT))
    if live_history is not None:
        followed_creators |= set(live_history.followed_creator_ids(user_id))
    recent_creators = set(
        live_personalization_provider.recent_positive_video_creator_ids(db, user_id, limit=_SIGNAL_LOOKUP_LIMIT)
    ) - followed_creators
    video_categories = live_personalization_provider.recent_positive_video_categories(db, user_id, limit=_SIGNAL_LOOKUP_LIMIT)
    preferred_categories = set(live_personalization_provider.preferred_live_categories(
        user_context=user_context, live_history=live_history, video_categories=video_categories,
        user_id=user_id, limit=_SIGNAL_LOOKUP_LIMIT,
    ))

    selected_ids: set[str] = set()
    ordered_content: list[Content] = []
    bucket_counts: dict[str, int] = {}

    def _take(bucket_name: str, rows: list[Content]) -> None:
        added = 0
        for content in rows:
            # Defensive only -- creator_id/category are NOT NULL columns, so a genuinely
            # malformed row should be unreachable in practice; one bad row must never break
            # the whole pool.
            if not content.creator_id or not content.category or content.content_id in selected_ids:
                continue
            selected_ids.add(content.content_id)
            ordered_content.append(content)
            added += 1
        bucket_counts[bucket_name] = added

    def _remaining() -> int:
        return max(0, limit - len(selected_ids))

    if followed_creators:
        _take("FOLLOWED_CREATOR", _query_active_live(
            db, extra_filter=Content.creator_id.in_(followed_creators), exclude_ids=selected_ids, limit=_remaining(),
        ))
    if recent_creators:
        _take("RECENT_CREATOR_INTERACTION", _query_active_live(
            db, extra_filter=Content.creator_id.in_(recent_creators), exclude_ids=selected_ids, limit=_remaining(),
        ))
    if preferred_categories:
        _take("PREFERRED_CATEGORY", _query_active_live(
            db, extra_filter=Content.category.in_(preferred_categories), exclude_ids=selected_ids, limit=_remaining(),
        ))
    if _remaining():
        # TRENDING_LIVE needs real dynamic state to evaluate "trending", which is only knowable
        # for content actually fetched -- so this looks at a broader evaluation pool (bounded
        # by _TRENDING_EVALUATION_POOL_CAP) rather than the DB filtering on momentum directly
        # (viewer/join momentum isn't a column; it's derived from Interaction rows).
        trending_pool = _query_active_live(
            db, exclude_ids=selected_ids, limit=min(_TRENDING_EVALUATION_POOL_CAP, max(_remaining(), _TRENDING_EVALUATION_POOL_CAP // 4)),
        )
        if trending_pool:
            pool_dynamic_state = compute_dynamic_state(db, [c.content_id for c in trending_pool], now=now)
            trending = sorted(
                (c for c in trending_pool
                 if pool_dynamic_state.get(c.content_id, NEUTRAL_DYNAMIC_STATE).viewer_growth_rate > TRENDING_GROWTH_RATE_THRESHOLD),
                key=lambda c: pool_dynamic_state[c.content_id].viewer_growth_rate, reverse=True,
            )
            _take("TRENDING_LIVE", trending[:_remaining()])
    _take("EXPLORATION", _query_active_live(db, exclude_ids=selected_ids, limit=_remaining()))

    dynamic_state_by_id = compute_dynamic_state(db, [content.content_id for content in ordered_content], now=now)
    candidates = [
        _to_candidate(
            content, now=now, live_history=live_history, user_id=user_id, followed_creators=followed_creators,
            dynamic_state=dynamic_state_by_id.get(content.content_id, NEUTRAL_DYNAMIC_STATE),
        )
        for content in ordered_content
    ]
    logger.debug(
        "local LIVE candidate pool built userId=%s count=%d followedCreator=%d recentCreator=%d "
        "preferredCategory=%d trending=%d exploration=%d",
        user_id, len(candidates), bucket_counts.get("FOLLOWED_CREATOR", 0), bucket_counts.get("RECENT_CREATOR_INTERACTION", 0),
        bucket_counts.get("PREFERRED_CATEGORY", 0), bucket_counts.get("TRENDING_LIVE", 0), bucket_counts.get("EXPLORATION", 0),
    )
    return candidates
