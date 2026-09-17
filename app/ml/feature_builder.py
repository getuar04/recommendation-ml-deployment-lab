import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

# Shared across app/ml/dataset_builder.py, app/services/user_profile_service.py, and this
# module so "completed"/"fast-skip" mean the same watch percentage everywhere a label or
# feature is derived from them -- a value changed in only one place would otherwise silently
# skew training/serving in the others.
COMPLETION_WATCH_PERCENTAGE_THRESHOLD = 90
FAST_SKIP_WATCH_PERCENTAGE_THRESHOLD = 20


class InteractionSignals(NamedTuple):
    """Canonical effective engagement flags for stored rows and request objects.

    Event-type aliases and explicit boolean columns are semantically equivalent. Reading
    both here keeps historical rows (whose booleans may be false) compatible while letting
    future ingestion persist the normalized booleans as well.
    """

    liked: bool
    shared: bool
    favorited: bool
    commented: bool
    creator_followed: bool


def interaction_signals(row: Any) -> InteractionSignals:
    event_type = getattr(row, "event_type", "")
    event_type = getattr(event_type, "value", event_type)
    return InteractionSignals(
        liked=bool(getattr(row, "liked", False) or event_type == "CONTENT_LIKED"),
        shared=bool(getattr(row, "shared", False) or event_type == "CONTENT_SHARED"),
        favorited=bool(getattr(row, "favorited", False) or event_type == "CONTENT_FAVORITED"),
        commented=bool(getattr(row, "commented", False) or event_type == "CONTENT_COMMENTED"),
        creator_followed=bool(
            getattr(row, "creator_followed", False) or event_type == "CREATOR_FOLLOWED"
        ),
    )


def watch_percentage(watch: float | None, duration: float | None) -> float | None:
    return round(watch / duration * 100, 4) if watch is not None and duration else None

def affinity_score(raw: float) -> float:
    return round(1 / (1 + math.exp(-raw / 10)), 6)

def build_profiles(rows, as_of: datetime | None = None):
    now = as_of or datetime.now(timezone.utc)
    profiles: defaultdict[Any, defaultdict[Any, defaultdict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for r in rows:
        signals = interaction_signals(r)
        d = profiles[r.user_id][r.category]
        d["interaction_count"] += 1
        d["impression_count"] += r.event_type == "VIDEO_IMPRESSION"
        d["watch_count"] += r.watch_percentage is not None
        if r.watch_percentage is not None:
            d["watch_sum"] += r.watch_percentage
        d["completion_count"] += r.event_type == "VIDEO_COMPLETED" or (r.watch_percentage or 0) >= COMPLETION_WATCH_PERCENTAGE_THRESHOLD
        d["skip_count"] += r.event_type == "VIDEO_SKIPPED"
        d["fast_skip_count"] += r.event_type == "VIDEO_SKIPPED" and (r.watch_percentage or 0) < FAST_SKIP_WATCH_PERCENTAGE_THRESHOLD
        d["like_count"] += signals.liked
        d["share_count"] += signals.shared
        d["favorite_count"] += signals.favorited
        d["comment_count"] += signals.commented
        d["creator_follow_count"] += signals.creator_followed
        d["rewatch_count"] += r.event_type == "VIDEO_REWATCHED"
        d["not_interested_count"] += r.event_type == "CONTENT_NOT_INTERESTED"
        ts = r.timestamp.replace(tzinfo=timezone.utc) if r.timestamp.tzinfo is None else r.timestamp
        d["recent_interaction_count"] += ts >= now - timedelta(days=7)
        d["last_interaction_timestamp"] = max(d.get("last_interaction_timestamp", ts), ts)
    for cats in profiles.values():
        for d in cats.values():
            d["average_watch_percentage"] = d["watch_sum"] / d["watch_count"] if d["watch_count"] else 0
            raw = d["completion_count"]*4+d["like_count"]*3+d["share_count"]*5+d["favorite_count"]*5+d["comment_count"]*2+d["creator_follow_count"]*6+d["rewatch_count"]*4-d["fast_skip_count"]*4-d["not_interested_count"]*8
            d["affinity_score"] = affinity_score(raw)
    return profiles


def content_popularity_score(rows: Any) -> float:
    """Real-engagement-derived popularity for one piece of VIDEO content, aggregated across
    ALL users' interactions with it -- the ownership/calculation-audit replacement for what
    used to be a static, caller-forgeable `ContentCreate.popularity_score` default that never
    updated after creation (see app.api.content_routes.ContentCreate and
    app.services.event_service.store_event, which calls this after every new real VIDEO
    interaction).

    Deliberately reuses, unmodified, the exact same weighted-signal formula
    `build_profiles()` already uses for its per-(user, category) `affinity_score` above (see
    its own `raw = ...` line) -- applied here across all interactions for one content_id
    instead of one user's interactions with one category -- and the same `affinity_score()`
    sigmoid squashing. This is a deliberate choice, not a coincidence: these weights are
    already the established, audited definition of "what counts as positive/negative
    engagement and how strongly" in this codebase; a popularity signal needs the same
    definition, not a second, independently invented one.

    Bounded in (0, 1). Zero evidence (`rows` empty, e.g. brand-new content) returns exactly
    0.5 -- not an arbitrary magic constant chosen to "look reasonable", but `affinity_score(0)`,
    the same principled "no evidence either way" neutral prior this codebase already uses
    elsewhere for an unmeasured affinity (e.g. the reranker's `session_category_affinity`
    default). This must NOT punish new content: `content.popularity_score` is only ever a
    *soft* ranking/bucket-sort signal (app.services.providers.video_candidate_provider's
    TRENDING bucket and within-bucket sort order) -- NEW_CONTENT/EXPLORATION candidate
    eligibility, semantic relevance, and freshness are computed independently of it.

    O(interaction count for this one content_id) per call -- a full recompute from the
    current real aggregate, not an incremental counter. Acceptable at local/dev scale (one
    indexed query per new VIDEO interaction on that content); a high-traffic production
    deployment would eventually want an incrementally-maintained counter instead of a full
    recompute per event -- a real future scaling concern, not addressed here."""
    completion = like = share = favorite = comment = follow = rewatch = fast_skip = not_interested = 0
    for r in rows:
        signals = interaction_signals(r)
        completion += r.event_type == "VIDEO_COMPLETED" or (r.watch_percentage or 0) >= COMPLETION_WATCH_PERCENTAGE_THRESHOLD
        fast_skip += r.event_type == "VIDEO_SKIPPED" and (r.watch_percentage or 0) < FAST_SKIP_WATCH_PERCENTAGE_THRESHOLD
        like += signals.liked
        share += signals.shared
        favorite += signals.favorited
        comment += signals.commented
        follow += signals.creator_followed
        rewatch += r.event_type == "VIDEO_REWATCHED"
        not_interested += r.event_type == "CONTENT_NOT_INTERESTED"
    raw = completion*4 + like*3 + share*5 + favorite*5 + comment*2 + follow*6 + rewatch*4 - fast_skip*4 - not_interested*8
    return affinity_score(raw)

# The only two EXPLICIT-rejection event types in app.schemas.event_schemas.EventType (every
# other event -- including VIDEO_SKIPPED -- is an ambient/implicit signal, not a dedicated "I
# don't want this" action). Kept here, not imported from EventType, to avoid a schemas<->ml
# import cycle; app.ml.dataset_builder's `update()` already hardcodes the CONTENT_NOT_INTERESTED
# string for the same reason (see its per-event affinity-delta formula).
EXPLICIT_NEGATIVE_EVENT_TYPES = frozenset({"CONTENT_NOT_INTERESTED", "LIVE_NOT_INTERESTED"})


def is_video_hard_seen_event(event_type: object) -> bool:
    """Canonical VIDEO HARD-SEEN/exposure policy -- the ONE source of truth every consumer
    that needs to know "did this interaction make the content ineligible for normal future
    candidate generation" must call, instead of independently re-deriving its own event-type
    list (the exact drift this function was added to eliminate --
    `app.services.providers.user_behavior_provider`'s authoritative `seen_content_ids`,
    `app.services.providers.video_candidate_provider`'s local-generation pre-bucketing
    exclusion, and its `Candidate.alreadySeen` metadata flag all call this now).

    HARD SEEN: every VIDEO EventType except an EXPLICIT_NEGATIVE_EVENT_TYPES rejection.
    Exposure is a logical prerequisite for every other event row to exist at all -- a
    VIDEO_IMPRESSION/STARTED/WATCHED/COMPLETED/SKIPPED/REWATCHED row, or a CONTENT_LIKED/
    SHARED/FAVORITED/COMMENTED/CREATOR_FOLLOWED row tied to this specific content_id, cannot
    be logged for content the user was never shown -- so none of these need a
    NEEDS_PRODUCT_DECISION carve-out; they are exposure by construction.

    NOT HARD SEEN: CONTENT_NOT_INTERESTED (and its LIVE counterpart, reachable only on rows
    this module's VIDEO-only callers have already filtered out before calling this). This is
    pre-existing, deliberate product intent (`target_for`'s own docstring above; also see
    `app.ml.dataset_builder.FeatureHistory.update`'s per-event affinity weighting), not a new
    rule invented here: CONTENT_NOT_INTERESTED is a dedicated, explicit "I don't want this"
    rejection, never proof the user was exposed to/consumed the content. It must still drive
    NEGATIVE PREFERENCE suppression via the existing affinity/reranker machinery -- a
    completely separate concern this function has no effect on either way -- just never
    through hard exclusion from future candidate generation.
    """
    return getattr(event_type, "value", event_type) not in EXPLICIT_NEGATIVE_EVENT_TYPES

TARGET_DEFINITION = (
    "positive (1): watch_percentage >= 70, or a VIDEO_COMPLETED event (even when "
    "watch_percentage cannot be derived -- matches this module's own completion_count/ "
    "_completed() precedent, which treats the event type alone as sufficient completion "
    "evidence), or liked/shared/favorited/creator_followed; "
    "negative (0): watch_percentage < 20 and none of liked/shared/favorited/creator_followed, "
    "OR an explicit rejection event (CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED) regardless of "
    "watch_percentage or any positive flag also set on the same row; "
    "neutral (excluded from training): everything else."
)


def target_for(row):
    # Explicit rejection takes precedence over everything else, including a high watch
    # percentage or a liked/shared/favorited/creator_followed flag also present on the same
    # row: CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED is a deliberate, single-purpose "I don't
    # want this" action, not an ambiguous engagement signal -- the training label must not let
    # watch-percentage ambiguity (or a contradictory flag) launder an explicit rejection into a
    # neutral or positive example. This matches this codebase's existing per-event affinity
    # weighting (app.ml.dataset_builder.FeatureHistory.update), which already treats
    # CONTENT_NOT_INTERESTED as the single strongest signal it defines (-8, larger in magnitude
    # than any positive weight alone).
    if row.event_type in EXPLICIT_NEGATIVE_EVENT_TYPES:
        return 0
    signals = interaction_signals(row)
    wp = row.watch_percentage
    # A VIDEO_COMPLETED event is sufficient evidence of a positive outcome on its own, even when
    # watch_percentage can't be derived (e.g. the client omitted watchTimeSeconds/
    # contentDurationSeconds) -- otherwise a genuinely completed watch silently contributes no
    # training label at all. This mirrors completion_count above and dataset_builder._completed(),
    # which already treat the event type alone as sufficient; target_for was the one place in
    # this module that didn't.
    positive = (wp is not None and wp >= 70) or row.event_type == "VIDEO_COMPLETED" or any((
        signals.liked, signals.shared, signals.favorited, signals.creator_followed,
    ))
    negative = wp is not None and wp < FAST_SKIP_WATCH_PERCENTAGE_THRESHOLD and not any((
        signals.liked, signals.shared, signals.favorited, signals.creator_followed,
    ))
    return 1 if positive else (0 if negative else None)

