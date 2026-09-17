"""Local, cross-format personalization SIGNALS for LIVE candidate retrieval (LIVE candidate-
personalization foundation): what RMS can honestly derive about a user from its own already-
stored local state, without a new synchronous Follow Service/UBS/Candidate Service call.

This module answers "what does RMS already know locally that could make LIVE candidate
MEMBERSHIP/priority reflect this user" -- it does not build `LiveCandidate` objects itself
(`app.services.providers.live_candidate_provider` does that) and does not score/rank anything.

Two signal families:
  - Creator-level: `followed_creator_ids` (any content type -- "follow" is a creator-level
    relationship, not a VIDEO- or LIVE-specific one) and `recent_positive_video_creator_ids`
    (Ylli's cross-format requirement: strong recent VIDEO engagement with a creator should be
    usable evidence for that creator's LIVE, even with zero LIVE history).
  - Category-level: `recent_positive_video_categories` plus `preferred_live_categories`, which
    merges onboarding interests (if persisted), real LIVE-history category affinity, and real
    VIDEO-history category affinity into one deduplicated, priority-ordered list.

"Positive" VIDEO engagement reuses `app.ml.feature_builder.target_for` verbatim (the SAME
definition the VIDEO training/ranking path already uses for "this row is real positive
evidence") -- never a second, independently-invented engagement threshold.

Does NOT invent a taxonomy mapping: category matching is a plain, case-normalized string
comparison against `Content.category`/`UserContext.interests` as they already exist today
(the legacy/bootstrap vocabulary) -- no canonical taxonomy is referenced or required.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import RECOMMENDATION_HISTORY_MAX_INTERACTIONS
from app.db.models import Content, Interaction
from app.ml.feature_builder import build_profiles, content_popularity_score, target_for
from app.ml.live_dataset_builder import LiveHistory
from app.ml.semantic_tokens import normalize_token
from app.schemas.recommendation_schemas import UserContext

__all__ = [
    "followed_creator_ids",
    "not_interested_stream_ids",
    "preferred_live_categories",
    "recent_positive_video_categories",
    "recent_positive_video_creator_ids",
    "video_bootstrap_affinities",
    "video_evidence_count",
]


def followed_creator_ids(db: Session, user_id: str, *, limit: int) -> list[str]:
    """Distinct creator ids this user has ever produced a `creator_followed=True` interaction
    row for, ANY content type (a VIDEO `CREATOR_FOLLOWED` and a LIVE `LIVE_CREATOR_FOLLOWED`
    are equally valid evidence of following that creator), most-recent-first. Deliberately
    de-duplicated in Python rather than `SELECT DISTINCT ... ORDER BY` (portable across this
    project's SQLite-test / Postgres-production engines without a Postgres-only
    `DISTINCT ON`)."""
    # Eventual-consistency audit (Task: content_pending evidence-leak audit): this query
    # bypasses app.db.repositories entirely (its own direct select(Interaction)), so it must
    # apply the same content_pending exclusion those functions apply -- see
    # app.db.repositories.interactions()'s own comment for why.
    rows = db.scalars(
        select(Interaction)
        .where(Interaction.user_id == user_id, Interaction.creator_followed.is_(True), Interaction.content_pending.is_(False))
        .order_by(Interaction.timestamp.desc(), Interaction.id.desc())
        .limit(RECOMMENDATION_HISTORY_MAX_INTERACTIONS)
    ).all()
    seen: list[str] = []
    for row in rows:
        if row.creator_id and row.creator_id not in seen:
            seen.append(row.creator_id)
        if len(seen) >= limit:
            break
    return seen


def not_interested_stream_ids(db: Session, user_id: str, *, limit: int) -> set[str]:
    """Stream (content) ids this user has explicitly sent a `LIVE_NOT_INTERESTED` event for.
    Used only for the bounded, STREAM-SPECIFIC reranking suppression in
    `app.ml.live_reranker.live_adjusted_score` (Part 13 of the dynamic-signal task) -- never a
    hard exclusion, and never generalized to a creator/category-level penalty (a genuinely
    unresolved product decision, deliberately left undecided rather than guessed at)."""
    # Eventual-consistency audit (Task: content_pending evidence-leak audit): see
    # followed_creator_ids's identical comment above for why this direct query needs the
    # same content_pending exclusion app.db.repositories's functions already apply.
    rows = db.scalars(
        select(Interaction)
        .where(Interaction.user_id == user_id, Interaction.event_type == "LIVE_NOT_INTERESTED", Interaction.content_pending.is_(False))
        .order_by(Interaction.timestamp.desc(), Interaction.id.desc())
        .limit(RECOMMENDATION_HISTORY_MAX_INTERACTIONS)
    ).all()
    ids: set[str] = set()
    for row in rows:
        ids.add(row.content_id)
        if len(ids) >= limit:
            break
    return ids


def _recent_video_rows(db: Session, user_id: str) -> list[Interaction]:
    """Bounded, most-recent-first VIDEO-only interaction rows for this user (positive AND
    negative -- unfiltered by engagement outcome). VIDEO-only via the same Content-lookup
    filter `app.services.providers.user_behavior_provider._local_state` and
    `app.services.providers.live_history_provider.load_live_history_for_user` already use:
    a row PROVABLY LIVE (its Content row exists and is content_type=="LIVE") is excluded; a
    row whose Content is missing/unavailable is kept ("unavailable" is not evidence of being
    LIVE, matching that same established precedent)."""
    # Eventual-consistency audit (Task: content_pending evidence-leak audit): see
    # followed_creator_ids's identical comment above for why this direct query needs the
    # same content_pending exclusion app.db.repositories's functions already apply -- this
    # feeds recent_positive_video_creator_ids/recent_positive_video_categories/
    # video_bootstrap_affinities/video_evidence_count below, every one of them cross-format
    # LIVE evidence.
    rows = db.scalars(
        select(Interaction)
        .where(Interaction.user_id == user_id, Interaction.content_pending.is_(False))
        .order_by(Interaction.timestamp.desc(), Interaction.id.desc())
        .limit(RECOMMENDATION_HISTORY_MAX_INTERACTIONS)
    ).all()
    if not rows:
        return []
    content_ids = {row.content_id for row in rows}
    content_by_id = {item.content_id: item for item in db.scalars(select(Content).where(Content.content_id.in_(content_ids))).all()}
    return [row for row in rows if getattr(content_by_id.get(row.content_id), "content_type", "VIDEO") != "LIVE"]


def _recent_positive_video_rows(db: Session, user_id: str) -> list[Interaction]:
    """Bounded, most-recent-first VIDEO interaction rows for this user with real positive
    engagement (`app.ml.feature_builder.target_for` == 1)."""
    return [row for row in _recent_video_rows(db, user_id) if target_for(row) == 1]


def recent_positive_video_creator_ids(db: Session, user_id: str, *, limit: int) -> list[str]:
    """Ylli's cross-format requirement: creators this user recently, positively engaged with
    via VIDEO (liked/shared/favorited/creator-followed/watch_percentage>=70, or
    VIDEO_COMPLETED) -- usable LIVE candidate-priority evidence even when this user has zero
    LIVE history at all, most-recent-first, de-duplicated."""
    seen: list[str] = []
    for row in _recent_positive_video_rows(db, user_id):
        if row.creator_id and row.creator_id not in seen:
            seen.append(row.creator_id)
        if len(seen) >= limit:
            break
    return seen


def recent_positive_video_categories(db: Session, user_id: str, *, limit: int) -> list[str]:
    """Categories this user recently, positively engaged with via VIDEO, most-recent-first,
    de-duplicated. Feeds `preferred_live_categories` below as the weakest-priority (cross-
    format) evidence tier."""
    seen: list[str] = []
    for row in _recent_positive_video_rows(db, user_id):
        category = (row.category or "").upper()
        if category and category not in seen:
            seen.append(category)
        if len(seen) >= limit:
            break
    return seen


def preferred_live_categories(
    *, user_context: UserContext | None, live_history: LiveHistory | None,
    video_categories: list[str], user_id: str, limit: int,
) -> list[str]:
    """Merges three category-preference sources into one deduplicated, priority-ordered list:
    1. Persisted onboarding interests (`UserContext.interests`) -- explicit, user-declared
       intent, already `normalize_token`-normalized (see `app.ml.reranker.
       explicit_interest_relevance`'s identical VIDEO-side precedent for this exact
       string-equality-only matching contract -- no taxonomy mapping involved).
    2. Real LIVE-history category affinity (`LiveHistory.top_categories`) -- direct-domain
       evidence.
    3. Real VIDEO-history category affinity (cross-format, weakest tier).

    An empty result is a legitimate outcome (a genuine cold-start user), never fabricated.
    """
    categories: list[str] = []
    if user_context is not None:
        categories.extend(user_context.interests)
    if live_history is not None:
        categories.extend(live_history.top_categories(user_id, limit=limit))
    categories.extend(video_categories)

    seen: list[str] = []
    for category in categories:
        token = normalize_token(category)
        if token and token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    return seen


def video_bootstrap_affinities(db: Session, user_id: str) -> dict[str, dict[str, float]]:
    """Cross-format LIVE-relevance bootstrap evidence (architecture requirement: "a LIVE
    stream may begin with very little LIVE-specific behavioral evidence... RMS should be
    able to bootstrap LIVE relevance using... historical VIDEO category interests, creator
    affinity from prior content interactions"). Computed ONCE per request (not per
    candidate) and handed to `app.ml.live_dataset_builder.live_feature_snapshot`, which
    only substitutes a value here in place of the flat neutral default when this user's
    real LIVE-specific evidence for that exact category/creator is zero -- never overrides
    genuine LIVE evidence.

    Returns `{"category": {CATEGORY: affinity, ...}, "creator": {creator_id: affinity, ...}}`,
    each affinity in [0, 1]. Deliberately reuses, unmodified, two already-established
    formulas -- never a third, independently-invented definition of engagement:
      - category: `app.ml.feature_builder.build_profiles()`'s own per-(user, category)
        `affinity_score` (the same VIDEO personalization-profile computation
        `app.services.providers.video_candidate_provider` already uses).
      - creator: `app.ml.feature_builder.content_popularity_score()`'s identical
        weighted-engagement formula, applied to this user's VIDEO rows for one creator_id
        instead of all users' rows for one content_id -- the formula is generic over "a set
        of interaction rows" regardless of what they are grouped by.
    An empty result for a user with no VIDEO evidence is legitimate (falls through to
    `live_feature_snapshot`'s own pre-existing 0.5 neutral default), never fabricated.
    """
    rows = _recent_video_rows(db, user_id)
    if not rows:
        return {"category": {}, "creator": {}}
    category_profile = build_profiles(rows).get(user_id, {})
    category_affinities = {
        category: stats["affinity_score"] for category, stats in category_profile.items() if category
    }
    rows_by_creator: dict[str, list[Interaction]] = {}
    for row in rows:
        if row.creator_id:
            rows_by_creator.setdefault(row.creator_id, []).append(row)
    creator_affinities = {
        creator_id: content_popularity_score(creator_rows) for creator_id, creator_rows in rows_by_creator.items()
    }
    return {"category": category_affinities, "creator": creator_affinities}


def video_evidence_count(db: Session, user_id: str) -> int:
    """This user's real VIDEO interaction count -- the cross-format evidence count LIVE
    lifecycle/strategy reporting falls back to when this user has zero real LIVE-native
    evidence (`app.services.live_recommendation_service.recommend_live`), reusing
    `app.services.recommendation_service.strategy_for`'s existing COLD_START/HYBRID/
    PERSONALISED_ML thresholds rather than inventing a LIVE-specific one."""
    return len(_recent_video_rows(db, user_id))
