"""Local VIDEO candidate generation, extracted from `app.api.candidate_routes.generate` so
BOTH that route and `app.services.recommendation_service.recommend()` (via
`app.services.providers.candidate_provider.resolve`) share the exact same implementation --
never duplicated, never called over HTTP by `recommend()`.

Preserves, byte-for-byte, the pre-existing business logic that used to live directly inside
the route handler: VIDEO-only/`is_active` filtering, the FOLLOWED_CREATOR/SESSION_INTEREST/
PREFERRED_CATEGORY/TRENDING/NEW_CONTENT/EXPLORATION source buckets (`SOURCE_ORDER`), their
quota/round-robin-fill logic, and the VIDEO-only interaction-history derivation this endpoint
has always used instead of the shared, non-VIDEO-scoped `user_profile_service` helpers.

Content Understanding / category inference (`app.ml.content_classifier`) never runs here:
this module only ever reads the already-persisted `Content.category` column, exactly like
the route it was extracted from always did -- classification happens once, at
`POST /contents` ingestion time (`app.services.content_enrichment_service`), never again on
this hot path.

Cold-start onboarding bootstrap (audited finding): PREFERRED_CATEGORY/EXPLORATION bucket
membership falls back to this user's persisted onboarding interests
(`app.services.user_context_provider.load_user_context`) ONLY when they have no real VIDEO
category-affinity evidence at all -- see `generate_video_candidates`'s own comment for the
exact precedence rule and why it mirrors `app.ml.reranker.explicit_interest_relevance`'s
existing cold-start-only gate rather than inventing a second, differently-behaved system.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import RECOMMENDATION_HISTORY_MAX_INTERACTIONS
from app.db.models import Content
from app.db.repositories import recent_interactions_for_ranking
from app.ml.dataset_builder import (
    MAX_CONTENT_AGE_HOURS,
    SESSION_MAX_EVENTS,
    SESSION_WINDOW,
)
from app.ml.feature_builder import build_profiles, is_video_hard_seen_event, target_for
from app.schemas.recommendation_schemas import Candidate
from app.services.user_context_provider import load_user_context

__all__ = ["SOURCE_ORDER", "VideoCandidate", "generate_video_candidates"]

# Unchanged from the pre-extraction route (see module docstring).
SOURCE_ORDER = ("FOLLOWED_CREATOR", "SESSION_INTEREST", "PREFERRED_CATEGORY", "TRENDING", "NEW_CONTENT", "EXPLORATION")
EXPLORATION_QUOTA_FRACTION = 0.5


class VideoCandidate(NamedTuple):
    """`source` is the bucket label (one of SOURCE_ORDER) this candidate was selected from --
    informational only (exposed by `POST /candidates/generate`'s response), never fed into
    `Candidate.candidate_source` (a different, SOCIAL/COLLABORATIVE-only concept -- see
    `app.schemas.recommendation_schemas.Candidate`'s own docstring)."""
    candidate: Candidate
    source: str


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _video_only_rows(rows: list, content_by_id: dict) -> list:
    """VIDEO-only interaction rows, given an already-fetched content_by_id map -- no I/O here,
    the caller owns the (single, bounded) Content query. `recent_interactions_for_ranking()`
    (shared repository) carries no VIDEO/LIVE discriminator of its own: event ingestion
    accepts LIVE_*/domain-ambiguous event types into this same table with no cross-check
    against the referenced content's own type. A row PROVABLY LIVE (its Content row exists and
    is explicitly content_type=="LIVE") is excluded; a row whose Content is missing/unavailable
    is left in unchanged -- mirrors training_service/user_behavior_provider/drift_service's
    identical compatibility rule: "unavailable" is not evidence of being LIVE."""
    return [row for row in rows if getattr(content_by_id.get(row.content_id), "content_type", "VIDEO") != "LIVE"]


def _session_preferred_categories(
    db: Session, user_id: str, now: datetime, content_by_id: dict, *, top_n: int = 2,
) -> set[str]:
    """Categories reflecting current-session intent ("what does this user want RIGHT NOW"),
    independent of `preferred` (long-term top VIDEO categories). Uses the exact same
    session-window definition as `app.ml.dataset_builder`'s session features (SESSION_WINDOW/
    SESSION_MAX_EVENTS) so "session interest" means the same thing to candidate sourcing as it
    does to the ranking model -- only positively-labeled session-window events count, ranked
    by frequency."""
    recent_rows = recent_interactions_for_ranking(db, user_id, limit=SESSION_MAX_EVENTS * 4)
    recent_rows = _video_only_rows(recent_rows, content_by_id)
    recent = [row for row in recent_rows if now - _utc(row.timestamp) <= SESSION_WINDOW]
    recent = recent[-SESSION_MAX_EVENTS:]
    positive_categories = Counter(row.category for row in recent if target_for(row) == 1)
    return {category for category, _ in positive_categories.most_common(top_n)}


def _to_candidate(content: Content, now: datetime, *, followed: bool, seen: bool, source: str) -> Candidate:
    created = _utc(content.created_at)
    # Clamped, unlike the pre-extraction route's plain dict output (which was never schema-
    # validated): constructing a real `Candidate` object enforces contentAgeHours<=
    # MAX_CONTENT_AGE_HOURS, the same cap app.ml.dataset_builder's own feature computation
    # already applies -- genuinely ancient content must not raise a ValidationError here.
    age_hours = min(MAX_CONTENT_AGE_HOURS, max(0.0, (now - created).total_seconds() / 3600))
    return Candidate.model_validate({
        "contentId": content.content_id, "creatorId": content.creator_id, "category": content.category,
        "contentPopularityScore": content.popularity_score, "contentAgeHours": round(age_hours, 2),
        "creatorFollowed": followed, "alreadySeen": seen,
        "title": content.title, "hashtags": content.hashtags, "topics": content.topics,
        "entities": content.entities, "subgenres": content.subgenres,
        # candidateSource/reason observability audit: carried on the Candidate object itself
        # (not discarded at the (candidate, source) tuple boundary the way it previously was
        # in app.services.providers.candidate_provider.resolve's LOCAL_GENERATION branch) so
        # it survives unchanged all the way to POST /recommendations' final response -- see
        # Candidate.local_bucket_source's own docstring.
        "localBucketSource": source,
    })


def generate_video_candidates(
    db: Session, user_id: str, *, limit: int, exclude_content_ids: frozenset[str] | None = None,
) -> list[VideoCandidate]:
    """Recommendation-eligible VIDEO candidates from local Postgres state, personalized by
    this user's own real interaction history (preferred/session-preferred categories,
    followed creators) exactly as `POST /candidates/generate` has always computed it -- see
    module docstring. Deterministic for identical DB state (stable sort by popularity_score
    within every bucket except NEW_CONTENT, which sorts by recency -- see the sort loop's own
    comment; stable bucket iteration order). Never raises for a cold-start user or an empty
    catalog -- both simply produce fewer/zero candidates.

    `exclude_content_ids` (default None -- `POST /candidates/generate`'s debug/analysis call
    site never passes it, so that endpoint keeps returning seen+unseen candidates with
    `alreadySeen` metadata, unchanged): when the caller passes the SAME authoritative
    already-seen set `recommend()` is about to hard-exclude
    (`app.services.recommendation_service.recommend`'s `seen_content_ids`), those content ids
    are removed from the eligible pool BEFORE bucket/quota selection instead of after -- so an
    already-seen item never occupies one of the `limit` slots only to be dropped later with no
    backfill. This is additive: bucket membership, quotas, the exploration cap, sort order and
    deduplication are computed exactly as before, just over a smaller starting `contents` list.

    History bounding (Task: unify VIDEO seen-state semantics, Part 9): bounded by
    `RECOMMENDATION_HISTORY_MAX_INTERACTIONS`, the SAME constant and query shape
    (`recent_interactions_for_ranking`) `app.services.providers.user_behavior_provider`'s
    authoritative history already uses -- config.py's own comment documents this as "shared
    bounds for the VIDEO ranking hot path's bounded interaction-history query", and this
    module IS part of that hot path. Previously unbounded (`interactions()`, no limit), which
    meant `preferred`/`followed`/the (now-fixed) `seen` metadata flag could each disagree with
    the authoritative, already-bounded ranking history about a user with more than that many
    total interactions. Bounded history is intentional replay/discovery behavior, not an
    accidental gap: content interacted with long enough ago to fall outside this window is
    deliberately eligible to resurface (repeated exposure WITHIN the window is a separate,
    already-handled concern -- `app.ml.replay_saturation_policy`, untouched by this change).
    """
    raw_history = recent_interactions_for_ranking(db, user_id, limit=RECOMMENDATION_HISTORY_MAX_INTERACTIONS)
    raw_content_ids = {row.content_id for row in raw_history}
    content_by_id = (
        {item.content_id: item for item in db.scalars(select(Content).where(Content.content_id.in_(raw_content_ids))).all()}
        if raw_content_ids else {}
    )
    history = _video_only_rows(raw_history, content_by_id)
    ranked_categories = sorted(
        build_profiles(history).get(user_id, {}).items(),
        key=lambda item: item[1]["affinity_score"], reverse=True,
    )
    preferred = {category for category, _ in ranked_categories[:2]}
    # Cold-start onboarding bootstrap (VIDEO onboarding candidate-generation gap, audited
    # finding): a user with NO real VIDEO category-affinity evidence at all previously
    # contributed nothing to PREFERRED_CATEGORY/EXPLORATION bucket membership -- their
    # persisted onboarding interests (app.services.user_context_provider.load_user_context)
    # only ever reached final RERANKING (app.ml.reranker.explicit_interest_relevance), never
    # candidate RETRIEVAL, so genuinely relevant content for a declared interest could be
    # excluded from the pool before the model ever saw it. This is a simple binary fallback,
    # not a blended weight (no product-approved weighting formula exists to invent one):
    # onboarding interests are used ONLY when `preferred` is empty, and real behavioral
    # evidence -- however small -- always wins outright and is never overridden. Mirrors
    # `app.services.providers.live_candidate_provider`'s identical existing precedent (it
    # already calls `load_user_context` for the same cross-source-composition purpose) and
    # reuses `explicit_interest_relevance`'s own "cold-start-only" precedence rule, so
    # candidate-generation-time and reranking-time onboarding influence never disagree about
    # WHEN onboarding should apply. `interests` is already `normalize_tokens`-normalized
    # (uppercase) by `UserContext` itself -- directly comparable to `Content.category`
    # (also stored upper-cased), no second normalizer.
    if not preferred:
        user_context = load_user_context(db, user_id)
        if user_context is not None and user_context.interests:
            preferred = set(user_context.interests)
    followed = {row.creator_id for row in history if row.creator_followed or row.event_type == "CREATOR_FOLLOWED"}
    # Canonical VIDEO hard-seen policy (app.ml.feature_builder.is_video_hard_seen_event), the
    # SAME one app.services.providers.user_behavior_provider's authoritative seen_content_ids
    # uses -- NOT "any Interaction row exists" (that used to make a CONTENT_NOT_INTERESTED-only
    # rejection set Candidate.alreadySeen=True here, which then hard-excluded the content via
    # recommend()'s already_seen_content_ids union, even though NOT_INTERESTED is deliberately
    # excluded from the authoritative definition -- a proven, now-fixed inconsistency).
    seen = {row.content_id for row in history if is_video_hard_seen_event(row.event_type)}
    # VIDEO-only, active-only: a LIVE-typed Content row must never enter this pool.
    contents = list(db.scalars(select(Content).where(Content.is_active.is_(True), Content.content_type == "VIDEO")).all())
    if exclude_content_ids:
        contents = [content for content in contents if content.content_id not in exclude_content_ids]
    now = datetime.now(timezone.utc)
    session_preferred = _session_preferred_categories(db, user_id, now, content_by_id)

    pools: dict[str, list[Content]] = defaultdict(list)
    for content in contents:
        created = _utc(content.created_at)
        if content.category in session_preferred: pools["SESSION_INTEREST"].append(content)
        if content.category in preferred: pools["PREFERRED_CATEGORY"].append(content)
        if content.creator_id in followed: pools["FOLLOWED_CREATOR"].append(content)
        if content.popularity_score >= 0.70: pools["TRENDING"].append(content)
        if (now - created).total_seconds() <= 72 * 3600: pools["NEW_CONTENT"].append(content)
        # Exploration means genuinely novel to this user right now: excluded from both the
        # long-term preferred categories AND whatever the current session is already about.
        if content.category not in preferred and content.category not in session_preferred:
            pools["EXPLORATION"].append(content)
    for source in SOURCE_ORDER:
        # NEW_CONTENT's entire purpose is freshness -- sorting it by popularity (like every
        # other bucket) silently defeats that: any catalog with more than `quota` items
        # already inside the 72h window AND already more popular than a brand-new (neutral
        # popularity_score) item pushes the genuinely fresh item below the bucket's own
        # first-pass quota cut, then TRENDING's backfill (earlier in SOURCE_ORDER, typically a
        # much larger pool) fills the rest of `limit` before NEW_CONTENT is ever revisited --
        # a real, reproduced starvation bug (Task: VIDEO fresh-content starvation audit), not
        # a hypothetical one. Recency (`created_at` descending) is the only sort key that
        # actually matches this bucket's name/intent; every other bucket's semantics
        # (FOLLOWED_CREATOR/SESSION_INTEREST/PREFERRED_CATEGORY/TRENDING/EXPLORATION) are about
        # relevance/popularity within an already-membership-filtered pool, where popularity
        # ordering is correct and unchanged.
        if source == "NEW_CONTENT":
            pools[source].sort(key=lambda item: _utc(item.created_at), reverse=True)
        else:
            pools[source].sort(key=lambda item: item.popularity_score, reverse=True)

    def _build(content: Content, source: str) -> VideoCandidate:
        return VideoCandidate(
            _to_candidate(content, now, followed=content.creator_id in followed, seen=content.content_id in seen,
                          source=source),
            source,
        )

    result: list[VideoCandidate] = []
    added: set[str] = set()
    base_quota = max(1, limit // len(SOURCE_ORDER))
    quota_by_source = {
        source: (max(1, int(base_quota * EXPLORATION_QUOTA_FRACTION)) if source == "EXPLORATION" else base_quota)
        for source in SOURCE_ORDER
    }
    counts: Counter = Counter()
    for source in SOURCE_ORDER:
        for content in pools[source]:
            if content.content_id in added: continue
            result.append(_build(content, source))
            added.add(content.content_id)
            counts[source] += 1
            if counts[source] >= quota_by_source[source]: break
    # Fill remaining capacity round-robin from every source, preserving deduplication.
    # EXPLORATION is the one exception: its quota is a hard cap here too (not just a
    # first-pass preference like every other source), otherwise a thin catalog would let
    # "exploration" flood the fill pass -- exploration must stay small, not dominate the feed.
    for source in SOURCE_ORDER:
        for content in pools[source]:
            if len(result) >= limit: break
            if source == "EXPLORATION" and counts[source] >= quota_by_source[source]: break
            if content.content_id not in added:
                result.append(_build(content, source))
                added.add(content.content_id)
                counts[source] += 1
    return result[:limit]
