"""Pure LIVE business reranking and heuristic explanation logic (no I/O).

Mirrors the VIDEO/`app.ml.reranker` split: kept separate from the online service so
the same logic is reusable for offline evaluation without risking drift.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

ALREADY_JOINED_PENALTY = 0.82
MAX_CREATOR_IN_TOP = 2
MAX_CONSECUTIVE_CATEGORY = 2

# Shared with app.services.providers.live_candidate_provider's TRENDING_LIVE bucket (dynamic-
# state foundation) -- one threshold for "this stream's real join-rate momentum counts as
# trending", not two independently-chosen numbers that could silently drift apart.
TRENDING_GROWTH_RATE_THRESHOLD = 0.15

# Recent-engagement reranking boost (dynamic-state foundation, Part 8): bounded, reranking-
# only -- the pre-existing 17-feature LIVE_FEATURES contract has no per-stream recent-
# engagement slot, and adding one would require retraining the active LIVE model, out of this
# task's scope. Gifts outweigh likes (a rarer, stronger engagement signal); log1p keeps one
# viral spike from ever dominating the model's own trained score.
MAX_ENGAGEMENT_BOOST = 0.15
GIFT_ENGAGEMENT_WEIGHT = 3

# Stream-specific LIVE_NOT_INTERESTED suppression (Part 13): a bounded penalty, never a VIDEO-
# style permanent hard exclusion -- applied only while the SAME stream the user explicitly
# rejected is still the one being scored, in this same request. Broader creator/category-level
# negative affinity for LIVE is a genuinely unresolved product decision, deliberately NOT
# implemented here (see this module's own callers for the full reasoning).
NOT_INTERESTED_STREAM_PENALTY = 0.5


def live_explanation(row: dict[str, Any], candidate: Any) -> str:
    """Transparent heuristic explanation; this is not model attribution or SHAP."""
    if candidate.creator_followed:
        return "FOLLOWED_CREATOR"
    if row["live_category_affinity"] >= .7:
        return "PREFERRED_LIVE_CATEGORY"
    if candidate.viewer_growth_rate > TRENDING_GROWTH_RATE_THRESHOLD:
        return "TRENDING_LIVE"
    if candidate.region_match:
        return "REGION_MATCH"
    return "EXPLORATION"


def live_engagement_boost(recent_likes: int, recent_gifts: int) -> float:
    """Bounded, log-scaled multiplicative boost >= 1.0 from real recent stream-level likes/
    gifts (app.services.providers.live_dynamic_state_provider). Capped at
    `1 + MAX_ENGAGEMENT_BOOST` -- a modest nudge on top of the model's own score, never
    allowed to override it."""
    weighted = recent_likes + recent_gifts * GIFT_ENGAGEMENT_WEIGHT
    if weighted <= 0:
        return 1.0
    return 1.0 + min(MAX_ENGAGEMENT_BOOST, math.log1p(weighted) * 0.05)


def live_adjusted_score(
    model_score: float, *, already_joined: bool, recent_likes: int = 0, recent_gifts: int = 0,
    not_interested_for_stream: bool = False,
) -> float:
    score = model_score * (ALREADY_JOINED_PENALTY if already_joined else 1.0)
    score *= live_engagement_boost(recent_likes, recent_gifts)
    if not_interested_for_stream:
        score *= NOT_INTERESTED_STREAM_PENALTY
    return score


def live_rerank(ranked: list[tuple[float, Any, str]], limit: int) -> list[tuple[float, Any, str]]:
    """Preserve adjusted-score order while capping creator repetition and consecutive-category
    runs -- but never under-fill below `min(limit, len(ranked))` just to hold that cap: a
    candidate skipped by either cap in the first pass is kept aside, and if the capped
    selection still falls short of `limit`, remaining slots are backfilled from those skipped
    candidates (still in their original score order) with the caps no longer enforced. This
    only ever activates when there is no genuinely diverse alternative left to prefer instead
    -- diversity still wins whenever a real alternative exists; it only yields to avoid
    returning fewer results than are actually available.

    Callers (`app.services.live_recommendation_service.recommend_live`) assign `rank`/display
    order directly from this function's return order and expect it to stay score-descending --
    the backfill step re-sorts by score after merging, so a skipped-then-backfilled item (which
    can have a higher original score than some items chosen before it) never ends up displayed
    out of score order.

    Ties are broken deterministically by the pre-rerank order, since `ranked` is
    expected to already be sorted descending by adjusted score.
    """
    chosen: list[tuple[float, Any, str]] = []
    skipped: list[tuple[float, Any, str]] = []
    creators: Counter[str] = Counter()
    for item in ranked:
        _, candidate, _ = item
        if creators[candidate.creator_id] >= MAX_CREATOR_IN_TOP:
            skipped.append(item)
            continue
        if len(chosen) >= MAX_CONSECUTIVE_CATEGORY and all(
            previous[1].category == candidate.category for previous in chosen[-MAX_CONSECUTIVE_CATEGORY:]
        ):
            skipped.append(item)
            continue
        chosen.append(item)
        creators[candidate.creator_id] += 1
        if len(chosen) >= limit:
            return chosen
    if not skipped:
        return chosen
    for item in skipped:
        if len(chosen) >= limit:
            break
        chosen.append(item)
    chosen.sort(key=lambda item: item[0], reverse=True)
    return chosen
