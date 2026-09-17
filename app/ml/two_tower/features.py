"""Candidate-independent User Tower / Content Tower input vectors.

Reuses, unmodified:
  - app.ml.dataset_builder.FeatureHistory (its internal `categories`/`tokens`/`creators`/
    `users` dicts, and its public `.features()`/`.update()` for the category-affinity numbers)
  - app.ml.feature_builder.affinity_score (the exact sigmoid used by the trained ranker)
  - app.ml.semantic_tokens.extract_title_tokens (identical title tokenization)

What's new here, and why: the ranker's own `FeatureHistory.features()` returns *paired*
(user, candidate) features -- e.g. `hashtag_affinity` only reflects tokens the ONE candidate
being scored happens to carry, and `category_affinity` is evaluated only for that candidate's
own category. A Two-Tower retrieval model needs the opposite: one user vector and one content
vector, each computed *independently*, so the user vector can be compared by dot product
against every candidate's content vector without re-deriving it per candidate. This module
builds those independent vectors:
  - the user vector aggregates the user's affinity across EVERY known category (not just one
    target category) plus a hashed summary of every semantic token they have ever positively/
    negatively interacted with;
  - the content vector is a one-hot category + hashed creator identity + hashed semantic tags
    + popularity, using no user-specific state at all.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

from app.ml.dataset_builder import SEMANTIC_FIELDS, FeatureHistory
from app.ml.feature_builder import affinity_score
from app.ml.semantic_tokens import extract_title_tokens

# Deliberately small (spec: "avoid unnecessary complexity", CPU-friendly PoC). Feature hashing
# (not a learned embedding table) keeps both towers' input dimensionality fixed regardless of
# catalog vocabulary size or the number of distinct creators -- the same tradeoff
# app.ml.dataset_builder's own semantic-affinity design already makes for the ranker (see that
# module's docstring), applied here to raw token/creator IDENTITY instead of aggregated affinity.
#
# Phase 3.1 tuning (measured, not guessed -- see scripts/run_two_tower_poc.py's diagnostics):
# at SEMANTIC_HASH_DIM=16 with ~320 distinct catalog tokens (~20 tokens/bucket), 14 of SPORT's
# 16 hash buckets were also used by FITNESS's vocabulary (87.5% bucket overlap) -- the semantic
# hash was actively confusable between adjacent categories instead of helping distinguish them.
# Doubling to 32 roughly halves the average tokens/bucket, cutting cross-category collisions.
#
# Phase 3.1 MUSIC-collision follow-up (measured, REJECTED): widening further to 64/16 was
# tried to address MUSIC's 100%-collided semantic buckets and 2/3-collided creator buckets, but
# it made retrieval strictly worse (Recall@5/10/20 dropped further, SPORT recovery weakened, and
# MUSIC still did not return to any demo user's Top-10) -- collision width is NOT the dominant
# cause of the MUSIC failure (see investigation notes), so 32/8 is kept pending further analysis
# rather than shipping an unproven, worse-performing change.
SEMANTIC_HASH_DIM = 32
CREATOR_HASH_DIM = 8

# Phase 3.1 tuning (measured): at equal weight, a content item's semantic-hash block had L2
# norm ~2.0-2.8 (varies with token count) against the one-hot category block's fixed norm of
# 1.0 -- the noisier, higher-cardinality hashed signal was structurally out-weighing the single
# clean, unambiguous category signal before either tower even started learning. CATEGORY_
# SIGNAL_WEIGHT boosts the one-hot block; SEMANTIC_HASH_WEIGHT/CREATOR_HASH_WEIGHT damp the
# hashed blocks -- applied identically on both towers (content's category one-hot vs. user's
# category-affinity blocks; content's semantic hash vs. user's semantic-interest hash) so the
# two sides remain comparable in the shared embedding space, not rebalanced independently.
#
# Phase 3.1 final selection (EXP4, chosen over a 2.0/0.4/0.4 first pass and nearby 1.4-1.6/
# 0.6-0.7 variants): 2.0/0.4/0.4 over-suppressed the semantic/creator blocks and left SPORT
# under-recovered for the SPORT-heavy demo user; EXP4's lighter 1.5x category boost with
# 0.6x semantic/creator damping gave the best balance of aggregate Recall@K, SPORT-heavy
# Top-10 recovery (rank #1-#2), and no MUSIC-heavy regression, at no extra epoch cost.
CATEGORY_SIGNAL_WEIGHT = 1.5
SEMANTIC_HASH_WEIGHT = 0.6
CREATOR_HASH_WEIGHT = 0.6
# Phase 3.1 tuning (measured): the raw log1p(count) aggregate stats (log_total_interactions,
# log_distinct_creators) reached magnitudes of 2.7-3.1 -- alone outweighing the ENTIRE 3x
# category-affinity block's L2 norm (2.67 for a 30-dim vector) with just 2 numbers that carry
# no category-specific information at all. Capped (divide-then-clip to roughly [0,1]) so overall
# user activity level remains a small, secondary signal instead of dominating by raw scale.
LOG_STAT_CAP_INTERACTIONS = 5.0
LOG_STAT_CAP_CREATORS = 3.0


def _hash_bucket(token: str, dim: int) -> tuple[int, float]:
    """Deterministic (not Python's randomized `hash()`) bucket + sign for the hashing trick --
    reproducible across processes/interpreters, required for the seed-42 determinism tests."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return bucket, sign


def _hashed_vector(weighted_tokens: Iterable[tuple[str, float]], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token, weight in weighted_tokens:
        if not token:
            continue
        bucket, sign = _hash_bucket(token, dim)
        vec[bucket] += sign * weight
    return vec


def content_tokens(content) -> list[str]:
    """Every normalized semantic token a content item carries, across all 5 semantic fields --
    mirrors app.ml.dataset_builder._token_fields' field set (SEMANTIC_FIELDS) without importing
    that private helper; title is tokenized via the same extract_title_tokens the ranker uses.
    Order-preserving dedup, matching _token_fields' own use of _dedupe."""
    tokens: list[str] = []
    for field in ("hashtags", "topics", "entities", "subgenres"):
        tokens.extend(getattr(content, field, None) or [])
    tokens.extend(extract_title_tokens(getattr(content, "title", None)))
    return list(dict.fromkeys(tokens))


# --------------------------------------------------------------------------------- user tower
USER_AGGREGATE_STAT_NAMES = [
    "log_total_interactions", "positive_ratio", "negative_ratio", "log_distinct_creators",
]


def user_vector_dim(categories: list[str]) -> int:
    return 3 * len(categories) + len(USER_AGGREGATE_STAT_NAMES) + SEMANTIC_HASH_DIM


def build_user_vector(history: FeatureHistory, user_id: str, categories: list[str]) -> np.ndarray:
    """Point-in-time safe: only reflects whatever `history` has accumulated via `.update()` so
    far -- callers doing point-in-time training-pair construction pass a history that has not
    yet seen the target event; callers doing live/offline retrieval pass a fully up-to-date
    history. Reuses `.features()` per category (public API, not a redesign) purely to read
    `category_affinity`/`recent_category_affinity`/`has_category_history` for each of the
    known categories -- creator_id/content_id/semantic args are irrelevant placeholders here
    since only the category-scoped fields are read back out."""
    category_affinity = np.zeros(len(categories), dtype=np.float32)
    recent_affinity = np.zeros(len(categories), dtype=np.float32)
    has_history = np.zeros(len(categories), dtype=np.float32)
    for i, category in enumerate(categories):
        feats = history.features(
            user_id=user_id, category=category, creator_id="__two_tower_probe__",
            content_id="__two_tower_probe__", timestamp=_now_placeholder(history, user_id),
            content_popularity_score=0.5, content_created_at=_now_placeholder(history, user_id),
        )
        category_affinity[i] = feats["category_affinity"]
        recent_affinity[i] = feats["recent_category_affinity"]
        has_history[i] = feats["has_category_history"]

    total_interactions = history.users.get(user_id, 0)
    total_positive = sum(cat["positive"] for (uid, _cat), cat in history.categories.items() if uid == user_id)
    total_negative = sum(cat["negative"] for (uid, _cat), cat in history.categories.items() if uid == user_id)
    distinct_creators = sum(1 for (uid, _cid), c in history.creators.items() if uid == user_id and c["interactions"] > 0)
    denom = max(1, total_positive + total_negative)
    stats = np.array([
        min(np.log1p(total_interactions) / LOG_STAT_CAP_INTERACTIONS, 1.0),
        total_positive / denom,
        total_negative / denom,
        min(np.log1p(distinct_creators) / LOG_STAT_CAP_CREATORS, 1.0),
    ], dtype=np.float32)

    weighted_tokens: list[tuple[str, float]] = []
    for field in SEMANTIC_FIELDS:
        for (uid, token), entry in history.tokens[field].items():
            if uid != user_id or entry["interactions"] == 0:
                continue
            weight = affinity_score(entry["raw"]) - 0.5  # signed deviation from neutral
            weighted_tokens.append((token, weight))
    semantic_vec = _hashed_vector(weighted_tokens, SEMANTIC_HASH_DIM) * SEMANTIC_HASH_WEIGHT

    return np.concatenate([category_affinity * CATEGORY_SIGNAL_WEIGHT, recent_affinity * CATEGORY_SIGNAL_WEIGHT,
                            has_history, stats, semantic_vec])


def _now_placeholder(history: FeatureHistory, user_id: str):
    """`.features()` requires a timestamp to window "recent"/"session" activity. Point-in-time
    correctness only depends on what has already been `.update()`-d into `history` -- the probe
    timestamp itself just needs to be safely at-or-after every event already applied, so it
    never accidentally excludes recent history from the RECENT_WINDOW/SESSION_WINDOW lookback.
    Uses the latest `last` timestamp recorded for this user's categories (falls back to a fixed
    epoch far in the future-safe direction is wrong for "recent window" math, so instead we use
    the max last-seen timestamp across this user's categories, or, if the user has no history
    yet, an arbitrary fixed constant -- neutral either way since every category feature is 0.5
    for a user with zero prior interactions regardless of the timestamp used)."""
    from datetime import datetime, timezone
    timestamps = [cat["last"] for (uid, _cat), cat in history.categories.items() if uid == user_id and cat["last"] is not None]
    return max(timestamps) if timestamps else datetime(2020, 1, 1, tzinfo=timezone.utc)


# ------------------------------------------------------------------------------ content tower
def content_vector_dim(categories: list[str]) -> int:
    return len(categories) + 1 + CREATOR_HASH_DIM + SEMANTIC_HASH_DIM


def build_content_vector(content, categories: list[str]) -> np.ndarray:
    """Uses only the content item's own attributes -- no user-specific state, so this can be
    (and, at retrieval time, is) precomputed once per content item and reused for every user."""
    category = (getattr(content, "category", None) or "").upper()
    onehot = np.array([1.0 if category == c else 0.0 for c in categories], dtype=np.float32) * CATEGORY_SIGNAL_WEIGHT
    popularity = np.array([float(getattr(content, "popularity_score", 0.5) or 0.5)], dtype=np.float32)
    creator_vec = _hashed_vector([(getattr(content, "creator_id", "") or "", 1.0)], CREATOR_HASH_DIM) * CREATOR_HASH_WEIGHT
    semantic_vec = _hashed_vector([(token, 1.0) for token in content_tokens(content)], SEMANTIC_HASH_DIM) * SEMANTIC_HASH_WEIGHT
    return np.concatenate([onehot, popularity, creator_vec, semantic_vec])
