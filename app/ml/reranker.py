"""Pure business reranking and heuristic explanation logic.

No I/O, no database access, no model loading: these functions only transform
already-scored candidate lists. That makes them directly reusable both by the
online recommendation services and by offline evaluation (see
`app.ml.reranking_eval`), so the metrics reported offline are guaranteed to
reflect the exact reranking behavior used in production rather than a
reimplementation that could drift from it.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any

from app.core.config import COHORT_HYBRID_STRATEGY_WEIGHT, COHORT_MAX_COLD_START_BOOST
from app.ml.semantic_tokens import extract_title_tokens, normalize_token

SEEN_PENALTY = 0.85
DIVERSITY_PENALTY = 0.96
MAX_CREATOR_IN_TOP_10 = 2
MAX_CONSECUTIVE_CATEGORY = 2

# Semantic (sub-category) evidence, VIDEO only -- see app.ml.dataset_builder.FeatureHistory.
# features() for how semantic_positive_match_count/semantic_negative_match_count/
# strongest_semantic_affinity are derived from strictly-prior hashtag/topic/entity/subgenre/
# title history. Purely multiplicative, matching this module's existing SEEN_PENALTY/
# DIVERSITY_PENALTY style -- a complementary business adjustment layered on top of the model
# score, never a replacement for it: two candidates that share category=SPORT but differ in
# title/hashtags/entities (e.g. one about a team the user has a real prior positive history
# with, one about a team they don't) can still end up with a meaningful score gap driven
# entirely by real behavioral evidence, not by which piece of content happens to be scored.
HIGH_SEMANTIC_AFFINITY_THRESHOLD = 0.75
SEMANTIC_AFFINITY_BOOST = 1.20
# SEMANTIC_MISMATCH_PENALTY is intentionally NOT applied in rerank() below anymore (minimal
# fix, forensic trace on CONTENT_NOT_INTERESTED over-suppression): the trained model already
# consumes semantic_negative_match_count/strongest_semantic_affinity/hashtag_affinity etc. as
# features, so this fixed multiplier was a second, independent application of the same signal
# the model had already scored -- for a candidate that merely shares a token with disliked
# content (e.g. a Djokovic video sharing "tennis" with a disliked Nadal video, never itself
# interacted with), this stacked on top of the model's own response and collapsed it almost as
# hard as the actually-disliked content. Kept defined (with `_has_semantic_mismatch` below) for
# observability/offline analysis, not deleted -- see reranking_eval for reuse.
SEMANTIC_MISMATCH_PENALTY = 0.55


def _has_high_semantic_affinity(features: dict[str, Any]) -> bool:
    return (
        features.get("semantic_positive_match_count", 0) > 0
        and features.get("strongest_semantic_affinity", 0.5) >= HIGH_SEMANTIC_AFFINITY_THRESHOLD
    )


def _has_semantic_mismatch(features: dict[str, Any]) -> bool:
    return (
        features.get("semantic_negative_match_count", 0) > 0
        and features.get("semantic_positive_match_count", 0) == 0
    )


# session_intent_confidence must clear this before a session reading is trusted enough to
# explain a ranking -- mirrors app.ml.dataset_builder's own confidence design ("one accidental
# interaction must not flip the feed" applies to the explanation, not just the score).
SESSION_INTENT_CONFIDENCE_THRESHOLD = 0.5
SESSION_AFFINITY_DEVIATION_THRESHOLD = 0.15  # |session_category_affinity - 0.5| must clear this to count as a real signal.

# Finalization spec Decision 2/3: explicit, request/session-scoped search intent. Purely
# reranker-side (never a model feature, never persisted) -- generic token-overlap + optional
# category match against whatever a candidate already carries, bounded so it can nudge but
# never dominate the model score. SEARCH_RELEVANCE_MATERIALITY_THRESHOLD gates the
# SEARCH_INTENT_MATCH reason the same way HIGH_SEMANTIC_AFFINITY_THRESHOLD already gates
# SEMANTIC_INTEREST_MATCH -- a token match must be substantial, not a single incidental hit.
SEARCH_RELEVANCE_BOOST_MAX = 0.35
SEARCH_RELEVANCE_MATERIALITY_THRESHOLD = 0.5

# "Search matching bug: generic multi-word queries" fix, isolated to search matching only
# (never touches app.ml.semantic_tokens, the global tokenizer training/serving both depend
# on -- see search_relevance's docstring for why). A candidate's compound metadata tokens
# (app.ml.semantic_tokens.normalize_token already joins a multi-word phrase like "Electronic
# Music" into one token, "ELECTRONIC_MUSIC", using "_" as the separator) are decomposed into
# their word components ONLY for this comparison -- the stored token itself is never mutated,
# only a local, search-side view of it. A component-only hit (query "MUSIC" against candidate
# "ELECTRONIC_MUSIC") counts for less than a whole-token exact hit (query "ELECTRONIC_MUSIC"
# against candidate "ELECTRONIC_MUSIC", or a plain single-word token matching another
# single-word token outright) -- Decision: "exact match > compound component match > no
# match" (Section 6 of the finalization spec this fixes).
COMPOUND_COMPONENT_MATCH_WEIGHT = 0.6


def _candidate_semantic_tokens(candidate: Any) -> set[str]:
    """Every normalized token a candidate carries (hashtags/topics/entities/subgenres are
    already normalized by app.schemas.recommendation_schemas.Candidate's validators; title is
    tokenized here via the exact same extract_title_tokens app.ml.dataset_builder uses for
    scoring) -- one flat pool, so matching never special-cases which field a token came from."""
    tokens: set[str] = set()
    tokens.update(getattr(candidate, "hashtags", None) or [])
    tokens.update(getattr(candidate, "topics", None) or [])
    tokens.update(getattr(candidate, "entities", None) or [])
    tokens.update(getattr(candidate, "subgenres", None) or [])
    tokens.update(extract_title_tokens(getattr(candidate, "title", None)))
    return tokens


def _compound_components(tokens: set[str]) -> set[str]:
    """Word-level components of every "_"-joined compound token in `tokens` (a single-word
    token's only "component" is itself, so this is always a superset of `tokens`). Splitting
    on the token alphabet's own existing separator -- never arbitrary substring slicing -- is
    what keeps this from turning into "ART matches PARTY": "PARTY" has no "_" in it, so its
    only component is "PARTY" itself, and "ART" != "PARTY"."""
    components: set[str] = set()
    for token in tokens:
        components.update(part for part in token.split("_") if part)
    return components


def _token_match_weight(search_token: str, candidate_tokens: set[str], candidate_components: set[str]) -> float:
    """1.0 for a whole-token exact hit (covers a single-word token matching another
    single-word token, and a compound search token like "ELECTRONIC_MUSIC" matching the same
    whole compound candidate token), COMPOUND_COMPONENT_MATCH_WEIGHT for a hit that only
    exists once the candidate's compound tokens are decomposed into components, 0.0 for no
    hit at all."""
    if search_token in candidate_tokens:
        return 1.0
    if search_token in candidate_components:
        return COMPOUND_COMPONENT_MATCH_WEIGHT
    return 0.0


def search_relevance(candidate: Any, search_intent: Any | None) -> float:
    """Bounded [0,1] search-intent relevance for one candidate. Generic token-overlap over the
    candidate's full normalized-token pool (category/title/hashtags/topics/entities/subgenres)
    against the search intent's own tokens -- no per-entity ("if COLDPLAY...") or per-category
    special-casing anywhere in this function; it works identically for any token/category.
    Returns 0.0 (no effect) when no search intent is supplied or it carries no tokens/category
    at all, so an absent searchIntent is always a pure no-op.

    Compound candidate tokens (e.g. "ELECTRONIC_MUSIC") also match a query token that is only
    one of their word components (e.g. "MUSIC") -- see _token_match_weight -- at reduced
    weight, so "live music" (which the shared title tokenizer reduces to just ["MUSIC"] --
    "live" is a stopword there, unchanged, see app.ml.semantic_tokens) still finds "Electronic
    Music"/"Pop Music"/"Rock Music" content generically, with no per-term/per-category rule."""
    if search_intent is None:
        return 0.0
    search_tokens: set[str] = set()
    search_tokens.update(search_intent.topics)
    search_tokens.update(search_intent.entities)
    search_tokens.update(search_intent.subgenres)
    search_tokens.update(extract_title_tokens(search_intent.query))
    match_ratio = 0.0
    if search_tokens:
        candidate_tokens = _candidate_semantic_tokens(candidate)
        candidate_components = _compound_components(candidate_tokens)
        total_weight = sum(_token_match_weight(token, candidate_tokens, candidate_components) for token in search_tokens)
        match_ratio = min(1.0, total_weight / len(search_tokens))
    category_match = 0.0
    if search_intent.category and search_intent.category.upper() == candidate.category.upper():
        category_match = 1.0
    relevance = min(1.0, max(match_ratio, 0.5 * category_match))
    return search_intent.confidence * relevance


# Finalization spec Decision 10, extended for related-user candidate generation: bounded,
# generic social/collaborative provenance. Reranker-only (never a model feature) -- credibility
# is a plain mean of three 0..1 caller-supplied signals, no per-user/per-relationship
# special-casing. The boost is multiplicative on top of model_score, which already encodes the
# user's real category/creator/semantic preference -- the same bounded-multiplier principle
# that keeps the semantic/search boosts from overriding a genuine dislike protects this one
# too: a small ceiling on an already-low base score stays low.
SOCIAL_RELEVANCE_BOOST_MAX = 0.25
SOCIAL_RELEVANCE_MATERIALITY_THRESHOLD = 0.6
# Both allowed candidateSource values (app.schemas.recommendation_schemas.
# ALLOWED_CANDIDATE_SOURCES) trigger the identical formula below -- SOCIAL vs COLLABORATIVE is
# a provenance label for the caller, not a different scoring policy; see that constant's
# docstring for why RMS deliberately does not rank them differently.
_SOCIAL_CANDIDATE_SOURCES = frozenset({"SOCIAL", "COLLABORATIVE"})


def social_relevance(candidate: Any, features: dict[str, Any]) -> float:
    """Bounded [0,1] social/collaborative relevance for one candidate. Zero (no effect) unless
    the candidate is explicitly SOCIAL- or COLLABORATIVE-sourced, the user has not already
    seen it, and social credibility context was actually supplied -- an ordinary catalog
    candidate is always a no-op here."""
    if getattr(candidate, "candidate_source", None) not in _SOCIAL_CANDIDATE_SOURCES:
        return 0.0
    if features.get("already_seen"):
        return 0.0
    social_context = getattr(candidate, "social_context", None)
    if social_context is None:
        return 0.0
    credibility = (
        social_context.interest_similarity + social_context.relationship_strength + social_context.source_user_engagement
    ) / 3
    return min(1.0, max(0.0, credibility))


# Finalization spec Decision 11: cold-start language/region compatibility ONLY -- no
# demographic stereotype of any kind. `userContext.age` is intentionally never read anywhere
# in this module (accepted by the schema for forward-compatibility, zero ranking influence).
# Region/language only ever matter (a) while the user is a genuine cold start (cold_start=True,
# the same user-level flag app.services.recommendation_service already computes) and (b) when
# the candidate itself carries language/regions metadata -- a candidate with none of that
# metadata (the common case today) is always a no-op, never a fabricated effect.
COLD_START_CONTEXT_BOOST_MAX = 0.08


def cold_start_context_relevance(candidate: Any, user_context: Any | None, cold_start: bool) -> float:
    """Bounded [0,1] language/region compatibility signal, cold-start only. Returns 0.0
    (no effect) whenever either side lacks the corresponding metadata, so a candidate/request
    that never mentions language or region is unaffected -- this is deliberately honest: no
    fake influence is applied just because a request happens to be cold-start."""
    if not cold_start or user_context is None:
        return 0.0
    matches = 0
    signals = 0
    candidate_language = getattr(candidate, "language", None)
    if user_context.language and candidate_language:
        signals += 1
        matches += int(user_context.language == candidate_language)
    candidate_regions = getattr(candidate, "regions", None) or []
    if user_context.region and candidate_regions:
        signals += 1
        matches += int(user_context.region in candidate_regions)
    return matches / signals if signals else 0.0


# Cold-start personalization: data-driven region/age cohort preference signal (finalization
# pass -- replaces the removed LOCAL_POC_REGIONAL_CATEGORY_COHORT_PRIOR hand-authored table).
# Deliberately separate from cold_start_context_relevance above: that function answers "is this
# CANDIDATE's own language/region metadata compatible with the user's" (candidate-level
# compatibility, unchanged); this one answers "does this resolved COHORT tend to prefer this
# CATEGORY", independent of whether the candidate carries any language/region metadata at all.
# Both are bounded [0,1], both cold-start-only, both additive in combined_multiplier below --
# neither duplicates the other's evidence.
#
# This module stays "no I/O, no database access" (see its own top docstring): it never computes,
# stores, or looks up cohort statistics itself. `cohort_profile` is an already-resolved
# app.ml.cohort_profile.CohortPreferenceProfile (or None) built by
# app.services.cohort_preference_provider from real, versioned, persisted aggregates over
# historical `interactions` data (app.services.cohort_aggregation_service) -- reranker only
# ever reads `cohort_profile.preferences[category]` back out. A profile with `reliable=False`
# (including `cohort_profile is None`, e.g. cohort preferences disabled/unconfigured/not yet
# built) always returns 0.0: missing or insufficient cohort data is a safe no-op, never a guess.
COLD_START_REGIONAL_COHORT_BOOST_MAX = COHORT_MAX_COLD_START_BOOST
COLD_START_REGIONAL_COHORT_MATERIALITY_THRESHOLD = 0.5

# Strategy-dependent cohort weight (spec §J: "personal history must override cohort
# assumptions"). COLD_START (0 prior interactions) gets full weight; PERSONALISED_ML (enough
# history for app.services.recommendation_service.strategy_for to consider the user
# established) gets none -- a user's own observed behavior always wins outright once the model
# has enough of it. HYBRID's weight is the one genuinely tunable middle ground (see
# app.core.config.COHORT_HYBRID_STRATEGY_WEIGHT).
COHORT_STRATEGY_WEIGHT: dict[str, float] = {
    "COLD_START": 1.0,
    "HYBRID": COHORT_HYBRID_STRATEGY_WEIGHT,
    "PERSONALISED_ML": 0.0,
}


def _cohort_strategy_weight(strategy: str | None, cold_start: bool) -> float:
    """`strategy` (app.services.recommendation_service.strategy_for's own COLD_START/HYBRID/
    PERSONALISED_ML labels) is the precise signal; an unrecognized or omitted strategy falls
    back to the coarser pre-existing `cold_start` boolean (1.0 when cold, 0.0 otherwise) so a
    caller that hasn't been updated to pass `strategy` yet (e.g. app.ml.reranking_eval, offline
    benchmarks) keeps its previous cohort behavior unchanged rather than silently losing it."""
    if strategy in COHORT_STRATEGY_WEIGHT:
        return COHORT_STRATEGY_WEIGHT[strategy]
    return 1.0 if cold_start else 0.0


def regional_cohort_relevance(candidate: Any, cohort_profile: Any | None, strategy_weight: float) -> float:
    """Bounded [0,1] cold-start-biased cohort-preference signal. Zero whenever
    `strategy_weight` is 0 (the user's own history already dominates -- see
    _cohort_strategy_weight) or `cohort_profile` is missing/unreliable; otherwise the
    profile's own already-shrunk category score, scaled by `strategy_weight` so a HYBRID user
    gets a proportionally smaller nudge than a genuine COLD_START user for the identical
    underlying cohort evidence."""
    if strategy_weight <= 0 or cohort_profile is None or not getattr(cohort_profile, "reliable", False):
        return 0.0
    preferences = getattr(cohort_profile, "preferences", None) or {}
    return preferences.get(candidate.category.upper(), 0.0) * strategy_weight


# Cold-start personalization upgrade: explicit onboarding-interest relevance. `user_context.
# interests` (app.schemas.recommendation_schemas.UserContext) is direct, user-declared intent --
# collected once at onboarding by whatever caller owns that flow (Feed/onboarding service, in
# production), supplied fresh on each request exactly like region/language already are. RMS does
# not persist it (no DB column, no migration) -- same request-scoped-only posture as every other
# UserContext field.
#
# Matching reuses this module's OWN existing token machinery (never a second, differently-
# behaved matcher, and never a substring match -- "ART" can never accidentally match "PARTY",
# since normalize_token/the token pool below only ever compare whole, underscore-delimited
# tokens): `normalize_token` is the exact function app.schemas.recommendation_schemas.UserContext.
# interests already normalized through at request-validation time, so an interest and a
# candidate's own category/hashtag/topic/entity/subgenre/title token can only ever match when
# they are the SAME normalized token. `_candidate_semantic_tokens`/`_compound_components`/
# `_token_match_weight` are the exact same helpers `search_relevance` above already uses for its
# own exact/compound-component token matching -- reused verbatim, not reimplemented.
#
# Three progressively weaker tiers, matching this task's own required ordering ("exact category
# match > exact topic/hashtag/entity match > weaker semantic/token overlap"):
#   1.0 -- an interest exactly equals the candidate's own category (the strongest, least
#          ambiguous signal: the user explicitly said "I want SPORT", and this candidate IS SPORT).
#   0.8 -- an interest exactly matches one of the candidate's own hashtag/topic/entity/subgenre/
#          title tokens (a specific declared interest, e.g. "FOOTBALL", found on this exact
#          candidate).
#   0.5 -- an interest only matches a candidate's compound token by COMPONENT (e.g. interest
#          "MUSIC" against a candidate hashtag "ELECTRONIC_MUSIC") -- real but weaker evidence,
#          the same reduced-confidence tier search_relevance already applies for this exact case.
#   0.0 -- no match at all; an unrecognized/unmatched interest is never guessed into a fabricated
#          relevance.
EXPLICIT_INTEREST_BOOST_MAX = 0.30
EXPLICIT_INTEREST_MATERIALITY_THRESHOLD = 0.5

_EXACT_CATEGORY_MATCH_RELEVANCE = 1.0
_EXACT_TOKEN_MATCH_RELEVANCE = 0.8
_COMPONENT_TOKEN_MATCH_RELEVANCE = 0.5


def explicit_interest_relevance(candidate: Any, user_context: Any | None, cold_start: bool) -> float:
    """Bounded [0,1] cold-start-only explicit-onboarding-interest relevance. Zero (no effect)
    unless the user is a genuine cold start and `user_context.interests` is non-empty -- an
    absent or empty interests list is always a pure no-op, exactly like every other relevance
    function in this module."""
    if not cold_start or user_context is None or not user_context.interests:
        return 0.0
    interests = user_context.interests  # already normalize_token-normalized by UserContext itself
    if normalize_token(candidate.category) in interests:
        return _EXACT_CATEGORY_MATCH_RELEVANCE
    candidate_tokens = _candidate_semantic_tokens(candidate)
    candidate_components = _compound_components(candidate_tokens)
    best_token_weight = max(
        (_token_match_weight(interest, candidate_tokens, candidate_components) for interest in interests),
        default=0.0,
    )
    if best_token_weight >= 1.0:
        return _EXACT_TOKEN_MATCH_RELEVANCE
    if best_token_weight > 0.0:
        return _COMPONENT_TOKEN_MATCH_RELEVANCE
    return 0.0


# Finalization spec Decision 12: content_age_hours already exists as a model feature, but its
# learned importance in the active model is statistically indistinguishable from zero (verified
# via an isolated retrain: meanImportance=-0.000872, stdImportance=0.002955 -- well within noise)
# -- the raw model provides no reliable "fresher is a little better" signal today. This is a
# small, generic, reranker-only nudge, NOT a feature/schema change and NOT a retrain: bounded to
# a modest advantage between otherwise-close candidates, and deliberately asymmetric ("old" never
# gets penalized, it just stops earning the bonus) so genuine personalization -- whose score gaps
# are typically far larger than FRESHNESS_BOOST_MAX -- always remains free to outweigh it.
FRESHNESS_BOOST_MAX = 0.05
FRESHNESS_FULL_CREDIT_HOURS = 24.0
FRESHNESS_ZERO_CREDIT_HOURS = 240.0


def freshness_relevance(candidate: Any) -> float:
    """[0,1], 1.0 for content newer than FRESHNESS_FULL_CREDIT_HOURS, linearly tapering to 0.0
    by FRESHNESS_ZERO_CREDIT_HOURS, 0.0 (never negative) beyond that -- "old" candidates simply
    stop earning a bonus, they are never actively penalized for age."""
    age = getattr(candidate, "content_age_hours", None)
    if age is None:
        return 0.0
    if age <= FRESHNESS_FULL_CREDIT_HOURS:
        return 1.0
    if age >= FRESHNESS_ZERO_CREDIT_HOURS:
        return 0.0
    return 1.0 - (age - FRESHNESS_FULL_CREDIT_HOURS) / (FRESHNESS_ZERO_CREDIT_HOURS - FRESHNESS_FULL_CREDIT_HOURS)


# Ranking-separation forensic investigation (2026-08-27): session_category_affinity and
# session_intent_confidence are already real trained-model FEATURES (app.ml.dataset_builder),
# so the raw model score does react to in-session evidence -- but unlike search/social/
# freshness, nothing translated a confidently-detected, above-neutral session preference into a
# reranker-side adjustment. A candidate whose active-session category is a strong, trusted
# preference had no bounded way to close a large raw-model-score gap against an unrelated,
# popularity-heavy candidate the way SEARCH_INTENT_MATCH/SOCIAL_RELEVANCE already can for their
# own signals. This is a positive-preference boost only, deliberately NOT paired with a
# same-signal penalty for the opposite (below-neutral) side: symmetric suppression would use the
# same threshold design to actively bury a category on evidence this module already treats as too
# thin to even report as an explanation, and was explicitly out of scope for this fix -- a
# separate, deliberate decision if ever pursued, not a default side effect of this one.
SESSION_INTEREST_BOOST_MAX = 0.30
# session_category_affinity is a sigmoid (feature_builder.affinity_score(raw/10)); realistic
# "strong same-category session" evidence (several completed/liked/shared interactions, per
# app.ml.dataset_builder's own SESSION_CONFIDENCE_SATURATION_WEIGHT calibration comment) lands
# well below the sigmoid's 1.0 asymptote -- 0.9 is already a very confident in-session reading,
# so deviation reaching 0.4 (affinity=0.9) is treated as full strength rather than requiring the
# asymptote itself, which would make the boost systematically under-fire on real traffic.
SESSION_AFFINITY_FULL_STRENGTH_DEVIATION = 0.40


def session_interest_relevance(features: dict[str, Any]) -> float:
    """Bounded [0,1] session-interest relevance for one candidate, in the candidate's OWN
    category -- session_category_affinity/session_intent_confidence are already computed
    per-candidate-category by app.ml.dataset_builder's FeatureHistory.features(), so this needs
    no separate "does this candidate match the active session category" check: a candidate in a
    category the session shows no evidence for simply carries neutral (0.5) affinity and 0.0
    confidence, and this returns 0.0 for it automatically.

    Gated on the EXACT SAME evidence explanation() already requires to report the SESSION_INTEREST
    reason (SESSION_INTENT_CONFIDENCE_THRESHOLD / SESSION_AFFINITY_DEVIATION_THRESHOLD) -- reusing
    the underlying feature values, never the `reason` string itself, so this can never drift out
    of sync with what "SESSION_INTEREST" is claiming to explain (one accidental interaction, or a
    single not-yet-trusted reading, produces the same 0.0 here as it does for the reason label).

    Positive-preference only: a confident session reading BELOW neutral (the session shows the
    user rejecting this category right now) returns 0.0, never a penalty -- see the module-level
    comment above this function for why a symmetric suppression side was deliberately not added."""
    if not features.get("has_session_activity"):
        return 0.0
    confidence = features.get("session_intent_confidence", 0.0)
    if confidence < SESSION_INTENT_CONFIDENCE_THRESHOLD:
        return 0.0
    deviation = features.get("session_category_affinity", 0.5) - 0.5
    if deviation < SESSION_AFFINITY_DEVIATION_THRESHOLD:
        return 0.0
    strength = min(1.0, deviation / SESSION_AFFINITY_FULL_STRENGTH_DEVIATION)
    return confidence * strength


def explanation(features: dict[str, Any], candidate: Any, cold_start: bool) -> str:
    """Transparent heuristic explanation; this is not model attribution or SHAP."""
    if features["creator_followed"]:
        return "FOLLOWED_CREATOR"
    # Negative-feedback session bug fix: was `abs(affinity - 0.5) >= threshold`, which fired
    # this reason for a strongly NEGATIVE session reading too (e.g. several same-category
    # explicit rejections) -- misleading callers into "SESSION_INTEREST" for a category the
    # session evidence actually shows the user rejecting. session_interest_relevance() (the
    # function that actually drives the score boost this reason is supposed to describe) was
    # already positive-only ("a confident session reading BELOW neutral... returns 0.0, never a
    # penalty" -- see its own docstring); this now matches that same positive-only gate exactly,
    # restoring the "EXACT SAME evidence" invariant this function's docstring already claims.
    if (
        features.get("has_session_activity")
        and features.get("session_intent_confidence", 0.0) >= SESSION_INTENT_CONFIDENCE_THRESHOLD
        and features.get("session_category_affinity", 0.5) - 0.5 >= SESSION_AFFINITY_DEVIATION_THRESHOLD
    ):
        return "SESSION_INTEREST"
    if _has_high_semantic_affinity(features):
        return "SEMANTIC_INTEREST_MATCH"
    if features["category_affinity"] >= 0.70:
        return "HIGH_CATEGORY_AFFINITY"
    if features["recent_category_watch_percentage"] >= 65:
        return "RECENT_CATEGORY_ACTIVITY"
    if candidate.content_popularity_score >= 0.80:
        return "POPULAR_CONTENT"
    # Bug fix: this used to fall back to "PREVIOUS_CATEGORY_INTEREST" for any warm user
    # (cold_start=False), even for a category the user has zero history in -- cold_start is
    # a user-level flag (any interactions at all), not a per-category one, so a SPORT-heavy
    # user browsing a candidate in a category they've never touched (has_category_history=0)
    # was reported as having "previous interest" in it, which is false. has_category_history
    # is the correct, per-category signal already computed by FeatureHistory.features() for
    # exactly this distinction; cold_start=True already implies has_category_history=0 for
    # every category, so this is a strict correction, not a behavior change for genuinely
    # cold-start users.
    # Merely having category history does not mean the user was interested in it: an
    # explicit rejection or a run of skips is history too. Only describe the fallback as
    # previous interest when the resulting affinity is actually above neutral; otherwise
    # the fallback explanation is exploration. This affects explanation text only, never
    # score or rank.
    has_positive_category_history = (
        features.get("has_category_history")
        and features.get("category_affinity", 0.5) > 0.5
    )
    return "PREVIOUS_CATEGORY_INTEREST" if has_positive_category_history else "EXPLORATION"


def _blend_multiplier_in_logit_space(base_score: float, multiplier: float) -> float:
    """Applies a boost/penalty `multiplier` (semantic/search/social/session-interest/cold-start/freshness
    combined) to `base_score` in logit space instead of multiplying the probability directly.

    Why (Phase 1 investigation 3, measured not guessed): multiplying an already-high
    probability by ANY boost saturates almost immediately against the `min(1.0, ...)` clamp --
    measured directly against a real trained model, 4/8 representative candidates (differing in
    real base model confidence by several points) landed on the EXACT same clamped 1.0 ceiling,
    erasing the underlying model-confidence gap the clamp is supposed to merely cap, not erase.
    Adding the equivalent adjustment in logit space instead compresses smoothly as the
    probability approaches 1.0 (or 0.0) -- a genuinely more-confident base score still ends up
    ranked above a genuinely less-confident one after the same nudge is applied to both, instead
    of both collapsing to an identical value and falling back to insertion-order artifacts
    (diversity-penalty exponents) for tie-breaking. `multiplier == 1.0` (no boost triggered,
    the common case) is a guaranteed exact no-op -- byte-identical to the previous multiplicative
    behavior whenever no semantic/search/social/session-interest/cold-start/freshness signal
    actually applies."""
    if multiplier == 1.0:
        return base_score
    p = min(max(base_score, 1e-9), 1 - 1e-9)
    logit = math.log(p / (1 - p)) + math.log(multiplier)
    return 1.0 / (1.0 + math.exp(-logit))


def rerank(
    scored: list[dict[str, Any]], limit: int, *,
    search_intent: Any | None = None, user_context: Any | None = None, cold_start: bool = False,
    user_id: str | None = None, cohort_profile: Any | None = None, strategy: str | None = None,
) -> list[dict[str, Any]]:
    """Preserve ML order while applying small exposure and diversity adjustments.

    Ties in `adjusted_score` are broken deterministically by the pre-rerank
    (model-score-descending, then insertion) order, since Python's sort is
    stable and `scored` is expected to already be sorted by model score.

    `search_intent` (finalization spec Decision 2/3): optional, request-scoped only -- never
    read from `features` (which is purely long-term/recent/session/creator/semantic model
    input), so it cannot leak into training and has no effect at all when omitted.

    `user_context`/`cold_start` (Decision 11): language/region compatibility, cold-start only;
    see cold_start_context_relevance. `cold_start` defaults to False so an existing caller that
    doesn't pass it gets zero cold-start-context effect, exactly like before this parameter
    existed.

    `user_id` (cold-start personalization upgrade): optional, used ONLY to seed the deterministic
    cold-start exploration pass below (see _apply_cold_start_exploration) -- never read from
    `features`, never sent to the model, never affects HYBRID/PERSONALISED_ML requests. Defaults
    to None so an existing caller that doesn't pass it gets zero exploration effect, exactly like
    every other additive parameter in this signature.

    `cohort_profile`/`strategy` (finalization pass, region/age cohort cold-start system): an
    already-resolved `app.ml.cohort_profile.CohortPreferenceProfile` (or None -- see
    app.services.cohort_preference_provider) and the caller's own COLD_START/HYBRID/
    PERSONALISED_ML strategy label (app.services.recommendation_service.strategy_for). Both
    default to None, in which case cohort weighting falls back to the coarser `cold_start`
    boolean (see _cohort_strategy_weight) -- an existing caller that only ever passed
    `cold_start` keeps working, just without the strategy-graduated weakening.
    """
    category_counts: Counter[str] = Counter()
    creator_counts: Counter[str] = Counter()
    adjusted: list[dict[str, Any]] = []
    for item in scored:
        features = item["features"]
        candidate = item["candidate"]
        score = item["model_score"]
        # Every individual boost/penalty's own detection condition, threshold, and magnitude
        # cap is completely unchanged from before -- only HOW they combine changed (see
        # _blend_multiplier_in_logit_space): accumulate one combined multiplier exactly as
        # this loop always has, then apply it once, in logit space, instead of chaining five
        # separate multiplications directly against the probability.
        combined_multiplier = 1.0
        if _has_high_semantic_affinity(features):
            combined_multiplier *= SEMANTIC_AFFINITY_BOOST
        # No SEMANTIC_MISMATCH_PENALTY here -- see its definition above for why (the model
        # already consumed this signal; this used to double-apply it).
        search = search_relevance(candidate, search_intent)
        if search > 0:
            combined_multiplier *= 1.0 + SEARCH_RELEVANCE_BOOST_MAX * search
        social = social_relevance(candidate, features)
        if social > 0:
            combined_multiplier *= 1.0 + SOCIAL_RELEVANCE_BOOST_MAX * social
        session_interest = session_interest_relevance(features)
        if session_interest > 0:
            combined_multiplier *= 1.0 + SESSION_INTEREST_BOOST_MAX * session_interest
        cold_start_context = cold_start_context_relevance(candidate, user_context, cold_start)
        if cold_start_context > 0:
            combined_multiplier *= 1.0 + COLD_START_CONTEXT_BOOST_MAX * cold_start_context
        explicit_interest = explicit_interest_relevance(candidate, user_context, cold_start)
        if explicit_interest > 0:
            combined_multiplier *= 1.0 + EXPLICIT_INTEREST_BOOST_MAX * explicit_interest
        cohort_strategy_weight = _cohort_strategy_weight(strategy, cold_start)
        regional_cohort = regional_cohort_relevance(candidate, cohort_profile, cohort_strategy_weight)
        if regional_cohort > 0:
            combined_multiplier *= 1.0 + COLD_START_REGIONAL_COHORT_BOOST_MAX * regional_cohort
        freshness = freshness_relevance(candidate)
        if freshness > 0:
            combined_multiplier *= 1.0 + FRESHNESS_BOOST_MAX * freshness
        score = _blend_multiplier_in_logit_space(score, combined_multiplier)
        score = min(1.0, max(0.0, score))
        score *= SEEN_PENALTY if features["already_seen"] else 1.0
        score *= DIVERSITY_PENALTY ** (category_counts[candidate.category] + creator_counts[candidate.creator_id])
        item["adjusted_score"] = score
        # Explicit stated intent (current search) outranks every other rerank-time reason, but
        # never a followed creator; explicit ONBOARDING interest (direct, user-declared, but a
        # one-time signal rather than the user's CURRENT stated intent) outranks the regional
        # cohort prior, which in turn outranks generic social provenance -- the cold-start
        # personalization precedence (current search intent > onboarding interest > region/
        # cohort > social > popularity/freshness) derived from this module's own pre-existing
        # ordering, extended by two rungs for the two new signals. Same priority as
        # explanation()'s own ordering, applied here because search/interest/social/regional
        # relevance is only known at rerank time (explanation() runs before rerank() in the
        # caller).
        if search >= SEARCH_RELEVANCE_MATERIALITY_THRESHOLD and item["reason"] != "FOLLOWED_CREATOR":
            item["reason"] = "SEARCH_INTENT_MATCH"
        elif explicit_interest >= EXPLICIT_INTEREST_MATERIALITY_THRESHOLD and item["reason"] not in ("FOLLOWED_CREATOR", "SEARCH_INTENT_MATCH"):
            item["reason"] = "ONBOARDING_INTEREST"
        elif regional_cohort >= COLD_START_REGIONAL_COHORT_MATERIALITY_THRESHOLD and item["reason"] not in ("FOLLOWED_CREATOR", "SEARCH_INTENT_MATCH", "ONBOARDING_INTEREST"):
            item["reason"] = "REGIONAL_INTEREST"
        elif social >= SOCIAL_RELEVANCE_MATERIALITY_THRESHOLD and item["reason"] not in ("FOLLOWED_CREATOR", "SEARCH_INTENT_MATCH", "ONBOARDING_INTEREST", "REGIONAL_INTEREST"):
            item["reason"] = "SOCIAL_RELEVANCE"
        category_counts[candidate.category] += 1
        creator_counts[candidate.creator_id] += 1
        adjusted.append(item)
    adjusted.sort(key=lambda item: item["adjusted_score"], reverse=True)

    # Single-pass constrained greedy over the FULL remaining pool at every slot (fix for the
    # "high-score candidate dumped near the tail" bug): the old implementation ran one greedy
    # pass that permanently skipped any candidate violating the creator-cap/consecutive-category
    # constraints at its scan position, then appended every such leftover in bulk AFTER all
    # diversity-preferred picks -- so a top-scoring candidate skipped once (e.g. its creator
    # already had 2 picks in the top 10) could land near rank 35+ despite nothing else being
    # wrong with it. Here, a skipped candidate stays at the front of `remaining` (still
    # score-sorted) and is re-evaluated at the very next slot, so it only ever falls a few slots
    # behind its raw-score position -- never to the tail of a large pool. Constraints relax
    # gracefully, slot by slot, in two tiers (never a bulk unconstrained dump):
    #   Tier 1: both creator-cap (only within the first 10 slots) and consecutive-category cap.
    #   Tier 2: if nothing satisfies Tier 1, relax the creator cap for this slot only, but keep
    #           enforcing the consecutive-category cap. Category diversity is relaxed second
    #           (not first): it is the explicit, user-visible "do not return N same-category
    #           videos in a row" requirement, whereas the creator cap exists to stop one single
    #           creator from over-dominating -- when an entire category happens to have only one
    #           creator (a real, tested scenario: 6 SPORT candidates all from the same creator),
    #           relaxing category first would let that creator's candidates pile up back-to-back
    #           the moment the creator cap starts blocking them, defeating the category cap
    #           entirely even though other-category candidates were still available to interleave
    #           with. Relaxing creator first keeps those other-category candidates in rotation.
    #   Tier 3: if nothing satisfies even that (the remaining pool is down to one category and/or
    #           one over-capped creator), take the best-remaining candidate by score outright.
    # No candidate IDs, no fixed category slots, no per-candidate special-casing anywhere here --
    # the same two generic rules as before (MAX_CREATOR_IN_TOP_10, MAX_CONSECUTIVE_CATEGORY),
    # just applied per-slot instead of once-then-dump. Every candidate in `adjusted` is placed
    # exactly once, so `limit == len(adjusted)` still returns every candidate back.
    remaining = list(adjusted)
    chosen: list[dict[str, Any]] = []
    creator_counts_top10: Counter[str] = Counter()

    def _creator_ok(candidate: Any) -> bool:
        return len(chosen) >= 10 or creator_counts_top10[candidate.creator_id] < MAX_CREATOR_IN_TOP_10

    def _category_ok(candidate: Any) -> bool:
        if len(chosen) < MAX_CONSECUTIVE_CATEGORY:
            return True
        return not all(previous["candidate"].category == candidate.category for previous in chosen[-MAX_CONSECUTIVE_CATEGORY:])

    while remaining and len(chosen) < limit:
        pick = next((i for i, item in enumerate(remaining) if _creator_ok(item["candidate"]) and _category_ok(item["candidate"])), None)
        if pick is None:
            pick = next((i for i, item in enumerate(remaining) if _category_ok(item["candidate"])), None)
        if pick is None:
            pick = 0
        item = remaining.pop(pick)
        chosen.append(item)
        creator_counts_top10[item["candidate"].creator_id] += 1

    if cold_start and user_id:
        chosen = _apply_cold_start_exploration(chosen, remaining, limit=limit, user_id=user_id)

    return chosen


# Cold-start personalization upgrade: genuine, bounded, deterministic exploration -- not merely
# a reason label. A cold-start slate built purely from the signals above can still collapse onto
# whichever category happens to score highest (popularity/freshness/cohort all correlate with
# the SAME few dominant categories); this reserves a small, bounded minority of TAIL slots for
# candidates from categories the exploit portion does not already cover, so a fresh user's very
# first feed is not a monoculture.
#
# COLD_START_EXPLORATION_FRACTION=0.15: a bounded minority (~1-2 slots in a 10-item feed), never
# a majority -- chosen so exploration can meaningfully surface an under-represented category
# without ever dominating the slate (Phase 10 quality guard: "exploration must not dominate").
COLD_START_EXPLORATION_FRACTION = 0.15

# Reasons an exploration slot must never evict -- these are explicit, confirmed signals the user
# (or the request) directly supplied; exploration only ever backfills UNDERSPECIFIED positions,
# never overrides a stated intent.
_EXPLORATION_PROTECTED_REASONS = frozenset({"FOLLOWED_CREATOR", "SEARCH_INTENT_MATCH"})


def _deterministic_unit_interval(*parts: str) -> float:
    """Stable pseudo-random value in [0, 1) from `parts` -- NEVER Python's built-in `hash()`
    (randomized per-process for str/bytes, so it would make exploration non-reproducible across
    restarts), and NEVER an invented request/impression identity RMS does not authoritatively
    own (see the cold-start audit: `X-Request-ID` is log-correlation only, never authoritative).
    Built only from inputs a caller already legitimately has -- here, `user_id` and a candidate's
    own `content_id` -- so the same user sees the same exploration pick for the same candidate
    pool on every request, without RMS inventing or persisting any new identity."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _apply_cold_start_exploration(
    chosen: list[dict[str, Any]], remaining: list[dict[str, Any]], *, limit: int, user_id: str,
) -> list[dict[str, Any]]:
    """Replaces a bounded, deterministic minority of `chosen`'s TAIL positions with genuinely
    different-category candidates from the leftover `remaining` pool. Never adds beyond `limit`
    (only swaps existing positions), never picks a candidate already in `chosen`, never evicts a
    protected-reason slot (see _EXPLORATION_PROTECTED_REASONS), and is a complete no-op whenever
    `remaining` is empty (e.g. `limit >= len(adjusted)`: every candidate was already returned,
    nothing left to explore with) -- exploration can only ever narrow the returned set's category
    concentration, never remove or reorder anything outside the reserved tail slots."""
    if not remaining:
        return chosen
    slots = round(limit * COLD_START_EXPLORATION_FRACTION)
    if slots <= 0:
        return chosen

    eligible_positions = [i for i in range(len(chosen) - 1, -1, -1) if chosen[i]["reason"] not in _EXPLORATION_PROTECTED_REASONS][:slots]
    if not eligible_positions:
        return chosen
    eligible_positions_set = set(eligible_positions)

    exploit_categories = {item["candidate"].category for i, item in enumerate(chosen) if i not in eligible_positions_set}
    novel_pool = [item for item in remaining if item["candidate"].category not in exploit_categories]
    if not novel_pool:
        return chosen
    novel_pool.sort(key=lambda item: _deterministic_unit_interval(user_id, item["candidate"].content_id))

    # Seeded from the FULL current `chosen` list (not just the exploit portion): an eligible
    # (tail) position not ultimately replaced still keeps its original occupant, so undercounting
    # its creator here could let a later replacement push that creator over MAX_CREATOR_IN_TOP_10.
    creator_counts = Counter(item["candidate"].creator_id for item in chosen)
    result = list(chosen)
    used_categories: set[str] = set()
    for position in sorted(eligible_positions):
        pick = next(
            (item for item in novel_pool
             if item["candidate"].category not in used_categories
             and (position >= 10 or creator_counts[item["candidate"].creator_id] < MAX_CREATOR_IN_TOP_10)),
            None,
        )
        if pick is None:
            continue
        novel_pool.remove(pick)
        used_categories.add(pick["candidate"].category)
        creator_counts[pick["candidate"].creator_id] += 1
        result[position] = pick
    return result
