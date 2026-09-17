"""Synthetic ranking-training groups for the ranking-native challenger experiment (Task 7).

Diagnosis (spec section 1/2): the real interaction dataset (app.ml.dataset_builder.build_dataset)
has NO true impression-candidate-group concept. Every row is one independent, already-happened
event; `candidate_group` (app.ml.dataset_builder.build_dataset) is `f"{user_id}:{date}"` -- an
approximate "same user, same day" bucket used only for app.ml.evaluator's ranking-metric
diagnostics, NOT a real "N candidates shown together, user picked one" impression set. There is
no impression-log table anywhere in this codebase's schema. Building ranking groups therefore
REQUIRES synthetic construction -- reusing historical rows as if they were true simultaneous
impressions would be pretending information that does not exist, so this module builds
controlled, point-in-time-safe synthetic groups instead.

One group = one (user, timestamp) context + several candidates scored against the SAME
point-in-time history (via app.ml.dataset_builder.FeatureHistory, the exact production feature
path -- never a second, parallel feature computation). Deliberately independent of both
scripts/generate_synthetic_data.py (the flat classifier-training generator) and
app.benchmark.scenarios (the independent benchmark) -- no shared content/creator/scenario IDs,
see tests/test_ranking_groups.py's independence tests.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.ml.dataset_builder import CATEGORICAL, NUMERIC, FeatureHistory
from app.ml.feature_builder import EXPLICIT_NEGATIVE_EVENT_TYPES
from app.ml.sample_weight_policy import (
    NEGATIVE_EXPLICIT_REJECTION_WEIGHT,
    NEGATIVE_IMPLICIT_WEIGHT,
    POSITIVE_BASE_WEIGHT,
    POSITIVE_COMPLETION_WEIGHT,
    POSITIVE_CREATOR_FOLLOWED_WEIGHT,
    POSITIVE_LIKE_WEIGHT,
    POSITIVE_SHARE_OR_FAVORITE_WEIGHT,
)

RANKING_GROUP_VERSION = "v3-multisignal"
_NOW = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)

# The ONE canonical source of the ranking-groups dataset production actually trains/evaluates
# XGBRanker on (app.ml.trainer.train_and_select_cross_family -> app.ml.ranker_trainer.
# train_ranking_challengers, both default to these exact values). Eligibility reconciliation
# finding: a challenger-tooling run that passes a DIFFERENT `seed` here builds a genuinely
# different synthetic dataset -- not a different view of the same one -- so its eligibility
# result is not comparable to production's unless it uses this same default. Keeping this as the
# single named default (rather than the literal `2027`/`40` repeated at each call site) makes
# that drift impossible to introduce silently; see tests/test_ranking_groups.py's coverage that
# every downstream default still resolves to this constant.
PRODUCTION_GROUP_SEED = 2027
PRODUCTION_REPLICAS_PER_ARCHETYPE = 40


@dataclass(frozen=True)
class _CandidateOutcome:
    """One candidate's SYNTHETIC observed outcome -- the source of its relevance grade AND
    sample weight, but NEVER fed into its own feature vector (see `_group_rows` below: every
    candidate in a group is scored against the SAME point-in-time history, computed before any
    candidate's own outcome is known)."""
    content_id: str
    creator_id: str
    category: str
    event_type: str
    watch_percentage: float
    liked: bool = False
    shared: bool = False
    favorited: bool = False
    creator_followed: bool = False
    hashtags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    subgenres: list[str] = field(default_factory=list)
    title: str | None = None
    content_popularity_score: float = 0.5
    content_age_hours: float = 24.0


def relevance_grade(outcome: _CandidateOutcome) -> int:
    """Centralized graded (0-4) relevance policy (spec section 4), derived ONLY from a
    candidate's own synthetic outcome fields -- never leaks anything about other candidates in
    the same group or about the user's future history.

        4 = exceptionally strong positive (shared/favorited, or followed the creator here)
        3 = strong positive (liked, or a near-full completion)
        2 = moderate positive (solid watch, no stronger signal)
        1 = weak/exploration relevance (some engagement, well short of a full watch)
        0 = negative/unwanted -- EITHER an explicit CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED
            rejection OR a generic fast-skip (spec section 5: both may be 0 since this ranking
            framework's relevance labels are non-negative; the distinction between them is
            instead preserved via `row_weight` below, not the label).
    """
    if outcome.event_type in EXPLICIT_NEGATIVE_EVENT_TYPES:
        return 0
    wp = outcome.watch_percentage
    if wp < 20 and not (outcome.liked or outcome.shared or outcome.favorited or outcome.creator_followed):
        return 0
    if outcome.shared or outcome.favorited or outcome.creator_followed:
        return 4
    if outcome.liked or wp >= 90:
        return 3
    if wp >= 70:
        return 2
    return 1


def row_weight(outcome: _CandidateOutcome) -> float:
    """Reuses app.ml.sample_weight_policy's exact constants (Task 5's measured, gate-swept
    values) so explicit rejection remains strictly stronger than a generic skip even though
    both share relevance grade 0 (spec section 5) -- one definition of "how strongly should
    this outcome influence training" anywhere in this codebase, not a second invented scale."""
    if outcome.event_type in EXPLICIT_NEGATIVE_EVENT_TYPES:
        return NEGATIVE_EXPLICIT_REJECTION_WEIGHT
    wp = outcome.watch_percentage
    if wp < 20 and not (outcome.liked or outcome.shared or outcome.favorited or outcome.creator_followed):
        return NEGATIVE_IMPLICIT_WEIGHT
    weight = POSITIVE_BASE_WEIGHT
    if wp >= 90:
        weight = max(weight, POSITIVE_COMPLETION_WEIGHT)
    if outcome.liked:
        weight = max(weight, POSITIVE_LIKE_WEIGHT)
    if outcome.shared or outcome.favorited:
        weight = max(weight, POSITIVE_SHARE_OR_FAVORITE_WEIGHT)
    if outcome.creator_followed:
        weight = max(weight, POSITIVE_CREATOR_FOLLOWED_WEIGHT)
    return weight


class _Row:
    __slots__ = (
        "category",
        "commented",
        "content_duration_seconds",
        "content_id",
        "creator_followed",
        "creator_id",
        "event_id",
        "event_type",
        "favorited",
        "liked",
        "shared",
        "timestamp",
        "user_id",
        "watch_percentage",
        "watch_time_seconds",
    )

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _interaction(outcome: _CandidateOutcome, *, when: datetime) -> _Row:
    return _Row(
        event_id=outcome.content_id, user_id="u", content_id=outcome.content_id, creator_id=outcome.creator_id,
        category=outcome.category, event_type=outcome.event_type, watch_time_seconds=outcome.watch_percentage,
        content_duration_seconds=100.0, watch_percentage=outcome.watch_percentage, liked=outcome.liked,
        shared=outcome.shared, favorited=outcome.favorited, commented=False, creator_followed=outcome.creator_followed,
        timestamp=when,
    )


class _TokenContent:
    def __init__(self, tokens: list[str], *, topics=None, entities=None, subgenres=None, title=None) -> None:
        self.hashtags = tokens
        self.topics = topics or []
        self.entities = entities or []
        self.subgenres = subgenres or []
        self.title = title


def _archetype_same_category_diff_semantic(
    rng: random.Random, replica: str, group_timestamp: datetime,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    """Hard negative: SPORT/Football liked vs. SPORT/Tennis repeatedly, explicitly rejected --
    plus a generic exploration distractor and an irrelevant-popular distractor."""
    history = FeatureHistory()
    for i in range(7):
        wp = rng.uniform(85, 98)
        history.update(_interaction(_CandidateOutcome(
            f"rg-scd-fb-hist-{replica}-{i}", f"rg-coach-{replica}", "SPORT", "VIDEO_COMPLETED", wp, liked=True,
        ), when=group_timestamp - timedelta(days=30 + i)), content=_TokenContent(["FOOTBALL"]))
    for i in range(3):
        history.update(_interaction(_CandidateOutcome(
            f"rg-scd-tn-hist-{replica}-{i}", f"rg-coach-tennis-{replica}", "SPORT", "CONTENT_NOT_INTERESTED",
            rng.uniform(1, 6),
        ), when=group_timestamp - timedelta(days=12 + i)), content=_TokenContent(["TENNIS"]))
    candidates = [
        _CandidateOutcome(f"rg-scd-cand-fb-{replica}", f"rg-coach-{replica}-2", "SPORT", "VIDEO_COMPLETED",
                           rng.uniform(88, 99), liked=True, hashtags=["FOOTBALL"]),
        _CandidateOutcome(f"rg-scd-cand-tn-{replica}", f"rg-coach-tennis-{replica}-2", "SPORT", "CONTENT_NOT_INTERESTED",
                           rng.uniform(1, 6), hashtags=["TENNIS"]),
        _CandidateOutcome(f"rg-scd-cand-explore-{replica}", f"rg-dj-{replica}", "MUSIC", "VIDEO_WATCHED",
                           rng.uniform(45, 65)),
        _CandidateOutcome(f"rg-scd-cand-popular-{replica}", f"rg-fun-{replica}", "COMEDY", "VIDEO_SKIPPED",
                           rng.uniform(5, 15)),
    ]
    return history, candidates


def _archetype_popularity_distractor(
    rng: random.Random, replica: str, group_timestamp: datetime,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    """Relevant low-popularity item vs. irrelevant extremely-popular item -- popularity alone
    must not outrank real behavioral relevance."""
    history = FeatureHistory()
    for i in range(6):
        wp = rng.uniform(85, 97)
        history.update(_interaction(_CandidateOutcome(
            f"rg-pop-hist-{replica}-{i}", f"rg-food-creator-{replica}", "FOOD", "VIDEO_COMPLETED", wp, liked=True,
        ), when=group_timestamp - timedelta(days=20 + i)))
    candidates = [
        _CandidateOutcome(f"rg-pop-cand-relevant-{replica}", f"rg-food-creator-{replica}-2", "FOOD",
                           "VIDEO_COMPLETED", rng.uniform(90, 99), liked=True),
        _CandidateOutcome(f"rg-pop-cand-popular-{replica}", f"rg-pop-creator-{replica}", "COMEDY",
                           "VIDEO_SKIPPED", rng.uniform(5, 18)),
        _CandidateOutcome(f"rg-pop-cand-neutral-{replica}", f"rg-news-creator-{replica}", "NEWS",
                           "VIDEO_WATCHED", rng.uniform(40, 60)),
    ]
    return history, candidates


def _archetype_creator_conflict(
    rng: random.Random, replica: str, group_timestamp: datetime,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    """Followed creator posting the wrong subtheme vs. an unknown creator with strong semantic
    relevance -- creator-follow alone must not outrank genuine content relevance."""
    history = FeatureHistory()
    for i in range(6):
        wp = rng.uniform(85, 96)
        history.update(_interaction(_CandidateOutcome(
            f"rg-cc-hist-{replica}-{i}", f"rg-followed-{replica}", "SPORT", "VIDEO_COMPLETED", wp, liked=True,
        ), when=group_timestamp - timedelta(days=25 + i)), content=_TokenContent(["BASKETBALL"]))
    history.update(_interaction(_CandidateOutcome(
        f"rg-cc-follow-{replica}", f"rg-followed-{replica}", "SPORT", "CREATOR_FOLLOWED", 95, creator_followed=True,
    ), when=group_timestamp - timedelta(days=24)))
    candidates = [
        # `creator_followed` here is deliberately NOT set on this outcome: the creator is
        # already followed (a pre-existing relationship fact, correctly reflected in this
        # candidate's FEATURE vector via history.followed_creators -- see the CREATOR_FOLLOWED
        # event seeded into `history` above), but that is a materially different thing from
        # THIS interaction itself being a positive follow action -- a VIDEO_SKIPPED outcome on
        # the wrong subtheme is still a real, low-relevance skip regardless of who posted it.
        _CandidateOutcome(f"rg-cc-cand-wrongsub-{replica}", f"rg-followed-{replica}", "SPORT", "VIDEO_SKIPPED",
                           rng.uniform(8, 18), hashtags=["RUGBY"]),
        _CandidateOutcome(f"rg-cc-cand-relevant-{replica}", f"rg-unknown-{replica}", "SPORT", "VIDEO_COMPLETED",
                           rng.uniform(88, 98), liked=True, hashtags=["BASKETBALL"]),
    ]
    return history, candidates


def _archetype_long_term_vs_recent(
    rng: random.Random, replica: str, group_timestamp: datetime,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    """Strong long-term SPORT history vs. a strong recent/session MUSIC shift -- recent signal
    must be able to compete with (not necessarily always beat) stale long-term history."""
    history = FeatureHistory()
    for i in range(8):
        wp = rng.uniform(85, 97)
        history.update(_interaction(_CandidateOutcome(
            f"rg-ltr-sport-{replica}-{i}", f"rg-sport-creator-{replica}", "SPORT", "VIDEO_COMPLETED", wp, liked=True,
        ), when=group_timestamp - timedelta(days=70 + i)))
    for i in range(4):
        wp = rng.uniform(88, 98)
        history.update(_interaction(_CandidateOutcome(
            f"rg-ltr-music-{replica}-{i}", f"rg-music-creator-{replica}", "MUSIC", "VIDEO_COMPLETED", wp,
            liked=True, shared=(i == 0),
        ), when=group_timestamp - timedelta(minutes=20 - i * 4)))
    candidates = [
        _CandidateOutcome(f"rg-ltr-cand-sport-{replica}", f"rg-sport-creator-{replica}-2", "SPORT",
                           "VIDEO_WATCHED", rng.uniform(55, 70)),
        _CandidateOutcome(f"rg-ltr-cand-music-{replica}", f"rg-music-creator-{replica}-2", "MUSIC",
                           "VIDEO_COMPLETED", rng.uniform(88, 98), liked=True),
        _CandidateOutcome(f"rg-ltr-cand-cold-{replica}", f"rg-cold-creator-{replica}", "NEWS",
                           "VIDEO_SKIPPED", rng.uniform(5, 15)),
    ]
    return history, candidates


def _archetype_long_term_positive_with_recent_negative_burst(
    rng: random.Random, replica: str, group_timestamp: datetime,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    """LONG_TERM_POSITIVE_WITH_RECENT_NEGATIVE_BURST: established, long-term positive SPORT
    history followed by a RECENT burst of generic (implicit) fast-skips in the SAME category --
    the exact behavioral family app.ml.eligibility._gate_negative probes (session-level
    suppression must be real, without erasing a large real positive sample down to "we know
    nothing about this person"). Deliberately NOT the gate's own literal fixture (different
    day/minute offsets, different watch-percentage ranges, randomized per replica) -- same
    behavior family, not a memorized copy; see tests/test_ranking_groups.py's independence
    tests. No CONTENT_NOT_INTERESTED anywhere here: fast-skips are deliberately ambiguous/
    implicit, mirroring the exact distinction app.ml.eligibility draws between `_gate_negative`
    (implicit) and `_gate_not_interested` (explicit) -- this archetype is the implicit-only one.
    """
    history = FeatureHistory()
    # Established long-term positive: 7 strong watches, ~35-77 days back -- well outside
    # RECENT_WINDOW (30 days) and SESSION_WINDOW (30 minutes), so only the STABLE, all-time
    # category_affinity/category_positive_count carry this signal forward. This is exactly
    # what condition C ("must stay above a cold stranger") depends on. Tagged FOOTBALL so the
    # "strong" candidate below has a REAL matched-token history to inherit, not just a bare
    # category match (see module-level fix note below).
    for i in range(7):
        wp = rng.uniform(84, 97)
        history.update(_interaction(_CandidateOutcome(
            f"rg-neg-hist-pos-{replica}-{i}", f"rg-neg-creator-{replica}", "SPORT", "VIDEO_COMPLETED", wp, liked=True,
        ), when=group_timestamp - timedelta(days=35 + i * 6 + rng.uniform(0, 3))), content=_TokenContent(["FOOTBALL"]))
    # Recent negative burst: 3 generic (implicit) fast-skips, minute-scale recency (within
    # SESSION_WINDOW), same category -- pulls recent_category_affinity/session_category_
    # affinity down without touching the explicit-rejection machinery at all. Because the
    # long-term positives above sit outside RECENT_WINDOW, these 3 skips are the ONLY recent-
    # window evidence, so recent_category_affinity ends up genuinely negative even though
    # all-time category_affinity stays clearly positive -- the exact asymmetry condition C
    # tests. Tagged TENNIS (a DIFFERENT subtheme from the positive history's FOOTBALL) -- the
    # "suppressed" candidate below inherits a genuinely negative matched-token history from
    # this, not just an ambient category-level mood.
    for i in range(3):
        wp = rng.uniform(3, 14)
        history.update(_interaction(_CandidateOutcome(
            f"rg-neg-hist-skip-{replica}-{i}", f"rg-neg-skip-creator-{replica}", "SPORT", "VIDEO_SKIPPED", wp,
        ), when=group_timestamp - timedelta(minutes=6 + i * 7 + rng.uniform(0, 2))), content=_TokenContent(["TENNIS"]))
    candidates = [
        # Unsuppressed-style: SPORT/Football, the SAME subtheme as the established positive
        # history -- inherits a genuinely high hashtag_affinity (not just category_affinity),
        # and is watched well despite the user's recent skip mood (relevance 3).
        _CandidateOutcome(f"rg-neg-cand-strong-{replica}", f"rg-neg-creator-{replica}-2", "SPORT",
                           "VIDEO_COMPLETED", rng.uniform(87, 98), liked=True, hashtags=["FOOTBALL"]),
        # Recently-suppressed-style: SPORT/Tennis, the SAME subtheme as the recent fast-skip
        # burst -- inherits a genuinely LOW (below-neutral) hashtag_affinity from those 3
        # skips, distinct from both "strong"'s high affinity and "weak"'s neutral one. Some
        # real engagement, clearly weaker than the strong match, but NOT another skip
        # (relevance 1, landing ABOVE the cold/irrelevant candidates below -- condition C's
        # ordering, spelled out as an actual candidate comparison, not just a feature probe).
        _CandidateOutcome(f"rg-neg-cand-suppressed-{replica}", f"rg-neg-skip-creator-{replica}-2", "SPORT",
                           "VIDEO_WATCHED", rng.uniform(42, 58), hashtags=["TENNIS"]),
        # SPORT/Basketball -- a subtheme with NO prior evidence at all (neutral 0.5 hashtag_
        # affinity, has_semantic_history=0 for this token): genuinely different features from
        # BOTH strong (positive) and suppressed (negative), landing between them (relevance 2)
        # -- "little/no prior semantic evidence" earns more trust than an actively-skipped
        # subtheme, but less than an established, positively-reinforced one.
        _CandidateOutcome(f"rg-neg-cand-weak-{replica}", f"rg-neg-creator-{replica}-3", "SPORT",
                           "VIDEO_WATCHED", rng.uniform(72, 82), hashtags=["BASKETBALL"]),
        # Completely cold, unrelated category -- must rank below every SPORT candidate above,
        # including the recently-suppressed one (relevance 0).
        _CandidateOutcome(f"rg-neg-cand-cold-{replica}", f"rg-neg-cold-creator-{replica}", "NEWS",
                           "VIDEO_SKIPPED", rng.uniform(4, 16)),
        # Popular-but-irrelevant distractor, matching this module's existing popularity-
        # distractor archetype's own convention (relevance 0).
        _CandidateOutcome(f"rg-neg-cand-popular-{replica}", f"rg-neg-pop-creator-{replica}", "COMEDY",
                           "VIDEO_SKIPPED", rng.uniform(5, 15)),
    ]
    return history, candidates


def _archetype_secondary_tiebreakers(
    rng: random.Random, replica: str, group_timestamp: datetime,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    """V2.1 residual-signal slate: one category/context, controlled candidate differences.

    Generic quality is useful in the first comparison, but deliberately contradicted by the
    strong-old and popular-negative candidates, so it can only become a bounded tie-breaker.
    Semantic dimensions are independently populated rather than globally painted onto every
    archetype (V2's measured source of spurious correlations and benchmark regression).
    """
    history = FeatureHistory()
    positive_content = _TokenContent(
        ["V21_TAG_POS"], topics=["V21_TOPIC_POS"], entities=["V21_ENTITY_POS"],
        subgenres=["V21_SUBGENRE_POS"], title="V21TITLEPOS",
    )
    negative_content = _TokenContent(
        ["V21_TAG_NEG"], topics=["V21_TOPIC_NEG"], entities=["V21_ENTITY_NEG"],
        subgenres=["V21_SUBGENRE_NEG"], title="V21TITLENEG",
    )
    for i in range(4):
        history.update(_interaction(_CandidateOutcome(
            f"rg-v21-pos-hist-{replica}-{i}", f"rg-v21-known-{replica}", "SPORT",
            "VIDEO_COMPLETED", rng.uniform(88, 97), liked=True,
        ), when=group_timestamp - timedelta(days=10 + i)), content=positive_content)
    for i in range(2):
        history.update(_interaction(_CandidateOutcome(
            f"rg-v21-neg-hist-{replica}-{i}", f"rg-v21-rejected-{replica}", "SPORT",
            "CONTENT_NOT_INTERESTED", rng.uniform(2, 7),
        ), when=group_timestamp - timedelta(days=4 + i)), content=negative_content)
    history.followed_creators.add(("u", f"rg-v21-followed-{replica}"))
    return history, [
        _CandidateOutcome(f"rg-v21-good-{replica}", f"rg-v21-new-{replica}", "SPORT",
                          event_type="VIDEO_COMPLETED", watch_percentage=94, liked=True,
                          topics=["V21_TOPIC_POS"], content_popularity_score=.70, content_age_hours=24),
        _CandidateOutcome(f"rg-v21-neutral-{replica}", f"rg-v21-neutral-{replica}", "SPORT",
                          event_type="VIDEO_WATCHED", watch_percentage=74,
                          entities=["V21_ENTITY_POS"], content_popularity_score=.40, content_age_hours=72),
        _CandidateOutcome(f"rg-v21-strong-old-{replica}", f"rg-v21-known-{replica}", "SPORT",
                          event_type="VIDEO_COMPLETED", watch_percentage=96, liked=True,
                          subgenres=["V21_SUBGENRE_POS"], content_popularity_score=.20, content_age_hours=480),
        _CandidateOutcome(f"rg-v21-followed-weak-{replica}", f"rg-v21-followed-{replica}", "SPORT",
                          event_type="VIDEO_WATCHED", watch_percentage=86, liked=True,
                          title="V21TITLEPOS", content_popularity_score=.55, content_age_hours=48),
        _CandidateOutcome(f"rg-v21-popular-negative-{replica}", f"rg-v21-rejected-{replica}", "SPORT",
                          event_type="CONTENT_NOT_INTERESTED", watch_percentage=4,
                          topics=["V21_TOPIC_NEG"], entities=["V21_ENTITY_NEG"],
                          subgenres=["V21_SUBGENRE_NEG"], title="V21TITLENEG",
                          content_popularity_score=.95, content_age_hours=1),
    ]


def _archetype_field_semantics(
    rng: random.Random, replica: str, group_timestamp: datetime,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    """One semantic field at a time, with both directional evidence and counterexamples.

    The relevance labels come only from the later synthetic outcomes.  Pre-ranking semantic
    facts are deliberately noisy: a positive match is usually useful, but one matched item is
    skipped and one negative-token item receives a strong outcome.  This prevents any field
    affinity from becoming a deterministic label proxy while still providing matched-context
    evidence for hashtags, topics, entities, subgenres and title across replicas.
    """
    return _field_semantic_group(rng, replica, group_timestamp, "hashtags")


def _field_semantic_group(
    rng: random.Random, replica: str, group_timestamp: datetime, field_name: str,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    positive_token = f"V3_{field_name.upper()}_POS"
    negative_token = f"V3_{field_name.upper()}_NEG"

    def content(token: str) -> _TokenContent:
        kwargs: dict[str, Any] = {"tokens": []}
        if field_name == "hashtags":
            kwargs["tokens"] = [token]
        elif field_name == "title":
            kwargs["title"] = token
        else:
            kwargs[field_name] = [token]
        return _TokenContent(**kwargs)

    def metadata(token: str) -> dict[str, Any]:
        if field_name == "title":
            return {"title": token}
        return {field_name: [token]}

    history = FeatureHistory()
    for i in range(5):
        history.update(_interaction(_CandidateOutcome(
            f"rg-v3-sem-pos-hist-{replica}-{i}", f"rg-v3-sem-cr-{replica}", "MUSIC",
            "VIDEO_COMPLETED", rng.uniform(88, 98), liked=True,
        ), when=group_timestamp - timedelta(days=12 + i)), content=content(positive_token))
    for i in range(3):
        history.update(_interaction(_CandidateOutcome(
            f"rg-v3-sem-neg-hist-{replica}-{i}", f"rg-v3-sem-neg-cr-{replica}", "MUSIC",
            "CONTENT_NOT_INTERESTED", rng.uniform(2, 8),
        ), when=group_timestamp - timedelta(days=5 + i)), content=content(negative_token))

    return history, [
        _CandidateOutcome(f"rg-v3-sem-match-{replica}", f"rg-v3-new-a-{replica}", "MUSIC",
                          "VIDEO_COMPLETED", 94, liked=True, content_popularity_score=.55,
                          content_age_hours=48, **metadata(positive_token)),
        _CandidateOutcome(f"rg-v3-sem-neutral-{replica}", f"rg-v3-new-b-{replica}", "MUSIC",
                          "VIDEO_WATCHED", 74, content_popularity_score=.55, content_age_hours=48),
        _CandidateOutcome(f"rg-v3-sem-reject-{replica}", f"rg-v3-new-c-{replica}", "MUSIC",
                          "CONTENT_NOT_INTERESTED", 4, content_popularity_score=.55,
                          content_age_hours=48, **metadata(negative_token)),
        # Counterexamples stop either semantic direction becoming a perfect grade rule.
        _CandidateOutcome(f"rg-v3-sem-pos-skip-{replica}", f"rg-v3-new-d-{replica}", "MUSIC",
                          "VIDEO_SKIPPED", 10, content_popularity_score=.20,
                          content_age_hours=480, **metadata(positive_token)),
        _CandidateOutcome(f"rg-v3-sem-neg-watch-{replica}", f"rg-v3-new-e-{replica}", "MUSIC",
                          "VIDEO_COMPLETED", 91, liked=True, content_popularity_score=.75,
                          content_age_hours=12, **metadata(negative_token)),
    ]


def _archetype_topic_semantics(rng: random.Random, replica: str, group_timestamp: datetime):
    return _field_semantic_group(rng, replica, group_timestamp, "topics")


def _archetype_entity_semantics(rng: random.Random, replica: str, group_timestamp: datetime):
    return _field_semantic_group(rng, replica, group_timestamp, "entities")


def _archetype_subgenre_semantics(rng: random.Random, replica: str, group_timestamp: datetime):
    return _field_semantic_group(rng, replica, group_timestamp, "subgenres")


def _archetype_title_semantics(rng: random.Random, replica: str, group_timestamp: datetime):
    return _field_semantic_group(rng, replica, group_timestamp, "title")


def _archetype_creator_residual(
    rng: random.Random, replica: str, group_timestamp: datetime,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    """Creator familiarity breaks a genuine tie but cannot rescue irrelevant content."""
    history = FeatureHistory()
    followed = f"rg-v3-creator-followed-{replica}"
    for i in range(4):
        history.update(_interaction(_CandidateOutcome(
            f"rg-v3-creator-hist-{replica}-{i}", followed, "SPORT", "VIDEO_COMPLETED",
            rng.uniform(85, 97), liked=i % 2 == 0,
        ), when=group_timestamp - timedelta(days=12 + i)), content=_TokenContent(["CYCLING"]))
    history.followed_creators.add(("u", followed))
    return history, [
        _CandidateOutcome(f"rg-v3-creator-known-{replica}", followed, "SPORT", "VIDEO_COMPLETED",
                          93, liked=True, hashtags=["CYCLING"], content_popularity_score=.50, content_age_hours=36),
        _CandidateOutcome(f"rg-v3-creator-unknown-{replica}", f"rg-v3-creator-new-{replica}", "SPORT",
                          "VIDEO_WATCHED", 78, hashtags=["CYCLING"], content_popularity_score=.50,
                          content_age_hours=36),
        _CandidateOutcome(f"rg-v3-creator-offtopic-{replica}", followed, "NEWS", "VIDEO_SKIPPED", 8,
                          content_popularity_score=.80, content_age_hours=4),
        _CandidateOutcome(f"rg-v3-creator-unrelated-{replica}", f"rg-v3-creator-other-{replica}", "COMEDY",
                          "CONTENT_NOT_INTERESTED", 3, content_popularity_score=.35, content_age_hours=96),
    ]


def _archetype_quality_residual(
    rng: random.Random, replica: str, group_timestamp: datetime,
) -> tuple[FeatureHistory, list[_CandidateOutcome]]:
    """Quality/freshness can resolve comparable relevance, never override personalization."""
    history = FeatureHistory()
    for i in range(6):
        history.update(_interaction(_CandidateOutcome(
            f"rg-v3-quality-hist-{replica}-{i}", f"rg-v3-quality-cr-{replica}", "FOOD",
            "VIDEO_COMPLETED", rng.uniform(84, 97), liked=i % 2 == 0,
        ), when=group_timestamp - timedelta(days=20 + i)))
    return history, [
        _CandidateOutcome(f"rg-v3-quality-fresh-{replica}", f"rg-v3-q-a-{replica}", "FOOD",
                          "VIDEO_COMPLETED", 92, liked=True, content_popularity_score=.80, content_age_hours=6),
        _CandidateOutcome(f"rg-v3-quality-stale-{replica}", f"rg-v3-q-b-{replica}", "FOOD",
                          "VIDEO_WATCHED", 76, content_popularity_score=.45, content_age_hours=360),
        _CandidateOutcome(f"rg-v3-quality-pop-skip-{replica}", f"rg-v3-q-c-{replica}", "NEWS",
                          "VIDEO_SKIPPED", 8, content_popularity_score=.95, content_age_hours=2),
        _CandidateOutcome(f"rg-v3-quality-old-relevant-{replica}", f"rg-v3-q-d-{replica}", "FOOD",
                          "VIDEO_COMPLETED", 94, liked=True, content_popularity_score=.20, content_age_hours=600),
        _CandidateOutcome(f"rg-v3-quality-fresh-irrelevant-{replica}", f"rg-v3-q-e-{replica}", "COMEDY",
                          "CONTENT_NOT_INTERESTED", 3, content_popularity_score=.70, content_age_hours=1),
    ]


_ARCHETYPES = (
    _archetype_same_category_diff_semantic,
    _archetype_popularity_distractor,
    _archetype_creator_conflict,
    _archetype_long_term_vs_recent,
    _archetype_long_term_positive_with_recent_negative_burst,
    _archetype_secondary_tiebreakers,
    _archetype_field_semantics,
    _archetype_topic_semantics,
    _archetype_entity_semantics,
    _archetype_subgenre_semantics,
    _archetype_title_semantics,
    _archetype_creator_residual,
    _archetype_quality_residual,
)


def _group_timestamp_for(query_index: int) -> datetime:
    """The ONE canonical source of a group's scoring timestamp -- every archetype receives this
    exact value (never the module's global `_NOW`) and must anchor all of its own historical
    events relative to it, so recent/session-window evidence stays genuinely recent regardless
    of how far into the generated dataset a given group falls. Chronologically increasing by
    query_index (1 minute apart) purely to give every group a distinct, strictly-ordered
    timestamp -- callers must not reinterpret this as real wall-clock spacing."""
    return _NOW + timedelta(minutes=query_index)


def build_ranking_groups(
    *, replicas_per_archetype: int = PRODUCTION_REPLICAS_PER_ARCHETYPE, seed: int = PRODUCTION_GROUP_SEED,
) -> pd.DataFrame:
    """Returns a DataFrame with CATEGORICAL+NUMERIC feature columns, plus `relevance` (0-4,
    int), `sample_weight` (float), `query_id` (str, one per group), `group_size` (int),
    `rankingGroupVersion`. Rows for the SAME group are contiguous (required by XGBRanker/
    LGBMRanker's `group` and CatBoostRanker's `group_id` fit conventions) and chronologically
    ordered across groups (`timestamp` strictly increasing group-to-group), matching this
    codebase's point-in-time discipline even though groups are independent of each other.
    """
    def _emit_group(
        history: FeatureHistory, candidates: list[_CandidateOutcome], query_index: int, group_timestamp: datetime,
    ) -> list[dict[str, Any]]:
        query_id = f"rg-{query_index:05d}"
        group_rows = []
        for candidate in candidates:
            # Every candidate in this group is scored against the SAME point-in-time
            # history (built entirely from the group's OWN prior events, all strictly before
            # `group_timestamp`) -- .features() is read-only, never mutates `history`, so
            # scoring candidate 2 cannot see candidate 1's own outcome.
            features = history.features(
                user_id="u", category=candidate.category, creator_id=candidate.creator_id,
                content_id=candidate.content_id, timestamp=group_timestamp,
                content_popularity_score=candidate.content_popularity_score,
                content_created_at=group_timestamp - timedelta(hours=candidate.content_age_hours),
                hashtags=candidate.hashtags, topics=candidate.topics, entities=candidate.entities,
                subgenres=candidate.subgenres, title=candidate.title,
            )
            group_rows.append({
                **features, "relevance": relevance_grade(candidate), "sample_weight": row_weight(candidate),
                "query_id": query_id, "timestamp": group_timestamp, "content_id": candidate.content_id,
            })
        group_rows.sort(key=lambda r: r["relevance"], reverse=True)  # deterministic within-group order
        return group_rows

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    query_index = 0
    for replica in range(replicas_per_archetype):
        for archetype in _ARCHETYPES:
            group_timestamp = _group_timestamp_for(query_index)
            history, candidates = archetype(rng, f"{seed}-{replica}", group_timestamp)
            rows.extend(_emit_group(history, candidates, query_index, group_timestamp))
            query_index += 1
    df = pd.DataFrame(rows)
    df["group_size"] = df.groupby("query_id")["query_id"].transform("size")
    df["rankingGroupVersion"] = RANKING_GROUP_VERSION
    return df[["query_id", "group_size", "timestamp", "content_id", "relevance", "sample_weight",
               "rankingGroupVersion", *CATEGORICAL, *NUMERIC]]


def group_sizes(df: pd.DataFrame) -> list[int]:
    """Group sizes in the exact row order `df` is already in -- callers must not reorder `df`
    between calling this and fitting a ranker on it (XGBRanker/LGBMRanker require contiguous,
    size-matching rows)."""
    return df.groupby("query_id", sort=False)["query_id"].size().tolist()


def per_group_weights(df: pd.DataFrame) -> list[float]:
    """Mean row-level `sample_weight` within each group, in the same group order `group_sizes`
    returns -- for estimators whose ranking objective only accepts one weight per query group,
    not per candidate (see app.ml.ranker_registry's `weight_granularity`; a real XGBoost API
    constraint, not a design choice). A group containing an explicit rejection (weight 2.0)
    still ends up with a higher mean than an all-neutral-negative group, so SOME of the
    explicit-vs-generic distinction survives even at this coarser granularity."""
    return df.groupby("query_id", sort=False)["sample_weight"].mean().tolist()
