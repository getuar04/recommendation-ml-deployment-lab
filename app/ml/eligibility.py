"""Generic behavioral eligibility gates for candidate model selection (finalization spec
Step 11/Step 3: "same dataset -> train all supported models -> evaluate behavioral gates ->
ELIGIBLE/NOT ELIGIBLE -> compare aggregate metrics among ELIGIBLE models only").

Every gate builds its probe candidates through `app.ml.dataset_builder.FeatureHistory` --
the exact same point-in-time feature-computation path production scoring uses -- with generic
placeholder identifiers ("CATEGORY_A"/"CREATOR_A"/"TOKEN_A", never a real-world name like
"SPORT"/"Barcelona"/a specific creator ID). This is deliberate on two fronts: production
selection logic must never contain named demo entities (those belong only in test/evaluation
fixtures), and reusing FeatureHistory (rather than hand-rolling a synthetic feature row from
scratch) guarantees every probe's feature combination is internally consistent -- e.g. strong
session evidence in a category naturally implies that category also has some interaction
history, exactly as it would for a real user, avoiding the out-of-distribution combinations a
hand-built "neutral row" can accidentally produce.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.ml.dataset_builder import FEATURES, FeatureHistory

_NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _score_pair(model: Any, row_a: dict[str, Any], row_b: dict[str, Any]) -> tuple[float, float]:
    frame = pd.DataFrame([row_a, row_b])[FEATURES]
    scores = model.predict_proba(frame)[:, 1]
    return float(scores[0]), float(scores[1])


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


def _interaction(content_id: str, creator_id: str, category: str, *, watch_percentage: float,
                  when: datetime, event_type: str | None = None, liked: bool = False, shared: bool = False,
                  favorited: bool = False, commented: bool = False, creator_followed: bool = False) -> _Row:
    if event_type is None:
        event_type = "VIDEO_COMPLETED" if watch_percentage >= 90 else (
            "VIDEO_SKIPPED" if watch_percentage < 20 else "VIDEO_WATCHED")
    return _Row(event_id=content_id, user_id="probe-user", content_id=content_id, creator_id=creator_id,
                category=category, event_type=event_type, watch_time_seconds=watch_percentage,
                content_duration_seconds=100.0, watch_percentage=watch_percentage,
                liked=liked, shared=shared, favorited=favorited, commented=commented,
                creator_followed=creator_followed, timestamp=when)


def _features_for(history: FeatureHistory, *, category: str, creator_id: str = "CREATOR_NEUTRAL",
                   hashtags=None, topics=None, entities=None, subgenres=None, title=None) -> dict[str, Any]:
    return history.features(
        user_id="probe-user", category=category, creator_id=creator_id, content_id="probe-candidate",
        timestamp=_NOW, content_popularity_score=0.5, content_created_at=_NOW - timedelta(hours=24),
        hashtags=hashtags, topics=topics, entities=entities, subgenres=subgenres, title=title,
    )


def _gate_long_term(model: Any) -> tuple[bool, float]:
    """Strong, long-term (40-90 day old) positive history in one category vs. a genuinely
    never-seen category."""
    history = FeatureHistory()
    for i in range(10):
        history.update(_interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=92,
                                     when=_NOW - timedelta(days=50 + i), liked=True))
    strong = _features_for(history, category="CATEGORY_A")
    never_seen = _features_for(history, category="CATEGORY_B")
    a, b = _score_pair(model, strong, never_seen)
    return a > b, a - b


def _gate_recent(model: Any) -> tuple[bool, float]:
    """Weak/stale long-term history in category A vs. strong recent (last 2 days) history in
    category B -- recent must outrank stale."""
    history = FeatureHistory()
    for i in range(4):
        history.update(_interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=55,
                                     when=_NOW - timedelta(days=60 + i)))
    for i in range(6):
        history.update(_interaction("c2", "CREATOR_B", "CATEGORY_B", watch_percentage=96,
                                     when=_NOW - timedelta(days=2, hours=i), liked=True, shared=True))
    recent_strong = _features_for(history, category="CATEGORY_B")
    stale = _features_for(history, category="CATEGORY_A")
    a, b = _score_pair(model, recent_strong, stale)
    return a > b, a - b


def _gate_session(model: Any) -> tuple[bool, float]:
    """Strong current-session evidence (last few minutes) in a category the user has no other
    history in must register a real, positive session_category_affinity/confidence reading."""
    history = FeatureHistory()
    for i in range(3):
        history.update(_interaction(f"c{i}", "CREATOR_A", "CATEGORY_A", watch_percentage=92,
                                     when=_NOW - timedelta(minutes=5 - i), liked=True))
    session_strong = _features_for(history, category="CATEGORY_A")
    passed = (
        session_strong["has_session_activity"] == 1
        and session_strong["session_category_affinity"] > 0.5
        and session_strong["session_intent_confidence"] >= 0.5
    )
    return passed, session_strong["session_category_affinity"] - 0.5


def _gate_negative(model: Any) -> tuple[bool, float]:
    """Strong long-term positive history in a category, then a burst of recent FAST-SKIPS
    (implicit rejection) in that SAME category: session-level suppression must be real (score
    below the same long-term history with neutral session state) without erasing the long-term
    history (score must stay above a genuinely cold candidate).

    Unchanged since before the Phase 1.5 gate-semantics split (see `_gate_not_interested`
    below): a handful of fast-skips is weak, ambiguous, session/candidate-quality-confounded
    evidence -- it can reasonably reflect bad candidate matches or a passing mood rather than a
    genuine category-level reversal, so it must not be allowed to fully erase a large real
    positive sample down to "we know nothing about this person". This invariant is specific to
    IMPLICIT rejection; explicit CONTENT_NOT_INTERESTED is tested separately below with a
    deliberately different invariant -- see that gate's docstring for why."""
    history_suppressed = FeatureHistory()
    for i in range(6):
        history_suppressed.update(_interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                                 when=_NOW - timedelta(days=50 + i), liked=True))
    for i in range(4):
        history_suppressed.update(_interaction(f"skip{i}", "CREATOR_A", "CATEGORY_A", watch_percentage=4,
                                                 when=_NOW - timedelta(minutes=12 - i * 3), event_type="VIDEO_SKIPPED"))
    suppressed = _features_for(history_suppressed, category="CATEGORY_A")

    history_unsuppressed = FeatureHistory()
    for i in range(6):
        history_unsuppressed.update(_interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                                   when=_NOW - timedelta(days=50 + i), liked=True))
    unsuppressed = _features_for(history_unsuppressed, category="CATEGORY_A")

    cold = _features_for(FeatureHistory(), category="CATEGORY_A")

    suppressed_score, unsuppressed_score = _score_pair(model, suppressed, unsuppressed)
    suppressed_score2, cold_score = _score_pair(model, suppressed, cold)
    session_suppressed = suppressed["session_category_affinity"] < 0.5
    # `unsuppressed_score` was already computed above for the returned margin but, until this
    # fix, was never actually checked -- a model that scores the fast-skip-suppressed candidate
    # *higher* than the same user's unsuppressed baseline (stronger negative evidence read as a
    # stronger positive signal) could still pass purely on "suppressed still beats a cold
    # stranger". Direction only, no magnitude threshold, so this stays generic across models.
    direction_correct = suppressed_score < unsuppressed_score
    passed = session_suppressed and direction_correct and suppressed_score2 > cold_score
    return passed, unsuppressed_score - suppressed_score


def _gate_not_interested(model: Any) -> tuple[bool, float]:
    """Strong long-term positive history in a category, then a burst of recent EXPLICIT
    CONTENT_NOT_INTERESTED events in that SAME category: suppression must be real and clear
    relative to the unsuppressed baseline -- deliberately NOT held to the same "must stay above
    a cold candidate" floor `_gate_negative` requires for implicit fast-skips.

    Why a different invariant (Phase 1.5 issue 2B; product-behavior analysis, not guessed):
    CONTENT_NOT_INTERESTED is a deliberate, low-noise, high-intent user action, not an ambiguous
    session/candidate-quality-confounded signal -- the weighting formula this codebase already
    uses agrees, weighting it -8 vs. a fast-skip's -4 (app.ml.dataset_builder.FeatureHistory.
    update), the single strongest negative weight the system defines. Continuing to guarantee
    "at least as relevant as a total stranger" for a category the user explicitly told us they
    are not interested in is a worse product outcome than under-serving it -- unlike an
    ambiguous fast-skip, there is no real case for treating this as noise that a large past
    positive sample should structurally outweigh. This gate instead requires only that
    suppression be real and clearly registered (session-level AND a measurable score drop from
    the unsuppressed baseline), which is the part of "negative feedback must suppress
    relevance" (see module docstring) that genuinely generalizes across both signal types --
    it does not require or forbid crossing the cold baseline either way, since product intent
    doesn't mandate a specific floor here, only that the suppression itself be real.

    Generic across categories/users/content (Decision 6, matching every other gate in this
    module): CATEGORY_A/CREATOR_A/c1/notInterested{i} are placeholders, never a named
    real-world entity."""
    history_suppressed = FeatureHistory()
    for i in range(6):
        history_suppressed.update(_interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                                 when=_NOW - timedelta(days=50 + i), liked=True))
    for i in range(4):
        history_suppressed.update(_interaction(f"notInterested{i}", "CREATOR_A", "CATEGORY_A", watch_percentage=4,
                                                 when=_NOW - timedelta(minutes=12 - i * 3),
                                                 event_type="CONTENT_NOT_INTERESTED"))
    suppressed = _features_for(history_suppressed, category="CATEGORY_A")

    history_unsuppressed = FeatureHistory()
    for i in range(6):
        history_unsuppressed.update(_interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                                   when=_NOW - timedelta(days=50 + i), liked=True))
    unsuppressed = _features_for(history_unsuppressed, category="CATEGORY_A")

    suppressed_score, unsuppressed_score = _score_pair(model, suppressed, unsuppressed)
    session_suppressed = suppressed["session_category_affinity"] < 0.5
    passed = session_suppressed and suppressed_score < unsuppressed_score
    return passed, unsuppressed_score - suppressed_score


def _gate_semantic(model: Any) -> tuple[bool, float]:
    """Real prior positive history with one generic token set vs. real prior negative history
    with a DIFFERENT generic token set, both within the same category (isolates the semantic
    signal from category_affinity, which stays identical for both probes)."""
    history = FeatureHistory()
    for i in range(6):
        history.update(_interaction("pos1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                     when=_NOW - timedelta(days=15 + i), liked=True),
                        content=_TokenContent(["TOKEN_A"]))
    for i in range(4):
        history.update(_interaction("neg1", "CREATOR_A", "CATEGORY_A", watch_percentage=4,
                                     when=_NOW - timedelta(days=12 + i), event_type="CONTENT_NOT_INTERESTED"),
                        content=_TokenContent(["TOKEN_B"]))
    matched = _features_for(history, category="CATEGORY_A", hashtags=["TOKEN_A"], topics=["TOKEN_A"],
                             entities=["TOKEN_A"], subgenres=["TOKEN_A"], title="TOKEN_A")
    mismatched = _features_for(history, category="CATEGORY_A", hashtags=["TOKEN_B"], topics=["TOKEN_B"],
                                entities=["TOKEN_B"], subgenres=["TOKEN_B"], title="TOKEN_B")
    a, b = _score_pair(model, matched, mismatched)
    return a > b, a - b


class _TokenContent:
    """Minimal duck-typed content object exposing the attributes app.ml.dataset_builder.
    _token_fields reads (hashtags/topics/entities/subgenres/title) -- all fields carry the
    SAME generic token so every semantic field contributes consistently, exactly like a real
    creator consistently tagging one video."""
    def __init__(self, tokens: list[str]) -> None:
        self.hashtags = tokens
        self.topics = tokens
        self.entities = tokens
        self.subgenres = tokens
        self.title = tokens[0]


def _gate_creator(model: Any) -> tuple[bool, float]:
    """Real prior positive/high-completion history with one generic creator vs. real prior
    negative/low-completion history with a different generic creator, same category."""
    history = FeatureHistory()
    for i in range(9):
        history.update(_interaction(f"pref{i}", "CREATOR_PREFERRED", "CATEGORY_A", watch_percentage=95,
                                     when=_NOW - timedelta(days=15 + i), liked=True, creator_followed=(i == 0)))
    for i in range(6):
        history.update(_interaction(f"weak{i}", "CREATOR_WEAK", "CATEGORY_A", watch_percentage=8,
                                     when=_NOW - timedelta(days=12 + i), event_type="VIDEO_SKIPPED"))
    preferred = _features_for(history, category="CATEGORY_A", creator_id="CREATOR_PREFERRED")
    weak = _features_for(history, category="CATEGORY_A", creator_id="CREATOR_WEAK")
    a, b = _score_pair(model, preferred, weak)
    return a > b, a - b


# Tolerance for the two gates below (already-seen preference; same-vs-diverse subtheme
# rejection localization): both probe a real, fitted classifier's response to features that
# are a small, second-order influence among ~35 others, not the dominant signal any single
# gate above tests in isolation -- measured directly against this project's real synthetic
# dataset across every candidate algorithm (LogisticRegression/RandomForest/XGBoost/LightGBM/
# CatBoost), the "wrong-direction" noise on both never exceeded ~0.10 for an otherwise
# healthy model, while a genuinely broken model (actively learning the opposite of the
# intended direction) produces a margin far larger than this. Tolerant by design (spec: "use
# tolerances, not exact scores") -- these two gates guard against a candidate materially
# preferring the wrong outcome, not against small, expected noise around a weak feature.
SOFT_GATE_TOLERANCE = 0.15


def _gate_already_seen(model: Any) -> tuple[bool, float]:
    """Real, strong long-term positive history in a category; scoring the SAME candidate
    already-seen vs. not-yet-seen must not materially favor the seen one (within
    SOFT_GATE_TOLERANCE). Production's already_seen deprioritization is guaranteed downstream
    regardless of the raw model (app.ml.reranker.SEEN_PENALTY, a fixed multiplier applied
    identically to every algorithm) -- this gate only guards against a candidate model
    actively learning a strong preference for already-seen content, not against small noise
    around this comparatively weak feature."""
    history = FeatureHistory()
    for i in range(8):
        history.update(_interaction(f"c{i}", "CREATOR_A", "CATEGORY_A", watch_percentage=92,
                                     when=_NOW - timedelta(days=20 + i), liked=True))
    unseen = history.features(user_id="probe-user", category="CATEGORY_A", creator_id="CREATOR_A",
                               content_id="probe-candidate", timestamp=_NOW, content_popularity_score=0.5,
                               content_created_at=_NOW - timedelta(hours=24), already_seen=False)
    seen = history.features(user_id="probe-user", category="CATEGORY_A", creator_id="CREATOR_A",
                             content_id="probe-candidate", timestamp=_NOW, content_popularity_score=0.5,
                             content_created_at=_NOW - timedelta(hours=24), already_seen=True)
    seen_score, unseen_score = _score_pair(model, seen, unseen)
    margin = unseen_score - seen_score
    return margin >= -SOFT_GATE_TOLERANCE, margin


def _gate_cold_start(model: Any) -> tuple[bool, float]:
    """Sanity check for a genuinely cold-start probe (zero prior history): the raw score must
    be finite and land in [0, 1] (predict_proba's own contract, checked defensively rather
    than assumed), and must respond monotonically to catalog popularity -- with no history to
    personalize against at all, a higher-popularity candidate must never score lower than an
    otherwise-identical lower-popularity one. Deliberately coarse (cold start has no
    personalization signal to check beyond this), not a precision requirement."""
    history = FeatureHistory()
    low_popularity = _features_for(history, category="CATEGORY_A")
    low_popularity["content_popularity_score"] = 0.2
    high_popularity = dict(low_popularity, content_popularity_score=0.9)
    low_score, high_score = _score_pair(model, low_popularity, high_popularity)
    finite_and_bounded = all(0.0 <= s <= 1.0 and not math.isnan(s) for s in (low_score, high_score))
    return finite_and_bounded and high_score >= low_score, high_score - low_score


def _subtheme_history(rejected_tokens: list[str]) -> FeatureHistory:
    """Shared builder for the redesigned `subthemeRejectionLocalization` probe below --
    positive baseline + a CONTENT_NOT_INTERESTED burst whose depth/recency mirror
    scripts/generate_synthetic_data.py's `_not_interested_streak_rows` (depth 1-2, burst
    anchored 40-90 days before "now", minute-scale spacing, long-term baseline further back
    still) rather than the pre-redesign probe's depth-4/10-13-day-old shape, which an
    OOD/nearest-neighbor check (see app.benchmark.diagnostics) showed sits far outside the
    real training distribution -- see this gate's own docstring for the measured comparison."""
    history = FeatureHistory()
    for i in range(7):
        history.update(_interaction("pos1", "CREATOR_A", "CATEGORY_A", watch_percentage=90,
                                     when=_NOW - timedelta(days=110 + i * 4), liked=True),
                        content=_TokenContent(["TOKEN_BASE"]))
    for i, token in enumerate(rejected_tokens):
        history.update(_interaction(f"rej{i}", "CREATOR_A", "CATEGORY_A", watch_percentage=5,
                                     when=_NOW - timedelta(days=60) + timedelta(minutes=i * 3),
                                     event_type="CONTENT_NOT_INTERESTED"),
                        content=_TokenContent([token]))
    return history


def _gate_subtheme_rejection_localization(model: Any) -> tuple[bool, float]:
    """Redesigned (Task 5 audit finding: the original depth-4/10-13-day-old probe measured
    nearestNeighborDistance~9.4 against real training data, ~2-27x farther than every other
    gate's probe side -- an OOD artifact, not a meaningful production behavior check). This
    version uses depth-2 bursts anchored ~60 days back (within
    scripts/generate_synthetic_data.py's NOT_INTERESTED_STREAK_ANCHOR_MIN/MAX_DAYS_AGO=40-90
    and NOT_INTERESTED_STREAK_DEPTHS=(1, 2) ranges) with the positive baseline pushed past
    RECENT_WINDOW (30 days), isolating the STABLE long-term category_affinity signal the
    repeat-penalty mechanism (app.ml.dataset_builder.CONTENT_NOT_INTERESTED_CATEGORY_REPEAT_
    PENALTY) actually targets, rather than an artifact of the recent-window feature.

    Two checks, both required:
      (a) spillover containment -- an unseen, never-rejected candidate in the SAME category
          should score no materially lower (within SOFT_GATE_TOLERANCE) under a same-subtheme-
          repeat history than under a diverse-subtheme-rejection history (the broad category
          should hold up better when only one subtheme was ever rejected).
      (b) candidate-level ranking (Task 5 spec: a real candidate comparison, not just an
          abstract feature probe) -- within the SAME (harder, diverse-rejection) history, a
          candidate matching the still-liked base subtheme must strictly outrank a candidate
          matching the actively, repeatedly-rejected subtheme. No tolerance: this is a direct
          analogue of the benchmark's "explicit rejected subtheme should not beat a strong
          relevant candidate" critical constraint, so it must hold exactly, not just on average.
    """
    same_subtheme = _subtheme_history(["TOKEN_REJECTED", "TOKEN_REJECTED"])
    diverse_subthemes = _subtheme_history(["TOKEN_REJECTED", "TOKEN_OTHER_B"])

    same_broad = _features_for(same_subtheme, category="CATEGORY_A", hashtags=["TOKEN_UNRELATED"])
    diverse_broad = _features_for(diverse_subthemes, category="CATEGORY_A", hashtags=["TOKEN_UNRELATED"])
    same_broad_score, diverse_broad_score = _score_pair(model, same_broad, diverse_broad)
    spillover_margin = same_broad_score - diverse_broad_score
    spillover_ok = spillover_margin >= -SOFT_GATE_TOLERANCE

    candidate_liked = _features_for(diverse_subthemes, category="CATEGORY_A", hashtags=["TOKEN_BASE"],
                                     topics=["TOKEN_BASE"], entities=["TOKEN_BASE"], subgenres=["TOKEN_BASE"],
                                     title="TOKEN_BASE")
    candidate_rejected = _features_for(diverse_subthemes, category="CATEGORY_A", hashtags=["TOKEN_REJECTED"],
                                        topics=["TOKEN_REJECTED"], entities=["TOKEN_REJECTED"],
                                        subgenres=["TOKEN_REJECTED"], title="TOKEN_REJECTED")
    liked_score, rejected_score = _score_pair(model, candidate_liked, candidate_rejected)
    ranking_margin = liked_score - rejected_score
    ranking_ok = ranking_margin > 0

    passed = spillover_ok and ranking_ok
    margin = min(spillover_margin, ranking_margin)
    return passed, margin


_GATES: dict[str, Any] = {
    "longTerm": _gate_long_term,
    "recent": _gate_recent,
    "session": _gate_session,
    "negative": _gate_negative,
    "notInterested": _gate_not_interested,
    "semantic": _gate_semantic,
    "creator": _gate_creator,
    "coldStart": _gate_cold_start,
    "subthemeRejectionLocalization": _gate_subtheme_rejection_localization,
}

GATE_NAMES: tuple[str, ...] = tuple(_GATES)

# `alreadySeen` (Task 5 audit finding, app.benchmark.layers.CLASSIFICATIONS): moved OUT of raw
# ModelBehavior eligibility -- app.ml.reranker.SEEN_PENALTY already guarantees seen-content
# deprioritization downstream, identically regardless of algorithm, so requiring the raw
# classifier to also learn this was redundant. The probe itself is kept (still useful as a
# pure raw-model observability signal -- see `evaluate_eligibility`'s `diagnosticGates`), but
# it no longer participates in `eligible`/HARD/SOFT gate accounting; its production home is now
# a RerankerPolicy check (app.ml.eligibility_policy) proving unseen > seen AFTER reranking.
_DIAGNOSTIC_GATES: dict[str, Any] = {
    "alreadySeen": _gate_already_seen,
}


def evaluate_eligibility(model: Any) -> dict[str, Any]:
    """Runs every raw ModelBehavior gate against `model.predict_proba` and returns a report:
    {"gates": {name: {"pass": bool, "margin": float}}, "eligible": bool,
    "diagnosticGates": {name: {"pass": bool, "margin": float}}}.

    `eligible` requires every gate in `gates` to pass -- kept as a pure diagnostic aggregate
    (e.g. still used to gate a single explicitly `TRAINING_ALGORITHM_LOCK`-ed candidate, where
    "selection" is trivial). Production model SELECTION among multiple candidates no longer
    uses this flag directly; see app.ml.gate_severity/app.ml.eligibility_policy for the
    HARD/SOFT-severity-aware policy this task introduces on top of it.

    `diagnosticGates` never affects `eligible` -- see `_DIAGNOSTIC_GATES` above."""
    gates: dict[str, dict[str, Any]] = {}
    for name, gate_fn in _GATES.items():
        passed, margin = gate_fn(model)
        gates[name] = {"pass": bool(passed), "margin": margin}
    diagnostic_gates: dict[str, dict[str, Any]] = {}
    for name, gate_fn in _DIAGNOSTIC_GATES.items():
        passed, margin = gate_fn(model)
        diagnostic_gates[name] = {"pass": bool(passed), "margin": margin}
    eligible = all(g["pass"] for g in gates.values())
    return {"gates": gates, "eligible": eligible, "diagnosticGates": diagnostic_gates}
