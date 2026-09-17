"""Structured inventory of every production behavioral gate (app.ml.eligibility), for
side-by-side comparison against app.benchmark scenarios/constraints.

Deliberately reuses app.ml.eligibility's own probe-construction primitives (`_interaction`,
`_features_for`, `FeatureHistory`, `_TokenContent`, `_NOW`) rather than re-implementing each
gate's history from scratch -- any drift between this inventory and the real gate would
otherwise be a second, independent (and potentially wrong) description of production
behavior. This module only ever READS what app.ml.eligibility already builds; it never
changes gate logic, thresholds, or tolerances.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from app.ml import eligibility as elig
from app.ml.dataset_builder import FeatureHistory


@dataclass(frozen=True)
class GateProbe:
    """One gate's actual probe pair -- the real feature dicts app.ml.eligibility scores,
    reconstructed via the exact same helper calls the gate itself uses."""
    name: str
    critical: bool  # every gate is currently mandatory (see eligibility.evaluate_eligibility);
                     # "critical" here distinguishes strict-comparison gates from the two
                     # explicitly tolerance-based ones (see `tolerant`).
    tolerant: bool
    tolerance: float
    history_description: str
    left_label: str
    right_label: str
    left_features: dict[str, Any]
    right_features: dict[str, Any]
    expected_relationship: str


def _long_term_probe() -> GateProbe:
    history = FeatureHistory()
    for i in range(10):
        history.update(elig._interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=92,
                                          when=elig._NOW - timedelta(days=50 + i), liked=True))
    return GateProbe(
        "longTerm", critical=True, tolerant=False, tolerance=0.0,
        history_description="10x CATEGORY_A, 40-59 days old, watch=92%, liked=True",
        left_label="CATEGORY_A (strong history)", right_label="CATEGORY_B (never seen)",
        left_features=elig._features_for(history, category="CATEGORY_A"),
        right_features=elig._features_for(history, category="CATEGORY_B"),
        expected_relationship="left > right",
    )


def _recent_probe() -> GateProbe:
    history = FeatureHistory()
    for i in range(4):
        history.update(elig._interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=55,
                                          when=elig._NOW - timedelta(days=60 + i)))
    for i in range(6):
        history.update(elig._interaction("c2", "CREATOR_B", "CATEGORY_B", watch_percentage=96,
                                          when=elig._NOW - timedelta(days=2, hours=i), liked=True, shared=True))
    return GateProbe(
        "recent", critical=True, tolerant=False, tolerance=0.0,
        history_description="4x CATEGORY_A ~60d old (weak, wp=55) + 6x CATEGORY_B last 2 days (strong, wp=96, liked+shared)",
        left_label="CATEGORY_B (recent strong)", right_label="CATEGORY_A (stale)",
        left_features=elig._features_for(history, category="CATEGORY_B"),
        right_features=elig._features_for(history, category="CATEGORY_A"),
        expected_relationship="left > right",
    )


def _negative_probe() -> GateProbe:
    suppressed_history = FeatureHistory()
    for i in range(6):
        suppressed_history.update(elig._interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                                      when=elig._NOW - timedelta(days=50 + i), liked=True))
    for i in range(4):
        suppressed_history.update(elig._interaction(f"skip{i}", "CREATOR_A", "CATEGORY_A", watch_percentage=4,
                                                      when=elig._NOW - timedelta(minutes=12 - i * 3), event_type="VIDEO_SKIPPED"))
    unsuppressed_history = FeatureHistory()
    for i in range(6):
        unsuppressed_history.update(elig._interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                                        when=elig._NOW - timedelta(days=50 + i), liked=True))
    return GateProbe(
        "negative", critical=True, tolerant=False, tolerance=0.0,
        history_description="6x CATEGORY_A strong (wp=95, liked) + 4x VIDEO_SKIPPED (implicit, wp=4) in the last 12 minutes",
        left_label="unsuppressed baseline", right_label="post-fast-skip-burst (same category)",
        left_features=elig._features_for(unsuppressed_history, category="CATEGORY_A"),
        right_features=elig._features_for(suppressed_history, category="CATEGORY_A"),
        expected_relationship="left > right (and right > cold stranger, checked separately)",
    )


def _not_interested_probe() -> GateProbe:
    suppressed_history = FeatureHistory()
    for i in range(6):
        suppressed_history.update(elig._interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                                      when=elig._NOW - timedelta(days=50 + i), liked=True))
    for i in range(4):
        suppressed_history.update(elig._interaction(f"notInterested{i}", "CREATOR_A", "CATEGORY_A", watch_percentage=4,
                                                      when=elig._NOW - timedelta(minutes=12 - i * 3),
                                                      event_type="CONTENT_NOT_INTERESTED"))
    unsuppressed_history = FeatureHistory()
    for i in range(6):
        unsuppressed_history.update(elig._interaction("c1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                                        when=elig._NOW - timedelta(days=50 + i), liked=True))
    return GateProbe(
        "notInterested", critical=True, tolerant=False, tolerance=0.0,
        history_description="6x CATEGORY_A strong (wp=95, liked) + 4x CONTENT_NOT_INTERESTED (explicit, wp=4) in the last 12 minutes",
        left_label="unsuppressed baseline", right_label="post-NOT_INTERESTED-burst (same category)",
        left_features=elig._features_for(unsuppressed_history, category="CATEGORY_A"),
        right_features=elig._features_for(suppressed_history, category="CATEGORY_A"),
        expected_relationship="left > right (no cold-baseline floor required, unlike `negative`)",
    )


def _semantic_probe() -> GateProbe:
    history = FeatureHistory()
    for i in range(6):
        history.update(elig._interaction("pos1", "CREATOR_A", "CATEGORY_A", watch_percentage=95,
                                          when=elig._NOW - timedelta(days=15 + i), liked=True),
                        content=elig._TokenContent(["TOKEN_A"]))
    for i in range(4):
        history.update(elig._interaction("neg1", "CREATOR_A", "CATEGORY_A", watch_percentage=4,
                                          when=elig._NOW - timedelta(days=12 + i), event_type="CONTENT_NOT_INTERESTED"),
                        content=elig._TokenContent(["TOKEN_B"]))
    return GateProbe(
        "semantic", critical=True, tolerant=False, tolerance=0.0,
        history_description="6x TOKEN_A positive (wp=95, liked) + 4x TOKEN_B rejected (CONTENT_NOT_INTERESTED), same category",
        left_label="candidate tagged TOKEN_A", right_label="candidate tagged TOKEN_B",
        left_features=elig._features_for(history, category="CATEGORY_A", hashtags=["TOKEN_A"], topics=["TOKEN_A"],
                                          entities=["TOKEN_A"], subgenres=["TOKEN_A"], title="TOKEN_A"),
        right_features=elig._features_for(history, category="CATEGORY_A", hashtags=["TOKEN_B"], topics=["TOKEN_B"],
                                           entities=["TOKEN_B"], subgenres=["TOKEN_B"], title="TOKEN_B"),
        expected_relationship="left > right",
    )


def _creator_probe() -> GateProbe:
    history = FeatureHistory()
    for i in range(9):
        history.update(elig._interaction(f"pref{i}", "CREATOR_PREFERRED", "CATEGORY_A", watch_percentage=95,
                                          when=elig._NOW - timedelta(days=15 + i), liked=True, creator_followed=(i == 0)))
    for i in range(6):
        history.update(elig._interaction(f"weak{i}", "CREATOR_WEAK", "CATEGORY_A", watch_percentage=8,
                                          when=elig._NOW - timedelta(days=12 + i), event_type="VIDEO_SKIPPED"))
    return GateProbe(
        "creator", critical=True, tolerant=False, tolerance=0.0,
        history_description="9x CREATOR_PREFERRED strong (wp=95, liked, followed once) + 6x CREATOR_WEAK skipped (wp=8), same category",
        left_label="candidate from CREATOR_PREFERRED", right_label="candidate from CREATOR_WEAK",
        left_features=elig._features_for(history, category="CATEGORY_A", creator_id="CREATOR_PREFERRED"),
        right_features=elig._features_for(history, category="CATEGORY_A", creator_id="CREATOR_WEAK"),
        expected_relationship="left > right",
    )


def _already_seen_probe() -> GateProbe:
    history = FeatureHistory()
    for i in range(8):
        history.update(elig._interaction(f"c{i}", "CREATOR_A", "CATEGORY_A", watch_percentage=92,
                                          when=elig._NOW - timedelta(days=20 + i), liked=True))
    content_created_at = elig._NOW - timedelta(hours=24)
    return GateProbe(
        "alreadySeen", critical=False, tolerant=True, tolerance=elig.SOFT_GATE_TOLERANCE,
        history_description="8x CATEGORY_A strong (wp=92, liked), 20-27 days old",
        left_label="unseen candidate", right_label="already-seen candidate (identical otherwise)",
        left_features=history.features(user_id="probe-user", category="CATEGORY_A", creator_id="CREATOR_A",
                                        content_id="probe-candidate", timestamp=elig._NOW,
                                        content_popularity_score=0.5, content_created_at=content_created_at,
                                        already_seen=False),
        right_features=history.features(user_id="probe-user", category="CATEGORY_A", creator_id="CREATOR_A",
                                         content_id="probe-candidate", timestamp=elig._NOW,
                                         content_popularity_score=0.5, content_created_at=content_created_at,
                                         already_seen=True),
        expected_relationship="left >= right - tolerance",
    )


def _cold_start_probe() -> GateProbe:
    history = FeatureHistory()
    low = elig._features_for(history, category="CATEGORY_A")
    low = dict(low, content_popularity_score=0.2)
    high = dict(low, content_popularity_score=0.9)
    return GateProbe(
        "coldStart", critical=True, tolerant=False, tolerance=0.0,
        history_description="Zero prior history (genuine cold start)",
        left_label="high popularity (0.9)", right_label="low popularity (0.2)",
        left_features=high, right_features=low,
        expected_relationship="left >= right (monotonic in popularity; finite/bounded score required)",
    )


def _subtheme_rejection_localization_probe() -> GateProbe:
    """Redesigned (Task 5) -- reconstructs check (a), the spillover comparison, via
    `elig._subtheme_history` (depth-2 bursts anchored ~60 days back, positive baseline pushed
    past RECENT_WINDOW; see app.ml.eligibility._gate_subtheme_rejection_localization's own
    docstring for the full rationale and the measured OOD-distance improvement over the
    pre-redesign depth-4/10-13-day-old probe this replaces). The gate's own check (b) -- a
    strict, no-tolerance candidate-level ranking within the diverse-subtheme history -- has no
    single left/right pair representable here; see that gate function directly for it."""
    same_subtheme = elig._subtheme_history(["TOKEN_REJECTED", "TOKEN_REJECTED"])
    diverse_subthemes = elig._subtheme_history(["TOKEN_REJECTED", "TOKEN_OTHER_B"])
    return GateProbe(
        "subthemeRejectionLocalization", critical=False, tolerant=True, tolerance=elig.SOFT_GATE_TOLERANCE,
        history_description="7x TOKEN_BASE positive (110-134 days old) + depth-2 CONTENT_NOT_INTERESTED "
                             "burst ~60 days old: SAME subtheme (TOKEN_REJECTED x2) vs. DIVERSE subthemes "
                             "(TOKEN_REJECTED + TOKEN_OTHER_B) -- spillover check (a) only; see docstring",
        left_label="same-subtheme-repeat history", right_label="diverse-subtheme-rejection history",
        left_features=elig._features_for(same_subtheme, category="CATEGORY_A", hashtags=["TOKEN_UNRELATED"]),
        right_features=elig._features_for(diverse_subthemes, category="CATEGORY_A", hashtags=["TOKEN_UNRELATED"]),
        expected_relationship="left >= right - tolerance",
    )


# `session` is omitted here: it is a pure feature-value sanity check (no model score
# comparison at all -- see app.ml.eligibility._gate_session), so it has no left/right score
# pair to inventory; see gate_inventory_report()'s own note about it.
_PROBE_BUILDERS = {
    "longTerm": _long_term_probe,
    "recent": _recent_probe,
    "negative": _negative_probe,
    "notInterested": _not_interested_probe,
    "semantic": _semantic_probe,
    "creator": _creator_probe,
    "alreadySeen": _already_seen_probe,
    "coldStart": _cold_start_probe,
    "subthemeRejectionLocalization": _subtheme_rejection_localization_probe,
}

ALL_GATE_NAMES = (*_PROBE_BUILDERS, "session")


def all_probes() -> dict[str, GateProbe]:
    """Every gate's reconstructed probe pair except `session` (a feature-value-only check;
    see module docstring)."""
    return {name: builder() for name, builder in _PROBE_BUILDERS.items()}
