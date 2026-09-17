"""Machine-readable dataset diagnostics for a point-in-time labeled training frame (as
produced by app.ml.dataset_builder.build_dataset). Pure computation over an already-built
DataFrame -- no I/O, no FastAPI/route dependency -- so it is directly reusable from
app.services.training_service (real training) and app.experiments.runner (offline
experiments) alike, and directly unit-testable.

This is diagnostic/observability only: it never changes what training does with the data
(see app.ml.sample_weight_policy for the module that actually acts on this same information).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml.dataset_builder import RECENT_WINDOW, SESSION_WINDOW
from app.ml.feature_builder import EXPLICIT_NEGATIVE_EVENT_TYPES

# Best-effort, prefix-based synthetic-scenario bucketing (offline diagnostics only -- never
# read by training/selection logic). Mirrors the user_id-prefix convention
# scripts/generate_synthetic_data.py's dedicated cohorts already use; any user_id not matching
# a known prefix (including every real, non-synthetic user_id) falls into "bulkOrOther", never
# an error.
_SCENARIO_PREFIXES: tuple[tuple[str, str], ...] = (
    ("coldstart-user-", "coldStart"),
    ("session-switch-user-", "sessionSwitch"),
    ("recent-shift-user-", "recentShift"),
    ("semantic-pref-user-", "semanticPreference"),
    ("creator-pref-user-", "creatorPreference"),
    ("fastskip-streak-user-", "fastSkipStreak"),
    ("notinterested-streak-user-", "notInterestedStreak"),
    ("creator-mismatch-user-", "creatorMismatch"),
)


def _scenario_bucket(user_id: str) -> str:
    for prefix, name in _SCENARIO_PREFIXES:
        if user_id.startswith(prefix):
            return name
    return "bulkOrOther"


def _counts_by(series: pd.Series) -> dict[str, int]:
    return {str(key): int(count) for key, count in series.value_counts().items()}


def dataset_diagnostics(df: pd.DataFrame, *, total_interaction_count: int | None = None) -> dict[str, Any]:
    """`df`: the labeled frame from app.ml.dataset_builder.build_dataset (must carry `target`
    plus TRAINING_METADATA_COLUMNS: event_type/event_watch_percentage/event_liked/event_shared/event_favorited/
    creator_followed). `total_interaction_count`, when given, is the count of RAW interaction
    rows build_dataset() was called with (before neutral-row exclusion) -- build_dataset()
    only ever emits labeled rows, so this is the only way to report how many rows were
    excluded as neutral; omit it (default None) when that count isn't available and
    `neutralRowsExcluded` will be reported as null rather than guessed.
    """
    labeled = len(df)
    if labeled == 0:
        return {
            "totalInteractions": total_interaction_count, "labeledRows": 0, "positiveRows": 0,
            "negativeRows": 0, "neutralRowsExcluded": (
                None if total_interaction_count is None else total_interaction_count
            ),
            "positiveRatio": None, "negativeRatio": None,
        }

    positive = int((df["target"] == 1).sum())
    negative = int((df["target"] == 0).sum())
    is_explicit_rejection = df["event_type"].isin(EXPLICIT_NEGATIVE_EVENT_TYPES)
    not_interested_count = int(is_explicit_rejection.sum())
    negative_df = df[df["target"] == 0]
    implicit_negative_count = int((~is_explicit_rejection[df["target"] == 0]).sum())

    positive_df = df[df["target"] == 1]
    high_watch_positive = int((positive_df["event_watch_percentage"].fillna(0) >= 90).sum())

    max_timestamp = df["timestamp"].max()
    age = max_timestamp - df["timestamp"]
    session_scale = int((age <= SESSION_WINDOW).sum())
    recent_scale = int(((age > SESSION_WINDOW) & (age <= RECENT_WINDOW)).sum())
    long_term_scale = int((age > RECENT_WINDOW).sum())

    scenario_counts = _counts_by(df["user_id"].map(_scenario_bucket))

    return {
        "totalInteractions": total_interaction_count,
        "labeledRows": labeled,
        "positiveRows": positive,
        "negativeRows": negative,
        "neutralRowsExcluded": (
            None if total_interaction_count is None else max(0, total_interaction_count - labeled)
        ),
        "positiveRatio": round(positive / labeled, 6),
        "negativeRatio": round(negative / labeled, 6),
        "negativeBreakdown": {
            "contentNotInterestedCount": not_interested_count,
            "contentNotInterestedPercentOfNegatives": round(not_interested_count / negative, 6) if negative else None,
            "implicitShortWatchNegativeCount": implicit_negative_count,
            "otherNegativeEventTypeCounts": _counts_by(negative_df.loc[~is_explicit_rejection[df["target"] == 0], "event_type"]),
        },
        "positiveBreakdown": {
            "highWatchPositiveCount": high_watch_positive,
            "likedCount": int(positive_df["event_liked"].sum()),
            "sharedCount": int(positive_df["event_shared"].sum()),
            "favoritedCount": int(positive_df["event_favorited"].sum()),
            "creatorFollowedCount": int(positive_df["event_creator_followed"].sum()),
            "otherPositiveEventTypeCounts": _counts_by(positive_df["event_type"]),
        },
        "uniqueUsers": int(df["user_id"].nunique()),
        "uniqueContents": int(df["content_id"].nunique()),
        "uniqueCreators": int(df["creator_id"].nunique()),
        "uniqueCategories": int(df["category"].nunique()),
        "rowsPerCategory": _counts_by(df["category"]),
        "rowsPerSyntheticScenario": scenario_counts,
        "temporalDistribution": {
            "sessionScaleRows": session_scale,
            "recentScaleRows": recent_scale,
            "longTermScaleRows": long_term_scale,
            "sessionWindow": str(SESSION_WINDOW),
            "recentWindow": str(RECENT_WINDOW),
            "note": "Buckets are relative to this dataset's own latest timestamp, not wall-clock now.",
        },
    }
