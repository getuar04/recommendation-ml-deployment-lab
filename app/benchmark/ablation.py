"""Controlled feature-group ablation -- diagnostic-only tooling to identify which group of
features drives a model's response between two known feature states (e.g. the NOT_INTERESTED
same-subtheme-repeat vs. diverse-subtheme-rejection probes).

Never used in production inference or training: this only ever takes two already-computed
feature dicts and re-scores hybrid combinations of them through an already-fitted model.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml.dataset_builder import FEATURES

# Every NUMERIC feature (app.ml.dataset_builder) assigned to exactly one interpretable group.
# `category` (the one CATEGORICAL feature) is deliberately never ablated here -- both probes
# in every paired comparison this module is used for already share the same category, so
# swapping it would test a different, unrelated question.
FEATURE_GROUPS: dict[str, list[str]] = {
    "longTermCategory": [
        "category_affinity", "average_category_watch_percentage",
        "category_completion_rate", "category_positive_count", "category_negative_count",
    ],
    "recentCategory": ["recent_category_affinity", "recent_category_watch_percentage", "recent_category_completion_rate"],
    "session": [
        "session_category_affinity", "session_positive_interaction_count", "session_negative_interaction_count",
        "session_category_streak_valence_matched",
        "last_interaction_category_match", "session_intent_confidence",
    ],
    "semantic": [
        "hashtag_affinity", "topic_affinity", "entity_affinity", "subgenre_affinity", "title_affinity",
        "semantic_positive_match_count", "semantic_negative_match_count", "has_semantic_history",
        "strongest_semantic_affinity", "average_semantic_affinity",
    ],
    "creator": ["has_creator_history", "creator_interaction_count", "creator_completion_rate", "creator_followed"],
    "contentPopularity": ["content_popularity_score", "content_age_hours"],
    "historyCounts": ["user_total_interaction_count"],
    "time": ["hour_of_day"],
    "seenState": ["already_seen"],
}

assert set().union(*FEATURE_GROUPS.values()) | {"category"} == set(FEATURES), (
    "FEATURE_GROUPS must partition every NUMERIC feature exactly once -- see app.ml.dataset_builder.FEATURES"
)


def ablate_group(baseline: dict[str, Any], modified: dict[str, Any], group: str) -> dict[str, Any]:
    """`baseline` with ONLY `group`'s features replaced by their `modified` values -- every
    other feature (including every other group, and `category`) stays at `baseline`."""
    result = dict(baseline)
    for feature in FEATURE_GROUPS[group]:
        result[feature] = modified.get(feature, baseline.get(feature))
    return result


def run_group_ablation(
    model: Any, baseline: dict[str, Any], modified: dict[str, Any], *, groups: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """For each feature group (default: all of them), scores baseline-with-only-that-group-
    changed and reports the resulting score delta vs. the pure baseline -- isolates which
    group of features is responsible for how much of the total baseline-vs-modified score gap.
    """
    group_names = groups or list(FEATURE_GROUPS)
    rows = [baseline] + [ablate_group(baseline, modified, group) for group in group_names] + [modified]
    frame = pd.DataFrame(rows)[FEATURES]
    scores = model.predict_proba(frame)[:, 1]
    baseline_score, modified_score = float(scores[0]), float(scores[-1])
    total_delta = modified_score - baseline_score

    result: dict[str, dict[str, float]] = {}
    for group, score in zip(group_names, scores[1:-1]):
        delta = float(score) - baseline_score
        result[group] = {
            "scoreDelta": round(delta, 6),
            "fractionOfTotalDelta": round(delta / total_delta, 4) if total_delta else 0.0,
        }
    result["_baselineScore"] = {"scoreDelta": 0.0, "fractionOfTotalDelta": 0.0}
    result["_totalDelta"] = {"scoreDelta": round(total_delta, 6), "fractionOfTotalDelta": 1.0}
    return result
