"""Diagnostic-only utilities for reconciling production behavioral gates against the
independent benchmark -- feature-delta summaries, simple interpretable out-of-distribution
checks, score-resolution/tie statistics, and a calibration before/after ranking comparison.

Nothing here is used at training or serving time; every function takes already-computed
feature dicts/scores/models and returns plain, JSON-friendly diagnostics.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.ml.dataset_builder import FEATURES, NUMERIC


def feature_delta_summary(
    left: dict[str, Any], right: dict[str, Any], *, min_abs_difference: float = 0.01,
) -> list[dict[str, Any]]:
    """Only NUMERIC features (see app.ml.dataset_builder) that differ by at least
    `min_abs_difference` -- a plain, material-differences-only report, not a 39-row dump."""
    rows: list[dict[str, Any]] = []
    for feature in NUMERIC:
        left_value = float(left.get(feature, 0.0) or 0.0)
        right_value = float(right.get(feature, 0.0) or 0.0)
        difference = left_value - right_value
        if abs(difference) >= min_abs_difference:
            rows.append({
                "feature": feature, "leftValue": round(left_value, 6), "rightValue": round(right_value, 6),
                "absoluteDifference": round(abs(difference), 6),
            })
    rows.sort(key=lambda row: row["absoluteDifference"], reverse=True)
    return rows


def _numeric_vector(features: dict[str, Any]) -> np.ndarray:
    return np.array([float(features.get(name, 0.0) or 0.0) for name in NUMERIC], dtype=float)


def feature_percentiles(reference_df: pd.DataFrame, probe_features: dict[str, Any]) -> dict[str, float]:
    """For each NUMERIC feature, what percentile of `reference_df` the probe's value falls
    at -- a simple, interpretable "how typical is this value" check. 0 or 100 marks a value
    at or beyond the extreme of the observed training range."""
    percentiles = {}
    for feature in NUMERIC:
        column = reference_df[feature].to_numpy(dtype=float)
        if len(column) == 0:
            continue
        value = float(probe_features.get(feature, 0.0) or 0.0)
        percentiles[feature] = round(float((column <= value).mean() * 100), 2)
    return percentiles


def nearest_neighbor_distance(
    reference_df: pd.DataFrame, probe_features: dict[str, Any], *, sample_size: int = 2000, seed: int = 42,
) -> dict[str, float]:
    """Normalized (z-scored per feature) Euclidean distance from the probe to its single
    nearest row in a random sample of `reference_df` -- a cheap, interpretable OOD signal.
    Never a new ML model: just a min-distance computation over NUMERIC features."""
    sample = reference_df if len(reference_df) <= sample_size else reference_df.sample(sample_size, random_state=seed)
    matrix = sample[NUMERIC].to_numpy(dtype=float)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0  # a constant column contributes 0 to every distance either way.
    normalized = (matrix - mean) / std
    probe_vector = (_numeric_vector(probe_features) - mean) / std
    distances = np.linalg.norm(normalized - probe_vector, axis=1)
    return {
        "nearestNeighborDistance": round(float(distances.min()), 4),
        "medianPairwiseScaleDistance": round(float(np.median(distances)), 4),
        "sampledRows": len(sample),
    }


def similar_row_fraction(
    reference_df: pd.DataFrame, probe_features: dict[str, Any], features_to_check: list[str], *, tolerance: float = 0.1,
) -> float:
    """Fraction of `reference_df` rows within `tolerance` of the probe on EVERY named feature
    -- "how much of training data looks like this probe", the plainest possible OOD check."""
    mask = np.ones(len(reference_df), dtype=bool)
    for feature in features_to_check:
        value = float(probe_features.get(feature, 0.0) or 0.0)
        mask &= (reference_df[feature].to_numpy(dtype=float) - value).__abs__() <= tolerance
    return round(float(mask.mean()) if len(reference_df) else 0.0, 6)


def score_resolution_stats(scores: list[float]) -> dict[str, Any]:
    """Distribution and tie-resolution diagnostics for a set of scores (raw or final) on one
    benchmark difficulty. Tree ensembles can produce coarse probability buckets -- many
    near-identical scores hurt ranking quality (NDCG) even when every pairwise CONSTRAINT
    still technically passes, since ties are broken arbitrarily."""
    array = np.array(scores, dtype=float)
    if array.size == 0:
        return {"count": 0}
    unique_count = int(np.unique(np.round(array, 6)).size)
    near_ties_1e4 = 0
    near_ties_1e3 = 0
    sorted_scores = np.sort(array)
    diffs = np.diff(sorted_scores)
    near_ties_1e4 = int((diffs < 1e-4).sum())
    near_ties_1e3 = int((diffs < 1e-3).sum())
    return {
        "count": int(array.size),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std()), 6),
        "uniqueScores": unique_count,
        "adjacentPairsWithin1e4": near_ties_1e4,
        "adjacentPairsWithin1e3": near_ties_1e3,
    }


def calibration_ranking_comparison(
    uncalibrated_model: Any, calibrated_model: Any, feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compares the SAME candidate pool's ranking before vs. after calibration -- calibration
    should improve probability quality without materially reordering candidates. Reports the
    number of pairwise rank inversions between the two orderings (0 = identical ranking)."""
    frame = pd.DataFrame(feature_rows)[FEATURES]
    uncalibrated_scores = uncalibrated_model.predict_proba(frame)[:, 1]
    calibrated_scores = calibrated_model.predict_proba(frame)[:, 1]

    uncalibrated_rank = pd.Series(uncalibrated_scores).rank(ascending=False, method="first").to_numpy()
    calibrated_rank = pd.Series(calibrated_scores).rank(ascending=False, method="first").to_numpy()

    n = len(feature_rows)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            before_order = uncalibrated_rank[i] < uncalibrated_rank[j]
            after_order = calibrated_rank[i] < calibrated_rank[j]
            if before_order != after_order:
                inversions += 1
    total_pairs = n * (n - 1) // 2

    return {
        "uncalibratedScores": [round(float(s), 6) for s in uncalibrated_scores],
        "calibratedScores": [round(float(s), 6) for s in calibrated_scores],
        "pairwiseInversions": inversions,
        "totalPairs": total_pairs,
        "inversionRate": round(inversions / total_pairs, 6) if total_pairs else 0.0,
        "rankingIdentical": inversions == 0,
    }
