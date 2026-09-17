"""Classification and slate-style recommendation evaluation metrics.

`evaluate()` reports two families of metrics that must not be confused:

- Probability-based metrics (ROC-AUC, PR-AUC, Brier score, log loss, calibration
  bins) are threshold-independent and always use raw predicted probabilities.
- Classification metrics (accuracy, precision, recall, F1, confusion matrix)
  depend on a decision threshold. Callers should pick that threshold on
  validation data with `select_threshold()` and pass it in explicitly; the
  default of 0.5 is only a fallback, never an assumption that it is optimal.

Ranking metrics (Precision@K, Recall@K, NDCG@K, HitRate@K, MRR; K in {5, 10}) are computed
per "candidate group" (e.g. one user-day slate). Groups with fewer than two candidates
cannot express a ranking and are skipped; groups without any positive label are excluded
from Recall@K/HitRate@K/MRR (undefined) but are still counted in the group-size diagnostics
so callers can judge how much the headline ranking numbers are supported by real slate
structure. Precision@K divides by min(K, group size) (undefined only for an empty group,
which cannot occur here), so it is always defined once a group clears the size-2 floor above.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5
_PROBABILITY_EPS = 1e-15


def _confusion(y: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    true_positive = int(np.sum((predicted == 1) & (y == 1)))
    false_positive = int(np.sum((predicted == 1) & (y == 0)))
    true_negative = int(np.sum((predicted == 0) & (y == 0)))
    false_negative = int(np.sum((predicted == 0) & (y == 1)))
    return {
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "trueNegative": true_negative,
        "falseNegative": false_negative,
    }


def _calibration_bins(y: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """Reliability diagram data: for each probability bin, predicted vs. observed rate."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    bin_index = np.clip(np.digitize(probability, edges[1:-1], right=True), 0, n_bins - 1)
    for i in range(n_bins):
        mask = bin_index == i
        count = int(mask.sum())
        bins.append({
            "binLower": round(float(edges[i]), 4),
            "binUpper": round(float(edges[i + 1]), 4),
            "count": count,
            "meanPredicted": round(float(probability[mask].mean()), 6) if count else None,
            "meanActual": round(float(y[mask].mean()), 6) if count else None,
        })
    return bins


def _rank_groups(y: np.ndarray, probability: np.ndarray, groups: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Sort each candidate group by descending probability; return ranked labels plus diagnostics."""
    sizes: list[int] = []
    ranked: list[np.ndarray] = []
    skipped_too_small = 0
    groups_without_positives = 0
    for group in sorted(set(groups.tolist())):
        indexes = np.where(groups == group)[0]
        size = len(indexes)
        if size < 2:  # A ranking metric is undefined for a one-item slate.
            skipped_too_small += 1
            continue
        sizes.append(size)
        order = indexes[np.argsort(-probability[indexes], kind="stable")]
        labels = y[order]
        if labels.sum() == 0:
            groups_without_positives += 1
        ranked.append(labels)
    diagnostics = {
        "totalGroups": len(set(groups.tolist())),
        "evaluatedGroups": len(ranked),
        "groupsSkippedTooSmall": skipped_too_small,
        "groupsWithoutPositives": groups_without_positives,
        "averageGroupSize": round(float(np.mean(sizes)), 4) if sizes else 0.0,
        "minGroupSize": int(np.min(sizes)) if sizes else 0,
        "maxGroupSize": int(np.max(sizes)) if sizes else 0,
        "p50GroupSize": round(float(np.percentile(sizes, 50)), 4) if sizes else 0.0,
        "p90GroupSize": round(float(np.percentile(sizes, 90)), 4) if sizes else 0.0,
    }
    return ranked, diagnostics


def _precision_at(ranked: list[np.ndarray], k: int) -> float:
    return float(np.mean([items[:k].sum() / min(k, len(items)) for items in ranked])) if ranked else 0.0


def _recall_at(ranked: list[np.ndarray], k: int) -> float:
    eligible = [items for items in ranked if items.sum()]
    return float(np.mean([items[:k].sum() / items.sum() for items in eligible])) if eligible else 0.0


def _ndcg_at(ranked: list[np.ndarray], k: int) -> float:
    values = []
    for items in ranked:
        dcg = sum(value / np.log2(index + 2) for index, value in enumerate(items[:k]))
        ideal = sum(value / np.log2(index + 2) for index, value in enumerate(sorted(items, reverse=True)[:k]))
        if ideal:
            values.append(dcg / ideal)
    return float(np.mean(values)) if values else 0.0


def _mrr(ranked: list[np.ndarray]) -> float:
    eligible = [items for items in ranked if items.sum()]
    if not eligible:
        return 0.0
    reciprocal_ranks = [1.0 / (int(np.argmax(items)) + 1) for items in eligible]
    return float(np.mean(reciprocal_ranks))


def _hit_rate_at(ranked: list[np.ndarray], k: int) -> float:
    """Fraction of groups with at least one relevant item in the top K -- undefined (skipped,
    like Recall@K/MRR) for a group with no positive label at all."""
    eligible = [items for items in ranked if items.sum()]
    return float(np.mean([float(items[:k].sum() > 0) for items in eligible])) if eligible else 0.0


def select_threshold(
    y: Iterable[int],
    probability: Iterable[float],
    *,
    objective: str = "f1",
    min_precision: float | None = None,
    candidate_thresholds: Iterable[float] | None = None,
) -> float:
    """Pick a decision threshold using validation-only labels/probabilities.

    objective="f1": maximize F1.
    objective="recall_at_min_precision": maximize recall subject to precision >= min_precision.
    Falls back to 0.5 when the data cannot support optimization (a single class).
    """
    y_array = np.asarray(list(y))
    probability_array = np.asarray(list(probability))
    if len(set(y_array.tolist())) < 2:
        return DEFAULT_THRESHOLD
    if candidate_thresholds is None:
        candidate_thresholds = np.unique(np.clip(probability_array, 0.0, 1.0))
        if candidate_thresholds.size == 0:
            return DEFAULT_THRESHOLD
    best_threshold = DEFAULT_THRESHOLD
    best_score = -1.0
    for threshold in candidate_thresholds:
        predicted = (probability_array >= threshold).astype(int)
        if objective == "f1":
            score = f1_score(y_array, predicted, zero_division=0)
        elif objective == "recall_at_min_precision":
            precision = precision_score(y_array, predicted, zero_division=0)
            if precision < (min_precision or 0.0):
                continue
            score = recall_score(y_array, predicted, zero_division=0)
        else:
            raise ValueError(f"Unknown threshold objective: {objective!r}")
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return best_threshold


def evaluate(
    y: Iterable[int],
    probability: Iterable[float],
    groups: Iterable[Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    bootstrap: bool = False,
    n_bootstrap: int = 200,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    # Renamed to *_arr rather than reassigning the Iterable-typed parameters: mypy cannot
    # narrow a parameter's declared type across a reassignment to a structurally-different
    # type (np.ndarray), so the rest of this function works from these instead.
    y_arr: np.ndarray = np.asarray(list(y))
    probability_arr: np.ndarray = np.clip(np.asarray(list(probability), dtype=float), 0.0, 1.0)
    groups_arr: np.ndarray = np.asarray(list(groups))
    predicted = (probability_arr >= threshold).astype(int)
    has_both_classes = len(set(y_arr.tolist())) > 1

    clipped_probability = np.clip(probability_arr, _PROBABILITY_EPS, 1 - _PROBABILITY_EPS)
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_arr, predicted),
        "precision": precision_score(y_arr, predicted, zero_division=0),
        "recall": recall_score(y_arr, predicted, zero_division=0),
        "f1Score": f1_score(y_arr, predicted, zero_division=0),
        "rocAuc": roc_auc_score(y_arr, probability_arr) if has_both_classes else 0.0,
        "prAuc": average_precision_score(y_arr, probability_arr) if has_both_classes else 0.0,
        "brierScore": brier_score_loss(y_arr, probability_arr),
        "logLoss": log_loss(y_arr, clipped_probability, labels=[0, 1]) if len(y_arr) else None,
        "decisionThreshold": float(threshold),
        "positiveClassRate": float(y_arr.mean()) if len(y_arr) else 0.0,
        "predictedPositiveRate": float(predicted.mean()) if len(predicted) else 0.0,
    }
    metrics["confusionMatrix"] = _confusion(y_arr, predicted)
    metrics["calibrationBins"] = _calibration_bins(y_arr, probability_arr)

    ranked, group_diagnostics = _rank_groups(y_arr, probability_arr, groups_arr)
    metrics.update(
        precisionAt5=_precision_at(ranked, 5),
        precisionAt10=_precision_at(ranked, 10),
        recallAt5=_recall_at(ranked, 5),
        recallAt10=_recall_at(ranked, 10),
        ndcgAt5=_ndcg_at(ranked, 5),
        ndcgAt10=_ndcg_at(ranked, 10),
        hitRateAt5=_hit_rate_at(ranked, 5),
        hitRateAt10=_hit_rate_at(ranked, 10),
        mrr=_mrr(ranked),
        evaluatedCandidateGroups=len(ranked),
        groupDiagnostics=group_diagnostics,
    )

    if bootstrap and len(y_arr) > 1:
        metrics["bootstrapConfidenceIntervals"] = _bootstrap_confidence_intervals(
            y_arr, probability_arr, threshold, n_bootstrap=n_bootstrap, seed=bootstrap_seed,
        )

    return _round_floats(metrics)


def _bootstrap_confidence_intervals(
    y: np.ndarray, probability: np.ndarray, threshold: float, *, n_bootstrap: int, seed: int,
) -> dict[str, list[float]]:
    """Deterministic (fixed-seed) row-resampling bootstrap for headline metrics."""
    rng = np.random.default_rng(seed)
    n = len(y)
    samples: dict[str, list[float]] = {"accuracy": [], "precision": [], "recall": [], "f1Score": [], "rocAuc": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], probability[idx]
        predicted_b = (pb >= threshold).astype(int)
        samples["accuracy"].append(accuracy_score(yb, predicted_b))
        samples["precision"].append(precision_score(yb, predicted_b, zero_division=0))
        samples["recall"].append(recall_score(yb, predicted_b, zero_division=0))
        samples["f1Score"].append(f1_score(yb, predicted_b, zero_division=0))
        samples["rocAuc"].append(roc_auc_score(yb, pb) if len(set(yb.tolist())) > 1 else 0.0)
    return {
        name: [round(float(np.percentile(values, 2.5)), 6), round(float(np.percentile(values, 97.5)), 6)]
        for name, values in samples.items()
    }


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_round_floats(inner) for inner in value]
    return value
