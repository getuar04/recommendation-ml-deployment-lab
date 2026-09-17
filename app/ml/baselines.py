"""Offline baselines so trained-model metrics show incremental value, not just absolute scores.

Every baseline produces a ranking/probability-shaped score array that can be fed straight
into `app.ml.evaluator.evaluate()`, so baselines and the trained model are scored with
exactly the same metric implementation. Ranking/probability metrics (ROC-AUC, PR-AUC,
Brier score, ...) are threshold-independent and therefore always directly comparable.
Classification metrics (accuracy/precision/recall/F1/confusion matrix) require a decision
threshold; each baseline gets its *own* threshold, selected only from a pre-test
threshold-tuning split (never from test labels) -- a baseline's score distribution (e.g.
the majority-class-prior baseline emits one constant value for every row) is not
comparable to the trained model's, so reusing the model's tuned threshold would be unfair.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from app.ml.evaluator import evaluate, select_threshold

DEFAULT_SCORE_COLUMNS: dict[str, str] = {"popularityOnly": "content_popularity_score", "categoryAffinityOnly": "category_affinity"}


def majority_class_prior_scores(train_target: Sequence[int] | np.ndarray, size: int) -> np.ndarray:
    """Constant score equal to the training-split positive rate (a class-prior/majority classifier)."""
    prior = float(np.mean(train_target)) if len(train_target) else 0.5
    return np.full(size, prior, dtype=float)


def _threshold_for(tuning_target: np.ndarray, tuning_scores: np.ndarray, *, objective: str) -> tuple[float, str]:
    if len(set(tuning_target.tolist())) > 1:
        return select_threshold(tuning_target, tuning_scores, objective=objective), "threshold-tuning split"
    return 0.5, "threshold-tuning split had a single class present; defaulted to 0.5"


def _evaluate_baseline(
    tuning_df: pd.DataFrame, test_df: pd.DataFrame, tuning_scores: np.ndarray, test_scores: np.ndarray,
    *, target_col: str, group_col: str, threshold_objective: str,
) -> dict[str, Any]:
    threshold, source = _threshold_for(tuning_df[target_col].to_numpy(), tuning_scores, objective=threshold_objective)
    metrics = evaluate(test_df[target_col].to_numpy(), test_scores, test_df[group_col].to_numpy(), threshold=threshold)
    metrics["decisionThreshold"] = threshold
    metrics["thresholdSelectionSource"] = source
    return metrics


def evaluate_baselines(
    train_target: Sequence[int] | np.ndarray,
    threshold_tuning_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = "target",
    group_col: str = "candidate_group",
    score_columns: Mapping[str, str] = DEFAULT_SCORE_COLUMNS,
    threshold_objective: str = "f1",
) -> dict[str, dict[str, Any]]:
    """Evaluate majority-class-prior plus any available heuristic score columns.

    Each baseline's classification threshold is selected only from `threshold_tuning_df`
    (never `test_df`'s labels) and recorded per-baseline in the returned metrics under
    `decisionThreshold`/`thresholdSelectionSource`, mirroring how the trained model's own
    threshold is tuned. `score_columns` maps a baseline name to a column already present
    in both DataFrames whose values are usable directly as a ranking/pseudo-probability
    score (e.g. popularity or category affinity, both already in [0, 1]).
    """
    results: dict[str, dict[str, Any]] = {
        "majorityClassPrior": _evaluate_baseline(
            threshold_tuning_df, test_df,
            majority_class_prior_scores(train_target, len(threshold_tuning_df)),
            majority_class_prior_scores(train_target, len(test_df)),
            target_col=target_col, group_col=group_col, threshold_objective=threshold_objective,
        ),
    }
    for name, column in dict(score_columns).items():
        if column in threshold_tuning_df.columns and column in test_df.columns:
            results[name] = _evaluate_baseline(
                threshold_tuning_df, test_df,
                threshold_tuning_df[column].to_numpy(dtype=float), test_df[column].to_numpy(dtype=float),
                target_col=target_col, group_col=group_col, threshold_objective=threshold_objective,
            )
    return results
