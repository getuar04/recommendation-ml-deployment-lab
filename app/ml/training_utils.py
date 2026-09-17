"""Small helpers shared by the VIDEO and LIVE trainers."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance as sklearn_permutation_importance


def has_both_classes(target: pd.Series) -> bool:
    return target.nunique() >= 2


def class_distribution(target: pd.Series) -> dict[str, int]:
    counts = target.value_counts().to_dict()
    return {"positive": int(counts.get(1, 0)), "negative": int(counts.get(0, 0))}


def permutation_importance_report(
    model: Any, x: pd.DataFrame, y: pd.Series, random_seed: int, *, feature_names: list[str],
) -> list[dict[str, Any]]:
    """Global permutation importance on held-out data.

    Model-agnostic and causally naive by construction (it only measures the drop in
    held-out ROC-AUC when a feature is shuffled). Report separately from, and never
    conflate with, heuristic recommendation reason codes or RandomForest's own
    (also non-causal) impurity-based `feature_importances_`.
    """
    result = sklearn_permutation_importance(model, x, y, n_repeats=5, random_state=random_seed, scoring="roc_auc", n_jobs=1)
    ranked = sorted(zip(feature_names, result.importances_mean, result.importances_std), key=lambda item: -item[1])
    return [
        {"feature": name, "meanImportance": round(float(mean), 6), "stdImportance": round(float(std), 6)}
        for name, mean, std in ranked
    ]
