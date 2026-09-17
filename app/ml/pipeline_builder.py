"""Shared sklearn preprocessing pipeline construction for the VIDEO and LIVE trainers."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# The fixed step name build_classifier_pipeline always uses for the estimator itself --
# centralized here so callers never hardcode the "model__" sklearn Pipeline fit-param prefix
# in more than one place.
_MODEL_STEP_NAME = "model"


def sample_weight_fit_params(sample_weight: np.ndarray | None) -> dict[str, Any]:
    """`Pipeline.fit(X, y, **fit_params)` keyword arguments to pass a per-row sample weight
    through to the underlying estimator step -- every one of this project's candidate
    algorithms (LogisticRegression/RandomForest/XGBoost/LightGBM/CatBoost) accepts
    `sample_weight` natively. Returns `{}` (no-op) when `sample_weight` is None, so a caller
    that opts out of weighting (app.core.config.TRAINING_USE_SAMPLE_WEIGHTS=false) needs no
    separate code path."""
    if sample_weight is None:
        return {}
    return {f"{_MODEL_STEP_NAME}__sample_weight": sample_weight}


def ranker_fit_params(
    *, group_param_name: str, group_values: Any, sample_weight: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """`Pipeline.fit(X, y, **fit_params)` keyword arguments for a ranking-objective estimator
    (Task 7 ranking-native challenger experiment; app.ml.ranker_registry). `group_param_name`
    is the underlying estimator's own fit-parameter name for its grouping concept -- "group"
    (a list of GROUP SIZES, in row order: XGBRanker/LGBMRanker) or "group_id" (a per-row GROUP
    IDENTIFIER array: CatBoostRanker) -- since the three libraries do not share one convention.
    `group_values` must already be in the exact row order the estimator will see (this
    function does not reorder or validate anything; see app.ml.ranking_groups.group_sizes)."""
    params = {f"{_MODEL_STEP_NAME}__{group_param_name}": group_values}
    if sample_weight is not None:
        params[f"{_MODEL_STEP_NAME}__sample_weight"] = sample_weight
    return params


def build_classifier_pipeline(model: Any, *, categorical: list[str], numeric: list[str], scale_numeric: bool) -> Pipeline:
    """Random Forest does not need feature scaling; Logistic Regression does.

    Phase 1.5 issue 2: RobustScaler (median/IQR), not StandardScaler (mean/std). Several
    features (all session_* features, has_semantic_history-adjacent counts, etc.) are near-
    constant across most rows -- only ~1-3% of rows carry any session-window activity at all --
    so their StandardScaler-fitted std is tiny, turning any genuinely realistic nonzero value
    (e.g. app.ml.eligibility's `negative` gate probe, or a real user's session-negative burst)
    into an extreme, logit-saturating outlier (measured 60-130 standard deviations). For a
    feature where more than 75% of training rows are exactly 0, RobustScaler's IQR is 0 too,
    which sklearn's zero-scale fallback resolves by leaving that feature's raw units unscaled
    (center=0, scale=1) rather than amplifying it -- generic, applies uniformly to every
    numeric feature (never a named/hardcoded feature list), and degrades gracefully to
    ordinary median/IQR scaling for every well-behaved (non-sparse) feature, unlike excluding
    specific features from scaling by name would."""
    numeric_step = RobustScaler() if scale_numeric else "passthrough"
    preprocessing = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", numeric_step, numeric),
    ])
    return Pipeline([("features", preprocessing), (_MODEL_STEP_NAME, model)])
