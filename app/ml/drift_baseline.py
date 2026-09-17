"""Drift baseline construction: a compact, JSON-serializable summary of the exact feature
DataFrame a model was trained on (the `train` split specifically -- the literal rows passed
to `model.fit()`, see `app.ml.trainer.train_models`/`app.ml.live_trainer.train_live_model`).

This module never touches raw identifiers (user/content/creator/event ids) and never stores
individual rows -- only bounded, aggregate per-feature statistics (see `build_baseline()`).
It has no FastAPI/DB dependency: pure pandas/numpy in, a plain dict out, so it is directly
unit-testable and reusable from both the VIDEO and LIVE trainers.

`app.ml.drift_detector` (not this module) turns a baseline plus a new observation batch
into an actual drift verdict -- this module only *describes* the training distribution.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.core.config import DRIFT_HISTOGRAM_BINS, DRIFT_MAX_CATEGORIES

BASELINE_SCHEMA_VERSION = "1.0.0"
QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _finite_values(series: pd.Series) -> np.ndarray:
    """Numeric values with NaN/+-Inf removed -- non-finite values are counted separately
    (see `missingCount` below) and never participate in min/max/mean/std/histogram/quantile
    computation, so one outlier `inf` cannot silently poison every other statistic."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _numeric_feature_baseline(series: pd.Series, *, bins: int) -> dict[str, Any]:
    """Deterministic aggregate stats for one numeric feature. Never raises on empty input,
    a constant feature, or a feature that is entirely NaN/Inf -- see the module-level test
    suite (`tests/test_drift_baseline.py`) for exactly which edge cases this covers."""
    total = len(series)
    finite = _finite_values(series)
    missing = total - len(finite)

    if finite.size == 0:
        return {
            "count": total, "missingCount": missing, "min": None, "max": None,
            "mean": None, "std": None, "binEdges": [], "binProportions": [],
            "quantiles": {f"p{int(level * 100)}": None for level in QUANTILE_LEVELS},
        }

    minimum, maximum = float(np.min(finite)), float(np.max(finite))
    # Interior edges only (bins-1 of them): np.digitize(x, edges, right=True) below implies
    # -inf/+inf at the two outer ends, mirroring app.ml.evaluator._calibration_bins's own
    # digitize convention -- an observed value outside [min, max] still lands in the first/
    # last bucket rather than raising, which is exactly what "out-of-range" detection (in
    # app.ml.drift_detector) needs to be able to measure against these same bins.
    if bins >= 2:
        quantile_levels = np.linspace(0.0, 1.0, bins + 1)[1:-1]
        edges = np.quantile(finite, quantile_levels) if finite.size else np.array([])
    else:
        edges = np.array([])
    bin_index = np.digitize(finite, edges, right=True) if edges.size else np.zeros(finite.size, dtype=int)
    n_buckets = edges.size + 1
    counts = np.bincount(bin_index, minlength=n_buckets)[:n_buckets]
    proportions = (counts / finite.size).tolist()

    quantiles = {
        f"p{int(level * 100)}": round(float(np.quantile(finite, level)), 6) for level in QUANTILE_LEVELS
    }
    return {
        "count": total, "missingCount": missing,
        "min": round(minimum, 6), "max": round(maximum, 6),
        "mean": round(float(np.mean(finite)), 6), "std": round(float(np.std(finite)), 6),
        "binEdges": [round(float(edge), 6) for edge in edges.tolist()],
        "binProportions": [round(float(p), 6) for p in proportions],
        "quantiles": quantiles,
    }


def _categorical_feature_baseline(series: pd.Series, *, max_categories: int) -> dict[str, Any]:
    """Bounded categorical distribution: only the top `max_categories` labels are retained
    by name (never the full unbounded label set), with everything else folded into a single
    `otherProportion`. This is the mechanism that keeps `driftBaseline` metadata size bounded
    regardless of how many distinct categories exist in the training data -- see
    `app.core.config.DRIFT_MAX_CATEGORIES`."""
    total = len(series)
    values = series.astype("string")
    missing = int(values.isna().sum())
    present = values.dropna()

    if present.empty:
        return {"count": total, "missingCount": missing, "topK": [], "otherProportion": 0.0}

    counts = present.value_counts()  # deterministic: pandas breaks ties by first-seen order
    top = counts.iloc[:max_categories]
    other_count = int(counts.iloc[max_categories:].sum())
    top_k = [
        {"value": str(label), "proportion": round(float(count) / len(present), 6)}
        for label, count in top.items()
    ]
    return {
        "count": total, "missingCount": missing, "topK": top_k,
        "otherProportion": round(other_count / len(present), 6),
    }


def build_baseline(
    df: pd.DataFrame,
    *,
    model_type: Literal["VIDEO", "LIVE"],
    numeric_features: list[str],
    categorical_features: list[str],
    histogram_bins: int = DRIFT_HISTOGRAM_BINS,
    max_categories: int = DRIFT_MAX_CATEGORIES,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a `driftBaseline` metadata block from the exact feature DataFrame a model was
    trained on. `df` must already contain only feature columns (plus, harmlessly, any extra
    columns -- only `numeric_features`/`categorical_features` are ever read); it must NOT
    contain raw identifier columns if the caller wants those excluded from consideration, but
    since this function only ever reads the named feature columns, an identifier column
    present in `df` for other purposes (e.g. `app.ml.dataset_builder.DIAGNOSTIC_COLUMNS`) is
    never touched or serialized here regardless.

    Determinism contract (corrective pass, Finding 3 -- stated precisely, not just implied):
    every field is a pure function of `df` and the config parameters above, INCLUDING
    `generatedAt`, when `generated_at` is supplied explicitly -- two calls with the same
    `df`/config/`generated_at` produce byte-identical output. `generated_at` defaults to the
    real wall-clock time at call time (`app.ml.trainer`/`app.ml.live_trainer`'s normal
    training-time usage never passes it explicitly), in which case -- and only in that case
    -- `generatedAt` is intentionally the one non-deterministic field; every other field is
    still deterministic for a fixed `df`/config regardless. Contains no raw identifiers, no
    raw rows, and no unbounded collections -- see `tests/test_drift_baseline.py` for the
    exact guarantees this is tested against.
    """
    feature_names = categorical_features + numeric_features
    return {
        "baselineVersion": BASELINE_SCHEMA_VERSION,
        "modelType": model_type,
        "generatedAt": (generated_at or datetime.now(timezone.utc)).isoformat(),
        "trainingSampleCount": len(df),
        "featureNames": list(feature_names),
        "categoricalFeatures": list(categorical_features),
        "numericFeatures": list(numeric_features),
        "numeric": {
            name: _numeric_feature_baseline(df[name] if name in df.columns else pd.Series([], dtype=float), bins=histogram_bins)
            for name in numeric_features
        },
        "categorical": {
            name: _categorical_feature_baseline(df[name] if name in df.columns else pd.Series([], dtype="string"), max_categories=max_categories)
            for name in categorical_features
        },
    }
