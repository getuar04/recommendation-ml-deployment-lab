"""Drift calculation: turns a `driftBaseline` (app.ml.drift_baseline.build_baseline) plus a
new batch of observed feature rows into a bounded, deterministic drift report.

No Kolmogorov-Smirnov test here, by design. A true two-sample KS test needs both samples'
raw values to build an empirical CDF; `driftBaseline` deliberately stores only bounded
aggregate histograms/quantiles/top-K distributions (see module docstring on
`app.ml.drift_baseline`), not raw training rows, so there is no baseline sample to run KS
against. Claiming a KS statistic here would misrepresent an aggregate-vs-aggregate
comparison as a true two-sample test. PSI and Jensen-Shannon divergence are used instead --
both are natively defined over binned/categorical proportions, which is exactly the shape
the baseline stores.

Pure pandas/numpy; no FastAPI, no DB, no filesystem access. `evaluate_drift()` is the single
entry point both VIDEO and LIVE call.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from app.core.config import (
    DRIFT_MIN_OBSERVATIONS,
    DRIFT_MISSING_RATE_CRITICAL_DELTA,
    DRIFT_MISSING_RATE_WARNING_DELTA,
    DRIFT_OTHER_RATE_CRITICAL_DELTA,
    DRIFT_OTHER_RATE_WARNING_DELTA,
    DRIFT_OUT_OF_RANGE_CRITICAL_RATE,
    DRIFT_OUT_OF_RANGE_WARNING_RATE,
    DRIFT_PROBABILITY_EPSILON,
    DRIFT_PSI_CRITICAL_THRESHOLD,
    DRIFT_PSI_WARNING_THRESHOLD,
)

DriftStatus = Literal["OK", "WARNING", "CRITICAL", "INSUFFICIENT_DATA"]
_STATUS_SEVERITY = {"OK": 0, "WARNING": 1, "CRITICAL": 2}


class DriftSchemaMismatchError(Exception):
    """Raised when the observation batch's columns don't match the baseline's recorded
    feature contract, or the baseline itself is missing required structure -- callers (the
    API layer) turn this into a safe, structured error response, never a raw traceback."""


def _worse(a: str, b: str) -> str:
    return a if _STATUS_SEVERITY.get(a, 0) >= _STATUS_SEVERITY.get(b, 0) else b


def _classify(value: float, *, warning: float, critical: float) -> Literal["OK", "WARNING", "CRITICAL"]:
    if value >= critical:
        return "CRITICAL"
    if value >= warning:
        return "WARNING"
    return "OK"


def population_stability_index(
    baseline_proportions: np.ndarray, observed_proportions: np.ndarray, *, epsilon: float = DRIFT_PROBABILITY_EPSILON,
) -> float:
    """Standard PSI: sum((q - p) * ln(q / p)) over aligned bins. Both inputs are floored at
    `epsilon` before use so a zero-probability bin on either side (a bin the baseline never
    populated, or one the new observations never populated) contributes a large-but-finite
    term instead of NaN/Inf from log(0) or division by zero. Never returns NaN/Inf."""
    if len(baseline_proportions) == 0 or len(observed_proportions) == 0:
        return 0.0
    p = np.clip(np.asarray(baseline_proportions, dtype=float), epsilon, None)
    q = np.clip(np.asarray(observed_proportions, dtype=float), epsilon, None)
    value = float(np.sum((q - p) * np.log(q / p)))
    return value if np.isfinite(value) else 0.0


def jensen_shannon_divergence(
    baseline_proportions: np.ndarray, observed_proportions: np.ndarray, *, epsilon: float = DRIFT_PROBABILITY_EPSILON,
) -> float:
    """Symmetric, bounded ([0, 1] after normalizing by ln(2)) alternative/complement to PSI
    for categorical distributions. Never returns NaN/Inf."""
    if len(baseline_proportions) == 0 or len(observed_proportions) == 0:
        return 0.0
    p = np.clip(np.asarray(baseline_proportions, dtype=float), epsilon, None)
    q = np.clip(np.asarray(observed_proportions, dtype=float), epsilon, None)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    js = 0.5 * kl_pm + 0.5 * kl_qm
    normalized = js / np.log(2)
    return float(np.clip(normalized, 0.0, 1.0)) if np.isfinite(normalized) else 0.0


def _observed_numeric_proportions(values: np.ndarray, edges: list[float]) -> np.ndarray:
    n_buckets = len(edges) + 1
    if values.size == 0:
        return np.zeros(n_buckets)
    bucket_index = np.digitize(values, np.asarray(edges), right=True) if edges else np.zeros(values.size, dtype=int)
    counts = np.bincount(bucket_index, minlength=n_buckets)[:n_buckets]
    return counts / values.size


def evaluate_numeric_feature(name: str, baseline: dict[str, Any], observed: pd.Series) -> dict[str, Any]:
    total = len(observed)
    numeric = pd.to_numeric(observed, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    observed_missing = total - int(finite.size)
    observed_missing_rate = observed_missing / total if total else 0.0

    baseline_count = baseline.get("count", 0) or 0
    baseline_missing_rate = (baseline.get("missingCount", 0) or 0) / baseline_count if baseline_count else 0.0
    missing_rate_delta = abs(observed_missing_rate - baseline_missing_rate)

    edges = baseline.get("binEdges") or []
    baseline_proportions = np.asarray(baseline.get("binProportions") or [])
    has_baseline_distribution = baseline_count > 0 and baseline_proportions.size > 0

    if has_baseline_distribution:
        observed_proportions = _observed_numeric_proportions(finite, edges)
        psi = population_stability_index(baseline_proportions, observed_proportions)
    else:
        observed_proportions, psi = np.zeros(0), 0.0

    baseline_min, baseline_max = baseline.get("min"), baseline.get("max")
    if baseline_min is not None and baseline_max is not None and finite.size:
        out_of_range_rate = float(np.mean((finite < baseline_min) | (finite > baseline_max)))
    else:
        out_of_range_rate = 0.0

    psi_status = _classify(psi, warning=DRIFT_PSI_WARNING_THRESHOLD, critical=DRIFT_PSI_CRITICAL_THRESHOLD)
    missing_status = _classify(missing_rate_delta, warning=DRIFT_MISSING_RATE_WARNING_DELTA, critical=DRIFT_MISSING_RATE_CRITICAL_DELTA)
    range_status = _classify(out_of_range_rate, warning=DRIFT_OUT_OF_RANGE_WARNING_RATE, critical=DRIFT_OUT_OF_RANGE_CRITICAL_RATE)
    status = _worse(_worse(psi_status, missing_status), range_status)

    return {
        "featureType": "numeric",
        "status": status,
        "psi": round(psi, 6),
        "observedCount": total,
        "observedMissingRate": round(observed_missing_rate, 6),
        "baselineMissingRate": round(baseline_missing_rate, 6),
        "missingRateDelta": round(missing_rate_delta, 6),
        "outOfRangeRate": round(out_of_range_rate, 6),
    }


def evaluate_categorical_feature(name: str, baseline: dict[str, Any], observed: pd.Series) -> dict[str, Any]:
    """Corrective pass (Finding 1): categorical tail drift is classified on the *change* in
    OTHER-bucket share (`abs(observedOtherRate - baselineOtherRate)`), never on the absolute
    observed OTHER rate. The absolute rate alone is not a drift signal: baseline OTHER
    already contains every training-time category that fell outside the retained top-K, so
    a high-cardinality feature can have a large, entirely legitimate baseline OTHER
    proportion -- an observation batch drawn from the exact same distribution as training
    would then also have a large OTHER share, and classifying that absolute value would
    flag WARNING/CRITICAL on a batch that has not drifted at all. There is deliberately no
    "unseen category" metric here: the baseline never stores the full training-time label
    set (only bounded top-K), so this code cannot actually tell a category that was
    genuinely never seen in training apart from one that was merely rare and fell into
    training's own OTHER bucket -- see `app.ml.drift_baseline` and README "Drift monitoring"."""
    total = len(observed)
    values = observed.astype("string")
    observed_missing = int(values.isna().sum())
    present = values.dropna()
    observed_missing_rate = observed_missing / total if total else 0.0

    baseline_count = baseline.get("count", 0) or 0
    baseline_missing_rate = (baseline.get("missingCount", 0) or 0) / baseline_count if baseline_count else 0.0
    missing_rate_delta = abs(observed_missing_rate - baseline_missing_rate)

    baseline_top = baseline.get("topK") or []
    baseline_labels = [entry["value"] for entry in baseline_top]
    baseline_other_rate = baseline.get("otherProportion", 0.0) or 0.0
    baseline_proportions = np.asarray([entry["proportion"] for entry in baseline_top] + [baseline_other_rate])

    if present.empty or (not baseline_labels and not baseline_other_rate):
        psi, js, observed_other_rate = 0.0, 0.0, 0.0
    else:
        observed_counts = present.value_counts()
        observed_present_total = len(present)
        observed_known = sum(int(observed_counts.get(label, 0)) for label in baseline_labels)
        observed_other = observed_present_total - observed_known
        observed_other_rate = observed_other / observed_present_total
        # PSI/JS still compare the *full* aligned Top-K + OTHER distribution (unchanged by
        # this correction): a shift between two named Top-K categories, or between a named
        # category and OTHER, is still detected here even when the OTHER share itself is
        # unchanged in aggregate.
        observed_proportions = np.asarray(
            [int(observed_counts.get(label, 0)) / observed_present_total for label in baseline_labels]
            + [observed_other_rate]
        )
        psi = population_stability_index(baseline_proportions, observed_proportions)
        js = jensen_shannon_divergence(baseline_proportions, observed_proportions)

    other_rate_delta = abs(observed_other_rate - baseline_other_rate)

    psi_status = _classify(psi, warning=DRIFT_PSI_WARNING_THRESHOLD, critical=DRIFT_PSI_CRITICAL_THRESHOLD)
    missing_status = _classify(missing_rate_delta, warning=DRIFT_MISSING_RATE_WARNING_DELTA, critical=DRIFT_MISSING_RATE_CRITICAL_DELTA)
    other_status = _classify(other_rate_delta, warning=DRIFT_OTHER_RATE_WARNING_DELTA, critical=DRIFT_OTHER_RATE_CRITICAL_DELTA)
    status = _worse(_worse(psi_status, missing_status), other_status)

    return {
        "featureType": "categorical",
        "status": status,
        "psi": round(psi, 6),
        "jsDivergence": round(js, 6),
        "observedCount": total,
        "observedMissingRate": round(observed_missing_rate, 6),
        "baselineMissingRate": round(baseline_missing_rate, 6),
        "missingRateDelta": round(missing_rate_delta, 6),
        "observedOtherRate": round(observed_other_rate, 6),
        "baselineOtherRate": round(baseline_other_rate, 6),
        "otherRateDelta": round(other_rate_delta, 6),
    }


def evaluate_drift(baseline: dict[str, Any], observations: pd.DataFrame) -> dict[str, Any]:
    """The single entry point: validates the observation batch against the baseline's
    recorded feature contract, then computes one report combining every feature's result.
    `observations` must contain a column per baseline feature name (extra columns are
    ignored, matching `app.ml.drift_baseline.build_baseline`'s own tolerance).

    Raises `DriftSchemaMismatchError` for a malformed baseline or an observation batch
    missing required feature columns -- never silently drops or ignores a required feature.
    """
    numeric_baselines = baseline.get("numeric")
    categorical_baselines = baseline.get("categorical")
    if not isinstance(numeric_baselines, dict) or not isinstance(categorical_baselines, dict):
        raise DriftSchemaMismatchError("Baseline is missing required 'numeric'/'categorical' structure.")

    feature_names = list(categorical_baselines) + list(numeric_baselines)
    missing_columns = [name for name in feature_names if name not in observations.columns]
    if missing_columns:
        raise DriftSchemaMismatchError(
            f"Observation batch is missing required feature columns: {sorted(missing_columns)}"
        )

    observation_count = len(observations)
    if observation_count < DRIFT_MIN_OBSERVATIONS:
        return {
            "status": "INSUFFICIENT_DATA",
            "observationCount": observation_count,
            "minObservationsRequired": DRIFT_MIN_OBSERVATIONS,
            "features": {},
        }

    features: dict[str, Any] = {}
    overall = "OK"
    for name, stats in categorical_baselines.items():
        result = evaluate_categorical_feature(name, stats, observations[name])
        features[name] = result
        overall = _worse(overall, result["status"])
    for name, stats in numeric_baselines.items():
        result = evaluate_numeric_feature(name, stats, observations[name])
        features[name] = result
        overall = _worse(overall, result["status"])

    return {
        "status": overall,
        "observationCount": observation_count,
        "minObservationsRequired": DRIFT_MIN_OBSERVATIONS,
        "features": features,
    }
