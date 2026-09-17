"""Typed request/response models for drift monitoring (app/api/drift_routes.py).

`DriftObservation.features` is deliberately a bounded `dict[str, float | str]`, not one
hardcoded Pydantic field per feature name. `app.ml.dataset_builder.FEATURES` /
`app.ml.live_feature_builder.LIVE_FEATURES` are the single source of truth for what a
feature row contains; duplicating every name here as a typed field would create a second
copy of that contract that could silently drift out of sync with it (the exact failure mode
this whole feature exists to catch elsewhere). `app.ml.drift_detector.evaluate_drift()`
validates the submitted key set against the baseline's own recorded feature names at
evaluation time and raises a safe, structured error on mismatch -- never a raw KeyError.
"""
from __future__ import annotations

import math
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import DRIFT_MAX_OBSERVATIONS

# The three per-feature statuses app.ml.drift_detector actually computes.
DriftFeatureStatus = Literal["OK", "WARNING", "CRITICAL"]
DriftFeatureType = Literal["numeric", "categorical"]

# Report-level status: the three above, plus states that mean "no per-feature computation
# happened at all" -- INSUFFICIENT_DATA (too few observations, app.ml.drift_detector),
# BASELINE_MISSING (artifact exists but predates drift monitoring, or was never (re)trained
# since), MODEL_NOT_READY (no artifact at all, or it fails validation), and
# OBSERVATION_SOURCE_UNAVAILABLE (GET .../drift/live specifically: LIVE has no local
# interaction log to reconstruct observations from -- see README "Drift monitoring" /
# app/api/drift_routes.py).
DriftReportStatus = Literal[
    "OK", "WARNING", "CRITICAL", "INSUFFICIENT_DATA",
    "BASELINE_MISSING", "MODEL_NOT_READY", "OBSERVATION_SOURCE_UNAVAILABLE",
]
DRIFT_REPORT_STATUSES = get_args(DriftReportStatus)


class DriftObservation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    features: dict[str, float | str] = Field(..., min_length=1, max_length=64)

    @field_validator("features")
    @classmethod
    def _features_are_safe(cls, value: dict[str, float | str]) -> dict[str, float | str]:
        for key, item in value.items():
            if not key.strip():
                raise ValueError("feature name must not be blank")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"feature {key!r} must be a finite number (got {item!r})")
            if isinstance(item, str) and not item.strip():
                raise ValueError(f"feature {key!r} must not be a blank string")
        return value


class DriftEvaluateRequest(BaseModel):
    """POST /model/drift/evaluate[/live] body: a bounded batch of already-computed feature
    rows -- e.g. sampled/logged by a real Recommendation Service wrapper at serving time.
    Never re-derived from raw interaction/user/content identifiers server-side; the caller
    supplies exactly the feature values to be evaluated, nothing else."""
    model_config = ConfigDict(populate_by_name=True)
    observations: list[DriftObservation] = Field(..., min_length=1, max_length=DRIFT_MAX_OBSERVATIONS)


class DriftFeatureResult(BaseModel):
    """`out_of_range_rate` is numeric-only; `js_divergence`/`observed_other_rate`/
    `baseline_other_rate`/`other_rate_delta` are categorical-only (see
    app.ml.drift_detector.evaluate_numeric_feature/evaluate_categorical_feature) -- the
    inapplicable set is always `None` for a given feature, never a fabricated 0.0. There is
    deliberately no "unseen category" field: this deployment cannot genuinely determine
    whether a category was absent from all training data (only bounded top-K is retained),
    so `otherRateDelta` (the change in OTHER-bucket share) is reported instead -- see
    app.ml.drift_detector's module docstring for why the raw/absolute OTHER rate alone is
    not a valid drift signal."""
    model_config = ConfigDict(populate_by_name=True)
    feature_type: DriftFeatureType = Field(alias="featureType")
    status: DriftFeatureStatus
    psi: float
    js_divergence: float | None = Field(None, alias="jsDivergence")
    observed_count: int = Field(alias="observedCount")
    observed_missing_rate: float = Field(alias="observedMissingRate")
    baseline_missing_rate: float = Field(alias="baselineMissingRate")
    missing_rate_delta: float = Field(alias="missingRateDelta")
    out_of_range_rate: float | None = Field(None, alias="outOfRangeRate")
    observed_other_rate: float | None = Field(None, alias="observedOtherRate")
    baseline_other_rate: float | None = Field(None, alias="baselineOtherRate")
    other_rate_delta: float | None = Field(None, alias="otherRateDelta")


class DriftReport(BaseModel):
    """The one response shape every drift endpoint returns. `features` is always present
    (empty `{}` when `status` is anything other than OK/WARNING/CRITICAL) and never contains
    raw observations -- only the bounded per-feature aggregate result above. Never includes a
    filesystem path or artifact checksum."""
    model_config = ConfigDict(populate_by_name=True)
    status: DriftReportStatus
    model_type: str = Field(alias="modelType")
    model_version: str | None = Field(None, alias="modelVersion")
    baseline_version: str | None = Field(None, alias="baselineVersion")
    baseline_generated_at: str | None = Field(None, alias="baselineGeneratedAt")
    baseline_training_sample_count: int | None = Field(None, alias="baselineTrainingSampleCount")
    observation_count: int = Field(alias="observationCount")
    # Corrective pass (Finding 2): only ever non-zero for GET /model/drift (VIDEO's local
    # reconstruction path) -- the count of prior interactions used to warm up FeatureHistory
    # for users appearing in the observation window, kept strictly distinct from
    # observationCount (warm-up rows never contribute a feature row of their own; see
    # app.ml.dataset_builder.build_feature_rows).
    warmup_count: int = Field(0, alias="warmupCount")
    min_observations_required: int = Field(alias="minObservationsRequired")
    observation_source: str = Field(alias="observationSource")
    message: str | None = None
    features: dict[str, DriftFeatureResult] = Field(default_factory=dict)
