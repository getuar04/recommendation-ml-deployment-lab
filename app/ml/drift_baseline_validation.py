"""Strict structural validation for a `driftBaseline` metadata block
(`app.ml.drift_baseline.build_baseline`'s output) before it is ever trusted for
evaluation.

`build_baseline` is the only writer of this structure and is already exhaustively
tested (`tests/test_drift_baseline.py`) to always produce a structurally sound
baseline from a real training run. This module exists for the *other* source of a
`driftBaseline` value: whatever an already-saved artifact's metadata.json happens to
contain -- a hand-edited fixture, a partially-written file from an interrupted save,
or a structure that predates a later change to this schema. A baseline with the right
feature *names* but empty or malformed per-feature statistics (e.g. `{}` for a
declared numeric feature, or a `count`/`missingCount` pair that claims finite rows
exist while every actual statistic is null) must not silently produce a misleading
OK -- it must be treated exactly like "no baseline at all", never partially trusted
and never an unhandled 500.

Pydantic models here are validate-only: never constructed programmatically elsewhere
and never serialized back out. `app.services.drift_service._validated_baseline` still
returns the original plain dict to `app.ml.drift_detector.evaluate_drift` unchanged
once validation passes, so this module cannot itself change what evaluate_drift reads.
"""
from __future__ import annotations

import itertools
import math
from datetime import datetime
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    ValidationError,
    model_validator,
)

from app.core.config import DRIFT_MAX_CATEGORIES
from app.ml.drift_baseline import BASELINE_SCHEMA_VERSION, QUANTILE_LEVELS

# Only the exact schema version this deployment's app.ml.drift_baseline currently
# writes is accepted -- there is no multi-version migration path yet, so a baseline
# claiming a different version is treated as unusable rather than guessed-at.
SUPPORTED_BASELINE_VERSIONS = {BASELINE_SCHEMA_VERSION}

_SUM_TOLERANCE = 1e-4  # generous vs. build_baseline's 6-decimal rounding (see drift_baseline.py)
_PROPORTION_UPPER_BOUND = 1.0 + 1e-6
_ORDER_TOLERANCE = 1e-9  # generous vs. 6-decimal rounding, for monotonic/range comparisons
_REQUIRED_QUANTILE_KEYS = frozenset(f"p{int(level * 100)}" for level in QUANTILE_LEVELS)


class BaselineStructureError(Exception):
    """Raised for any structurally invalid `driftBaseline`. Callers
    (`app.services.drift_service._validated_baseline`) treat this identically to a
    wholly-missing baseline (`BaselineMissing` -> `BASELINE_MISSING`), never a 500."""


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_plain_int(value: Any) -> Any:
    """Rejects the exact inputs `int` field coercion would otherwise silently accept:
    `bool` (a Python `int` subclass), numeric strings, and floats (even integral ones
    like `10.0`). A `driftBaseline`'s `count`/`missingCount`/`trainingSampleCount` are
    always plain JSON integers when genuinely produced by `app.ml.drift_baseline`, so
    anything else here is itself evidence of a malformed/hand-edited baseline."""
    if isinstance(value, bool) or not isinstance(value, int):
        # ValueError, not TypeError: a `BeforeValidator` must raise ValueError (or
        # AssertionError) for pydantic to wrap it as a field ValidationError -- a TypeError
        # here propagates raw past pydantic entirely, bypassing `validate_baseline_structure`'s
        # `except ValidationError` and turning a routine malformed-baseline case into an
        # unhandled 500 instead of BASELINE_MISSING.
        raise ValueError("must be a plain integer, not a bool, string, or float")  # noqa: TRY004
    return value


_StrictInt = Annotated[int, BeforeValidator(_require_plain_int)]


def _quantile_level(key: str) -> int:
    return int(key[1:])


class _NumericFeatureBaseline(BaseModel):
    model_config = ConfigDict(extra="allow")
    count: _StrictInt
    missingCount: _StrictInt
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    binEdges: list[float]
    binProportions: list[float]
    quantiles: dict[str, float | None] | None = None

    @model_validator(mode="after")
    def _check(self) -> _NumericFeatureBaseline:
        if self.count < 0 or self.missingCount < 0:
            raise ValueError("count/missingCount must be non-negative")
        if self.missingCount > self.count:
            raise ValueError("missingCount must not exceed count")

        finite_count = self.count - self.missingCount
        if finite_count == 0:
            # The one explicitly supported empty representation
            # (app.ml.drift_baseline._numeric_feature_baseline's finite.size == 0 branch):
            # empty edges/proportions, null min/max/mean/std, every required quantile key
            # present and null. Gated on `count - missingCount`, never on whether
            # binProportions/binEdges merely happen to be empty -- a baseline claiming
            # finite rows (count > missingCount) with no actual statistics is malformed,
            # not "empty", and must be rejected below instead.
            if self.binEdges or self.binProportions:
                raise ValueError("a feature with count - missingCount == 0 must have empty binEdges/binProportions")
            if any(value is not None for value in (self.min, self.max, self.mean, self.std)):
                raise ValueError("a feature with count - missingCount == 0 must have null min/max/mean/std")
            if self.quantiles is None or set(self.quantiles) != _REQUIRED_QUANTILE_KEYS:
                raise ValueError(f"a feature with count - missingCount == 0 must have exactly these quantile keys: {sorted(_REQUIRED_QUANTILE_KEYS)}")
            if any(value is not None for value in self.quantiles.values()):
                raise ValueError("a feature with count - missingCount == 0 must have null-valued quantiles")
            return self

        # finite_count > 0: a real distribution is required -- min/max/mean/std, the
        # complete quantile set, and a bin histogram must all actually be present.
        for label, value in (("min", self.min), ("max", self.max), ("mean", self.mean), ("std", self.std)):
            if not _is_finite_number(value):
                raise ValueError(f"{label} must be a finite number when count - missingCount > 0")
        minimum, maximum, mean = self.min, self.max, self.mean  # narrowed non-None by the finite check above
        assert minimum is not None and maximum is not None and mean is not None
        if minimum > maximum:
            raise ValueError("min must be <= max")
        if not (minimum - _ORDER_TOLERANCE <= mean <= maximum + _ORDER_TOLERANCE):
            raise ValueError("mean must be between min and max")
        if self.std is None or self.std < 0:
            raise ValueError("std must be a non-negative finite number when count - missingCount > 0")

        if self.quantiles is None or set(self.quantiles) != _REQUIRED_QUANTILE_KEYS:
            raise ValueError(f"required quantile keys missing when count - missingCount > 0: {sorted(_REQUIRED_QUANTILE_KEYS)}")
        ordered_keys = sorted(self.quantiles, key=_quantile_level)
        ordered_values: list[float] = []
        for key in ordered_keys:
            value = self.quantiles[key]
            if not _is_finite_number(value):
                raise ValueError(f"quantile {key!r} must be a finite number when count - missingCount > 0")
            assert value is not None
            if not (minimum - _ORDER_TOLERANCE <= value <= maximum + _ORDER_TOLERANCE):
                raise ValueError(f"quantile {key!r} must be within [min, max]")
            ordered_values.append(value)
        if any(later < earlier - _ORDER_TOLERANCE for earlier, later in itertools.pairwise(ordered_values)):
            raise ValueError("quantiles must be monotonically non-decreasing")

        if not self.binProportions:
            raise ValueError("binProportions must be present when count - missingCount > 0")
        if not all(_is_finite_number(edge) for edge in self.binEdges):
            raise ValueError("binEdges must be finite")
        if any(later < earlier - _ORDER_TOLERANCE for earlier, later in itertools.pairwise(self.binEdges)):
            raise ValueError("binEdges must be monotonically non-decreasing")
        if len(self.binProportions) != len(self.binEdges) + 1:
            raise ValueError("binProportions length must equal len(binEdges) + 1 when count - missingCount > 0")
        if not all(_is_finite_number(p) and 0.0 <= p <= _PROPORTION_UPPER_BOUND for p in self.binProportions):
            raise ValueError("binProportions must be finite, non-negative, and bounded to [0, 1]")
        if not math.isclose(sum(self.binProportions), 1.0, abs_tol=_SUM_TOLERANCE):
            raise ValueError("binProportions must sum to approximately 1")
        return self


class _CategoricalTopKEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    value: str
    proportion: float

    @model_validator(mode="after")
    def _check(self) -> _CategoricalTopKEntry:
        if not self.value.strip():
            raise ValueError("topK entry value must not be blank")
        if not _is_finite_number(self.proportion) or not (0.0 <= self.proportion <= _PROPORTION_UPPER_BOUND):
            raise ValueError("topK entry proportion must be finite and non-negative")
        return self


class _CategoricalFeatureBaseline(BaseModel):
    model_config = ConfigDict(extra="allow")
    count: _StrictInt
    missingCount: _StrictInt
    topK: list[_CategoricalTopKEntry]
    otherProportion: float

    @model_validator(mode="after")
    def _check(self) -> _CategoricalFeatureBaseline:
        if self.count < 0 or self.missingCount < 0:
            raise ValueError("count/missingCount must be non-negative")
        if self.missingCount > self.count:
            raise ValueError("missingCount must not exceed count")
        # Defensive bound, not a re-check of the exact max_categories a past training run
        # used (that value is not recorded on the baseline itself): catches a corrupted/
        # unbounded topK, not a legitimate historical baseline built with a smaller config.
        if len(self.topK) > DRIFT_MAX_CATEGORIES:
            raise ValueError("topK is not a bounded list")
        values = [entry.value for entry in self.topK]
        if len(values) != len(set(values)):
            raise ValueError("topK values must be unique")
        if not _is_finite_number(self.otherProportion) or not (0.0 <= self.otherProportion <= _PROPORTION_UPPER_BOUND):
            raise ValueError("otherProportion must be finite and within [0, 1]")

        # Gated on `count - missingCount`, exactly like the numeric feature above: a
        # categorical feature claiming present (non-missing) rows must show a real
        # distribution, never fall back to the empty representation just because topK
        # happens to be empty.
        present_count = self.count - self.missingCount
        if present_count == 0:
            # The one explicitly supported empty representation
            # (app.ml.drift_baseline._categorical_feature_baseline's present.empty branch).
            if self.topK or not math.isclose(self.otherProportion, 0.0, abs_tol=_SUM_TOLERANCE):
                raise ValueError("a feature with count - missingCount == 0 must have empty topK and zero otherProportion")
        else:
            total = sum(entry.proportion for entry in self.topK) + self.otherProportion
            if not math.isclose(total, 1.0, abs_tol=_SUM_TOLERANCE):
                raise ValueError("topK proportions + otherProportion must sum to approximately 1 when count - missingCount > 0")
        return self


class _DriftBaselineModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    baselineVersion: str
    modelType: str
    generatedAt: str
    trainingSampleCount: _StrictInt
    featureNames: list[str]
    categoricalFeatures: list[str]
    numericFeatures: list[str]
    numeric: dict[str, _NumericFeatureBaseline]
    categorical: dict[str, _CategoricalFeatureBaseline]

    @model_validator(mode="after")
    def _check(self) -> _DriftBaselineModel:
        if self.baselineVersion not in SUPPORTED_BASELINE_VERSIONS:
            raise ValueError(f"unsupported baselineVersion {self.baselineVersion!r}")
        if self.trainingSampleCount < 0:
            raise ValueError("trainingSampleCount must be a non-negative integer")
        try:
            datetime.fromisoformat(self.generatedAt)
        except ValueError as exc:
            raise ValueError("generatedAt must be a parseable ISO-8601 timestamp") from exc

        if len(self.featureNames) != len(set(self.featureNames)):
            raise ValueError("featureNames must not contain duplicates")
        numeric_keys, categorical_keys = set(self.numeric), set(self.categorical)
        if set(self.categoricalFeatures) != categorical_keys:
            raise ValueError("categoricalFeatures must agree with the categorical baseline's keys")
        if set(self.numericFeatures) != numeric_keys:
            raise ValueError("numericFeatures must agree with the numeric baseline's keys")
        if numeric_keys & categorical_keys:
            raise ValueError("numeric and categorical feature sets must be disjoint")
        if set(self.featureNames) != numeric_keys | categorical_keys:
            raise ValueError("featureNames must exactly match the union of numeric and categorical features")

        # Every per-feature `count` describes the same training run this baseline as a
        # whole was built from (app.ml.drift_baseline.build_baseline passes the identical
        # `df` -- and therefore the identical row count -- into every feature's stats), so
        # a feature whose own `count` disagrees with `trainingSampleCount` is internally
        # inconsistent, not a legitimate per-feature difference.
        for name, numeric_stats in self.numeric.items():
            if numeric_stats.count != self.trainingSampleCount:
                raise ValueError(
                    f"numeric feature {name!r} count ({numeric_stats.count}) does not match trainingSampleCount ({self.trainingSampleCount})"
                )
        for name, categorical_stats in self.categorical.items():
            if categorical_stats.count != self.trainingSampleCount:
                raise ValueError(
                    f"categorical feature {name!r} count ({categorical_stats.count}) does not match trainingSampleCount ({self.trainingSampleCount})"
                )
        return self


def validate_baseline_structure(baseline: dict[str, Any], *, model_type: str, expected_features: list[str]) -> None:
    """Raises `BaselineStructureError` unless `baseline` is a complete, internally
    consistent `driftBaseline` for `model_type` whose feature set exactly matches
    `expected_features`. Returns `None` (the caller keeps using its own original
    `baseline` dict) -- this function only ever validates, never transforms."""
    try:
        parsed = _DriftBaselineModel.model_validate(baseline)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        raise BaselineStructureError(f"driftBaseline structure is invalid at {location or '<root>'}: {first['msg']}") from exc

    if parsed.modelType != model_type:
        raise BaselineStructureError(
            f"driftBaseline is recorded for a different model type ({parsed.modelType!r}) than this endpoint ({model_type!r})"
        )
    declared = set(parsed.featureNames)
    if declared != set(expected_features):
        raise BaselineStructureError("driftBaseline feature set does not match the current feature contract")
