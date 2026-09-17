"""Loads feature rows in the model's expected column order for inference.

Sanitizes non-finite numeric values (NaN/inf) defensively at the serving boundary: none
of `FeatureHistory.features()`'s own arithmetic should produce them given validated
candidate input, but a broken upstream caller must degrade to a neutral score rather
than crash the request or silently corrupt the ranking.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.ml.dataset_builder import FEATURES, NUMERIC

NEUTRAL_VALUE = 0.0


def sanitize_numeric(rows: list[dict[str, Any]], numeric_features: list[str]) -> list[dict[str, Any]]:
    """Shared non-finite-value sanitizer for any feature-row model input (VIDEO or LIVE):
    replaces None/NaN/inf in `numeric_features` with `NEUTRAL_VALUE` so a broken upstream
    caller degrades to a neutral score rather than crashing the request or corrupting the
    ranking. Callers pass their own numeric feature list rather than this module assuming
    VIDEO's `NUMERIC`, so it stays correct for any feature schema."""
    sanitized = []
    for row in rows:
        clean = dict(row)
        for key in numeric_features:
            value = clean.get(key)
            if value is None or (isinstance(value, (int, float)) and not math.isfinite(value)):
                clean[key] = NEUTRAL_VALUE
        sanitized.append(clean)
    return sanitized


def probabilities(model, rows: list[dict[str, Any]]):
    frame = pd.DataFrame(sanitize_numeric(rows, NUMERIC))[FEATURES]
    return model.predict_proba(frame)[:, 1]
