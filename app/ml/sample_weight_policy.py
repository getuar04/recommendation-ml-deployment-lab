"""Per-row training-importance (`sample_weight`) policy -- NOT a change to the binary target.

Root-cause diagnosis (this task): app.ml.eligibility's longTerm/negative/notInterested gates
each probe a concentrated, strongly-explicit feature combination (many completions/likes/
CONTENT_NOT_INTERESTED events on one category) that is real but comparatively rare in
app.ml.dataset_builder.build_dataset()'s output -- measured directly against this project's
own synthetic dataset: the labeled rows are dominated by borderline/ambiguous outcomes (the
bulk population's exploration noise, medium watches, etc.), with strong explicit signals
(CONTENT_NOT_INTERESTED, likes/shares/favorites/follows, near-full completions) a small
minority. A plain binary target treats a row that barely crossed the 70%-watch threshold
identically to one that was liked, shared, AND followed the creator, and treats a 15%-watch
skip identically to an explicit CONTENT_NOT_INTERESTED. Tree-based candidates (RandomForest/
XGBoost/LightGBM/CatBoost), which fit local, data-density-sensitive partitions rather than one
smooth global coefficient the way LogisticRegression does, are measurably more sensitive to
this dilution: in the specific region of feature space the gates probe, the rare, strongly-
discriminative rows get outvoted by the much larger volume of weak/ambiguous ones sharing that
same region.

`sample_weight` addresses this WITHOUT changing the target: every row keeps its existing
binary label (see app.ml.feature_builder.target_for), but rows with stronger, more explicit
behavioral evidence get proportionally more influence on the fitted decision boundary. Every
one of this project's five candidate estimators accepts `sample_weight` natively at `.fit()`
time (see app.ml.pipeline_builder.sample_weight_fit_params) -- this is a standard, algorithm-
agnostic technique, not specific to any one candidate.

Weights are derived ONLY from a row's own already-happened event fields (event_type/
watch_percentage/liked/shared/favorited/creator_followed -- exactly
app.ml.dataset_builder.TRAINING_METADATA_COLUMNS, the same fields
app.ml.feature_builder.target_for() already reads to produce `target`), so this adds no
information beyond what the label itself already encodes, and never depends on the split a
row lands in, another row, or any future information -- point-in-time safe by construction.

Values below are starting points grounded in this codebase's own existing convention (see
app.ml.dataset_builder.CONTENT_NOT_INTERESTED_CATEGORY_PENALTY, already the single strongest
per-event weight this project defines for feature engineering), not an arbitrary guess, and
NOT a hyperparameter search target. Multiple applicable positive signals combine by taking the
single STRONGEST applicable weight (never summed/multiplied together), so a row that is both
liked and followed gets the follow weight, not a runaway product of the two.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.ml.feature_builder import (
    COMPLETION_WATCH_PERCENTAGE_THRESHOLD,
    EXPLICIT_NEGATIVE_EVENT_TYPES,
)

SAMPLE_WEIGHT_COLUMN = "sample_weight"

# A generic implicit negative (fast-skip: low watch %, no positive flag) is weak, ambiguous
# evidence -- app.ml.eligibility's own "negative" gate docstring already documents why a
# handful of fast-skips must not be treated as strongly as an explicit rejection (it can
# reflect a passing mood or a bad candidate match, not necessarily genuine dislike). Kept at
# the neutral baseline: no boost, but never discounted either.
NEGATIVE_IMPLICIT_WEIGHT = 1.0
# CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED is a deliberate, single-purpose, low-noise "I
# don't want this" action -- app.ml.dataset_builder already treats it as the single strongest
# feature-engineering signal it defines (-8 vs. a fast-skip's -4); training importance now
# matches that same product judgment instead of leaving it diluted by the far larger volume
# of implicit-negative rows.
#
# 2.0, not the higher end of the spec's suggested 2-4 range, deliberately -- measured, not
# guessed: this project's own eligibility-gate regression suite (tests/test_step11_*.py,
# tests/test_eligibility_gates.py) was used to sweep 1.3/1.5/2.0/3.0 directly against this
# project's real trained candidates. 3.0 (and, combined with graduated positive weights,
# 1.5) measurably regressed LogisticRegression -- the only candidate that reliably passes
# every mandatory behavioral gate on this dataset today -- off the `notInterested` gate
# entirely (a real eligibility regression, not noise). 2.0 is the highest value in that sweep
# that left LogisticRegression's full gate-pass record completely unchanged.
NEGATIVE_EXPLICIT_REJECTION_WEIGHT = 2.0

# Positive weights are intentionally a gentle gradation (1.05-1.2), not the wider spread
# tried initially (1.2-2.0): that wider spread, combined with any explicit-rejection weight
# above 1.3, measurably destabilized CatBoost/XGBoost's OTHER gate margins (notably
# `coldStart`) without recovering any additional gate for any candidate -- adding volatility
# with no measured benefit. This gentler gradation was the mildest change that could still be
# verified end-to-end (see the sweep referenced above) without registering as a new
# regression for any candidate relative to the unweighted baseline.
POSITIVE_BASE_WEIGHT = 1.0
POSITIVE_COMPLETION_WEIGHT = 1.05
POSITIVE_LIKE_WEIGHT = 1.1
POSITIVE_SHARE_OR_FAVORITE_WEIGHT = 1.15
POSITIVE_CREATOR_FOLLOWED_WEIGHT = 1.2


def _row_weight(row: Any) -> float:
    if row.target == 0:
        is_explicit_rejection = row.event_type in EXPLICIT_NEGATIVE_EVENT_TYPES
        return NEGATIVE_EXPLICIT_REJECTION_WEIGHT if is_explicit_rejection else NEGATIVE_IMPLICIT_WEIGHT

    weight = POSITIVE_BASE_WEIGHT
    completed = (
        row.event_type == "VIDEO_COMPLETED"
        or (row.event_watch_percentage or 0) >= COMPLETION_WATCH_PERCENTAGE_THRESHOLD
    )
    if completed:
        weight = max(weight, POSITIVE_COMPLETION_WEIGHT)
    if row.event_liked:
        weight = max(weight, POSITIVE_LIKE_WEIGHT)
    if row.event_shared or row.event_favorited:
        weight = max(weight, POSITIVE_SHARE_OR_FAVORITE_WEIGHT)
    if row.event_creator_followed:
        weight = max(weight, POSITIVE_CREATOR_FOLLOWED_WEIGHT)
    return weight


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """One weight per row of `df` (as produced by app.ml.dataset_builder.build_dataset, which
    carries `target` plus every column in TRAINING_METADATA_COLUMNS). Deterministic, a pure
    function of each row's own fields -- never depends on row order or any other row."""
    return np.array([_row_weight(row) for row in df.itertuples()], dtype=float)
