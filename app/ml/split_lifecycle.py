"""Shared chronological split lifecycle for the VIDEO and LIVE trainers.

Preferred sequence (five distinct, group-preserving, strictly-ordered chronological
partitions -- see `app.ml.splitting` for the non-overlap guarantee):

    train -> modelSelection -> calibration -> thresholdTuning -> test

Data usage only ever moves forward in time:
  - `train` fits each candidate model.
  - `modelSelection` picks the best candidate (never touches calibration/threshold/test).
  - `calibration` calibrates the frozen selected model (never reused for threshold tuning
    unless a small-data fallback below is in effect).
  - `thresholdTuning` tunes the decision threshold on the *calibrated* model's probabilities
    (never touches test).
  - `test` is evaluated exactly once, at the end, and never influences any of the above.

When the dataset cannot support all five as distinct, non-empty, class-valid partitions,
`resolve_split_lifecycle` falls back to a coarser, still-chronological split and records
exactly which roles were merged -- never silently reusing a split without saying so.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.ml.splitting import InsufficientSplitDataError, chronological_group_split
from app.ml.training_utils import has_both_classes

CANONICAL_SPLIT_NAMES = ("train", "modelSelection", "calibration", "thresholdTuning", "test")

# Ordered from most to least statistically ideal. Each entry is
# (description, ratios, aliases), where `aliases` maps a canonical role not physically
# present in `ratios` to the name of the physical split whose rows it reuses.
SPLIT_ATTEMPTS: tuple[tuple[str, dict[str, float], dict[str, str]], ...] = (
    (
        ("5-way chronological split: train/modelSelection/calibration/thresholdTuning/test "
         "are five distinct, non-overlapping partitions."),
        {"train": .40, "modelSelection": .15, "calibration": .15, "thresholdTuning": .10, "test": .20},
        {},
    ),
    (
        ("4-way fallback: insufficient data for a dedicated threshold-tuning split; "
         "thresholdTuning reuses the calibration split."),
        {"train": .50, "modelSelection": .15, "calibration": .15, "test": .20},
        {"thresholdTuning": "calibration"},
    ),
    (
        ("3-way fallback: insufficient data for dedicated calibration or threshold-tuning "
         "splits; both reuse the modelSelection split."),
        {"train": .60, "modelSelection": .20, "test": .20},
        {"calibration": "modelSelection", "thresholdTuning": "modelSelection"},
    ),
)

# Every canonical role needs both classes for scientifically meaningful evaluation, not just
# the two (train, calibration) that would hard-error otherwise: PR-AUC model selection needs
# both to compare candidates meaningfully; F1 threshold tuning needs both to optimize a
# genuine precision/recall trade-off rather than degenerately predicting one class; and the
# final test evaluation needs both so ROC-AUC/PR-AUC/classification metrics describe real
# discrimination rather than a fabricated placeholder (see `app.ml.evaluator.evaluate`'s
# `rocAuc: 0.0` single-class fallback, which must never be reached from the trainer).
REQUIRE_BOTH_CLASSES = CANONICAL_SPLIT_NAMES


class InsufficientLifecycleDataError(Exception):
    """No split attempt (5-way, 4-way, or 3-way fallback) could produce a valid
    train/modelSelection/calibration/thresholdTuning/test lifecycle for this dataset."""


def resolve_split_lifecycle(
    df: pd.DataFrame, *, group_col: str = "candidate_group", timestamp_col: str = "timestamp",
) -> dict[str, Any]:
    """Resolve the best available chronological split lifecycle for `df`.

    Returns a dict with:
      - "splits": the five canonical-name DataFrames (train/modelSelection/calibration/
        thresholdTuning/test); under a fallback, two or more of these point at the exact
        same underlying rows (see "roleMapping").
      - "physicalSplits": only the *distinct* DataFrames actually produced by
        `chronological_group_split` for the winning attempt -- use this for honest
        row-count/class-distribution reporting so shared rows are never double-counted.
      - "roleMapping": canonical name -> physical split name backing it.
      - "description": human-readable description of which attempt was used.
      - "usedFallback": bool, True unless the full 5-way split succeeded.
      - "ratios": the ratios dict used for the winning (physical) attempt.

    Raises `InsufficientLifecycleDataError` if no attempt can produce non-empty,
    class-valid splits -- callers should translate this into their domain-specific
    "insufficient training data" exception.
    """
    failures: list[str] = []
    for description, ratios, aliases in SPLIT_ATTEMPTS:
        try:
            physical = chronological_group_split(df, ratios=ratios, group_col=group_col, timestamp_col=timestamp_col)
        except InsufficientSplitDataError as exc:
            failures.append(f"[{description}] {exc}")
            continue

        role_mapping = {name: aliases.get(name, name) for name in CANONICAL_SPLIT_NAMES}
        canonical = {name: physical[role_mapping[name]] for name in CANONICAL_SPLIT_NAMES}

        invalid = [name for name in REQUIRE_BOTH_CLASSES if not has_both_classes(canonical[name].target)]
        if invalid:
            failures.append(f"[{description}] split(s) {invalid} lack both classes (positive and negative)")
            continue

        return {
            "splits": canonical,
            "physicalSplits": physical,
            "roleMapping": role_mapping,
            "description": description,
            "usedFallback": bool(aliases),
            "ratios": ratios,
        }

    raise InsufficientLifecycleDataError(
        "No chronological split configuration (5-way, or the documented 4-way/3-way "
        "fallbacks) could produce non-empty, class-balanced train/calibration splits for "
        "this dataset. Attempts:\n" + "\n".join(failures)
    )
