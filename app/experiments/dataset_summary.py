"""One reusable dataset-summary implementation (spec §9/§13/§46), used by experiment run
reports so training metadata, experiment records, and comparison reports never compute this
independently and drift apart.

Every summary handles: empty data, neutral (unlabeled) rows, missing timestamps, missing
optional identity columns, division by zero, and single-class datasets -- see boundary tests
in tests/test_experiment_dataset_summary.py.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from app.ml.feature_builder import target_for


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def video_dataset_summary(rows: Iterable[Any]) -> dict:
    """`rows`: Interaction ORM rows (or any object exposing the same attributes)."""
    rows = list(rows)
    total = len(rows)
    if total == 0:
        return {
            "totalInteractions": 0, "labelledSamples": 0, "neutralRowsExcluded": 0,
            "positiveSamples": 0, "negativeSamples": 0, "positiveRatio": None, "negativeRatio": None,
            "uniqueUsers": 0, "uniqueContents": 0, "uniqueCreators": 0,
            "earliestTimestamp": None, "latestTimestamp": None,
        }
    labels = [target_for(row) for row in rows]
    positive = sum(1 for label in labels if label == 1)
    negative = sum(1 for label in labels if label == 0)
    neutral = sum(1 for label in labels if label is None)
    labelled = positive + negative
    timestamps = [row.timestamp for row in rows if getattr(row, "timestamp", None) is not None]
    return {
        "totalInteractions": total, "labelledSamples": labelled, "neutralRowsExcluded": neutral,
        "positiveSamples": positive, "negativeSamples": negative,
        "positiveRatio": _ratio(positive, labelled), "negativeRatio": _ratio(negative, labelled),
        "uniqueUsers": len({row.user_id for row in rows if getattr(row, "user_id", None)}),
        "uniqueContents": len({row.content_id for row in rows if getattr(row, "content_id", None)}),
        "uniqueCreators": len({row.creator_id for row in rows if getattr(row, "creator_id", None)}),
        "earliestTimestamp": min(timestamps).isoformat() if timestamps else None,
        "latestTimestamp": max(timestamps).isoformat() if timestamps else None,
    }


def live_dataset_summary(dataset: pd.DataFrame, *, is_synthetic: bool) -> dict:
    """`dataset`: the same shape `app.ml.live_trainer.train_live_model` accepts/generates --
    already label-filtered (neutral rows dropped at generation time), with a `target` column
    of 0/1 and (usually) a `timestamp` column."""
    total = len(dataset)
    base = {"isSynthetic": is_synthetic}
    if total == 0 or "target" not in dataset.columns:
        return {
            **base, "totalRows": total, "labelledSamples": 0, "neutralRowsExcluded": 0,
            "positiveSamples": 0, "negativeSamples": 0, "positiveRatio": None, "negativeRatio": None,
            "earliestTimestamp": None, "latestTimestamp": None,
        }
    positive = int((dataset["target"] == 1).sum())
    negative = int((dataset["target"] == 0).sum())
    labelled = positive + negative
    if "timestamp" in dataset.columns and len(dataset["timestamp"]):
        earliest = dataset["timestamp"].min()
        latest = dataset["timestamp"].max()
        earliest = earliest.isoformat() if hasattr(earliest, "isoformat") else str(earliest)
        latest = latest.isoformat() if hasattr(latest, "isoformat") else str(latest)
    else:
        earliest = latest = None
    return {
        **base, "totalRows": total, "labelledSamples": labelled, "neutralRowsExcluded": total - labelled,
        "positiveSamples": positive, "negativeSamples": negative,
        "positiveRatio": _ratio(positive, labelled), "negativeRatio": _ratio(negative, labelled),
        "earliestTimestamp": earliest, "latestTimestamp": latest,
    }
