"""Bridges the Content Understanding dataset foundation (`app.ml.content_dataset_schema`/
`content_dataset_io`/`content_dataset_eligibility`) onto the classifier's own training input
shape (a `text`/`category` pandas DataFrame, see `app.ml.content_classifier.train_classifier`)
-- WITHOUT changing the classifier's artifact/model representation at all: same pipeline,
same fixed `CATEGORIES` vocabulary, same feature text encoding.

Release-boundary integration (Task: content classifier release boundary audit): the
classifier training script previously read `app.ml.content_classifier_data`'s synthetic
bootstrap dataset directly, with no path for real HUMAN_REVIEWED/PSEUDO_LABELED/
WEAK_SUPERVISION data to ever reach it, even once the real dataset foundation stopped being
empty. This module is the missing structural integration point: it reads real
`DatasetRecord`s (when any exist), narrows them to training-eligible
(`app.ml.content_dataset_eligibility.select_training_set`, reused verbatim -- never a second,
differently-behaved eligibility policy) AND taxonomy-compatible (`_is_taxonomy_compatible`)
records, then combines them with the synthetic bootstrap dataset -- always producing an
HONEST `dataset_source` composition (real record counts by `LabelSource`, whether synthetic
was included) to attach to the trained artifact's metadata via
`app.ml.content_classifier.train_classifier`'s own `dataset_source` parameter. It never
claims synthetic (or synthetic-augmented) holdout accuracy as production-quality evidence --
`dataset_source["productionQualityEvidence"]` is unconditionally `False` here; only a
dedicated, deliberate real-evaluation pipeline against the HUMAN_REVIEWED gold set
(`app.ml.content_dataset_eligibility.select_gold_evaluation_set`) could ever justify `True`,
and that pipeline does not exist yet (a separate, future task).

Currently a no-op in practice: `data/content_understanding/real/dataset.jsonl` has 0 records
(see its own `manifest.json`), so `build_training_dataframe()` today returns output
byte-identical to calling `app.ml.content_classifier_data.build_dataframe()` directly --
this module changes nothing about what actually gets trained until real data exists. Does
NOT retrain anything itself and is not invoked by anything automatically; only
`scripts/train_content_classifier.py` calls it, exactly as before.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import CONTENT_UNDERSTANDING_REAL_DATASET_PATH
from app.ml.content_classifier import (
    CATEGORIES,
    CATEGORY_TAXONOMY_VERSION,
    feature_text,
)
from app.ml.content_dataset_eligibility import select_training_set
from app.ml.content_dataset_io import load_jsonl
from app.ml.content_dataset_schema import DatasetRecord, LabelSource

__all__ = ["build_training_dataframe"]


def _is_taxonomy_compatible(record: DatasetRecord) -> bool:
    """A `DatasetRecord.primary_category` is a free string under WHATEVER `taxonomy_version`
    it was labeled with (the dataset foundation is deliberately taxonomy-agnostic -- no
    taxonomy is product-approved yet, see `data/content_understanding/README.md`). The
    classifier, however, is a FIXED multi-class model over exactly `CATEGORIES`
    (`CATEGORY_TAXONOMY_VERSION`) -- a record labeled under a different taxonomy version
    (e.g. a future, product-approved one) is never silently coerced into this one; it is
    simply not usable as a training example for the CURRENT classifier artifact until it is
    re-labeled, or this classifier is deliberately retrained against the new taxonomy (a
    separate, future decision -- see this module's own docstring)."""
    return (
        record.taxonomy_version == CATEGORY_TAXONOMY_VERSION
        and record.primary_category is not None
        and record.primary_category.strip().upper() in CATEGORIES
    )


def _rows_from_records(records: list[DatasetRecord]) -> list[dict[str, Any]]:
    return [
        {"text": feature_text(record.title, record.hashtags), "category": (record.primary_category or "").strip().upper()}
        for record in records
    ]


def build_training_dataframe(
    *, real_dataset_path: Path | None = None, allow_synthetic_fallback: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Returns `(dataframe, dataset_source)`.

    `dataframe` has the same `text`/`category` columns `train_classifier` has always
    required. `dataset_source` is an honest, computed-from-the-actual-input composition
    dict, meant to be passed straight through to `train_classifier(..., dataset_source=...)`.

    Real records are loaded from `real_dataset_path` (default
    `CONTENT_UNDERSTANDING_REAL_DATASET_PATH`), then narrowed to training-eligible AND
    taxonomy-compatible records (see module docstring).

    `allow_synthetic_fallback=True` (default -- matches today's only real caller,
    `scripts/train_content_classifier.py`, which must keep working for local development
    regardless of real-dataset state): the synthetic bootstrap dataset
    (`app.ml.content_classifier_data.build_dataframe`) is ALWAYS additionally included --
    real rows augment it, never silently replace it, until a deliberate future policy
    decision changes that. Set `allow_synthetic_fallback=False` to train on real rows only
    (raises `ValueError` when none are eligible -- never silently falls through to an empty
    dataset)."""
    from app.ml.content_classifier_data import build_dataframe as _synthetic_dataframe

    path = real_dataset_path or CONTENT_UNDERSTANDING_REAL_DATASET_PATH
    load_result = load_jsonl(path)
    eligible_real = [record for record in select_training_set(load_result.records) if _is_taxonomy_compatible(record)]

    dataset_source: dict[str, Any] = {
        "realDatasetPath": str(path),
        "realRecordsLoaded": len(load_result.records),
        "realRecordsLoadErrors": len(load_result.errors),
        "realRecordsTrainingEligible": len(eligible_real),
        "realRecordsBySource": dict(Counter(record.label_source.value for record in eligible_real)),
        "syntheticBootstrapIncluded": False,
    }

    if not eligible_real and not allow_synthetic_fallback:
        raise ValueError(
            f"No training-eligible, taxonomy-compatible real records found at {path!s} "
            "and allow_synthetic_fallback=False."
        )

    frames: list[pd.DataFrame] = []
    if eligible_real:
        frames.append(pd.DataFrame(_rows_from_records(eligible_real)))
    if allow_synthetic_fallback:
        frames.append(_synthetic_dataframe())
        dataset_source["syntheticBootstrapIncluded"] = True

    dataset = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    dataset_source["totalTrainingRows"] = len(dataset)
    dataset_source["containsAnyRealData"] = bool(eligible_real)
    dataset_source["containsAnyHumanReviewedData"] = any(
        record.label_source is LabelSource.HUMAN_REVIEWED for record in eligible_real
    )
    # Never auto-claimed True here -- see module docstring for exactly what would justify it
    # (a dedicated gold-evaluation pipeline, which does not exist yet).
    dataset_source["productionQualityEvidence"] = False
    dataset_source["note"] = (
        "productionQualityEvidence is always False from this function: a real-augmented or "
        "synthetic-only training set's own holdout accuracy is never sufficient production "
        "evidence on its own. See realRecordsBySource/containsAnyHumanReviewedData for the "
        "actual composition; a real quality claim requires a dedicated evaluation against "
        "app.ml.content_dataset_eligibility.select_gold_evaluation_set (not implemented yet)."
    )
    return dataset, dataset_source
