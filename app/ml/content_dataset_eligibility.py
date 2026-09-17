"""Training/evaluation eligibility policy for `app.ml.content_dataset_schema.DatasetRecord`.

This module answers exactly two questions per record -- is it eligible for classifier
TRAINING, and is it eligible for classifier QUALITY EVALUATION (the "gold" set) -- and does
NOT train, score, or select any actual classifier. See the dataset-foundation task's Part 14
(eligibility) and Part 15 (gold-evaluation guardrail); the guardrail requirement is enforced
by `select_gold_evaluation_set` below plus this module's own tests.

Policy (by `LabelSource`):
    HUMAN_REVIEWED   -- training + gold eligible ONLY once `review_status` is REVIEWED
                        (an unreviewed/disputed human label is not yet trustworthy).
    SYNTHETIC        -- training eligible (development/regression use, e.g. the existing
                        bootstrap dataset), never gold eligible.
    PSEUDO_LABELED   -- training eligible (explicit policy: training-only), never gold
                        eligible, regardless of confidence.
    WEAK_SUPERVISION -- training eligible only when a label_confidence was actually recorded
                        (an unweighted weak label carries no usable signal strength); never
                        gold eligible.

HUMAN_REVIEWED + REVIEWED is the best controlled label source this repository can produce --
it is NOT asserted to be perfect ground truth, just the only source trusted for gold
evaluation.
"""
from __future__ import annotations

from app.ml.content_dataset_schema import DatasetRecord, LabelSource, ReviewStatus


def is_training_eligible(record: DatasetRecord) -> bool:
    if record.label_source is LabelSource.HUMAN_REVIEWED:
        return record.review_status is ReviewStatus.REVIEWED
    if record.label_source is LabelSource.SYNTHETIC:
        return True
    if record.label_source is LabelSource.PSEUDO_LABELED:
        return True
    if record.label_source is LabelSource.WEAK_SUPERVISION:
        return record.label_confidence is not None
    return False


def is_gold_evaluation_eligible(record: DatasetRecord) -> bool:
    """The only path to True: HUMAN_REVIEWED and review_status REVIEWED. Every other source
    -- SYNTHETIC, PSEUDO_LABELED, WEAK_SUPERVISION, or an unreviewed/disputed human label --
    is unconditionally excluded from the gold evaluation set."""
    return record.label_source is LabelSource.HUMAN_REVIEWED and record.review_status is ReviewStatus.REVIEWED


def select_training_set(records: list[DatasetRecord]) -> list[DatasetRecord]:
    return [record for record in records if is_training_eligible(record)]


def select_gold_evaluation_set(records: list[DatasetRecord]) -> list[DatasetRecord]:
    """The explicit guardrail: callers building a production-quality evaluation set must go
    through this function rather than filtering ad hoc, so the exclusion rule lives in one
    place and is covered by this module's own tests (synthetic/pseudo/weak-supervision/
    unreviewed records never appear in the return value)."""
    return [record for record in records if is_gold_evaluation_eligible(record)]
