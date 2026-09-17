"""Controlled-experiment feature groups for the negative-feedback feature representation task.

Every feature named below is already computed unconditionally by
`app.ml.dataset_builder.FeatureHistory.features()` -- this module adds NO new computation, it
only names which already-computed columns belong to which experimental group, so
`scripts/run_negative_feature_experiment.py` (and tests) can build an alternate `numeric`
feature list for `app.ml.pipeline_builder.build_classifier_pipeline` without a second feature-
computation path. None of these are part of production `app.ml.dataset_builder.NUMERIC`/
`FEATURES` yet -- see that experiment script's report for the evidence gate before permanent
adoption.

    NEG_A: category-level explicit-rejection count + bounded recency-decayed strength.
    NEG_B: semantic-level explicit-rejection match count + the single bounded
        candidate_negative_semantic_match scalar.
    NEG_C: positive-vs-negative balance features -- purely derived from EXISTING
        category_positive_count/category_negative_count/semantic_positive_match_count/
        semantic_negative_match_count, no new FeatureHistory state at all.
"""
from __future__ import annotations

NEG_A: tuple[str, ...] = ("category_explicit_rejection_count", "recent_explicit_rejection_strength")
NEG_B: tuple[str, ...] = ("semantic_explicit_rejection_match_count", "candidate_negative_semantic_match")
NEG_C: tuple[str, ...] = ("category_positive_minus_negative", "semantic_positive_minus_negative")
NEG_V2: tuple[str, ...] = NEG_A + NEG_B + NEG_C

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "BASELINE": (),
    "NEG_A": NEG_A,
    "NEG_B": NEG_B,
    "NEG_C": NEG_C,
    "NEG_V2": NEG_V2,
}
