"""Score adapter: makes a ranking-objective pipeline (app.ml.ranker_registry) speak the exact
`predict_proba(frame) -> ndarray[:, 2]` contract every other part of this codebase already
assumes -- app.ml.eligibility's gates, app.ml.predictor.probabilities (serving), app.benchmark's
raw-score capture, and app.ml.reranker's `_blend_multiplier_in_logit_space` (which specifically
requires `0 < model_score < 1`) all call `model.predict_proba(frame)[:, 1]` and nothing else.
Wrapping a fitted ranker in `RankerScorer` means every one of those call sites needs ZERO
changes to accept a ranker model -- "the smallest clean adapter necessary" (Task 7 spec section
7/25), not a parallel serving path.

Classifier output is `P(positive engagement)`; a ranker's raw output is an unbounded relative-
relevance score with no probability semantics -- `RankerScorer` never calls it or documents it
as a probability (spec section 8: "Do not label ranker output as probability"). Internally it
is always referred to as `model_score`/`raw_score`, and the [0,1] value it exposes through
`predict_proba` is explicitly a NORMALIZED score, not a calibrated probability -- no
`CalibratedClassifierCV` is ever applied to a ranker (spec section 9).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

NORMALIZATION_STRATEGY = "sigmoid_scaled_by_training_score_std"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipped before exponentiating -- a numerically stable sigmoid, standard practice: an
    # unclipped exp(-x) can overflow (RuntimeWarning, though still numerically correct at the
    # 0.0/1.0 limits) for a raw score many scales away from the training distribution (e.g. an
    # eligibility-gate probe deliberately far from typical training data).
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class RankerScorer:
    """Wraps one FITTED ranking-objective `sklearn.pipeline.Pipeline` (built by
    app.ml.pipeline_builder.build_classifier_pipeline around an XGBRanker/LGBMRanker/
    CatBoostRanker estimator) plus a fixed, deterministic normalization `scale` computed ONCE
    from training data (never from a live request's candidate pool -- spec section 9: pool-
    dependent normalization "changes score semantics request-by-request", explicitly the
    method to avoid). `scale` is the standard deviation of raw scores on the same calibration-
    analogous split a classifier would be calibrated on; dividing by it before the sigmoid
    keeps the transform in a well-behaved input range regardless of a given library's raw-score
    magnitude convention.

    `predict_proba(X)` is STRICTLY monotonic in the raw score (sigmoid composed with a
    positive-scale division is a strictly increasing function) -- see
    tests/test_ranker_adapter.py's zero-inversions proof. Never reorders candidates relative to
    the raw ranker output; only rescales them into (0, 1) so downstream probability-shaped math
    (app.ml.reranker's logit blending, app.ml.eligibility's margin comparisons) stays valid.
    """

    def __init__(self, pipeline: Any, *, scale: float, algorithm_name: str) -> None:
        if scale <= 0:
            raise ValueError(f"RankerScorer scale must be positive (got {scale!r}) -- a degenerate/zero-variance "
                              "training score distribution cannot be normalized meaningfully.")
        self.pipeline = pipeline
        self.scale = float(scale)
        self.algorithm_name = algorithm_name
        self.model_family = "ranker"
        self.score_semantics = "relative_relevance_score"
        self.normalization_strategy = NORMALIZATION_STRATEGY

    def raw_scores(self, frame: pd.DataFrame) -> np.ndarray:
        """The ranker's own unbounded relevance score, unnormalized -- never a probability."""
        return np.asarray(self.pipeline.predict(frame), dtype=float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """`ndarray[:, 2]`, matching sklearn's `predict_proba` shape convention exactly, so
        every existing `model.predict_proba(frame)[:, 1]` call site works unmodified. Column 1
        is the sigmoid-normalized score (NOT a calibrated probability -- see module docstring);
        column 0 is its complement, purely so the 2-column shape holds, never itself meaningful."""
        normalized = _sigmoid(self.raw_scores(frame) / self.scale)
        return np.column_stack([1.0 - normalized, normalized])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Passthrough to the raw ranker score -- kept for callers that want the unnormalized
        ranking score directly (e.g. offline NDCG diagnostics), never used by serving."""
        return self.raw_scores(frame)


def compute_normalization_scale(raw_scores: np.ndarray, *, minimum_scale: float = 1e-6) -> float:
    """Standard deviation of `raw_scores` (typically the calibration-analogous split's scores),
    floored at `minimum_scale` so a degenerate (near-constant) score distribution never divides
    by ~0 and blows up the sigmoid -- the resulting near-flat normalized scores are still a
    faithful (if uninformative) representation of a ranker that produced almost no separation
    on that data, not a crash."""
    std = float(np.std(raw_scores)) if len(raw_scores) else 0.0
    return max(std, minimum_scale)


def pairwise_inversions(scores_a: np.ndarray, scores_b: np.ndarray) -> int:
    """Number of pairs whose RELATIVE ORDER differs between `scores_a` and `scores_b` (same
    length, same row order) -- used to prove normalization introduces zero ranking inversions."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    n = len(a)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (a[i] < a[j]) != (b[i] < b[j]):
                inversions += 1
    return inversions
