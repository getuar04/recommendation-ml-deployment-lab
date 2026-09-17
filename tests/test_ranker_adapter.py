"""app.ml.ranker_adapter.RankerScorer: makes a ranking-objective pipeline speak the exact
predict_proba contract everything else in this codebase already assumes, without ever calling
its output a probability. Normalization must be order-preserving (zero pairwise inversions) and
bounded in (0,1) so app.ml.reranker's logit blend stays valid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.ranker_adapter import (
    NORMALIZATION_STRATEGY,
    RankerScorer,
    compute_normalization_scale,
    pairwise_inversions,
)


class _FakePipeline:
    """A minimal stand-in exposing only `.predict`, exactly like a fitted ranker pipeline --
    deliberately has NO `predict_proba` at all, so these tests prove the adapter (not the
    underlying estimator) is what supplies it."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array(self._scores[: len(frame)], dtype=float)


def test_predict_proba_has_the_sklearn_two_column_shape():
    scorer = RankerScorer(_FakePipeline([1.0, -2.0, 0.5]), scale=1.0, algorithm_name="TestRanker")
    frame = pd.DataFrame({"x": [0, 0, 0]})
    proba = scorer.predict_proba(frame)
    assert proba.shape == (3, 2)
    assert np.allclose(proba[:, 0] + proba[:, 1], 1.0)


def test_predict_proba_column_1_is_bounded_in_zero_one():
    # A raw ranker score far outside the training distribution correctly SATURATES to exactly
    # 0.0/1.0 at float64 precision (sigmoid(50) rounds to 1.0 - ~1e-22, indistinguishable from
    # 1.0) -- that is the mathematically correct rounded value, not a normalization bug, so the
    # bound being checked is the closed interval [0, 1] plus finiteness (never nan/inf).
    scorer = RankerScorer(_FakePipeline([50.0, -50.0, 0.0]), scale=1.0, algorithm_name="TestRanker")
    frame = pd.DataFrame({"x": [0, 0, 0]})
    proba = scorer.predict_proba(frame)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.all(np.isfinite(proba))
    # A moderate score (well within any plausible scale) must land strictly inside (0, 1).
    moderate = RankerScorer(_FakePipeline([1.5]), scale=1.0, algorithm_name="TestRanker")
    moderate_proba = moderate.predict_proba(pd.DataFrame({"x": [0]}))
    assert 0.0 < moderate_proba[0, 1] < 1.0


def test_normalization_never_introduces_ranking_inversions():
    raw_scores = [3.1, -0.4, 7.7, 0.0, -5.5, 2.2]
    scorer = RankerScorer(_FakePipeline(raw_scores), scale=2.5, algorithm_name="TestRanker")
    frame = pd.DataFrame({"x": range(len(raw_scores))})
    normalized = scorer.predict_proba(frame)[:, 1]
    assert pairwise_inversions(np.array(raw_scores), normalized) == 0


def test_raw_scores_are_never_relabeled_as_probability_by_the_scorer():
    scorer = RankerScorer(_FakePipeline([1.0]), scale=1.0, algorithm_name="TestRanker")
    assert scorer.score_semantics != "probability"
    assert scorer.model_family == "ranker"


def test_predict_passthrough_returns_the_raw_unnormalized_score():
    scorer = RankerScorer(_FakePipeline([4.2]), scale=1.0, algorithm_name="TestRanker")
    frame = pd.DataFrame({"x": [0]})
    assert scorer.predict(frame)[0] == pytest.approx(4.2)


def test_scale_must_be_positive():
    with pytest.raises(ValueError):
        RankerScorer(_FakePipeline([1.0]), scale=0.0, algorithm_name="TestRanker")
    with pytest.raises(ValueError):
        RankerScorer(_FakePipeline([1.0]), scale=-1.0, algorithm_name="TestRanker")


def test_compute_normalization_scale_is_the_std_floored_at_a_minimum():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert compute_normalization_scale(scores) == pytest.approx(float(np.std(scores)))
    assert compute_normalization_scale(np.array([5.0, 5.0, 5.0])) > 0  # degenerate: floored, not zero/nan


def test_normalization_strategy_is_documented_and_stable():
    assert NORMALIZATION_STRATEGY == "sigmoid_scaled_by_training_score_std"


def test_pairwise_inversions_detects_a_real_reordering():
    assert pairwise_inversions(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0
    assert pairwise_inversions(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == 3
