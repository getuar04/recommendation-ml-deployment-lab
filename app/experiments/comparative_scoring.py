"""Experiment-only raw-score capture for the comparative pipeline (spec section 4/6: preserve
`rawModelScore` -- the model's pre-reranking probability -- separately from the adjusted,
post-reranking score the public API returns).

This module does not reimplement feature construction, prediction, or reranking. It calls the
real, completely unmodified `app.services.recommendation_service.recommend(db, request)` --
the exact function `POST /api/v1/recommendation-ml-service/recommendations` calls -- and observes the return value of
`app.ml.predictor.probabilities(...)` in transit via a temporary wrapper, removed immediately
afterward regardless of outcome. `recommend()` itself is never edited, never monkeypatched,
and produces exactly the response it always would; only one function's return value is
additionally recorded on the way past.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from app.ml import predictor as predictor_module
from app.services import recommendation_service

# recommendation_service.py does `from app.ml.predictor import probabilities` -- a name
# import that binds its OWN module-level reference at import time. Patching
# `app.ml.predictor.probabilities` (the module attribute) would therefore have no effect on
# what `recommend()` actually calls; the reference that must be wrapped is
# `recommendation_service.probabilities` specifically. `predictor_module` is kept imported
# too, purely so `original = recommendation_service.probabilities` below is unambiguously
# understood to be `app.ml.predictor.probabilities`'s real implementation.
assert recommendation_service.probabilities is predictor_module.probabilities


class RawScoreCaptureError(Exception):
    """The number of raw probabilities captured from `app.ml.predictor.probabilities` did not
    match the number of candidates in the request, meaning `recommend()` filtered out or
    deduplicated at least one candidate -- a positional (candidate -> score) mapping would
    silently mis-attribute scores, so this is raised instead. This capture path is only valid
    for well-formed, already-deduplicated candidate lists (e.g.
    `app.experiments.fixed_scenario.PRIMARY_FIXED_CANDIDATES`), which is verified here, not
    assumed."""


@contextmanager
def _capture_raw_probabilities():
    captured: dict[str, Any] = {}
    original = recommendation_service.probabilities

    def _wrapped(model, rows):
        scores = original(model, rows)
        captured["scores"] = scores
        return scores

    recommendation_service.probabilities = _wrapped
    try:
        yield captured
    finally:
        recommendation_service.probabilities = original


def score_with_raw_capture(db, request) -> tuple[dict[str, Any], dict[str, float]]:
    """Calls the real, unmodified `recommendation_service.recommend(db, request)` and returns
    `(response, raw_scores_by_content_id)`.

    `response` is exactly what `recommend()` always returns (userId, modelVersion, strategy,
    interactionCount, recommendations -- the same shape `POST /recommendations` serves).

    `raw_scores_by_content_id` maps each requested candidate's `contentId` to the model's raw,
    pre-reranking probability, in [0, 1]. Raises `RawScoreCaptureError` rather than guessing
    if `recommend()` filtered any candidate out (see that exception's docstring).
    """
    with _capture_raw_probabilities() as captured:
        response = recommendation_service.recommend(db, request)

    scores = captured.get("scores")
    if scores is None:
        # No candidates were scored at all (e.g. every candidate was invalid) -- recommend()
        # already returns an empty recommendations list in that case; nothing to attribute.
        return response, {}
    if len(scores) != len(request.candidates):
        raise RawScoreCaptureError(
            f"recommend() scored {len(scores)} candidates but the request had "
            f"{len(request.candidates)}; refusing to positionally attribute raw scores to "
            f"content IDs. This candidate list must be pre-validated (unique, non-empty "
            f"content_id/category) so recommend()'s internal validity filter never drops "
            f"anything."
        )
    return response, {
        candidate.content_id: float(score)
        for candidate, score in zip(request.candidates, scores)
    }
