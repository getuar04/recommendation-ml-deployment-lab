"""Central, deterministic model-selection policy for VIDEO training (app.ml.trainer).

The served product is a Top-N slate, not a bare binary classification, so ranking/
recommendation quality is weighted ahead of generic classification accuracy: `selection_score`
is a normalized, weighted blend of ranking metrics (NDCG@10, Precision@10, Recall@10) and
PR-AUC (threshold-independent, appropriate for an imbalanced target) -- see SELECTION_WEIGHTS.
Every one of those fields is already normalized to [0, 1] by `app.ml.evaluator.evaluate()`,
so a plain weighted sum needs no additional rescaling.

This module has no opinion on behavioral eligibility -- `app.ml.eligibility` decides which
candidates are even allowed to win; this module only ranks among whatever pool the caller
passes in (the eligible candidates, or -- the no-eligible-candidate fallback -- every trained
candidate; see `app.ml.trainer.train_models`).
"""
from __future__ import annotations

from typing import Any

SELECTION_WEIGHTS: dict[str, float] = {
    "ndcgAt10": 0.35,
    "precisionAt10": 0.25,
    "prAuc": 0.25,
    "recallAt10": 0.15,
}


def selection_score(metrics: dict[str, Any]) -> float:
    return sum(weight * float(metrics.get(name) or 0.0) for name, weight in SELECTION_WEIGHTS.items())


def _tie_break_key(name: str, comparison: dict[str, dict[str, Any]], scores: dict[str, float]) -> tuple[Any, ...]:
    """Deterministic total order, best-first:

        1. higher selection_score
        2. higher prAuc (diagnostic in its own right, and the sharpest single tiebreaker
           between two candidates whose blended score happens to land close together)
        3. lower logLoss (calibration quality)
        4. shorter trainingDurationSeconds -- a simpler/faster model wins when quality is
           otherwise indistinguishable (see app.ml.trainer, which records this per candidate)
        5. algorithm name, alphabetically -- final, always-deterministic fallback.
    """
    metrics = comparison[name]
    log_loss = metrics.get("logLoss")
    return (
        -scores[name],
        -float(metrics.get("prAuc") or 0.0),
        float(log_loss) if log_loss is not None else float("inf"),
        float(metrics.get("trainingDurationSeconds") or 0.0),
        name,
    )


def select_winner(candidate_names: list[str], comparison: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pick the winner among `candidate_names` using `selection_score`, with the fully
    deterministic tie-break chain documented in `_tie_break_key`.

    Returns:
        {
          "selected": winning algorithm name,
          "scores": {name: selection_score, ...},
          "rankedOrder": [name, ...] best-first, the exact order this policy ranked every
              candidate in `candidate_names`,
          "weights": the SELECTION_WEIGHTS blend used,
        }
    -- enough to fully explain why a given model won.

    Kept unchanged for `app.ml.trainer.train_models`'s single-restricted-algorithm path (where
    exactly one candidate is ever in the pool, so this scoring choice cannot change the
    outcome) and any caller still comparing on modelSelection-split metrics alone. Production,
    multi-candidate selection now uses `select_winner_v2` below -- see that function's
    docstring for why (app.ml.quality_scorer/app.ml.eligibility_policy).
    """
    if not candidate_names:
        raise ValueError("select_winner requires at least one candidate name.")
    scores = {name: selection_score(comparison[name]) for name in candidate_names}
    ranked = sorted(candidate_names, key=lambda name: _tie_break_key(name, comparison, scores))
    return {"selected": ranked[0], "scores": scores, "rankedOrder": ranked, "weights": dict(SELECTION_WEIGHTS)}


def _tie_break_key_v2(
    name: str, comparison: dict[str, dict[str, Any]], quality_scores: dict[str, dict[str, Any]],
) -> tuple[Any, ...]:
    """Same deterministic shape as `_tie_break_key`, but ranked by PHASE 2 qualityScore
    (app.ml.quality_scorer) instead of the old modelSelection-split-only `selection_score`."""
    metrics = comparison[name]
    log_loss = metrics.get("logLoss")
    return (
        -quality_scores[name]["score"],
        -float(metrics.get("prAuc") or 0.0),
        float(log_loss) if log_loss is not None else float("inf"),
        float(metrics.get("trainingDurationSeconds") or 0.0),
        name,
    )


def select_winner_v2(
    candidate_names: list[str],
    eligibility_decisions: dict[str, dict[str, Any]],
    quality_scores: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Layered selection (Task 5), two phases:

        PHASE 1 (app.ml.eligibility_policy.decide, already computed by the caller into
            `eligibility_decisions`): restrict the pool to eligible candidates. If NO
            candidate is eligible, fall back to ranking every candidate anyway (so
            training/diagnostics can still complete) -- exactly mirroring `select_winner`'s
            existing no-eligible-candidate fallback semantics -- but `eligibleSelection=False`
            is returned so the caller can still refuse to promote it.
        PHASE 2 (app.ml.quality_scorer, already computed by the caller into `quality_scores`):
            rank the eligible pool by qualityScore, with the same style of fully deterministic
            tie-break chain `select_winner` already uses (see `_tie_break_key_v2`).

    Returns everything needed to audit why a candidate won without hiding disagreement (Task 5
    spec): {"selected", "eligibleSelection", "eligibleCandidates", "rankedOrder",
    "qualityScores", "eligibilityDecisions"}.
    """
    if not candidate_names:
        raise ValueError("select_winner_v2 requires at least one candidate name.")
    eligible_names = [name for name in candidate_names if eligibility_decisions[name]["eligible"]]
    eligible_selection = bool(eligible_names)
    pool = eligible_names if eligible_selection else list(candidate_names)
    ranked = sorted(pool, key=lambda name: _tie_break_key_v2(name, comparison, quality_scores))
    return {
        "selected": ranked[0],
        "eligibleSelection": eligible_selection,
        "eligibleCandidates": eligible_names,
        "rankedOrder": ranked,
        "qualityScores": {name: quality_scores[name]["score"] for name in pool},
        "eligibilityDecisions": eligibility_decisions,
    }
