"""PHASE 2 -- recommendation-quality scoring among ELIGIBLE candidates (Task 5). Answers "among
acceptable models, which produces the best recommendation ranking?" -- kept strictly separate
from PHASE 1 eligibility (app.ml.eligibility_policy), which already decided "acceptable enough
to deploy" using different, non-normalized signals (hard gate pass/fail, critical-constraint
rate). Ineligible candidates never reach this function in production selection (app.ml.
model_selection.select_winner_v2 only scores the eligible pool, or -- the no-eligible-candidate
fallback -- every candidate, exactly mirroring the old policy's fallback semantics).

Re-evaluated weighting (Task 5 spec: do not reuse the old 0.35/0.25/0.25/0.15 NDCG/Precision/
PR-AUC/Recall blend blindly). End-to-end reranked ranking quality now dominates -- it measures
what a real user's feed actually looks like, unlike the old blend's modelSelection-split binary
metrics, which never saw the reranker at all. HARD/ADVERSARIAL matter more than MEDIUM (spec:
"HARD and ADVERSARIAL should matter more than EASY" -- EASY is excluded entirely, see app.ml.
candidate_evaluation.SELECTION_DIFFICULTIES). PR-AUC remains useful (threshold-independent,
appropriate for the imbalanced target) but no longer dominates a feed-ranking evaluation.
`accuracy`/F1 are deliberately excluded (spec: "Do not include accuracy as a major selection
signal"); Precision@10/Recall@10 (from the SAME modelSelection-split evaluation the old policy
used) are kept as small secondary ranking-quality signals, not the headline.
"""
from __future__ import annotations

from typing import Any

QUALITY_WEIGHTS: dict[str, float] = {
    "hardFinalNdcg": 0.30,
    "adversarialFinalNdcg": 0.20,
    "criticalPassRate": 0.20,
    "mediumFinalNdcg": 0.15,
    "prAuc": 0.10,
    "precisionAt10": 0.025,
    "recallAt10": 0.025,
}

if round(sum(QUALITY_WEIGHTS.values()), 9) != 1.0:
    raise AssertionError("QUALITY_WEIGHTS must sum to 1.0")


def quality_score(comparison_metrics: dict[str, Any], candidate_eval: dict[str, Any]) -> dict[str, Any]:
    """`comparison_metrics`: one candidate's app.ml.evaluator.evaluate(...) result (the
    modelSelection-split binary/ranking metrics already computed during fitting).
    `candidate_eval`: app.ml.candidate_evaluation.evaluate_candidate(...)'s return value.

    Returns {"score": float, "components": {...}, "weights": dict(QUALITY_WEIGHTS)} -- the
    exact per-component breakdown is always persisted (Task 5 spec: never collapse selection
    into one opaque number; a future developer must be able to see why a candidate won)."""
    by_difficulty = candidate_eval["endToEnd"]["byDifficulty"]
    critical_passed = candidate_eval["endToEnd"]["criticalPassed"]
    critical_total = candidate_eval["endToEnd"]["criticalTotal"]

    components = {
        "hardFinalNdcg": by_difficulty.get("HARD", {}).get("finalNdcgAt10", 0.0),
        "adversarialFinalNdcg": by_difficulty.get("ADVERSARIAL", {}).get("finalNdcgAt10", 0.0),
        "criticalPassRate": (critical_passed / critical_total) if critical_total else 0.0,
        "mediumFinalNdcg": by_difficulty.get("MEDIUM", {}).get("finalNdcgAt10", 0.0),
        "prAuc": float(comparison_metrics.get("prAuc") or 0.0),
        "precisionAt10": float(comparison_metrics.get("precisionAt10") or 0.0),
        "recallAt10": float(comparison_metrics.get("recallAt10") or 0.0),
    }
    score = sum(QUALITY_WEIGHTS[name] * components[name] for name in QUALITY_WEIGHTS)
    return {"score": round(score, 6), "components": {k: round(v, 6) for k, v in components.items()},
            "weights": dict(QUALITY_WEIGHTS)}


# Task 7 (ranking-native challenger experiment): classification metrics (PR-AUC/Precision@10/
# Recall@10, all from a modelSelection-split BINARY evaluation) are not meaningful for a raw
# ranking-objective model's output and must not be forced to 0.0 -- that would structurally
# undercount a ranker by up to RANKER_QUALITY_WEIGHTS's excluded 0.15 weight for a reason
# having nothing to do with its actual ranking quality (spec section 18: "Selection metadata
# must clearly indicate unavailable/non-applicable metrics", not silently zero them). This
# renormalizes QUALITY_WEIGHTS's four END_TO_END-only components (the ones that ARE meaningful
# for any model, classifier or ranker) to sum to 1.0 on their own.
_RANKER_APPLICABLE = ("hardFinalNdcg", "adversarialFinalNdcg", "criticalPassRate", "mediumFinalNdcg")
_RANKER_APPLICABLE_WEIGHT_SUM = sum(QUALITY_WEIGHTS[name] for name in _RANKER_APPLICABLE)
RANKER_QUALITY_WEIGHTS: dict[str, float] = {
    name: round(QUALITY_WEIGHTS[name] / _RANKER_APPLICABLE_WEIGHT_SUM, 6) for name in _RANKER_APPLICABLE
}
if round(sum(RANKER_QUALITY_WEIGHTS.values()), 6) != 1.0:
    raise AssertionError("RANKER_QUALITY_WEIGHTS must sum to 1.0")


def quality_score_for_ranker(candidate_eval: dict[str, Any]) -> dict[str, Any]:
    """Same END_TO_END components as `quality_score`, reweighted over just the four that apply
    to any model regardless of training objective (see RANKER_QUALITY_WEIGHTS above). NOT
    directly comparable in absolute magnitude to a classifier's `quality_score` (different
    weight normalization) -- comparable only in relative ranking-quality terms, which is what
    this task's central question actually needs."""
    by_difficulty = candidate_eval["endToEnd"]["byDifficulty"]
    critical_passed = candidate_eval["endToEnd"]["criticalPassed"]
    critical_total = candidate_eval["endToEnd"]["criticalTotal"]
    components = {
        "hardFinalNdcg": by_difficulty.get("HARD", {}).get("finalNdcgAt10", 0.0),
        "adversarialFinalNdcg": by_difficulty.get("ADVERSARIAL", {}).get("finalNdcgAt10", 0.0),
        "criticalPassRate": (critical_passed / critical_total) if critical_total else 0.0,
        "mediumFinalNdcg": by_difficulty.get("MEDIUM", {}).get("finalNdcgAt10", 0.0),
    }
    score = sum(RANKER_QUALITY_WEIGHTS[name] * components[name] for name in RANKER_QUALITY_WEIGHTS)
    return {"score": round(score, 6), "components": {k: round(v, 6) for k, v in components.items()},
            "weights": dict(RANKER_QUALITY_WEIGHTS),
            "notApplicable": ["prAuc", "precisionAt10", "recallAt10"]}


# XGBRanker promotion task: `quality_score`'s FULL 7-component score and `quality_score_for_
# ranker`'s 4-component score are NOT directly comparable in absolute magnitude (the docstring
# above already says so) -- the 4 END_TO_END components are worth only 0.85 of a classifier's
# total but 1.0 of a ranker's, so comparing those two numbers directly structurally inflates
# whichever family lacks prAuc/precisionAt10/recallAt10, regardless of true relative quality.
# `quality_score_for_ranker` never actually reads `comparison_metrics` -- it is already a pure
# function of `candidate_eval["endToEnd"]`, which every candidate (classifier or ranker) has,
# since both go through the identical `app.ml.candidate_evaluation.evaluate_candidate` benchmark
# pipeline. Reusing it unmodified, for BOTH families, is therefore the smallest principled fix:
# no new weights invented, no existing weights retuned, and the SAME formula applied identically
# regardless of family is what makes the comparison fair. Each candidate's own family-native
# `quality_score`/`quality_score_for_ranker` result (with prAuc/etc. where applicable) remains
# in its metadata as a diagnostic; only cross-family SELECTION uses this score.
quality_score_cross_family = quality_score_for_ranker
