"""Graded-relevance ranking metrics for one benchmark scenario (one real candidate group --
never computed over arbitrary unrelated rows the way a naive metric would).

Deliberately separate from app.ml.evaluator, which only ever sees the model's *binary*
training target: a benchmark scenario's truth is a graded relevance judgment (0-4, see
RELEVANCE_SCALE), a different concept entirely from what the model was trained to predict.
NDCG here uses the exact same linear (not exponential 2^rel-1) DCG form app.ml.evaluator
already uses, for one consistent "what does NDCG mean in this codebase" answer. Precision/
Recall/HitRate/MRR treat any candidate at or above RELEVANT_THRESHOLD as "relevant" -- a
documented simplification, not a hidden one.
"""
from __future__ import annotations

import math

RELEVANCE_SCALE = {
    4: "highly relevant -- unseen, strong category+semantic match, current preference",
    3: "relevant -- matches long-term or recent/session preference",
    2: "weakly relevant -- secondary interest or reasonable exploration",
    1: "neutral -- borderline (e.g. popular-but-irrelevant, or otherwise-good-but-seen)",
    0: "irrelevant or explicitly unwanted (e.g. a repeatedly-rejected subtheme)",
}
RELEVANT_THRESHOLD = 2


def _dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def ndcg_at(ranked_relevances: list[int], k: int) -> float:
    ideal = _dcg(sorted(ranked_relevances, reverse=True), k)
    return _dcg(ranked_relevances, k) / ideal if ideal else 0.0


def precision_at(ranked_relevances: list[int], k: int) -> float:
    if not ranked_relevances:
        return 0.0
    top = ranked_relevances[:k]
    relevant = sum(1 for rel in top if rel >= RELEVANT_THRESHOLD)
    return relevant / min(k, len(ranked_relevances))


def recall_at(ranked_relevances: list[int], k: int) -> float:
    total_relevant = sum(1 for rel in ranked_relevances if rel >= RELEVANT_THRESHOLD)
    if not total_relevant:
        return 0.0
    hit = sum(1 for rel in ranked_relevances[:k] if rel >= RELEVANT_THRESHOLD)
    return hit / total_relevant


def hit_rate_at(ranked_relevances: list[int], k: int) -> float:
    total_relevant = sum(1 for rel in ranked_relevances if rel >= RELEVANT_THRESHOLD)
    if not total_relevant:
        return 0.0
    return 1.0 if any(rel >= RELEVANT_THRESHOLD for rel in ranked_relevances[:k]) else 0.0


def mrr(ranked_relevances: list[int]) -> float:
    for index, rel in enumerate(ranked_relevances):
        if rel >= RELEVANT_THRESHOLD:
            return 1.0 / (index + 1)
    return 0.0


def compute_ranking_metrics(ranked_relevances: list[int]) -> dict[str, float]:
    """`ranked_relevances`: the scenario's graded relevance labels, reordered into whatever
    order a ranking stage (raw model, or final reranked) actually produced -- one real
    candidate group per call."""
    return {
        "ndcgAt5": round(ndcg_at(ranked_relevances, 5), 6),
        "ndcgAt10": round(ndcg_at(ranked_relevances, 10), 6),
        "precisionAt5": round(precision_at(ranked_relevances, 5), 6),
        "precisionAt10": round(precision_at(ranked_relevances, 10), 6),
        "recallAt5": round(recall_at(ranked_relevances, 5), 6),
        "recallAt10": round(recall_at(ranked_relevances, 10), 6),
        "hitRateAt5": round(hit_rate_at(ranked_relevances, 5), 6),
        "hitRateAt10": round(hit_rate_at(ranked_relevances, 10), 6),
        "mrr": round(mrr(ranked_relevances), 6),
    }


def diversity_diagnostics(categories: list[str], creators: list[str], k: int = 10) -> dict[str, float | int]:
    """Diagnostic-only diversity signals over the top-`k` of a ranking (see app.benchmark's
    dominant-category scenario) -- never a model-selection metric, just an observability
    check that a legitimately dominant category isn't being pathologically fragmented."""
    top_categories = categories[:k]
    top_creators = creators[:k]
    denom = max(1, len(top_categories))
    top_category_share = (
        max((top_categories.count(c) for c in set(top_categories)), default=0) / denom
    )
    return {
        "topCategoryShare": round(top_category_share, 6),
        "uniqueCategories": len(set(top_categories)),
        "uniqueCreators": len(set(top_creators)),
    }
