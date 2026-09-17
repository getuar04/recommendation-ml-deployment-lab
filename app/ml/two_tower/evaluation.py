"""Retrieval evaluation (Step 7): Recall@K / HitRate@K, NOT a ranker-comparison metric.

Two-Tower is evaluated as a retrieval model here -- "can it find the one item the user
actually positively interacted with, somewhere in the Top-K, out of the WHOLE content
catalog" -- never against LogisticRegression/RandomForest PR-AUC, which answers a different
question (given a candidate, how well-calibrated is its score) for a different stage of the
pipeline (ranking a small pre-filtered candidate list, not searching the full catalog).

Each eval example has exactly one relevant (positive) target content per query in this PoC, so
Recall@K and HitRate@K are numerically identical here (both = fraction of queries whose target
appears in the Top-K) -- reported as one metric, not two, to avoid implying a difference that
does not exist for this evaluation design. Precision@K is not reported for the same reason: with
exactly one relevant item, Precision@K = Recall@K / K, which is not a separate, meaningful
signal.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from app.ml.two_tower.dataset import Example
from app.ml.two_tower.model import TwoTowerModel
from app.ml.two_tower.retrieval import ContentIndex, embed_user, retrieve_top_k


@dataclass
class RetrievalMetrics:
    k_values: list[int]
    recall_at_k: dict[int, float]
    queries_evaluated: int


def evaluate_recall_at_k(
    model: TwoTowerModel, eval_examples: list[Example], index: ContentIndex, k_values: list[int],
) -> RetrievalMetrics:
    """For every POSITIVE eval example (label == 1), retrieves Top-max(k_values) content ids for
    that example's own (point-in-time) user embedding and checks whether the example's own
    target content_id appears within each K. Negative eval examples are not queries here (there
    is no "relevant item" to recall for a disliked interaction) -- they exist in the labeled
    dataset for the training objective, not for this retrieval metric."""
    positives = [ex for ex in eval_examples if ex.label == 1]
    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    for ex in positives:
        user_embedding = embed_user(model, ex.user_vector)
        ranked = retrieve_top_k(user_embedding, index, max_k)
        ranked_ids = [cid for cid, _score in ranked]
        for k in k_values:
            if ex.content_id in ranked_ids[:k]:
                hits[k] += 1
    n = len(positives)
    recall = {k: (hits[k] / n if n else 0.0) for k in k_values}
    return RetrievalMetrics(k_values=k_values, recall_at_k=recall, queries_evaluated=n)


@dataclass
class FullRetrievalReport:
    """Extends RetrievalMetrics (Recall@K) with the ranking-quality/diversity metrics this
    module previously lacked (see scripts/run_two_tower_offline_validation.py, which had
    prototyped the same computations ad hoc, outside this reusable module -- consolidated
    here so any caller gets them from one tested place instead of re-deriving them)."""
    k_values: list[int]
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_10: float
    coverage: float  # fraction of the indexed catalog that ever appeared in a Top-max(k_values) across all queries
    top_item_frequency: list[tuple[str, float]] = field(default_factory=list)  # (content_id, fraction of queries), most frequent first
    queries_evaluated: int = 0


def evaluate_full(
    model: TwoTowerModel, eval_examples: list[Example], index: ContentIndex, k_values: list[int],
    *, top_items_reported: int = 10,
) -> FullRetrievalReport:
    """Recall@K (same definition as evaluate_recall_at_k) plus MRR, NDCG@10, catalog coverage,
    and the most-frequently-retrieved items -- all from ONE ranked retrieval per query (exact
    rank against the FULL indexed catalog, not just Top-max(k_values)), so MRR/NDCG@10 are
    exact, not approximated by truncating to max(k_values).

    NDCG@10 uses IDCG=1 (exactly one relevant item per query, matching this module's existing
    "Recall@K and HitRate@K are numerically identical here" design note), so
    NDCG@10 == DCG@10 == 1/log2(rank+1) when the relevant item's rank <= 10, else 0."""
    positives = [ex for ex in eval_examples if ex.label == 1]
    n = len(positives)
    max_k = max(k_values)
    catalog_size = len(index.content_ids)
    hits = {k: 0 for k in k_values}
    reciprocal_ranks: list[float] = []
    ndcg_sum = 0.0
    retrieved_counts: Counter[str] = Counter()

    for ex in positives:
        user_embedding = embed_user(model, ex.user_vector)
        ranked = retrieve_top_k(user_embedding, index, catalog_size)  # full ranking -> exact rank
        ranked_ids = [cid for cid, _score in ranked]
        retrieved_counts.update(ranked_ids[:max_k])
        for k in k_values:
            if ex.content_id in ranked_ids[:k]:
                hits[k] += 1
        rank = ranked_ids.index(ex.content_id) + 1 if ex.content_id in ranked_ids else None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if rank is not None and rank <= 10:
            ndcg_sum += 1.0 / math.log2(rank + 1)

    recall = {k: (hits[k] / n if n else 0.0) for k in k_values}
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    ndcg10 = ndcg_sum / n if n else 0.0
    coverage = (len(retrieved_counts) / catalog_size) if catalog_size else 0.0
    top_items = [(cid, count / n) for cid, count in retrieved_counts.most_common(top_items_reported)] if n else []

    return FullRetrievalReport(
        k_values=k_values, recall_at_k=recall, mrr=mrr, ndcg_at_10=ndcg10, coverage=coverage,
        top_item_frequency=top_items, queries_evaluated=n,
    )


def cohort_thresholds(prior_interaction_counts: list[int]) -> tuple[int, int]:
    """Deterministic sparse/medium/high cohort split points (33rd/66th percentile of this
    run's own prior-interaction-count distribution at query time) -- never hardcoded
    thresholds, so cohorts always reflect the actual data being evaluated."""
    counts_sorted = sorted(prior_interaction_counts)
    if not counts_sorted:
        return (0, 0)
    p33 = counts_sorted[len(counts_sorted) // 3]
    p66 = counts_sorted[2 * len(counts_sorted) // 3]
    return (p33, p66)


def cohort_of(prior_interaction_count: int, p33: int, p66: int) -> str:
    if prior_interaction_count <= p33:
        return "sparse"
    if prior_interaction_count <= p66:
        return "medium"
    return "high"
