"""Offline pre-/post-reranking diagnostics.

Model metrics in `app.ml.evaluator` are computed on raw model probabilities,
before the business reranking (`app.ml.reranker.rerank`) that production
actually serves. This module measures the relevance/diversity trade-off that
reranking introduces, by running the exact same `rerank()` function used
online over each evaluation candidate group and comparing Top-K composition
before and after.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from app.ml.reranker import rerank


def _concentration(values: list[str]) -> float:
    """Herfindahl-style concentration in (0, 1]; 1.0 means a single value fills the slate."""
    if not values:
        return 0.0
    counts = Counter(values)
    n = len(values)
    return float(sum((count / n) ** 2 for count in counts.values()))


def _top_k_diagnostics(items: list[dict[str, Any]], k: int) -> dict[str, Any]:
    top = items[:k]
    categories = [item["candidate"].category for item in top]
    creators = [item["candidate"].creator_id for item in top]
    denom = max(1, len(top))
    return {
        "uniqueCategoryCount": len(set(categories)),
        "uniqueCreatorCount": len(set(creators)),
        "seenContentRate": sum(bool(item["features"]["already_seen"]) for item in top) / denom,
        "categoryConcentration": _concentration(categories),
        "creatorConcentration": _concentration(creators),
        "precisionAtK": sum(int(item["target"]) for item in top) / denom,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {key: round(float(np.mean([row[key] for row in rows])), 6) for key in rows[0]}


def evaluate_reranking(
    df: pd.DataFrame, probability: np.ndarray, *, k: int = 10, group_col: str = "candidate_group",
) -> dict[str, Any]:
    """Compare Top-K slate composition before vs. after production reranking, averaged per candidate group.

    Requires `df` to carry `category`, `creator_id`, `content_id`, `content_popularity_score`,
    `already_seen` and `target` columns, as produced by `app.ml.dataset_builder.build_dataset`.
    """
    working = df.reset_index(drop=True)
    pre_rows, post_rows = [], []
    groups_evaluated = 0
    for _, group_df in working.groupby(group_col, sort=False):
        if len(group_df) < 2:
            continue
        groups_evaluated += 1
        scored = [
            {
                "candidate": SimpleNamespace(
                    content_id=row.content_id, category=row.category, creator_id=row.creator_id,
                    content_popularity_score=row.content_popularity_score,
                ),
                "features": {"already_seen": row.already_seen},
                "model_score": float(probability[index]),
                "target": row.target,
            }
            for index, row in zip(group_df.index, group_df.itertuples())
        ]
        scored.sort(key=lambda item: item["model_score"], reverse=True)
        pre_rows.append(_top_k_diagnostics(scored, k))
        reranked = rerank([dict(item) for item in scored], k)
        post_rows.append(_top_k_diagnostics(reranked, k))

    return {
        "evaluatedGroups": groups_evaluated,
        "k": k,
        "preRerank": _aggregate(pre_rows),
        "postRerank": _aggregate(post_rows),
    }
