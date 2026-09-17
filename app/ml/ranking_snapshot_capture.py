"""Pure construction boundary connecting real serving-time scoring output to
`app.ml.ranking_snapshot.RankingSnapshotBuilder`.

No I/O, no database access: this module only reads the SAME `scored`/`chosen` dict items
`app.services.recommendation_service.recommend()` already builds from real model scoring and
`app.ml.reranker.rerank()` output, and turns each ranked candidate into an immutable
`RankedCandidateSnapshot`. Never re-scores, never re-ranks, never mutates its inputs -- what a
caller does with the returned list (persist it, discard it, hold it in memory for a test) is
entirely the caller's decision; see `app.ml.ranking_snapshot_repository` for that boundary.

`internal_correlation_id` is NEVER an authoritative business/training request id. RMS does not
own that identity today -- there is no confirmed Feed/Candidate/Event-Tracking contract that
assigns RMS a real feed-request, served-slate, or impression id (see CLAUDE real-data
integration prep, sections 4/5). At most this is the transport-level `X-Request-ID`
(`app.core.request_context`) this call happened to run under: a log-correlation value only,
never trusted, never authorization, never guaranteed stable across retries. It is prefixed
with `INTERNAL_CORRELATION_ONLY_PREFIX` specifically so a stored/logged snapshot can never be
mistaken for one keyed by a real training group id. Passing `internal_correlation_id=None`
(every caller that does not have a real HTTP request context -- offline benchmarks, the
existing test suite, direct `recommend()` callers) skips capture entirely: nothing is built,
nothing is recorded, and the caller's behavior is byte-identical to before this module existed.

Model-score staging note (do not add a fake stage): production has exactly ONE model-output
value per candidate today (`app.ml.predictor.probabilities`, an sklearn-style `predict_proba`
call) -- there is no separate pre-calibration "raw" score distinct from what serving actually
uses. `raw_model_score`/`normalized_model_score` below therefore intentionally carry the
IDENTICAL value; inventing a fabricated second stage would violate the "do not pretend a stage
exists" requirement this module was built against. Likewise, `final_score` below is exactly
the business/reranker-adjusted score `app.ml.reranker.rerank()` already computes (`adjusted_score`)
-- there is no further rescoring after the diversity-constrained slate selection, only
reordering, so a separate "business-adjusted score" field would just duplicate `final_score`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from app.ml.ranking_snapshot import RankedCandidateSnapshot, RankingSnapshotBuilder

INTERNAL_CORRELATION_ONLY_PREFIX = "internal-correlation-only:"


def build_ranking_decision_snapshots(
    *,
    internal_correlation_id: str | None,
    ranking_timestamp: datetime,
    model_version: str,
    feature_names: Sequence[str],
    user_snapshot: Mapping[str, Any],
    pre_rerank_order: Sequence[Mapping[str, Any]],
    final_order: Sequence[Mapping[str, Any]],
) -> list[RankedCandidateSnapshot]:
    """`pre_rerank_order` is `recommend()`'s own `scored` list (model-score-descending, before
    `app.ml.reranker.rerank()` runs); `final_order` is `rerank()`'s own `chosen` return value,
    in final slate order. Each item in both is the exact dict `recommend()` already holds
    (`candidate`/`features`/`model_score`, plus `adjusted_score` on `final_order` items) --
    never a re-derived or re-fetched copy.

    Returns `[]` (no construction attempted at all) when `internal_correlation_id` is falsy --
    see module docstring."""
    if not internal_correlation_id:
        return []
    correlation_id = f"{INTERNAL_CORRELATION_ONLY_PREFIX}{internal_correlation_id}"
    artifact_metadata = {"modelVersion": model_version, "featureNames": list(feature_names)}
    pre_rerank_rank_by_content_id = {
        item["candidate"].content_id: rank for rank, item in enumerate(pre_rerank_order, 1)
    }
    snapshots: list[RankedCandidateSnapshot] = []
    for final_rank, item in enumerate(final_order, 1):
        candidate = item["candidate"]
        # FeatureHistory.features() returns FEATURES plus several diagnostic-only extras
        # (has_category_history, category_interaction_count, ... -- see
        # app.ml.dataset_builder.NUMERIC's own comments on each) that were evaluated and
        # deliberately excluded from the active model contract. RankingSnapshotBuilder enforces
        # an EXACT feature-name match against that contract, so only the FEATURES subset is
        # ever passed through -- never the full diagnostic dict, and never a silently-widened
        # snapshot schema.
        features = item["features"]
        active_features = {name: features[name] for name in feature_names}
        model_score = float(item["model_score"])
        snapshots.append(RankingSnapshotBuilder.build(
            request_id=correlation_id,
            ranking_timestamp=ranking_timestamp,
            artifact_metadata=artifact_metadata,
            user_snapshot=user_snapshot,
            candidate_snapshot={
                "contentId": candidate.content_id,
                "creatorId": candidate.creator_id,
                # Must match `features["category"]` exactly (RankingSnapshotBuilder enforces
                # this) -- FeatureHistory.features() upper-cases category internally, so the
                # already-computed feature value is used here rather than the candidate's own
                # (possibly differently-cased) request field.
                "category": features["category"],
                "alreadySeen": bool(features.get("already_seen")),
                "candidateSource": getattr(candidate, "candidate_source", None),
            },
            features=active_features,
            raw_model_score=model_score,
            normalized_model_score=model_score,
            pre_rerank_rank=pre_rerank_rank_by_content_id[candidate.content_id],
            final_score=float(item["adjusted_score"]),
            final_rank=final_rank,
        ))
    return snapshots
