"""Pure builder: valid, trainable `RealRankingDecision`s -> XGBRanker-compatible rows.

No I/O, no external storage, no model fitting, not wired into any HTTP path or into
`app.ml.trainer`/`app.ml.ranker_trainer`. This is the future real-data builder foundation only
-- see `app.ml.real_ranker_trainer` for the guard that will eventually call this once a real
data source exists.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from app.ml.dataset_builder import FEATURES
from app.ml.real_ranking_decision import ObservationState, RealRankingDecision
from app.ml.real_ranking_group_validation import GroupValidationOutcome, validate_group


@dataclass(frozen=True, slots=True)
class RealRankingDatasetRow:
    """One labeled training row. `features` contains ONLY the exact production feature schema
    (same keys/order as `app.ml.dataset_builder.FEATURES`) -- qid/content_id/relevance/weight are
    kept on separate fields, never mixed into `features`, so a caller cannot accidentally select
    metadata into a model's input matrix."""

    ranking_group_id: str
    ranking_timestamp: str
    content_id: str
    category: str
    features: dict[str, str | float]
    relevance: int
    sample_weight: float | None


@dataclass(frozen=True, slots=True)
class RealRankingDatasetBuildResult:
    rows: tuple[RealRankingDatasetRow, ...]
    # Row-order-aligned group sizes (XGBRanker's `group` fit-parameter convention -- see
    # app.ml.ranking_groups.group_sizes for the same convention on the synthetic path). Rows for
    # the same ranking_group_id are contiguous by construction (one decision at a time, in
    # input order), never interleaved.
    group_sizes: tuple[int, ...]
    ranking_group_ids_in_order: tuple[str, ...]
    excluded_group_ids: tuple[str, ...]
    excluded_group_reasons: dict[str, tuple[str, ...]]


def build_real_ranking_dataset(decisions: Sequence[RealRankingDecision]) -> RealRankingDatasetBuildResult:
    """Excludes (never raises for) any group that is not VALID_TRAINABLE -- tracked in
    `excluded_group_ids`/`excluded_group_reasons` for diagnostics, never silently dropped without
    a reason. Within a trainable group, only RESOLVED candidates become labeled rows: an
    unresolved/censored/not-observed candidate is excluded from the fit, never mapped to
    relevance 0."""
    rows: list[RealRankingDatasetRow] = []
    group_sizes: list[int] = []
    ids_in_order: list[str] = []
    excluded_ids: list[str] = []
    excluded_reasons: dict[str, tuple[str, ...]] = {}

    for decision in decisions:
        result = validate_group(decision)
        if result.outcome is not GroupValidationOutcome.VALID_TRAINABLE:
            excluded_ids.append(decision.ranking_group_id)
            excluded_reasons[decision.ranking_group_id] = result.reasons
            continue

        group_rows: list[RealRankingDatasetRow] = []
        for candidate in decision.candidates:
            if candidate.observation_state is not ObservationState.RESOLVED:
                continue
            # RealCandidateObservation.__post_init__ guarantees resolved_relevance is set
            # whenever observation_state is RESOLVED -- see app.ml.real_ranking_decision.
            assert candidate.resolved_relevance is not None
            group_rows.append(RealRankingDatasetRow(
                ranking_group_id=decision.ranking_group_id,
                ranking_timestamp=decision.ranking_timestamp,
                content_id=candidate.snapshot.content_id,
                category=candidate.snapshot.category,
                # The exact frozen T0 feature vector -- never recomputed from any current/later
                # state (see app.ml.ranking_snapshot.FeatureVectorSnapshot: immutable, and
                # RankingSnapshotBuilder enforces the exact FEATURES schema at construction).
                features=candidate.snapshot.features.as_dict(),
                relevance=candidate.resolved_relevance,
                sample_weight=candidate.resolved_weight,
            ))
        rows.extend(group_rows)
        group_sizes.append(len(group_rows))
        ids_in_order.append(decision.ranking_group_id)

    return RealRankingDatasetBuildResult(
        rows=tuple(rows), group_sizes=tuple(group_sizes), ranking_group_ids_in_order=tuple(ids_in_order),
        excluded_group_ids=tuple(excluded_ids), excluded_group_reasons=excluded_reasons,
    )


def to_split_ready_frame(result: RealRankingDatasetBuildResult) -> pd.DataFrame:
    """Adapts `result.rows` into the exact `candidate_group`/`timestamp` column shape
    `app.ml.splitting.chronological_group_split` already expects -- reuses the existing
    group-preserving chronological splitter instead of building a second one. Feature columns
    follow the exact `app.ml.dataset_builder.FEATURES` name/order."""
    return pd.DataFrame([
        {
            "candidate_group": row.ranking_group_id,
            "timestamp": pd.Timestamp(row.ranking_timestamp),
            "content_id": row.content_id,
            "relevance": row.relevance,
            "sample_weight": row.sample_weight,
            **{name: row.features[name] for name in FEATURES},
        }
        for row in result.rows
    ])
