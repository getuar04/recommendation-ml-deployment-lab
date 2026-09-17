"""Immutable, persistence-neutral snapshots for future real ranking groups.

This module deliberately has no dependency on RecommendationService or FeatureHistory.
Callers must supply an authoritative request id and both scoring stages explicitly; until
that cross-service contract exists, snapshots are constructed and tested but not persisted.

Serving-time construction: see `app.ml.ranking_snapshot_capture` for the boundary that turns
real `recommend()` scoring output into `RankedCandidateSnapshot` instances, and
`app.ml.ranking_snapshot_repository` for the (currently no-op) storage port. `request_id` on
each snapshot is, today, only ever an internal-correlation-only value (never authoritative --
see that module) -- once a real feed/recommendation-request identity is confirmed, snapshots
sharing the SAME authoritative `request_id` are exactly what a future real
`app.ml.ranking_groups`-style ranking/training group would key on, replacing that module's
synthetic `query_id`. No such grouping exists yet; this is the integration point it will use.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.ml.dataset_builder import FEATURES, NUMERIC

FEATURE_NAMES = tuple(FEATURES)
FEATURE_SCHEMA_VERSION = "video-ranking-features-sha256:" + hashlib.sha256(
    json.dumps(FEATURE_NAMES, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-blank string")
    return value


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive_rank(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _canonical_json(value: Mapping[str, Any], field: str) -> str:
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain finite JSON values") from exc
    return encoded


@dataclass(frozen=True, slots=True)
class FeatureVectorSnapshot:
    """The exact ordered model input, with lossless JSON float round-tripping."""

    schema_version: str
    names: tuple[str, ...]
    values: tuple[str | float, ...]

    @classmethod
    def build(cls, features: Mapping[str, Any]) -> FeatureVectorSnapshot:
        missing = [name for name in FEATURE_NAMES if name not in features]
        extras = [name for name in features if name not in FEATURE_NAMES]
        if missing or extras:
            raise ValueError(f"feature schema mismatch: missing={missing}, extras={extras}")

        category = features["category"]
        if not isinstance(category, str) or not category.strip():
            raise ValueError("feature category must be a non-blank string")
        numeric = tuple(_finite_float(features[name], f"feature {name}") for name in NUMERIC)
        return cls(FEATURE_SCHEMA_VERSION, FEATURE_NAMES, (category, *numeric))

    def as_dict(self) -> dict[str, str | float]:
        return dict(zip(self.names, self.values))

    def to_payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "names": list(self.names),
            "values": list(self.values),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FeatureVectorSnapshot:
        if payload.get("schemaVersion") != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported feature schema version")
        if tuple(payload.get("names", ())) != FEATURE_NAMES:
            raise ValueError("feature names or order do not match the active schema")
        raw_values = payload.get("values")
        if not isinstance(raw_values, list) or len(raw_values) != len(FEATURE_NAMES):
            raise ValueError(f"feature vector must contain exactly {len(FEATURE_NAMES)} values")
        return cls.build(dict(zip(FEATURE_NAMES, raw_values)))


@dataclass(frozen=True, slots=True)
class RankedCandidateSnapshot:
    """One candidate's complete scoring trace at a single recommendation instant."""

    request_id: str
    ranking_timestamp: str
    model_version: str
    user_id: str
    content_id: str
    creator_id: str
    category: str
    user_snapshot_json: str
    candidate_snapshot_json: str
    features: FeatureVectorSnapshot
    raw_model_score: float
    normalized_model_score: float
    pre_rerank_rank: int
    final_score: float
    final_rank: int

    def to_json(self) -> str:
        payload = {
            "requestId": self.request_id,
            "rankingTimestamp": self.ranking_timestamp,
            "modelVersion": self.model_version,
            "userId": self.user_id,
            "contentId": self.content_id,
            "creatorId": self.creator_id,
            "category": self.category,
            "userSnapshot": json.loads(self.user_snapshot_json),
            "candidateSnapshot": json.loads(self.candidate_snapshot_json),
            "features": self.features.to_payload(),
            "rawModelScore": self.raw_model_score,
            "normalizedModelScore": self.normalized_model_score,
            "preRerankRank": self.pre_rerank_rank,
            "finalScore": self.final_score,
            "finalRank": self.final_rank,
        }
        return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, encoded: str) -> RankedCandidateSnapshot:
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise TypeError("ranking snapshot must be a JSON object")
        ranking_timestamp = payload.get("rankingTimestamp")
        user_snapshot = payload.get("userSnapshot")
        candidate_snapshot = payload.get("candidateSnapshot")
        if not isinstance(ranking_timestamp, str):
            raise TypeError("rankingTimestamp must be a string")
        if not isinstance(user_snapshot, dict) or not isinstance(candidate_snapshot, dict):
            raise TypeError("userSnapshot and candidateSnapshot must be objects")
        return RankingSnapshotBuilder.build(
            request_id=payload.get("requestId"),
            ranking_timestamp=ranking_timestamp,
            artifact_metadata={
                "modelVersion": payload.get("modelVersion"),
                "featureNames": payload.get("features", {}).get("names"),
            },
            user_snapshot=user_snapshot,
            candidate_snapshot=candidate_snapshot,
            features=FeatureVectorSnapshot.from_payload(payload.get("features", {})).as_dict(),
            raw_model_score=payload.get("rawModelScore"),
            normalized_model_score=payload.get("normalizedModelScore"),
            pre_rerank_rank=payload.get("preRerankRank"),
            final_score=payload.get("finalScore"),
            final_rank=payload.get("finalRank"),
        )


class RankingSnapshotBuilder:
    """Pure construction boundary; performs no I/O, training, or serving mutation."""

    @staticmethod
    def build(
        *,
        request_id: object,
        ranking_timestamp: datetime | str,
        artifact_metadata: Mapping[str, Any],
        user_snapshot: Mapping[str, Any],
        candidate_snapshot: Mapping[str, Any],
        features: Mapping[str, Any],
        raw_model_score: object,
        normalized_model_score: object,
        pre_rerank_rank: object,
        final_score: object,
        final_rank: object,
    ) -> RankedCandidateSnapshot:
        if tuple(artifact_metadata.get("featureNames", ())) != FEATURE_NAMES:
            raise ValueError("artifact featureNames do not match the snapshot feature schema")
        model_version = _required_text(artifact_metadata.get("modelVersion"), "modelVersion")

        if isinstance(ranking_timestamp, datetime):
            if ranking_timestamp.tzinfo is None or ranking_timestamp.utcoffset() is None:
                raise ValueError("ranking_timestamp must be timezone-aware")
            timestamp = ranking_timestamp.isoformat()
        elif isinstance(ranking_timestamp, str):
            try:
                parsed = datetime.fromisoformat(ranking_timestamp)
            except ValueError as exc:
                raise ValueError("ranking_timestamp must be ISO-8601") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("ranking_timestamp must be timezone-aware")
            timestamp = parsed.isoformat()
        else:
            raise TypeError("ranking_timestamp must be a datetime or ISO-8601 string")

        user_id = _required_text(user_snapshot.get("userId"), "userSnapshot.userId")
        content_id = _required_text(candidate_snapshot.get("contentId"), "candidateSnapshot.contentId")
        creator_id = _required_text(candidate_snapshot.get("creatorId"), "candidateSnapshot.creatorId")
        category = _required_text(candidate_snapshot.get("category"), "candidateSnapshot.category")
        feature_vector = FeatureVectorSnapshot.build(features)
        if feature_vector.as_dict()["category"] != category:
            raise ValueError("candidate category does not match the feature vector category")

        return RankedCandidateSnapshot(
            request_id=_required_text(request_id, "request_id"),
            ranking_timestamp=timestamp,
            model_version=model_version,
            user_id=user_id,
            content_id=content_id,
            creator_id=creator_id,
            category=category,
            user_snapshot_json=_canonical_json(user_snapshot, "user_snapshot"),
            candidate_snapshot_json=_canonical_json(candidate_snapshot, "candidate_snapshot"),
            features=feature_vector,
            raw_model_score=_finite_float(raw_model_score, "raw_model_score"),
            normalized_model_score=_finite_float(normalized_model_score, "normalized_model_score"),
            pre_rerank_rank=_positive_rank(pre_rerank_rank, "pre_rerank_rank"),
            final_score=_finite_float(final_score, "final_score"),
            final_rank=_positive_rank(final_rank, "final_rank"),
        )
