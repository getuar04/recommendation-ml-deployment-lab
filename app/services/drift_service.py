"""Orchestrates drift evaluation: loads the active artifact's metadata through the existing
`model_store.check_artifact_compatibility()` contract (schema/feature/checksum-validated,
but never deserializes the model itself -- drift only needs `driftBaseline` from metadata,
not to run inference, so this avoids the one-time cost of a full `joblib.load()` per
request), then delegates the actual statistics to `app.ml.drift_detector`.

Exceptions mirror `app.services.recommendation_service`'s `ModelNotTrained`/
`ModelArtifactInvalid` shape so `app.api.drift_routes` can reuse the same kind of
try/except mapping every other model-control route already uses -- no new error-handling
pattern introduced for this one feature.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import (
    DRIFT_MIN_OBSERVATIONS,
    DRIFT_WARMUP_MAX_ROWS,
    LIVE_METADATA_PATH,
    LIVE_MODEL_PATH,
    METADATA_PATH,
    MODEL_PATH,
)
from app.db.models import Content
from app.db.repositories import (
    recent_interactions_for_drift,
    warmup_interactions_for_drift,
)
from app.ml import model_store
from app.ml.dataset_builder import FEATURES, build_feature_rows
from app.ml.drift_baseline_validation import (
    BaselineStructureError,
    validate_baseline_structure,
)
from app.ml.drift_detector import evaluate_drift
from app.ml.live_feature_builder import LIVE_FEATURES

__all__ = [
    "BaselineMissing",
    "ModelNotReady",
    "live_drift_baseline_status",
    "live_drift_from_observations",
    "video_drift_from_observations",
    "video_drift_from_recent_interactions",
]


class ModelNotReady(Exception):
    """No usable active artifact exists at all: never trained, or trained but corrupted/
    incompatible with the current feature contract. `reason` is one of "NOT_TRAINED" /
    "INCOMPATIBLE" / "CORRUPTED", mirroring app.services.recommendation_service's
    ModelArtifactInvalid.reason values (minus "BUSY" -- this reads metadata only via
    check_artifact_compatibility(), which has no retry/stable-read race to report)."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class BaselineMissing(Exception):
    """No *usable* drift baseline exists on an otherwise-valid, servable active artifact --
    either because it genuinely has none (trained before drift monitoring was added), or
    because what it has is malformed/inconsistent with the current feature contract (see
    `_validated_baseline` below) and must not be trusted. Either way this is a reportable,
    expected state, not a crash: `model_type`/`model_version` are always attached, since
    valid artifact metadata was already loaded before this was raised (corrective pass,
    Finding 4) -- `GET /model/drift`'s BASELINE_MISSING response must not lose the model
    version it already knows just because the baseline itself is unusable."""

    def __init__(self, message: str, *, model_type: str, model_version: str | None) -> None:
        super().__init__(message)
        self.model_type = model_type
        self.model_version = model_version


def _active_metadata(*, model_path, metadata_path, expected_features: list[str]) -> dict[str, Any]:
    try:
        return model_store.check_artifact_compatibility(expected_features, model_path=model_path, metadata_path=metadata_path)
    except model_store.ArtifactNotFoundError as exc:
        raise ModelNotReady("No trained model artifact exists.", reason="NOT_TRAINED") from exc
    except model_store.ArtifactIncompatibleError as exc:
        raise ModelNotReady(str(exc), reason="INCOMPATIBLE") from exc
    except model_store.ArtifactCorruptedError as exc:
        raise ModelNotReady(str(exc), reason="CORRUPTED") from exc


def _validated_baseline(metadata: dict[str, Any], *, model_type: str, expected_features: list[str]) -> dict[str, Any]:
    """Returns `metadata["driftBaseline"]` only if it is structurally sound AND actually
    describes *this* domain's *current* feature contract -- general-review hardening added
    in the corrective pass: a baseline whose `modelType` doesn't match the endpoint it was
    read from, whose `featureNames`/`numeric`/`categorical` keys don't agree with the
    feature set this deployment currently expects (e.g. a baseline left over from an older,
    since-changed feature set), or whose per-feature statistics are themselves malformed
    (right feature names, but empty/inconsistent numbers underneath -- see
    `app.ml.drift_baseline_validation`), is exactly the kind of malformed input that must
    never silently produce a misleading OK -- it is treated the same as "no baseline", not
    partially trusted."""
    model_version = metadata.get("modelVersion")
    baseline = metadata.get("driftBaseline")
    if not isinstance(baseline, dict):
        raise BaselineMissing(
            "Active model artifact has no drift baseline (it was trained before drift monitoring was added; retrain to add one).",
            model_type=model_type, model_version=model_version,
        )
    try:
        validate_baseline_structure(baseline, model_type=model_type, expected_features=expected_features)
    except BaselineStructureError as exc:
        raise BaselineMissing(
            f"Active model artifact's drift baseline is malformed or does not match the current "
            f"feature contract; treating it as unusable (retrain to refresh it). Reason: {exc}",
            model_type=model_type, model_version=model_version,
        ) from exc
    return baseline


def _finalize(
    report: dict[str, Any], metadata: dict[str, Any], baseline: dict[str, Any], *, observation_source: str, message: str | None = None,
) -> dict[str, Any]:
    return {
        **report,
        "modelType": baseline.get("modelType", "UNKNOWN"),
        "modelVersion": metadata.get("modelVersion"),
        "baselineVersion": baseline.get("baselineVersion"),
        "baselineGeneratedAt": baseline.get("generatedAt"),
        "baselineTrainingSampleCount": baseline.get("trainingSampleCount"),
        "observationSource": observation_source,
        "message": message,
    }


def _observations_to_frame(observations: list[dict[str, float | str]]) -> pd.DataFrame:
    return pd.DataFrame(observations)


def video_drift_from_recent_interactions(db: Session, *, limit: int) -> dict[str, Any]:
    """GET /model/drift: the only endpoint that sources its own observations (bounded,
    recent, cross-user local VIDEO interactions -- see app.db.repositories.
    recent_interactions_for_drift) rather than requiring the caller to supply them. LIVE has
    no equivalent (see live_drift_baseline_status below): there is no local LIVE interaction
    log to reconstruct from.

    Corrective pass (Finding 2): the observation window alone is not enough to compute most
    VIDEO features correctly -- see app.ml.dataset_builder.build_feature_rows's docstring
    for the full per-feature audit of which ones need prior history. A second, bounded
    warm-up query (single query, not one per user -- see
    app.db.repositories.warmup_interactions_for_drift) primes FeatureHistory with each
    window user's prior interactions before any observation-window feature is computed;
    warm-up rows update history but never themselves become an evaluated observation.
    """
    metadata = _active_metadata(model_path=MODEL_PATH, metadata_path=METADATA_PATH, expected_features=FEATURES)
    baseline = _validated_baseline(metadata, model_type="VIDEO", expected_features=FEATURES)
    raw_window_rows = recent_interactions_for_drift(db, limit=limit)
    # VIDEO-only observation window: `interactions` (app.db.repositories, shared by name but
    # with no other caller of recent_interactions_for_drift/warmup_interactions_for_drift --
    # LIVE has no local interaction log to reconstruct from at all, see this module's own
    # live_drift_baseline_status docstring) has no VIDEO/LIVE discriminator of its own -- event
    # ingestion accepts LIVE_*/domain-ambiguous event types into this same table with no
    # cross-check against the referenced content's own type. A row PROVABLY LIVE must not enter
    # VIDEO drift's observation window or its warm-up history; a row whose Content is missing/
    # unavailable is left in unchanged (mirrors training_service/user_behavior_provider's
    # identical fix) -- "unavailable" is not evidence of being LIVE.
    raw_content_ids = {row.content_id for row in raw_window_rows}
    raw_content_by_id = (
        {item.content_id: item for item in db.scalars(select(Content).where(Content.content_id.in_(raw_content_ids))).all()}
        if raw_content_ids else {}
    )
    window_rows = [
        row for row in raw_window_rows
        if getattr(raw_content_by_id.get(row.content_id), "content_type", "VIDEO") != "LIVE"
    ]
    if window_rows:
        boundary = window_rows[0]  # oldest row in the window (ascending order)
        user_ids = {row.user_id for row in window_rows}
        raw_warmup_rows = warmup_interactions_for_drift(
            db, user_ids=user_ids, before_timestamp=boundary.timestamp, before_id=boundary.id,
            limit=DRIFT_WARMUP_MAX_ROWS,
        )
        # A second, bounded content lookup -- warm-up rows can reference content outside the
        # observation window, so this is a genuinely new (but still single, non-N+1) query,
        # needed only to apply the same VIDEO-only filter to warm-up history.
        warmup_content_ids = {row.content_id for row in raw_warmup_rows} - raw_content_by_id.keys()
        warmup_content_by_id = (
            {item.content_id: item for item in db.scalars(select(Content).where(Content.content_id.in_(warmup_content_ids))).all()}
            if warmup_content_ids else {}
        )
        warmup_rows = [
            row for row in raw_warmup_rows
            if getattr(raw_content_by_id.get(row.content_id) or warmup_content_by_id.get(row.content_id), "content_type", "VIDEO") != "LIVE"
        ]
        # `raw_content_by_id` already covers every window_rows content_id (built from the raw,
        # pre-filter set) -- reused directly, no second window-scoped fetch needed.
        content_by_id = raw_content_by_id
    else:
        warmup_rows = []
        content_by_id = {}
    observations = build_feature_rows(window_rows, content_by_id, warmup_rows=warmup_rows)
    report = evaluate_drift(baseline, observations)
    report["warmupCount"] = len(warmup_rows)
    return _finalize(report, metadata, baseline, observation_source="recent_local_video_interactions")


def video_drift_from_observations(observations: list[dict[str, float | str]]) -> dict[str, Any]:
    metadata = _active_metadata(model_path=MODEL_PATH, metadata_path=METADATA_PATH, expected_features=FEATURES)
    baseline = _validated_baseline(metadata, model_type="VIDEO", expected_features=FEATURES)
    report = evaluate_drift(baseline, _observations_to_frame(observations))
    return _finalize(report, metadata, baseline, observation_source="caller_supplied")


def live_drift_baseline_status() -> dict[str, Any]:
    """GET /model/drift/live: LIVE has no local interaction table at all (confirmed against
    app/db/models.py -- there is no LiveInteraction table), so there is nothing this
    endpoint can reconstruct on its own. It reports baseline presence/version only, with
    status OBSERVATION_SOURCE_UNAVAILABLE, and points the caller at the POST endpoint --
    it never fabricates observations from the synthetic data LIVE training itself may have
    used (see driftBaseline.trainingDataProvenance) and presents them as real traffic."""
    metadata = _active_metadata(model_path=LIVE_MODEL_PATH, metadata_path=LIVE_METADATA_PATH, expected_features=LIVE_FEATURES)
    baseline = _validated_baseline(metadata, model_type="LIVE", expected_features=LIVE_FEATURES)
    report = {"status": "OBSERVATION_SOURCE_UNAVAILABLE", "observationCount": 0, "minObservationsRequired": DRIFT_MIN_OBSERVATIONS, "features": {}}
    return _finalize(
        report, metadata, baseline, observation_source="none_available",
        message="LIVE has no local interaction log to reconstruct observations from. "
                "Submit explicitly-observed LIVE feature rows to POST /model/drift/evaluate/live instead.",
    )


def live_drift_from_observations(observations: list[dict[str, float | str]]) -> dict[str, Any]:
    metadata = _active_metadata(model_path=LIVE_MODEL_PATH, metadata_path=LIVE_METADATA_PATH, expected_features=LIVE_FEATURES)
    baseline = _validated_baseline(metadata, model_type="LIVE", expected_features=LIVE_FEATURES)
    report = evaluate_drift(baseline, _observations_to_frame(observations))
    return _finalize(report, metadata, baseline, observation_source="caller_supplied")
