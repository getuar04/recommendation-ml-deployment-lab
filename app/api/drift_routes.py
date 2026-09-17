"""Drift monitoring endpoints. Thin orchestration only -- every statistic is computed in
`app.ml.drift_baseline`/`app.ml.drift_detector` (pure, FastAPI-independent), and artifact/
observation-source handling lives in `app.services.drift_service`; this module's only job is
HTTP mapping (status codes are not used to carry drift severity -- every endpoint below
returns 200 with a typed `DriftReport` whose own `status` field carries OK/WARNING/CRITICAL/
INSUFFICIENT_DATA/BASELINE_MISSING/MODEL_NOT_READY/OBSERVATION_SOURCE_UNAVAILABLE, mirroring
how GET /model/status already reports READY/MISSING/INCOMPATIBLE/CORRUPTED at 200 rather than
as HTTP errors -- a monitoring dashboard should never have to branch on HTTP status to render
"here is the current state"). Malformed *requests* (oversized/malformed observation batches,
an observation batch whose feature keys don't match this deployment's actual feature
contract) are the one exception and still return 422, consistent with every other endpoint's
request-validation behavior.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import (
    DRIFT_MAX_OBSERVATIONS,
    DRIFT_MIN_OBSERVATIONS,
    DRIFT_OBSERVATION_DEFAULT_LIMIT,
)
from app.db.database import get_db
from app.ml.drift_detector import DriftSchemaMismatchError
from app.schemas.drift_schemas import DriftEvaluateRequest, DriftReport
from app.services import drift_service

router = APIRouter(tags=["model-control"])

_MODEL_NOT_READY_REASON_CODE = {
    "NOT_TRAINED": "MODEL_NOT_TRAINED",
    "INCOMPATIBLE": "MODEL_ARTIFACT_INVALID",
    "CORRUPTED": "MODEL_ARTIFACT_INVALID",
}


def _not_ready_report(exc: drift_service.ModelNotReady, *, model_type: str) -> dict:
    return {
        "status": "MODEL_NOT_READY", "modelType": model_type, "modelVersion": None,
        "baselineVersion": None, "baselineGeneratedAt": None, "baselineTrainingSampleCount": None,
        "observationCount": 0, "minObservationsRequired": DRIFT_MIN_OBSERVATIONS,
        "observationSource": "none", "message": f"{_MODEL_NOT_READY_REASON_CODE.get(exc.reason, 'MODEL_NOT_READY')}: {exc}",
        "features": {},
    }


def _baseline_missing_report(exc: drift_service.BaselineMissing) -> dict:
    # Corrective pass (Finding 4): model_type/model_version come from the exception itself
    # now, not a hardcoded None -- valid artifact metadata was already loaded before
    # BaselineMissing was raised (app.services.drift_service._validated_baseline), so the
    # active model version must not be discarded just because its baseline is unusable.
    return {
        "status": "BASELINE_MISSING", "modelType": exc.model_type, "modelVersion": exc.model_version,
        "baselineVersion": None, "baselineGeneratedAt": None, "baselineTrainingSampleCount": None,
        "observationCount": 0, "minObservationsRequired": DRIFT_MIN_OBSERVATIONS,
        "observationSource": "none", "message": str(exc), "features": {},
    }


_DRIFT_EXAMPLE = {
    "status": "OK", "modelType": "VIDEO", "modelVersion": "recommendation-prod-20260730143000",
    "baselineVersion": "1.0.0", "baselineGeneratedAt": "2026-07-30T14:30:00+00:00",
    "baselineTrainingSampleCount": 4434, "observationCount": 500, "warmupCount": 213,
    "minObservationsRequired": 30, "observationSource": "recent_local_video_interactions", "message": None,
    "features": {"category_affinity": {"featureType": "numeric", "status": "OK", "psi": 0.012,
                                        "jsDivergence": None, "observedCount": 500,
                                        "observedMissingRate": 0.0, "baselineMissingRate": 0.0,
                                        "missingRateDelta": 0.0, "outOfRangeRate": 0.0,
                                        "observedOtherRate": None, "baselineOtherRate": None, "otherRateDelta": None}},
}


@router.get("/model/drift", response_model=DriftReport, responses={200: {"content": {"application/json": {"example": _DRIFT_EXAMPLE}}}})
def video_drift(
    db: Session = Depends(get_db),
    limit: int = Query(DRIFT_OBSERVATION_DEFAULT_LIMIT, ge=DRIFT_MIN_OBSERVATIONS, le=DRIFT_MAX_OBSERVATIONS),
):
    """Reconstructs up to `limit` of the most recent local VIDEO interactions (across all
    users, bounded, real if present -- see app.db.repositories.recent_interactions_for_drift)
    into feature rows and evaluates them against the active model's training-time baseline.
    Never raises for a missing model/baseline/insufficient data -- see module docstring."""
    try:
        return drift_service.video_drift_from_recent_interactions(db, limit=limit)
    except drift_service.ModelNotReady as exc:
        return _not_ready_report(exc, model_type="VIDEO")
    except drift_service.BaselineMissing as exc:
        return _baseline_missing_report(exc)


@router.post("/model/drift/evaluate", response_model=DriftReport)
def video_drift_evaluate(request: DriftEvaluateRequest):
    """Evaluates a caller-supplied, bounded batch of already-computed VIDEO feature rows
    against the active model's training-time baseline -- the production-shaped path: no
    local database read at all, matching userProfile's "caller supplies pre-computed data"
    convention on POST /recommendations."""
    try:
        return drift_service.video_drift_from_observations([observation.features for observation in request.observations])
    except drift_service.ModelNotReady as exc:
        return _not_ready_report(exc, model_type="VIDEO")
    except drift_service.BaselineMissing as exc:
        return _baseline_missing_report(exc)
    except DriftSchemaMismatchError as exc:
        raise HTTPException(422, detail={"error": "DRIFT_SCHEMA_MISMATCH", "message": str(exc)}) from exc


@router.get("/model/drift/live", response_model=DriftReport)
def live_drift():
    """LIVE has no local interaction log to reconstruct observations from (see
    app.services.drift_service.live_drift_baseline_status) -- this reports baseline
    presence/version only, with status OBSERVATION_SOURCE_UNAVAILABLE, and never fabricates
    a report from LIVE's synthetic training data presented as real traffic."""
    try:
        return drift_service.live_drift_baseline_status()
    except drift_service.ModelNotReady as exc:
        return _not_ready_report(exc, model_type="LIVE")
    except drift_service.BaselineMissing as exc:
        return _baseline_missing_report(exc)


@router.post("/model/drift/evaluate/live", response_model=DriftReport)
def live_drift_evaluate(request: DriftEvaluateRequest):
    """The only way to get a real LIVE drift evaluation: a caller-supplied, bounded batch of
    already-observed LIVE feature rows (see app.ml.live_feature_builder.LIVE_FEATURES)."""
    try:
        return drift_service.live_drift_from_observations([observation.features for observation in request.observations])
    except drift_service.ModelNotReady as exc:
        return _not_ready_report(exc, model_type="LIVE")
    except drift_service.BaselineMissing as exc:
        return _baseline_missing_report(exc)
    except DriftSchemaMismatchError as exc:
        raise HTTPException(422, detail={"error": "DRIFT_SCHEMA_MISMATCH", "message": str(exc)}) from exc
