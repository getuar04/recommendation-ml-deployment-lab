import json
import uuid
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import (
    LIVE_METADATA_PATH,
    LIVE_MODEL_PATH,
    METADATA_PATH,
    MODEL_PATH,
)
from app.core.logging import logger
from app.db.database import SessionLocal, get_db
from app.ml import artifact_lifecycle, model_store
from app.ml.dataset_builder import CATEGORICAL, FEATURES
from app.ml.live_feature_builder import LIVE_CATEGORICAL, LIVE_FEATURES
from app.ml.live_trainer import InsufficientLiveData
from app.schemas.response_models import (
    ErrorResponse,
    ModelMetricsSummaryResponse,
    ModelStatusResponse,
    ModelVersionsResponse,
    TrainingJobResponse,
)
from app.services import job_store, training_lock
from app.services.live_training_service import train_live as run_live_training
from app.services.training_service import InsufficientData, NoEligibleModel, train

router=APIRouter(tags=["model-control"])

def _fail_job_safely(db,job_id:str,model_type:str,error_code:str,message:str)->None:
    """Mark a training job FAILED, tolerating a `db` session left in an aborted-transaction
    state by whatever raised inside train()/train_live_model() (e.g. a real
    psycopg.errors.UndefinedColumn from a stale schema poisons the rest of that Postgres
    transaction -- every further statement on the same session raises
    psycopg.errors.InFailedSqlTransaction until ROLLBACK). Without the rollback below, the
    job_store.mark_failed() call itself would raise, escape unhandled, and leave the job
    stuck RUNNING forever with its training lock never released -- exactly the incident this
    function exists to prevent. Best-effort: if mark_failed still fails after the rollback
    (e.g. the connection is truly gone), that is logged, not re-raised, so the caller's
    `finally` can still attempt to release the lock."""
    try:
        db.rollback()
    except Exception:  # noqa: BLE001 -- best-effort per this function's docstring: must not mask the original training failure
        logger.exception("rollback before marking job %s FAILED also failed (modelType=%s)",job_id,model_type)
    try:
        job_store.mark_failed(db,job_id,error_code,message)
        logger.info("training failed jobId=%s modelType=%s errorCode=%s",job_id,model_type,error_code)
    except Exception:  # noqa: BLE001 -- best-effort per this function's docstring: must not mask the original training failure
        logger.exception("failed to persist FAILED status for job %s (modelType=%s, errorCode=%s)",job_id,model_type,error_code)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001 -- best-effort per this function's docstring: must not mask the original training failure
            logger.exception("rollback after failed mark_failed for job %s also failed",job_id)


def _release_lock_safely(db,model_type:str,job_id:str)->None:
    """Best-effort lock release, tolerating the same aborted-transaction state
    `_fail_job_safely` guards against -- a job that already failed to persist its own FAILED
    status must still not leave the training lock held forever."""
    try:
        training_lock.release(db,model_type,job_id)
    except Exception:  # noqa: BLE001 -- best-effort per this function's docstring: must not leave the training lock held forever
        logger.exception("failed to release %s training lock for job %s",model_type,job_id)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001 -- best-effort per this function's docstring: must not leave the training lock held forever
            logger.exception("rollback after failed lock release for job %s also failed",job_id)


def _run_training_job(job_id:str):
    db=SessionLocal()
    logger.info("training started jobId=%s modelType=VIDEO",job_id)
    try:
        job_store.mark_running(db,job_id)
        try:
            metadata=train(db)
            job_store.mark_succeeded(db,job_id,metadata)
            logger.info("training succeeded jobId=%s modelType=VIDEO modelVersion=%s",job_id,metadata.get("modelVersion"))
        except InsufficientData as e:
            _fail_job_safely(db,job_id,"VIDEO","INSUFFICIENT_TRAINING_DATA",str(e))
        except NoEligibleModel as e:
            _fail_job_safely(db,job_id,"VIDEO","NO_ELIGIBLE_MODEL",str(e))
        except artifact_lifecycle.PromotionValidationError as e:
            _fail_job_safely(db,job_id,"VIDEO","MODEL_PROMOTION_FAILED",str(e))
        except Exception as e:  # noqa: BLE001 -- background job: any unanticipated training failure must still mark the job FAILED, not crash silently
            logger.exception("training failed jobId=%s modelType=VIDEO errorCode=TRAINING_FAILED",job_id)
            _fail_job_safely(db,job_id,"VIDEO","TRAINING_FAILED",str(e))
    finally:
        _release_lock_safely(db,"VIDEO",job_id)
        db.close()

def _run_live_training_job(job_id:str,mode:Literal["synthetic","real"]="synthetic"):
    db=SessionLocal()
    logger.info("training started jobId=%s modelType=LIVE mode=%s",job_id,mode)
    try:
        job_store.mark_running(db,job_id)
        try:
            metadata=run_live_training(db,mode=mode)
            job_store.mark_succeeded(db,job_id,metadata)
            logger.info("training succeeded jobId=%s modelType=LIVE modelVersion=%s",job_id,metadata.get("modelVersion"))
        except InsufficientLiveData as e:
            _fail_job_safely(db,job_id,"LIVE","INSUFFICIENT_LIVE_TRAINING_DATA",str(e))
        except artifact_lifecycle.PromotionValidationError as e:
            _fail_job_safely(db,job_id,"LIVE","LIVE_MODEL_PROMOTION_FAILED",str(e))
        except Exception as e:  # noqa: BLE001 -- background job: any unanticipated training failure must still mark the job FAILED, not crash silently
            logger.exception("training failed jobId=%s modelType=LIVE errorCode=LIVE_TRAINING_FAILED",job_id)
            _fail_job_safely(db,job_id,"LIVE","LIVE_TRAINING_FAILED",str(e))
    finally:
        _release_lock_safely(db,"LIVE",job_id)
        db.close()

_TRAIN_ACCEPTED_EXAMPLE = {"jobId": "a400555e10234d748474a690226cabda", "status": "PENDING"}
_TRAIN_UNAUTHORIZED_EXAMPLE = {"error": "UNAUTHORIZED", "message": "A valid X-Internal-API-Key header is required for this operation.", "requestId": "..."}
_TRAIN_CONFLICT_EXAMPLE = {"error": "TRAINING_ALREADY_RUNNING", "message": "A VIDEO training job is already running.", "requestId": "..."}


_TRAIN_RESPONSES: dict[int | str, dict[str, Any]] = {202: {"content": {"application/json": {"example": _TRAIN_ACCEPTED_EXAMPLE}}},
                    401: {"model": ErrorResponse, "content": {"application/json": {"example": _TRAIN_UNAUTHORIZED_EXAMPLE}}},
                    409: {"model": ErrorResponse, "content": {"application/json": {"example": _TRAIN_CONFLICT_EXAMPLE}}}}


def _start_video_training(background_tasks:BackgroundTasks,db:Session)->dict:
    job_id=job_store.create_job(db,"VIDEO_TRAINING")
    logger.info("training accepted jobId=%s modelType=VIDEO",job_id)
    try:
        training_lock.acquire(db,"VIDEO",job_id)
    except training_lock.TrainingAlreadyRunningError:
        job_store.mark_failed(db,job_id,"TRAINING_ALREADY_RUNNING","A VIDEO training job is already running.")
        raise HTTPException(409,detail={"error":"TRAINING_ALREADY_RUNNING","message":"A VIDEO training job is already running."})
    background_tasks.add_task(_run_training_job,job_id)
    return {"jobId":job_id,"status":"PENDING"}


@router.post("/model/train",status_code=202,responses=_TRAIN_RESPONSES)
def train_endpoint(background_tasks:BackgroundTasks,db:Session=Depends(get_db)):
    """Starts training as a background job and returns immediately; poll
    GET .../jobs/{jobId} for the result. Example values above are illustrative only, not
    measured results."""
    return _start_video_training(background_tasks,db)


@router.post("/model/train/async",status_code=202,responses=_TRAIN_RESPONSES,deprecated=True)
def train_endpoint_async(background_tasks:BackgroundTasks,db:Session=Depends(get_db)):
    """Spec §31/§63: backward-compatible alias for POST /model/train using the exact same
    job creation, lock acquisition, and background execution -- no training logic is
    duplicated. `/model/train` is preserved unchanged for existing callers. A separate
    always-blocking synchronous endpoint was deliberately not added: real VIDEO training on
    this service's own experiment data has taken up to ~49s (see README's experiment
    results), and blocking a single HTTP worker thread for that long is a worse outcome than
    the already-fast async poll loop this service uses end-to-end (see tests/test_end_to_end.py).
    Example values above are illustrative only, not measured results."""
    return _start_video_training(background_tasks,db)

@router.post("/model/train/live",status_code=202)
def train_live(background_tasks:BackgroundTasks,db:Session=Depends(get_db),mode:Literal["synthetic","real"]="synthetic"):
    """`mode` (query param, default "synthetic", unchanged pre-existing behavior): "synthetic"
    trains on app.ml.live_synthetic_data's deterministic PoC/dev dataset, exactly as before.
    "real" trains on this service's own stored LIVE interactions
    (app.services.live_training_service, app.ml.live_dataset_builder) -- if that real dataset
    is too small, the job fails with INSUFFICIENT_LIVE_TRAINING_DATA; it never silently falls
    back to synthetic."""
    job_id=job_store.create_job(db,"LIVE_TRAINING")
    logger.info("training accepted jobId=%s modelType=LIVE mode=%s",job_id,mode)
    try:
        training_lock.acquire(db,"LIVE",job_id)
    except training_lock.TrainingAlreadyRunningError:
        job_store.mark_failed(db,job_id,"TRAINING_ALREADY_RUNNING","A LIVE training job is already running.")
        raise HTTPException(409,detail={"error":"TRAINING_ALREADY_RUNNING","message":"A LIVE training job is already running."})
    background_tasks.add_task(_run_live_training_job,job_id,mode)
    return {"jobId":job_id,"status":"PENDING"}

_JOB_PENDING_EXAMPLE = {"jobId": "abc123", "jobType": "VIDEO_TRAINING", "status": "PENDING", "result": None, "error": None,
                       "createdAt": "2026-07-30T10:00:00+00:00", "updatedAt": "2026-07-30T10:00:00+00:00"}
_JOB_SUCCEEDED_EXAMPLE = {"jobId": "abc123", "jobType": "VIDEO_TRAINING", "status": "SUCCEEDED",
                         "result": {"modelVersion": "recommendation-prod-20260730143000"}, "error": None,
                         "createdAt": "2026-07-30T10:00:00+00:00", "updatedAt": "2026-07-30T10:00:15+00:00"}
_JOB_FAILED_EXAMPLE = {"jobId": "abc123", "jobType": "VIDEO_TRAINING", "status": "FAILED", "result": None,
                      "error": {"error": "INSUFFICIENT_TRAINING_DATA", "message": "At least 100 labeled interactions containing positive and negative samples are required."},
                      "createdAt": "2026-07-30T10:00:00+00:00", "updatedAt": "2026-07-30T10:00:04+00:00"}


@router.get("/model/train/jobs/{job_id}", response_model=TrainingJobResponse,
            responses={200: {"content": {"application/json": {"examples": {
                "pending": {"summary": "Job accepted, not yet started", "value": _JOB_PENDING_EXAMPLE},
                "succeeded": {"summary": "Training completed successfully", "value": _JOB_SUCCEEDED_EXAMPLE},
                "failed": {"summary": "Training failed with a safe, structured error", "value": _JOB_FAILED_EXAMPLE},
            }}}}, 404: {"model": ErrorResponse, "content": {"application/json": {"example": {"error": "JOB_NOT_FOUND", "message": "No training job with that id exists.", "requestId": "..."}}}}})
def train_job_status(job_id:str,db:Session=Depends(get_db)):
    """`status` is PENDING/RUNNING/SUCCEEDED/FAILED. Never exposes a stack trace or internal
    exception text -- `error` (when present) is a stable code plus a safe message. Example
    values above are illustrative only, not measured results."""
    job=job_store.get_job(db,job_id)
    if job is None: raise HTTPException(404,detail={"error":"JOB_NOT_FOUND","message":"No training job with that id exists."})
    return job

def _with_model_lock(db:Session,model_type:str,op_name:str):
    """Acquires the SAME per-model-type `training_lock` row a training job holds for its
    whole duration (see `_run_training_job`/`_run_live_training_job`'s `finally`-guarded
    release above) -- promote() and rollback() both mutate the identical active/previous
    file pair via unprotected `os.replace` calls (app.ml.artifact_lifecycle), so a rollback
    racing an in-flight promotion (or a second concurrent rollback) can interleave those
    renames and corrupt the active/previous pairing (contained, not prevented, by
    model_store's checksum fail-closed check on the next load -- an avoidable outage, not
    silent corruption). Reusing `training_lock` here -- rather than a second, parallel lock
    -- means a rollback simply cannot run while that model type's training holds the lock,
    exactly like a second training job cannot."""
    job_id=f"{op_name}-{uuid.uuid4().hex}"
    try:
        training_lock.acquire(db,model_type,job_id)
    except training_lock.TrainingAlreadyRunningError:
        raise HTTPException(409,detail={"error":"MODEL_OPERATION_IN_PROGRESS",
                                        "message":f"Another {model_type} training/rollback operation is already in progress."})
    return job_id


@router.post("/model/rollback")
def rollback_video(db:Session=Depends(get_db)):
    from app.ml import model_cache
    job_id=_with_model_lock(db,"VIDEO","rollback")
    try:
        paths=artifact_lifecycle.sibling_paths(MODEL_PATH,METADATA_PATH)
        try:
            metadata=artifact_lifecycle.rollback(
                active_model_path=MODEL_PATH,active_metadata_path=METADATA_PATH,
                previous_model_path=paths["previous_model"],previous_metadata_path=paths["previous_metadata"],
                feature_names=FEATURES,categorical_features=CATEGORICAL,
            )
        except model_store.ArtifactNotFoundError:
            raise HTTPException(404,detail={"error":"NO_PREVIOUS_MODEL","message":"No previous model artifact exists to roll back to."})
        except artifact_lifecycle.PromotionValidationError as e:
            raise HTTPException(409,detail={"error":"ROLLBACK_FAILED","message":str(e)})
    finally:
        training_lock.release(db,"VIDEO",job_id)
    model_cache.video_cache.invalidate()
    logger.info("rollback modelType=VIDEO modelVersion=%s",metadata.get("modelVersion"))
    return {"status":"ROLLED_BACK","modelVersion":metadata.get("modelVersion")}

@router.post("/model/rollback/live")
def rollback_live(db:Session=Depends(get_db)):
    from app.ml import model_cache
    job_id=_with_model_lock(db,"LIVE","rollback")
    try:
        paths=artifact_lifecycle.sibling_paths(LIVE_MODEL_PATH,LIVE_METADATA_PATH)
        try:
            metadata=artifact_lifecycle.rollback(
                active_model_path=LIVE_MODEL_PATH,active_metadata_path=LIVE_METADATA_PATH,
                previous_model_path=paths["previous_model"],previous_metadata_path=paths["previous_metadata"],
                feature_names=LIVE_FEATURES,categorical_features=LIVE_CATEGORICAL,
            )
        except model_store.ArtifactNotFoundError:
            raise HTTPException(404,detail={"error":"NO_PREVIOUS_LIVE_MODEL","message":"No previous LIVE model artifact exists to roll back to."})
        except artifact_lifecycle.PromotionValidationError as e:
            raise HTTPException(409,detail={"error":"ROLLBACK_FAILED","message":str(e)})
    finally:
        training_lock.release(db,"LIVE",job_id)
    model_cache.live_cache.invalidate()
    logger.info("rollback modelType=LIVE modelVersion=%s",metadata.get("modelVersion"))
    return {"status":"ROLLED_BACK","modelVersion":metadata.get("modelVersion")}

@router.get("/model/metrics")
def metrics():
    # Validates the same schema/feature/checksum contract inference uses (without paying
    # to deserialize the model): stale/incompatible/corrupted metadata is never returned
    # as if it were ready to serve.
    try:
        return model_store.check_artifact_compatibility(FEATURES,model_path=MODEL_PATH,metadata_path=METADATA_PATH)
    except model_store.ArtifactNotFoundError:
        raise HTTPException(404,detail={"error":"MODEL_NOT_TRAINED","message":"No trained model metadata exists."})
    except model_store.ArtifactIncompatibleError as e:
        raise HTTPException(503,detail={"error":"MODEL_ARTIFACT_INVALID","reason":"INCOMPATIBLE","message":str(e)})
    except model_store.ArtifactCorruptedError as e:
        raise HTTPException(503,detail={"error":"MODEL_ARTIFACT_INVALID","reason":"CORRUPTED","message":str(e)})

_CLASSIFIER_SUMMARY_METRIC_KEYS=("accuracy","precision","recall","f1Score","prAuc","rocAuc",
                                  "precisionAt5","recallAt10","ndcgAt10")
# XGBRanker promotion task: a ranker's `metrics` (app.ml.artifact_metadata.build_ranker_metadata)
# has none of the classifier keys above -- filtering to them alone would silently return an
# EMPTY metrics object for a ranker winner, which is honest but useless. These are the
# ranker-native fields worth surfacing instead; never merged with the classifier keys (a
# candidate is always exactly one family, never both).
_RANKER_SUMMARY_METRIC_KEYS=("ndcgByDifficulty","criticalPassRate","unseenBeatsSeenPassed")


def _summary_from(metadata:dict)->dict:
    """Spec §35: concise operational summary — never the full ~900-line metadata object.
    Family-conditional (XGBRanker promotion task): a classifier gets classifier-shaped
    diagnostic metrics, a ranker gets ranker-shaped ones — never a mix, and never a fabricated
    0/None classifier metric for a ranker."""
    metrics=metadata.get("metrics",{}) or {}
    family=metadata.get("modelFamily") or "classifier"
    metric_keys=_RANKER_SUMMARY_METRIC_KEYS if family=="ranker" else _CLASSIFIER_SUMMARY_METRIC_KEYS
    return {"modelVersion":metadata.get("modelVersion"),"selectedModel":metadata.get("selectedModel"),
            "modelFamily":family,
            "trainedAt":metadata.get("trainedAt"),"trainingSamples":metadata.get("trainingSamples"),
            "testSamples":metadata.get("testSamples"),"decisionThreshold":metadata.get("decisionThreshold"),
            "trainingDurationSeconds":metadata.get("trainingDurationSeconds"),
            "metrics":{key:metrics.get(key) for key in metric_keys if key in metrics}}

def _status_payload(cache,model_path,metadata_path,expected_features)->dict:
    """Spec §36: READY / MISSING / INCOMPATIBLE / CORRUPTED, plus version info when ready."""
    state=cache.peek_status(model_path,metadata_path,expected_features=expected_features)
    payload={"status":state}
    if state=="READY":
        try:
            metadata=model_store.load_metadata(metadata_path=metadata_path)
            payload.update(modelVersion=metadata.get("modelVersion"),selectedModel=metadata.get("selectedModel"),
                           modelFamily=metadata.get("modelFamily") or "classifier",trainedAt=metadata.get("trainedAt"))
        except (OSError, json.JSONDecodeError):
            # cache.peek_status() already confirmed READY, so metadata_path existed and
            # validated moments ago -- a read/parse failure here means it was deleted or
            # corrupted in the gap since (a real race, not a normal state). Still report
            # READY with no version fields (unchanged response contract; the caller has
            # already committed to serving from the in-memory model this status describes),
            # but log it so a genuine corruption/race is never silently invisible.
            logger.warning("model metadata unreadable at status time despite READY cache state: %s",metadata_path)
    elif state=="INCOMPATIBLE":
        payload.update(errorCode="MODEL_ARTIFACT_INCOMPATIBLE",
                       message="The stored model artifact is not compatible with the current service version.")
    elif state=="CORRUPTED":
        payload.update(errorCode="MODEL_ARTIFACT_CORRUPTED",
                       message="The stored model artifact failed integrity validation.")
    return payload

_MODEL_STATUS_READY_EXAMPLE = {"status": "READY", "modelVersion": "recommendation-prod-20260730143000", "selectedModel": "LogisticRegression", "trainedAt": "2026-07-30T14:30:00+00:00"}
_MODEL_STATUS_MISSING_EXAMPLE = {"status": "MISSING"}
_MODEL_STATUS_INCOMPATIBLE_EXAMPLE = {"status": "INCOMPATIBLE", "errorCode": "MODEL_ARTIFACT_INCOMPATIBLE", "message": "The stored model artifact is not compatible with the current service version."}
_MODEL_STATUS_CORRUPTED_EXAMPLE = {"status": "CORRUPTED", "errorCode": "MODEL_ARTIFACT_CORRUPTED", "message": "The stored model artifact failed integrity validation."}
_MODEL_STATUS_RESPONSES: dict[int | str, dict[str, Any]] = {200: {"content": {"application/json": {"examples": {
    "ready": {"summary": "Valid, servable artifact", "value": _MODEL_STATUS_READY_EXAMPLE},
    "missing": {"summary": "Never trained yet", "value": _MODEL_STATUS_MISSING_EXAMPLE},
    "incompatible": {"summary": "Artifact predates the current schema/feature contract", "value": _MODEL_STATUS_INCOMPATIBLE_EXAMPLE},
    "corrupted": {"summary": "Checksum or deserialization failure", "value": _MODEL_STATUS_CORRUPTED_EXAMPLE},
}}}}}


@router.get("/model/status", response_model=ModelStatusResponse, response_model_exclude_none=True, responses=_MODEL_STATUS_RESPONSES)
def model_status():
    """READY/MISSING/INCOMPATIBLE/CORRUPTED -- never leaks a filesystem path or checksum.
    Example values above are illustrative only, not measured results."""
    from app.ml import model_cache
    return _status_payload(model_cache.video_cache,MODEL_PATH,METADATA_PATH,FEATURES)

@router.get("/model/status/live", response_model=ModelStatusResponse, response_model_exclude_none=True, responses=_MODEL_STATUS_RESPONSES)
def live_model_status():
    from app.ml import model_cache
    return _status_payload(model_cache.live_cache,LIVE_MODEL_PATH,LIVE_METADATA_PATH,LIVE_FEATURES)

_METRICS_SUMMARY_EXAMPLE = {"modelVersion": "recommendation-prod-20260730143000", "selectedModel": "RandomForestClassifier",
                          "trainedAt": "2026-07-30T14:30:00+00:00", "trainingSamples": 4000, "testSamples": 800,
                          "decisionThreshold": 0.47, "trainingDurationSeconds": 14.8,
                          "metrics": {"f1Score": 0.84, "prAuc": 0.88, "rocAuc": 0.89}}


@router.get("/model/metrics/summary", response_model=ModelMetricsSummaryResponse, responses={200: {"content": {"application/json": {"example": _METRICS_SUMMARY_EXAMPLE}}}})
def metrics_summary():
    """Concise operational summary -- the full ~900-field metadata object is at
    GET /model/metrics instead. Example values above are illustrative only, not measured results."""
    try:
        return _summary_from(model_store.check_artifact_compatibility(FEATURES,model_path=MODEL_PATH,metadata_path=METADATA_PATH))
    except model_store.ArtifactNotFoundError:
        raise HTTPException(404,detail={"error":"MODEL_NOT_TRAINED","message":"No trained model metadata exists."})
    except model_store.ArtifactIncompatibleError as e:
        raise HTTPException(503,detail={"error":"MODEL_ARTIFACT_INVALID","reason":"INCOMPATIBLE","message":str(e)})
    except model_store.ArtifactCorruptedError as e:
        raise HTTPException(503,detail={"error":"MODEL_ARTIFACT_INVALID","reason":"CORRUPTED","message":str(e)})

@router.get("/model/metrics/summary/live", response_model=ModelMetricsSummaryResponse)
def live_metrics_summary():
    try:
        return _summary_from(model_store.check_artifact_compatibility(LIVE_FEATURES,model_path=LIVE_MODEL_PATH,metadata_path=LIVE_METADATA_PATH))
    except model_store.ArtifactNotFoundError:
        raise HTTPException(404,detail={"error":"LIVE_MODEL_NOT_TRAINED","message":"Train the LIVE model before requesting metrics."})
    except model_store.ArtifactIncompatibleError as e:
        raise HTTPException(503,detail={"error":"LIVE_MODEL_ARTIFACT_INVALID","reason":"INCOMPATIBLE","message":str(e)})
    except model_store.ArtifactCorruptedError as e:
        raise HTTPException(503,detail={"error":"LIVE_MODEL_ARTIFACT_INVALID","reason":"CORRUPTED","message":str(e)})

@router.get("/model/metrics/live")
def live_metrics():
    try:
        return model_store.check_artifact_compatibility(LIVE_FEATURES,model_path=LIVE_MODEL_PATH,metadata_path=LIVE_METADATA_PATH)
    except model_store.ArtifactNotFoundError:
        raise HTTPException(404,detail={"error":"LIVE_MODEL_NOT_TRAINED","message":"Train the LIVE model before requesting metrics."})
    except model_store.ArtifactIncompatibleError as e:
        raise HTTPException(503,detail={"error":"LIVE_MODEL_ARTIFACT_INVALID","reason":"INCOMPATIBLE","message":str(e)})
    except model_store.ArtifactCorruptedError as e:
        raise HTTPException(503,detail={"error":"LIVE_MODEL_ARTIFACT_INVALID","reason":"CORRUPTED","message":str(e)})


def _version_slot(metadata_path)->dict|None:
    """A version slot's summary, or None if that slot has no readable metadata (never
    raised as an error -- MISSING/CANDIDATE/PREVIOUS are all normal, expected states).
    Never leaks a filesystem path or raw exception text."""
    try:
        metadata=model_store.load_metadata(metadata_path=metadata_path)
    except FileNotFoundError:
        return None  # empty slot -- the normal, expected case this docstring describes
    except (OSError, json.JSONDecodeError):
        # The slot has a file but it's unreadable/corrupted -- not an empty-slot state.
        # Still return None (unchanged response contract), but make it observable.
        logger.warning("version slot metadata unreadable/corrupted at %s",metadata_path)
        return None
    if not isinstance(metadata,dict):
        return None
    return {"modelVersion":metadata.get("modelVersion"),"selectedModel":metadata.get("selectedModel"),
            "modelFamily":metadata.get("modelFamily") or "classifier",
            "trainedAt":metadata.get("trainedAt"),"promotedAt":metadata.get("promotedAt"),
            "rolledBackAt":metadata.get("rolledBackAt")}


def _versions_payload(model_path,metadata_path)->dict:
    """Spec §39: ACTIVE/CANDIDATE/ARCHIVED (here: previous) version info, read-only, no
    path input accepted from the caller -- always the service's own fixed artifact paths."""
    paths=artifact_lifecycle.sibling_paths(model_path,metadata_path)
    return {"active":_version_slot(metadata_path),
            "previous":_version_slot(paths["previous_metadata"]),
            "candidate":_version_slot(paths["candidate_metadata"])}


_MODEL_VERSIONS_EXAMPLE = {"active": {"modelVersion": "recommendation-prod-20260730143000", "selectedModel": "RandomForestClassifier",
                                     "trainedAt": "2026-07-30T14:30:00+00:00", "promotedAt": "2026-07-30T14:30:01+00:00", "rolledBackAt": None},
                          "previous": None, "candidate": None}


@router.get("/model/versions", response_model=ModelVersionsResponse, responses={200: {"content": {"application/json": {"example": _MODEL_VERSIONS_EXAMPLE}}}})
def model_versions():
    """Read-only view of the active/previous/candidate artifact slots used by the
    promotion/rollback lifecycle (app.ml.artifact_lifecycle). Any slot with no artifact
    present is null, never an error. Example values above are illustrative only, not
    measured results."""
    return _versions_payload(MODEL_PATH,METADATA_PATH)


@router.get("/model/versions/live", response_model=ModelVersionsResponse, responses={200: {"content": {"application/json": {"example": _MODEL_VERSIONS_EXAMPLE}}}})
def live_model_versions():
    return _versions_payload(LIVE_MODEL_PATH,LIVE_METADATA_PATH)
