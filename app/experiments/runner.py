"""Orchestrates one experiment run: isolated SQLite database + isolated model artifact
paths, synthetic dataset generation from a definition, real training via the existing
VIDEO/LIVE trainers (app.services.training_service.train / app.ml.live_trainer.train_live_model
-- no training logic duplicated here), and an immutable JSON report.

`work_dir` holds ephemeral, run-scoped scratch state (the isolated SQLite file and the
trained model binary) -- never written into `EXPERIMENT_DIR`, since experiment reports must
never contain model binaries. `experiment_dir` is where the JSON report itself is persisted
(this is `EXPERIMENT_DIR` in real usage, or a test's `tmp_path`). Callers own `work_dir`'s
lifecycle (e.g. a `tempfile.TemporaryDirectory` in the CLI entry point); this module never
deletes it, so a failed run's artifacts remain inspectable.

Policy on report-write failure after a successful training run (spec §11): if training
itself SUCCEEDED (a valid artifact was produced and promoted) but writing the experiment
report then fails, that is treated as *operational metadata* failure, not a training
failure -- the already-promoted model is never rolled back or invalidated because a report
file couldn't be written. `run_experiment` re-raises in that case so the caller (the CLI
script) sees a nonzero exit and a clear message, but the underlying training result is
already durable and correct.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core.logging import logger
from app.db.database import Base
from app.db.models import Interaction
from app.experiments.dataset_generation import generate_experiment_dataset
from app.experiments.dataset_summary import live_dataset_summary, video_dataset_summary
from app.experiments.definitions import ExperimentDefinition
from app.experiments.failures import (
    ExperimentStageError,
    insufficient_data_failure,
    stage_failure,
)
from app.experiments.report_store import write_report

__all__ = ["InsufficientExperimentDataError", "RunOutcome", "run_experiment"]


class InsufficientExperimentDataError(Exception):
    pass


@dataclass
class RunOutcome:
    status: str  # "SUCCEEDED" | "FAILED"
    run_id: str
    report_path: Path | None
    error_code: str | None = None
    error_message: str | None = None
    # Populated only on SUCCEEDED: the isolated artifact paths this run trained into, so a
    # caller (e.g. scripts/benchmark_recommendation.py) can measure model loading/inference
    # against the exact same artifact without retraining or duplicating training logic.
    # Always inside `work_dir`, never repository `models/`. `db_path` is VIDEO-only (LIVE
    # trains from an in-memory DataFrame, no database involved).
    model_path: Path | None = None
    metadata_path: Path | None = None
    db_path: Path | None = None


def _metric_subset(metadata: dict, keys: tuple[str, ...]) -> dict:
    metrics = metadata.get("metrics") or {}
    return {key: metrics.get(key) for key in keys}


def _write_report_safely(experiment_dir: Path, report: dict, *, run_id: str, experiment_id: str) -> Path | None:
    """Used only for FAILURE reports: if persisting the failure report itself fails, that
    must not raise a second exception out of an already-failing run (which could mask the
    real failure) -- log it safely and return None rather than a path."""
    try:
        return write_report(experiment_dir, report)
    except Exception:  # noqa: BLE001 -- must not raise a second exception out of an already-failing run (see docstring)
        logger.exception("failed to persist experiment FAILURE report runId=%s experimentId=%s", run_id, experiment_id)
        return None


def run_experiment(
    definition: ExperimentDefinition, *, domain: Literal["video", "live"], experiment_dir: Path, work_dir: Path,
) -> RunOutcome:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # 16 hex chars (64 bits) rather than the full 32-char uuid4().hex: still excellent
    # collision resistance for local experiment tracking (not a cryptographic security
    # requirement), and short enough to avoid Windows MAX_PATH (260 chars) when combined
    # with a deeply-nested work/experiment directory.
    run_id = uuid.uuid4().hex[:16]
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    model_type = domain.upper()

    base = {
        "runId": run_id, "experimentId": definition.experiment_id, "datasetVersion": definition.dataset_version,
        "modelType": model_type, "reportKind": "TRAINING",
        "synthetic": definition.synthetic, "startedAt": started_at.isoformat(),
    }

    try:
        if domain == "video":
            metadata, dataset, model_path, metadata_path, db_path, dataset_seconds = _run_video(definition, work_dir, run_id)
        else:
            metadata, dataset, model_path, metadata_path, db_path, dataset_seconds = _run_live(definition, work_dir, run_id)
    except ExperimentStageError as exc:
        # Full detail (including the original exception, which may contain a path, a DSN, or
        # other internal detail) is logged server-side only, tied to runId/experimentId.
        logger.exception("experiment run failed runId=%s experimentId=%s stage=%s",
                          run_id, definition.experiment_id, exc.stage, exc_info=exc.original)
        safe = exc.safe_failure
        report = {
            **base, "status": "FAILED", "completedAt": datetime.now(timezone.utc).isoformat(),
            "trainingDurationSeconds": round(time.perf_counter() - started_perf, 3),
            "totalExperimentSeconds": round(time.perf_counter() - started_perf, 3),
            "errorCode": safe.code, "errorMessage": safe.message,
        }
        report_path = _write_report_safely(experiment_dir, report, run_id=run_id, experiment_id=definition.experiment_id)
        return RunOutcome(status="FAILED", run_id=run_id, report_path=report_path,
                           error_code=safe.code, error_message=safe.message)
    except Exception:  # noqa: BLE001 -- catch-all for anything not already classified into a pipeline stage (see comment below)
        # Anything not already classified into a pipeline stage -- still never persists
        # str(exc); full detail goes to the log only.
        logger.exception("experiment run failed (unclassified) runId=%s experimentId=%s", run_id, definition.experiment_id)
        safe = stage_failure("TRAINING")
        report = {
            **base, "status": "FAILED", "completedAt": datetime.now(timezone.utc).isoformat(),
            "trainingDurationSeconds": round(time.perf_counter() - started_perf, 3),
            "totalExperimentSeconds": round(time.perf_counter() - started_perf, 3),
            "errorCode": safe.code, "errorMessage": safe.message,
        }
        report_path = _write_report_safely(experiment_dir, report, run_id=run_id, experiment_id=definition.experiment_id)
        return RunOutcome(status="FAILED", run_id=run_id, report_path=report_path,
                           error_code=safe.code, error_message=safe.message)

    try:
        artifact_size = model_path.stat().st_size
    except OSError:
        artifact_size = None

    # Distinguish "training time" (the real trainer's own duration -- split-lifecycle fit,
    # calibration, threshold tuning, evaluation; from metadata, unaffected by this module)
    # from "complete experiment time" (this function's total wall time: synthetic dataset
    # generation + isolation setup + training + artifact promotion + report write) -- the
    # two answer different questions and neither should be presented as the other.
    report = {
        **base, "status": "SUCCEEDED", "completedAt": datetime.now(timezone.utc).isoformat(),
        "modelVersion": metadata.get("modelVersion"), "selectedModel": metadata.get("selectedModel"),
        "decisionThreshold": metadata.get("decisionThreshold"),
        "trainingDurationSeconds": metadata.get("trainingDurationSeconds"),
        "datasetGenerationSeconds": dataset_seconds,
        "totalExperimentSeconds": round(time.perf_counter() - started_perf, 3),
        "metrics": _metric_subset(metadata, ("accuracy", "precision", "recall", "f1Score", "prAuc", "rocAuc")),
        "rankingMetrics": _metric_subset(metadata, ("precisionAt5", "recallAt10", "ndcgAt10")),
        "dataset": dataset,
        "artifactSizeBytes": artifact_size,
        "datasetSource": metadata.get("datasetSource"),
        "warnings": [],
    }
    # Report-write failure policy: the model was already trained and promoted successfully
    # by this point, and this function does not (and must not) attempt to undo that. The
    # failure is still never allowed to leak raw exception text -- it's logged safely and
    # re-raised as a clean, safe exception (`from None` deliberately breaks the chain so a
    # caller that does str() on it never sees the original).
    try:
        report_path = write_report(experiment_dir, report)
    except Exception:  # noqa: BLE001 -- report-write failure policy: the model is already trained/promoted, must not be undone (see comment above)
        logger.exception("failed to persist experiment SUCCESS report runId=%s experimentId=%s (model already promoted)",
                          run_id, definition.experiment_id)
        safe = stage_failure("REPORT")
        raise RuntimeError(safe.message) from None
    return RunOutcome(status="SUCCEEDED", run_id=run_id, report_path=report_path,
                       model_path=model_path, metadata_path=metadata_path, db_path=db_path)


def _check_sufficient(dataset: dict) -> None:
    if dataset["labelledSamples"] < 100 or dataset["positiveSamples"] == 0 or dataset["negativeSamples"] == 0:
        original = InsufficientExperimentDataError(
            f"Only {dataset['labelledSamples']} labelled rows with "
            f"{dataset['positiveSamples']} positive / {dataset['negativeSamples']} negative "
            "(need >=100 labelled rows containing both classes)."
        )
        safe = insufficient_data_failure(
            labelled=dataset["labelledSamples"], positive=dataset["positiveSamples"], negative=dataset["negativeSamples"],
        )
        raise ExperimentStageError("INSUFFICIENT_DATA", original, safe_failure=safe)


def _run_video(definition: ExperimentDefinition, work_dir: Path, run_id: str):
    db_path = work_dir / f"{run_id}.db"
    try:
        engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        db = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()
    except Exception as exc:
        raise ExperimentStageError("CONFIGURATION", exc) from exc
    try:
        try:
            generation_started = time.perf_counter()
            generate_experiment_dataset(db, definition)
            rows = db.scalars(select(Interaction)).all()
            dataset_seconds = round(time.perf_counter() - generation_started, 3)
        except Exception as exc:
            raise ExperimentStageError("GENERATION", exc) from exc
        dataset = video_dataset_summary(rows)
        _check_sufficient(dataset)

        # Files live directly in work_dir (no extra "<runId>-model" subdirectory level):
        # work_dir is already unique per run in every call site, so the run_id prefix on
        # the filenames alone is sufficient for uniqueness -- and every path segment saved
        # here matters, because model_store.save()'s own tmp-metadata filename (which must
        # not be touched -- it's core ML persistence, not this module's concern) is fairly
        # long, and a deeply-nested work_dir (e.g. under a long pytest basetemp) can
        # otherwise exceed Windows MAX_PATH (260 chars).
        model_dir = work_dir
        model_path, metadata_path = model_dir / f"{run_id}-model.joblib", model_dir / f"{run_id}-model.json"

        from app.ml import model_store
        from app.services import training_service
        original_paths = (model_store.MODEL_PATH, model_store.METADATA_PATH, model_store.MODEL_DIR)
        model_store.MODEL_PATH, model_store.METADATA_PATH, model_store.MODEL_DIR = model_path, metadata_path, model_dir
        try:
            metadata = training_service.train(db)
        except training_service.InsufficientData as exc:
            raise ExperimentStageError("INSUFFICIENT_DATA", exc) from exc
        except model_store.ArtifactError as exc:
            raise ExperimentStageError("ARTIFACT", exc) from exc
        except Exception as exc:
            from app.ml import artifact_lifecycle
            stage = "ARTIFACT" if isinstance(exc, artifact_lifecycle.PromotionValidationError) else "TRAINING"
            raise ExperimentStageError(stage, exc) from exc
        finally:
            model_store.MODEL_PATH, model_store.METADATA_PATH, model_store.MODEL_DIR = original_paths
        return metadata, dataset, model_path, metadata_path, db_path, dataset_seconds
    finally:
        db.close()
        # dispose(), not just close(): SQLAlchemy's connection pool can otherwise keep the
        # underlying SQLite file handle open on Windows, which silently blocks a later
        # shutil.rmtree(work_dir, ignore_errors=True) from actually removing it. A fresh
        # engine can still be opened against the same db_path afterward (e.g. the
        # benchmark script re-reading it) -- disposing this one doesn't affect that.
        engine.dispose()


def _run_live(definition: ExperimentDefinition, work_dir: Path, run_id: str):
    from app.ml.live_synthetic_data import generate_synthetic_live_dataset
    try:
        generation_started = time.perf_counter()
        live_df = generate_synthetic_live_dataset(definition.interactions, seed=definition.seed)
        dataset = live_dataset_summary(live_df, is_synthetic=True)
        dataset_seconds = round(time.perf_counter() - generation_started, 3)
    except Exception as exc:
        raise ExperimentStageError("GENERATION", exc) from exc
    _check_sufficient(dataset)

    # See the matching comment in _run_video: no extra subdirectory level, to stay clear of
    # Windows MAX_PATH under a deeply-nested work_dir.
    model_dir = work_dir
    model_path, metadata_path = model_dir / f"{run_id}-live.joblib", model_dir / f"{run_id}-live.json"

    from app.ml import live_trainer, model_store
    original_paths = (live_trainer.LIVE_MODEL_PATH, live_trainer.LIVE_METADATA_PATH, live_trainer.MODEL_DIR)
    live_trainer.LIVE_MODEL_PATH, live_trainer.LIVE_METADATA_PATH, live_trainer.MODEL_DIR = model_path, metadata_path, model_dir
    try:
        metadata = live_trainer.train_live_model(
            dataset=live_df, random_seed=definition.seed, dataset_provenance="synthetic",
        )
    except live_trainer.InsufficientLiveData as exc:
        raise ExperimentStageError("INSUFFICIENT_DATA", exc) from exc
    except model_store.ArtifactError as exc:
        raise ExperimentStageError("ARTIFACT", exc) from exc
    except Exception as exc:
        from app.ml import artifact_lifecycle
        stage = "ARTIFACT" if isinstance(exc, artifact_lifecycle.PromotionValidationError) else "TRAINING"
        raise ExperimentStageError(stage, exc) from exc
    finally:
        live_trainer.LIVE_MODEL_PATH, live_trainer.LIVE_METADATA_PATH, live_trainer.MODEL_DIR = original_paths
    return metadata, dataset, model_path, metadata_path, None, dataset_seconds
