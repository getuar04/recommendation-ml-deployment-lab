"""Session 5 (spec §60): actual, low-overhead performance measurement through the real code
paths -- never hard-coded, never against the real running stack or repository `models/`.

Measures, all via `time.perf_counter()`, reported in seconds:
  - datasetGenerationSeconds / trainingDurationSeconds (via the existing experiment runner)
  - coldModelLoadSeconds: model_cache after an explicit invalidate() -- full validated load
  - warmModelLoadSeconds: the same cache entry, already warm -- NOT equivalent to cold startup
  - candidateGenerationSeconds: the real candidate-sourcing flow
  - recommendationInferenceSeconds: the real scoring + reranking flow (separate from candidate generation)
  - artifactSizeBytes

Usage:
    python -m scripts.benchmark_recommendation --experiment small_balanced --domain video --describe
    python -m scripts.benchmark_recommendation --experiment small_balanced --domain video
    python -m scripts.benchmark_recommendation --experiment small_balanced --domain live

Trains into an isolated work directory (never repository `models/`) and, unless
--no-report is passed, persists one new, distinct experiment report (via the same
`write_report()` every other run uses -- never edits an existing report) under
--experiment-dir (defaults to EXPERIMENT_DIR) with a "performance" block attached.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from app.experiments.definitions import DEFINITIONS

BENCHMARK_CANDIDATE_COUNT = 20


def _describe(experiment_id: str, domain: str) -> int:
    definition = DEFINITIONS[experiment_id]
    print(f"experimentId={definition.experiment_id} domain={domain} synthetic=True")
    print("Would train via the real pipeline into an isolated work directory (never repository models/), then measure:")
    print("  - datasetGenerationSeconds, trainingDurationSeconds (already part of every experiment report)")
    print("  - coldModelLoadSeconds (model_cache after invalidate()) vs warmModelLoadSeconds (same cache entry, warm)")
    print("  - candidateGenerationSeconds (the real candidate flow)")
    print("  - recommendationInferenceSeconds (the real scoring + reranking flow)")
    print("No training, no writes, in --describe mode.")
    return 0


def _measure_video(model_path: Path, metadata_path: Path, db_path: Path) -> dict:
    import time as _time

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.api.candidate_routes import generate as generate_candidates
    from app.ml import model_cache, model_store
    from app.ml.dataset_builder import FEATURES
    from app.schemas.candidate_schemas import CandidateGenerationRequest
    from app.schemas.recommendation_schemas import Candidate as RecCandidate
    from app.schemas.recommendation_schemas import RecommendationRequest
    from app.services import recommendation_service
    from app.services.recommendation_service import recommend

    original = (model_store.MODEL_PATH, model_store.METADATA_PATH, recommendation_service.MODEL_PATH)
    model_store.MODEL_PATH = model_path
    model_store.METADATA_PATH = metadata_path
    recommendation_service.MODEL_PATH = model_path
    try:
        model_cache.video_cache.invalidate()
        started = _time.perf_counter()
        model_cache.video_cache.get(model_path, metadata_path, expected_features=FEATURES)
        cold_seconds = round(_time.perf_counter() - started, 6)

        started = _time.perf_counter()
        model_cache.video_cache.get(model_path, metadata_path, expected_features=FEATURES)
        warm_seconds = round(_time.perf_counter() - started, 6)

        engine = create_engine(f"sqlite:///{db_path}")
        db = sessionmaker(bind=engine)()
        try:
            candidate_request = CandidateGenerationRequest(userId="benchmark-user", limit=BENCHMARK_CANDIDATE_COUNT)
            started = _time.perf_counter()
            candidate_response = generate_candidates(candidate_request, db)
            candidate_seconds = round(_time.perf_counter() - started, 6)
            raw_candidates = candidate_response["candidates"]
            candidate_count = len(raw_candidates)

            rec_candidates = [
                RecCandidate(
                    contentId=c["contentId"], creatorId=c["creatorId"], category=c["category"],
                    contentPopularityScore=c["contentPopularityScore"], contentAgeHours=c["contentAgeHours"],
                    creatorFollowed=c["creatorFollowed"], alreadySeen=c["alreadySeen"],
                    title=c.get("title"), hashtags=c.get("hashtags", []), topics=c.get("topics", []),
                    entities=c.get("entities", []), subgenres=c.get("subgenres", []),
                    language=None, candidateSource=None, socialContext=None, localBucketSource=None,
                )
                for c in raw_candidates
            ]
            inference_seconds = None
            if rec_candidates:
                request = RecommendationRequest(userId="benchmark-user", limit=min(10, len(rec_candidates)), candidates=rec_candidates,
                                                 userProfile=None, searchIntent=None, userContext=None)
                started = _time.perf_counter()
                recommend(db, request)
                inference_seconds = round(_time.perf_counter() - started, 6)
        finally:
            db.close()
            engine.dispose()  # release the SQLite file handle so the caller's work_dir cleanup can actually delete it
    finally:
        model_store.MODEL_PATH, model_store.METADATA_PATH, recommendation_service.MODEL_PATH = original

    return {
        "coldModelLoadSeconds": cold_seconds, "warmModelLoadSeconds": warm_seconds,
        "candidateGenerationSeconds": candidate_seconds, "candidateCount": candidate_count,
        "recommendationInferenceSeconds": inference_seconds,
    }


def _benchmark_live_candidates() -> list:
    """Small, deterministic, hand-authored LIVE candidates for benchmark timing only -- not
    part of any experiment dataset, not persisted, clearly synthetic."""
    from app.schemas.live_schemas import LiveCandidate, LiveStatus
    return [
        LiveCandidate(
            streamId=f"bench-stream-{i}", creatorId=f"bench-creator-{i % 5}", category="GAMING", status=LiveStatus.ACTIVE,
            currentViewerCount=100 + i * 10, viewerGrowthRate=0.05, liveAgeMinutes=15.0,
            creatorFollowed=(i % 3 == 0), previousLiveInteractions=i % 4, previousLiveWatchTime=float(i * 20),
            region="eu-central-1", language="en", regionMatch=True, languageMatch=True, alreadyJoined=False,
            startedAt=None,
        )
        for i in range(BENCHMARK_CANDIDATE_COUNT)
    ]


def _measure_live(model_path: Path, metadata_path: Path) -> dict:
    import time as _time

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.api.candidate_routes import generate_live as generate_live_candidates
    from app.db.database import Base
    from app.ml import live_trainer, model_cache
    from app.ml.live_feature_builder import LIVE_FEATURES
    from app.schemas.candidate_schemas import LiveCandidateGenerationRequest
    from app.schemas.live_schemas import LiveRecommendationRequest
    from app.services import live_recommendation_service
    from app.services.live_recommendation_service import recommend_live

    original = (live_trainer.LIVE_MODEL_PATH, live_trainer.LIVE_METADATA_PATH,
                live_recommendation_service.LIVE_MODEL_PATH, live_recommendation_service.LIVE_METADATA_PATH)
    live_trainer.LIVE_MODEL_PATH = model_path
    live_trainer.LIVE_METADATA_PATH = metadata_path
    live_recommendation_service.LIVE_MODEL_PATH = model_path
    live_recommendation_service.LIVE_METADATA_PATH = metadata_path
    # In-memory only: this benchmark measures the caller-supplied-history fallback path (the
    # real-history lookup this service now performs on serving) -- no isolated on-disk DB is
    # needed since no real LIVE history is being measured here, unlike VIDEO's db_path above.
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        model_cache.live_cache.invalidate()
        started = _time.perf_counter()
        model_cache.live_cache.get(model_path, metadata_path, expected_features=LIVE_FEATURES)
        cold_seconds = round(_time.perf_counter() - started, 6)

        started = _time.perf_counter()
        model_cache.live_cache.get(model_path, metadata_path, expected_features=LIVE_FEATURES)
        warm_seconds = round(_time.perf_counter() - started, 6)

        streams = _benchmark_live_candidates()
        candidate_request = LiveCandidateGenerationRequest(userId="benchmark-user", limit=BENCHMARK_CANDIDATE_COUNT, streams=streams)
        started = _time.perf_counter()
        candidate_response = generate_live_candidates(candidate_request)
        candidate_seconds = round(_time.perf_counter() - started, 6)
        candidate_count = len(candidate_response["candidates"])

        rec_request = LiveRecommendationRequest(userId="benchmark-user", limit=min(10, len(streams)), candidates=streams)
        started = _time.perf_counter()
        recommend_live(rec_request, db)
        inference_seconds = round(_time.perf_counter() - started, 6)
    finally:
        db.close()
        engine.dispose()
        (live_trainer.LIVE_MODEL_PATH, live_trainer.LIVE_METADATA_PATH,
         live_recommendation_service.LIVE_MODEL_PATH, live_recommendation_service.LIVE_METADATA_PATH) = original

    return {
        "coldModelLoadSeconds": cold_seconds, "warmModelLoadSeconds": warm_seconds,
        "candidateGenerationSeconds": candidate_seconds, "candidateCount": candidate_count,
        "recommendationInferenceSeconds": inference_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=sorted(DEFINITIONS))
    parser.add_argument("--domain", choices=("video", "live"), required=True)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--experiment-dir", default=None)
    parser.add_argument("--no-report", action="store_true", help="Print JSON only; do not persist a new experiment report.")
    args = parser.parse_args(argv)

    if args.describe:
        return _describe(args.experiment, args.domain)

    from app.experiments.report_store import write_report
    from app.experiments.runner import run_experiment

    definition = DEFINITIONS[args.experiment]
    if args.experiment_dir is not None:
        experiment_dir = Path(args.experiment_dir)
    else:
        from app.core.config import EXPERIMENT_DIR
        experiment_dir = EXPERIMENT_DIR
    work_dir = Path(tempfile.mkdtemp(prefix="benchmark-run-"))

    try:
        print(f"Training '{definition.experiment_id}' (domain={args.domain}) into an isolated work directory...")
        outcome = run_experiment(definition, domain=args.domain, experiment_dir=experiment_dir, work_dir=work_dir)
        if outcome.status != "SUCCEEDED":
            result = {"status": outcome.status, "errorCode": outcome.error_code, "errorMessage": outcome.error_message}
            print(__import__("json").dumps(result, indent=2))
            return 1

        print("Training succeeded; measuring model load / candidate generation / inference...")
        # A SUCCEEDED outcome always sets model_path/metadata_path/report_path -- only a FAILED
        # run (already returned above) leaves them None. db_path is VIDEO-only by design (LIVE
        # training uses no SQLite database at all) -- see RunOutcome / app.experiments.runner.
        assert outcome.model_path is not None and outcome.metadata_path is not None
        assert outcome.report_path is not None
        if args.domain == "video":
            assert outcome.db_path is not None
            performance = _measure_video(outcome.model_path, outcome.metadata_path, outcome.db_path)
        else:
            performance = _measure_live(outcome.model_path, outcome.metadata_path)

        success_result = {"status": "SUCCEEDED", "runId": outcome.run_id, "experimentId": definition.experiment_id,
                           "domain": args.domain, "performance": performance}

        if not args.no_report:
            import json as _json
            base_report = _json.loads(outcome.report_path.read_text(encoding="utf-8"))
            benchmark_report = {
                **base_report,
                "runId": outcome.run_id + "-bench",
                "reportKind": "BENCHMARK",
                "sourceRunId": outcome.run_id,
                "performance": performance,
            }
            report_path = write_report(experiment_dir, benchmark_report)
            success_result["benchmarkReportPath"] = report_path.name  # filename only -- never a full path

        print(__import__("json").dumps(success_result, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
