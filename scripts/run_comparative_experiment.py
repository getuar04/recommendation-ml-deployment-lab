"""Run one comparative experiment ("same algorithm, different dataset") end-to-end against
the REAL running application (real Postgres, real async training job/lock, real model cache,
real `/recommendations` scoring) -- never an isolated temp SQLite database, unlike
`scripts/run_experiment.py`. Meant to run inside the app container, in the comparative
environment for exactly one dataset (see README "Comparative Experiments" and
`.env.experiment-*.env` / `.env.experiment-*-v2.env`):

    docker compose --env-file .env --env-file .env.experiment-sport.env exec app \
        python -m scripts.run_comparative_experiment --dataset sport --seed

`--experiment-version` defaults to `v1`, so the command above -- the exact command documented
before the v2 rigor pass existed -- is unchanged: same seeding logic, same single fixed
scenario, same report shape, no invariant gate, no raw-score capture. Pass
`--experiment-version v2` for the rigor-pass pipeline (a v2 profile,
`.env.experiment-sport-v2.env`): fail-fast pre-training invariants, a symmetric primary
scenario (raw model scores captured via `app.experiments.comparative_scoring`, parity-checked
against the real HTTP response) plus a separate secondary business-reranking scenario, and a
fail-fast post-training invariant gate before the report is written.

v1 steps (all against the real HTTP API on `--base-url`, no training/prediction logic
reimplemented):

1. (optional, `--seed`) seed the dataset.
2. `POST /model/train` (the real async job).
3. Poll `GET /model/train/jobs/{jobId}` until SUCCEEDED or FAILED.
4. `GET /model/metrics` for the full persisted artifact metadata.
5. `GET /model/status` to confirm the artifact is READY.
6. `POST /recommendations` with the one fixed scenario.
7. Assemble and persist one JSON report under `<EXPERIMENT_DIR>/comparative/<experimentId>/`.
8. Print the report JSON to stdout for host-side capture.

v2 adds, between steps 5 and 7: a pre-training invariant gate (before step 2), the primary
scenario scored both via the real HTTP endpoint (parity reference) and in-process with raw
score capture (`app.experiments.comparative_scoring.score_with_raw_capture`), a parity
assertion between the two, a separate secondary-scenario HTTP call, and a post-training
invariant gate immediately before the report is written -- on any invariant failure, nothing
is written and the process exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.experiments.comparative_definitions import (
    COMPARATIVE_DEFINITIONS,
    ComparativeExperimentDefinition,
)
from app.experiments.comparative_definitions_v2 import COMPARATIVE_DEFINITIONS_V2

_DEFINITIONS_BY_VERSION = {"v1": COMPARATIVE_DEFINITIONS, "v2": COMPARATIVE_DEFINITIONS_V2}

DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_TRAIN_TIMEOUT_SECONDS = 300.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 60.0


def _locked_algorithm_hyperparameters(algorithm: str, model_random_state: int) -> dict[str, Any]:
    """The real, currently-effective hyperparameters for the locked algorithm -- read from
    the actual `app.ml.trainer` candidate-model constructor (not a hand-copied literal that
    could silently drift from what training really used), via `get_params()`."""
    from app.ml.trainer import _candidate_models
    pipeline = _candidate_models(model_random_state, only=algorithm)[algorithm]
    return pipeline.named_steps["model"].get_params()


def _poll_job(client, base_url: str, job_id: str, *, timeout_seconds: float, poll_interval_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = client.get(f"{base_url}/model/train/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"SUCCEEDED", "FAILED"}:
            return job
        if time.monotonic() > deadline:
            raise TimeoutError(f"Training job {job_id} did not reach SUCCEEDED/FAILED within {timeout_seconds}s (last status={job['status']}).")
        time.sleep(poll_interval_seconds)


def _run_v1(
    definition: ComparativeExperimentDefinition, *, dataset_key: str, base_url: str, experiment_dir: Path,
    do_seed: bool, train_timeout_seconds: float, poll_interval_seconds: float, http_timeout_seconds: float,
) -> dict[str, Any]:
    """Unchanged from before the v2 rigor pass -- byte-for-byte the same steps, same report
    shape, no invariant gate, no raw-score capture, single fixed scenario."""
    import httpx

    from app.experiments.comparative_dataset_generation import category_distribution
    from app.experiments.dataset_summary import video_dataset_summary
    from app.experiments.fixed_scenario import (
        FIXED_CANDIDATE_CATEGORIES,
        FIXED_RECOMMENDATION_LIMIT,
        FIXED_TEST_USER_ID,
        fixed_recommendation_request,
    )
    from app.experiments.report_store import write_report
    from app.ml.trainer import RANDOM_SEED as MODEL_RANDOM_STATE

    dataset_summary: dict[str, Any] | None = None
    realized_categories: dict[str, float] | None = None
    if do_seed:
        from scripts.seed_comparative_dataset import _seed as seed_dataset
        seed_dataset(dataset_key, version="v1")

    started_perf = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    with httpx.Client(timeout=http_timeout_seconds) as client:
        from sqlalchemy import select

        from app.db.database import SessionLocal
        from app.db.models import Interaction
        db = SessionLocal()
        try:
            rows = db.scalars(
                select(Interaction).where(Interaction.event_id.like(f"syn-cmp-{definition.experiment_id}-%"))
            ).all()
            dataset_summary = video_dataset_summary(rows)
            realized_categories = category_distribution(rows)
        finally:
            db.close()

        train_response = client.post(f"{base_url}/model/train")
        train_response.raise_for_status()
        job_id = train_response.json()["jobId"]
        job = _poll_job(client, base_url, job_id, timeout_seconds=train_timeout_seconds, poll_interval_seconds=poll_interval_seconds)

        if job["status"] != "SUCCEEDED":
            report = {
                "experimentId": definition.experiment_id, "runId": job_id, "modelType": "VIDEO",
                "reportKind": "COMPARATIVE_TRAINING", "status": "FAILED",
                "datasetVersion": definition.dataset_version, "synthetic": True,
                "startedAt": started_at, "completedAt": datetime.now(timezone.utc).isoformat(),
                "totalExperimentSeconds": round(time.perf_counter() - started_perf, 3),
                "errorCode": job.get("error", {}).get("error"), "errorMessage": job.get("error", {}).get("message"),
                "dataset": dataset_summary, "categoryDistribution": realized_categories,
            }
            path = write_report(experiment_dir, report)
            report["_reportPath"] = str(path)
            return report

        metrics_response = client.get(f"{base_url}/model/metrics")
        metrics_response.raise_for_status()
        metadata = metrics_response.json()

        status_response = client.get(f"{base_url}/model/status")
        status_response.raise_for_status()
        model_status = status_response.json()

        recommend_response = client.post(f"{base_url}/recommendations", json=fixed_recommendation_request())
        recommend_response.raise_for_status()
        recommendations = recommend_response.json()

    report = {
        "experimentId": definition.experiment_id, "runId": job_id, "modelType": "VIDEO",
        "reportKind": "COMPARATIVE_TRAINING", "status": "SUCCEEDED",
        "datasetVersion": definition.dataset_version, "synthetic": True,
        "startedAt": started_at, "completedAt": datetime.now(timezone.utc).isoformat(),
        "totalExperimentSeconds": round(time.perf_counter() - started_perf, 3),

        "datasetName": definition.experiment_id,
        "datasetProvenance": "synthetic (app.experiments.comparative_dataset_generation)",
        "dominantCategory": definition.dominant_category,
        "categoryMappingNote": definition.category_mapping_note or None,
        "categoryWeightsTarget": definition.category_weights,
        "categoryDistributionRealized": realized_categories,
        "dataset": dataset_summary,

        "algorithm": definition.algorithm,
        "algorithmHyperparameters": _locked_algorithm_hyperparameters(definition.algorithm, MODEL_RANDOM_STATE),
        "datasetGenerationSeed": definition.seed,
        "modelRandomState": MODEL_RANDOM_STATE,
        "modelComparison": metadata.get("modelComparison"),
        "selectedModel": metadata.get("selectedModel"),

        "modelVersion": metadata.get("modelVersion"),
        "trainingSamples": metadata.get("trainingSamples"),
        "modelSelectionSamples": metadata.get("modelSelectionSamples"),
        "calibrationSamples": metadata.get("calibrationSamples"),
        "thresholdTuningSamples": metadata.get("thresholdTuningSamples"),
        "testSamples": metadata.get("testSamples"),
        "trainingDurationSeconds": metadata.get("trainingDurationSeconds"),
        "decisionThreshold": metadata.get("decisionThreshold"),
        "metrics": metadata.get("metrics"),
        "rankingMetrics": {
            key: metadata.get("metrics", {}).get(key)
            for key in ("precisionAt5", "recallAt10", "ndcgAt10") if key in (metadata.get("metrics") or {})
        },
        "artifactChecksum": metadata.get("artifactChecksum"),
        "modelStatus": model_status,

        "fixedScenario": {
            "userId": FIXED_TEST_USER_ID, "limit": FIXED_RECOMMENDATION_LIMIT,
            "candidateCategories": FIXED_CANDIDATE_CATEGORIES, "candidateCount": len(fixed_recommendation_request()["candidates"]),
        },
        "recommendations": recommendations,
        "top10": recommendations.get("recommendations", [])[:10],
    }
    path = write_report(experiment_dir, report)
    report["_reportPath"] = str(path)
    return report


def _run_v2(
    definition: ComparativeExperimentDefinition, *, dataset_key: str, base_url: str, experiment_dir: Path,
    do_seed: bool, train_timeout_seconds: float, poll_interval_seconds: float, http_timeout_seconds: float,
) -> dict[str, Any]:
    import httpx
    from sqlalchemy import select

    from app.db.database import SessionLocal
    from app.db.models import Interaction
    from app.experiments.comparative_dataset_generation import category_distribution
    from app.experiments.comparative_invariants import (
        validate_parity,
        validate_pre_training,
        validate_report_invariants_v2,
    )
    from app.experiments.comparative_scoring import score_with_raw_capture
    from app.experiments.dataset_summary import video_dataset_summary
    from app.experiments.fixed_scenario import (
        FIXED_TEST_USER_ID,
        PRIMARY_CANDIDATE_CATEGORIES,
        PRIMARY_FIXED_CANDIDATES,
        PRIMARY_RECOMMENDATION_LIMIT,
        primary_fixed_recommendation_request,
        secondary_fixed_recommendation_request,
    )
    from app.experiments.report_store import write_report
    from app.ml.trainer import RANDOM_SEED as MODEL_RANDOM_STATE
    from app.schemas.recommendation_schemas import RecommendationRequest

    if do_seed:
        from scripts.seed_comparative_dataset import _seed as seed_dataset
        seed_dataset(dataset_key, version="v2")

    started_perf = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    db = SessionLocal()
    try:
        # Pre-training gate (spec section 1/5/6): EXPERIMENT_ID must match, no foreign
        # comparative rows, realized class balance must exactly equal the configured target.
        validate_pre_training(db, definition, version="v2")

        rows = db.scalars(
            select(Interaction).where(Interaction.event_id.like(f"syn-cmp-{definition.experiment_id}-%"))
        ).all()
        dataset_summary = video_dataset_summary(rows)
        realized_categories = category_distribution(rows)

        with httpx.Client(timeout=http_timeout_seconds) as client:
            train_response = client.post(f"{base_url}/model/train")
            train_response.raise_for_status()
            job_id = train_response.json()["jobId"]
            job = _poll_job(client, base_url, job_id, timeout_seconds=train_timeout_seconds, poll_interval_seconds=poll_interval_seconds)

            if job["status"] != "SUCCEEDED":
                report = {
                    "experimentId": definition.experiment_id, "runId": job_id, "modelType": "VIDEO",
                    "reportKind": "COMPARATIVE_TRAINING_V2", "status": "FAILED", "experimentVersion": "v2",
                    "datasetVersion": definition.dataset_version, "synthetic": True,
                    "startedAt": started_at, "completedAt": datetime.now(timezone.utc).isoformat(),
                    "totalExperimentSeconds": round(time.perf_counter() - started_perf, 3),
                    "errorCode": job.get("error", {}).get("error"), "errorMessage": job.get("error", {}).get("message"),
                    "dataset": dataset_summary, "categoryDistribution": realized_categories,
                }
                path = write_report(experiment_dir, report)
                report["_reportPath"] = str(path)
                return report

            metrics_response = client.get(f"{base_url}/model/metrics")
            metrics_response.raise_for_status()
            metadata = metrics_response.json()

            status_response = client.get(f"{base_url}/model/status")
            status_response.raise_for_status()
            model_status = status_response.json()

            # Primary (symmetric) scenario via the real HTTP endpoint -- the canonical
            # adjusted-score/rank/reason/strategy evidence, and the parity reference.
            primary_http_response = client.post(f"{base_url}/recommendations", json=primary_fixed_recommendation_request()).json()

            # Secondary (asymmetric, business-reranking) scenario -- real HTTP only, reported
            # separately, never consumed by the ML-learning verdict.
            secondary_http_response = client.post(f"{base_url}/recommendations", json=secondary_fixed_recommendation_request()).json()

        # Primary scenario again, in-process, with raw (pre-reranking) score capture -- calls
        # the exact same recommend() the HTTP endpoint calls (see
        # app.experiments.comparative_scoring); nothing here reimplements feature
        # construction, prediction, or reranking.
        primary_request = RecommendationRequest.model_validate(primary_fixed_recommendation_request())
        primary_in_process_response, raw_scores = score_with_raw_capture(db, primary_request)

        # Parity gate (spec section 4/7): the in-process path and the real HTTP path must
        # agree exactly -- same modelVersion, strategy, adjusted scores, order, and reasons.
        validate_parity(primary_in_process_response, primary_http_response)
    finally:
        db.close()

    expected_hyperparameters = _locked_algorithm_hyperparameters(definition.algorithm, MODEL_RANDOM_STATE)
    expected_fixed_scenario = {
        "userId": FIXED_TEST_USER_ID, "limit": PRIMARY_RECOMMENDATION_LIMIT,
        "candidateCategories": PRIMARY_CANDIDATE_CATEGORIES, "candidateCount": len(PRIMARY_FIXED_CANDIDATES),
    }

    report = {
        "experimentId": definition.experiment_id, "runId": job_id, "modelType": "VIDEO",
        "reportKind": "COMPARATIVE_TRAINING_V2", "status": "SUCCEEDED", "experimentVersion": "v2",
        "datasetVersion": definition.dataset_version, "synthetic": True,
        "startedAt": started_at, "completedAt": datetime.now(timezone.utc).isoformat(),
        "totalExperimentSeconds": round(time.perf_counter() - started_perf, 3),

        # 1. Experimental controls
        "algorithm": definition.algorithm,
        "algorithmHyperparameters": expected_hyperparameters,
        "datasetGenerationSeed": definition.seed,
        "modelRandomState": MODEL_RANDOM_STATE,
        "featureNames": metadata.get("featureNames"),
        "splitStrategy": metadata.get("splitStrategy"),
        "splitLifecycleUsedFallback": metadata.get("splitLifecycleUsedFallback"),
        "fixedScenario": expected_fixed_scenario,

        # 2. Dataset treatment
        "datasetName": definition.experiment_id,
        "datasetProvenance": "synthetic (app.experiments.comparative_dataset_generation, v2: fixed reference timestamp + deterministic post-generation class-balance stratification)",
        "dominantCategory": definition.dominant_category,
        "categoryMappingNote": definition.category_mapping_note or None,
        "categoryWeightsTarget": definition.category_weights,

        # 3. Realized dataset statistics
        "categoryDistributionRealized": realized_categories,
        "dataset": dataset_summary,

        # Model identity/artifact evidence
        "modelComparison": metadata.get("modelComparison"),
        "selectedModel": metadata.get("selectedModel"),
        "modelVersion": metadata.get("modelVersion"),
        "trainingSamples": metadata.get("trainingSamples"),
        "modelSelectionSamples": metadata.get("modelSelectionSamples"),
        "calibrationSamples": metadata.get("calibrationSamples"),
        "thresholdTuningSamples": metadata.get("thresholdTuningSamples"),
        "testSamples": metadata.get("testSamples"),
        "trainingDurationSeconds": metadata.get("trainingDurationSeconds"),
        "decisionThreshold": metadata.get("decisionThreshold"),
        "metrics": metadata.get("metrics"),
        "rankingMetrics": {
            key: metadata.get("metrics", {}).get(key)
            for key in ("precisionAt5", "recallAt10", "ndcgAt10") if key in (metadata.get("metrics") or {})
        },
        "artifactChecksum": metadata.get("artifactChecksum"),
        "modelStatus": model_status,

        # 4. Raw model-score evidence (primary scenario only -- this is what the verdict uses)
        "rawModelScores": raw_scores,

        # 5. Post-reranking evidence (primary scenario's real, adjusted response)
        "primaryScenario": {
            "response": primary_http_response,
            "rawModelScores": raw_scores,
            "note": "Symmetric candidates (identical popularity/age/creatorFollowed=false/alreadySeen=false, one per category) -- the only scenario used to compute the ML-learning verdict, via rawModelScores.",
        },

        # Secondary (business-reranking) evidence -- reported, never used for the verdict.
        "secondaryScenario": {
            "response": secondary_http_response,
            "note": "Asymmetric candidates (varied popularity/age, one creatorFollowed=true, one alreadySeen=true) -- exercises business-reranking logic only. Excluded from the ML-learning verdict.",
        },
    }

    # Post-training invariant gate (spec section 5/7): must pass before the report is written.
    # On failure this raises and propagates uncaught -- no report is written, non-zero exit.
    validate_report_invariants_v2(
        report, definition, expected_hyperparameters=expected_hyperparameters, expected_fixed_scenario=expected_fixed_scenario,
    )

    path = write_report(experiment_dir, report)
    report["_reportPath"] = str(path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(COMPARATIVE_DEFINITIONS))
    parser.add_argument("--experiment-version", choices=("v1", "v2"), default="v1",
                         help="Defaults to v1 (the original, unchanged documented command). Pass v2 for the rigor-pass pipeline.")
    parser.add_argument("--base-url", default=None, help="Defaults to http://localhost:$PORT/api/v1/recommendation-ml-service in this container.")
    parser.add_argument("--experiment-dir", default=None, help="Defaults to <EXPERIMENT_DIR>/comparative.")
    parser.add_argument("--seed", action="store_true", help="Seed the dataset first (idempotent) before training.")
    parser.add_argument("--train-timeout-seconds", type=float, default=DEFAULT_TRAIN_TIMEOUT_SECONDS)
    parser.add_argument("--poll-interval-seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--http-timeout-seconds", type=float, default=DEFAULT_HTTP_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    if args.base_url is not None:
        base_url = args.base_url
    else:
        import os
        base_url = f"http://localhost:{os.getenv('PORT', '3500')}/api/v1/recommendation-ml-service"

    if args.experiment_dir is not None:
        experiment_dir = Path(args.experiment_dir)
    else:
        from app.core.config import EXPERIMENT_DIR
        experiment_dir = EXPERIMENT_DIR / "comparative"

    definition = _DEFINITIONS_BY_VERSION[args.experiment_version][args.dataset]
    run = _run_v1 if args.experiment_version == "v1" else _run_v2
    print(f"Running comparative experiment '{definition.experiment_id}' (version={args.experiment_version}) against {base_url} ...", file=sys.stderr)
    report = run(
        definition, dataset_key=args.dataset, base_url=base_url, experiment_dir=experiment_dir, do_seed=args.seed,
        train_timeout_seconds=args.train_timeout_seconds, poll_interval_seconds=args.poll_interval_seconds,
        http_timeout_seconds=args.http_timeout_seconds,
    )
    print(json.dumps(report, indent=2, default=str))
    print(f"status={report['status']} reportPath={report.get('_reportPath')}", file=sys.stderr)
    return 0 if report["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
