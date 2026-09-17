"""Read-only experiment report listing/comparison (spec §13). No mutation endpoints exist
here -- reports are written only by `app.experiments.runner`, invoked via
`python -m scripts.run_experiment`, never through the HTTP API. Uses the same central
request-ID/error-contract handling as every other route (app/main.py) automatically.
"""
from fastapi import APIRouter, Query

from app.core.config import EXPERIMENT_DIR
from app.experiments.comparison import build_comparison
from app.experiments.report_store import list_reports
from app.schemas.limits import (
    EXPERIMENT_LIMIT_DEFAULT,
    EXPERIMENT_LIMIT_MAX,
    EXPERIMENT_LIMIT_MIN,
    EXPERIMENT_OFFSET_MAX,
    EXPERIMENT_OFFSET_MIN,
)

router = APIRouter(tags=["experiments"])

_EXAMPLE_RESPONSE = {
    "total": 2, "limit": 50, "offset": 0, "returned": 2, "hasMore": False,
    "succeededCount": 1, "failedOrIncompleteCount": 1,
    "trainingRunCount": 1, "benchmarkRecordCount": 0,
    "experiments": [{
        "experimentId": "small-balanced-v1", "runId": "a1b2c3d4e5f6a1b2", "datasetVersion": "small-balanced-v1",
        "modelType": "VIDEO", "reportKind": "TRAINING", "sourceRunId": None,
        "selectedModel": "LogisticRegression", "datasetSize": 1180,
        "uniqueUsers": 25, "uniqueContents": 40, "positiveRatio": 0.31,
        "prAuc": 0.71, "rocAuc": 0.68, "f1Score": 0.6, "precision": 0.58, "recall": 0.62,
        "precisionAt5": 0.4, "recallAt10": 0.55, "ndcgAt10": 0.52,
        "trainingDurationSeconds": 2.1, "artifactSizeBytes": 18422, "synthetic": True, "warnings": [],
    }],
    "recommendedByDomain": {
        "VIDEO": {"experimentId": "small-balanced-v1", "runId": "a1b2c3d4e5f6a1b2",
                  "reason": "Highest weighted score across PR-AUC, NDCG@10, dataset size, and class balance."},
        "LIVE": None,
    },
    "comparabilityWarning": None,
}


@router.get("/experiments", responses={200: {"content": {"application/json": {"example": _EXAMPLE_RESPONSE}}}})
def list_experiments(
    limit: int = Query(EXPERIMENT_LIMIT_DEFAULT, ge=EXPERIMENT_LIMIT_MIN, le=EXPERIMENT_LIMIT_MAX,
                        description="Maximum number of reports to include in this page."),
    offset: int = Query(EXPERIMENT_OFFSET_MIN, ge=EXPERIMENT_OFFSET_MIN, le=EXPERIMENT_OFFSET_MAX,
                         description="Number of reports (most recent first) to skip before this page."),
):
    """List and compare stored experiment run reports (most recently started first,
    deterministic ordering). Corrupt/malformed report files are silently skipped (see
    app.experiments.report_store). Never exposes filesystem paths, secrets, or fabricated
    metrics. TRAINING runs and linked BENCHMARK records are counted separately, and only
    unique trusted TRAINING runs can be recommended. VIDEO and LIVE runs are never combined
    into one misleading recommendation -- see `recommendedByDomain`/`comparabilityWarning`."""
    reports = list_reports(EXPERIMENT_DIR)
    reports.sort(key=lambda report: (report.get("startedAt") or "", report.get("runId") or ""), reverse=True)
    total = len(reports)
    page = reports[offset:offset + limit]
    comparison = build_comparison(page)
    return {
        "total": total, "limit": limit, "offset": offset, "returned": len(page),
        "hasMore": offset + len(page) < total,
        **comparison,
    }
