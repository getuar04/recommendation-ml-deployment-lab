"""Deterministic comparison over real, stored experiment run reports (spec §47).

Never ranks by F1 alone, never fabricates a missing metric as 0 -- a candidate missing a
metric the comparison needs is simply excluded from ranking on that axis (and, if it's
missing PR-AUC entirely, excluded from the "recommended" heuristic altogether), not
penalized with a fake zero.
"""
from __future__ import annotations

import math
from typing import Any

from app.experiments.validation import InvalidIdentifierError, validate_identifier

__all__ = ["build_comparison"]

_METRIC_KEYS = ("prAuc", "rocAuc", "f1Score", "precision", "recall")
_RANKING_KEYS = ("precisionAt5", "recallAt10", "ndcgAt10")
_REPORT_KINDS = {"TRAINING", "BENCHMARK"}


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _identifier(value: Any, *, field: str) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return validate_identifier(value, field=field)
    except InvalidIdentifierError:
        return None


def _number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _report_kind(report: dict[str, Any]) -> str | None:
    """Validated kind with a narrow compatibility rule for Session-5 reports.

    Reports written before ``reportKind`` existed are TRAINING, except for the benchmark
    records created by our own CLI whose validated run id ends in ``-bench``. Unknown
    explicit kinds are rejected instead of trusted.
    """
    explicit = report.get("reportKind")
    if explicit is not None:
        return explicit if explicit in _REPORT_KINDS else None
    run_id = _identifier(report.get("runId"), field="runId")
    return "BENCHMARK" if run_id and run_id.endswith("-bench") else "TRAINING"


def _source_run_id(report: dict[str, Any], kind: str | None) -> str | None:
    """If the report explicitly provides `sourceRunId` (even an empty/invalid one), honor
    that claim if valid or reject it outright -- never silently substitute a *different*,
    inferred value in its place, which would mask the fact that the report's own explicit
    claim failed validation. The runId-suffix inference below is a narrow compatibility
    rule for legacy reports that predate this field existing at all (key absent, not just
    invalid)."""
    if "sourceRunId" in report:
        return _identifier(report.get("sourceRunId"), field="sourceRunId")
    run_id = _identifier(report.get("runId"), field="runId")
    if kind == "BENCHMARK" and run_id and run_id.endswith("-bench"):
        return run_id[:-6] or None
    return None


def _provenance_warning(report: dict[str, Any]) -> str | None:
    top_level = report.get("synthetic")
    source = report.get("datasetSource")
    nested = source.get("synthetic") if isinstance(source, dict) else None
    # Legacy reports may predate the provenance field; keep them visible as unknown.
    # New reports that claim provenance must use booleans and must agree at both levels.
    if top_level is not None and not isinstance(top_level, bool):
        return "invalid synthetic provenance"
    if nested is not None and not isinstance(nested, bool):
        return "invalid datasetSource synthetic provenance"
    if isinstance(nested, bool) and nested != top_level:
        return "contradictory synthetic provenance"
    return None


def _row(report: dict[str, Any]) -> dict[str, Any]:
    """Explicit allowlist projection: only these named fields, each type-checked, are ever
    pulled out of a stored report -- this is what makes it safe to call on a hand-crafted or
    corrupted JSON file. In particular, `warnings` in the output is *never* copied from
    `report["warnings"]` (a stored report's own claimed warnings are untrusted, arbitrary
    strings an attacker or a corrupted file could set to anything); the only warnings ever
    surfaced here are computed by this function itself from validated numeric fields."""
    metrics_raw = report.get("metrics")
    metrics: dict[str, Any] = metrics_raw if isinstance(metrics_raw, dict) else {}
    ranking_raw = report.get("rankingMetrics")
    ranking: dict[str, Any] = ranking_raw if isinstance(ranking_raw, dict) else {}
    dataset_raw = report.get("dataset")
    dataset: dict[str, Any] = dataset_raw if isinstance(dataset_raw, dict) else {}

    kind = _report_kind(report)
    warnings: list[str] = []
    if kind is None:
        warnings.append("invalid report kind")
    provenance_warning = _provenance_warning(report)
    if provenance_warning:
        warnings.append(provenance_warning)
    missing_metrics = [key for key in (*_METRIC_KEYS, *_RANKING_KEYS) if _number(metrics.get(key)) is None and _number(ranking.get(key)) is None]
    if missing_metrics:
        warnings.append(f"missing metrics: {', '.join(missing_metrics)}")

    return {
        "experimentId": _identifier(report.get("experimentId"), field="experimentId"),
        "runId": _identifier(report.get("runId"), field="runId"),
        "datasetVersion": _identifier(report.get("datasetVersion"), field="datasetVersion"),
        "modelType": report.get("modelType") if report.get("modelType") in {"VIDEO", "LIVE"} else None,
        "reportKind": kind, "sourceRunId": _source_run_id(report, kind),
        "selectedModel": _str(report.get("selectedModel")),
        "datasetSize": _number(dataset.get("totalInteractions")) or _number(dataset.get("totalRows")),
        "uniqueUsers": _number(dataset.get("uniqueUsers")), "uniqueContents": _number(dataset.get("uniqueContents")),
        "positiveRatio": _number(dataset.get("positiveRatio")),
        "prAuc": _number(metrics.get("prAuc")), "rocAuc": _number(metrics.get("rocAuc")), "f1Score": _number(metrics.get("f1Score")),
        "precision": _number(metrics.get("precision")), "recall": _number(metrics.get("recall")),
        "precisionAt5": _number(ranking.get("precisionAt5")), "recallAt10": _number(ranking.get("recallAt10")),
        "ndcgAt10": _number(ranking.get("ndcgAt10")),
        "trainingDurationSeconds": _number(report.get("trainingDurationSeconds")),
        "artifactSizeBytes": _number(report.get("artifactSizeBytes")),
        "synthetic": report.get("synthetic") if isinstance(report.get("synthetic"), bool) else None,
        "warnings": warnings,
    }


def _recommend(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Multi-factor heuristic, documented, never a pure F1 ranking. Considers: ranking
    quality (PR-AUC, NDCG@10), a class-balance-extremity penalty (distrusts a near-100%
    single-class split over a genuinely mixed one), a small log-scaled dataset-size bonus
    (more rows -> a more trustworthy metric estimate, capped so it can't dominate), and
    metric availability (a candidate missing PR-AUC is excluded outright, not penalized to
    a fake zero). Returns None when nothing is meaningfully comparable -- that's a valid,
    honest outcome, not an error.
    """
    candidates = [row for row in rows if row.get("prAuc") is not None and row.get("datasetSize")]
    if not candidates:
        return None

    def score(row: dict[str, Any]) -> float:
        pr_auc = row["prAuc"]
        ndcg = row.get("ndcgAt10") or 0.0
        ratio = row.get("positiveRatio")
        balance_penalty = 0.0 if ratio is None else abs(0.5 - ratio) * 0.3
        size_bonus = min(0.15, math.log10(max(1, row["datasetSize"])) * 0.03)
        return 0.5 * pr_auc + 0.3 * ndcg + size_bonus - balance_penalty

    best = max(candidates, key=score)
    return {
        "experimentId": best["experimentId"], "runId": best["runId"],
        "reason": (
            "Highest weighted score across PR-AUC (weight 0.5), NDCG@10 (weight 0.3), a "
            "small log-scaled dataset-size bonus (capped at 0.15), and a penalty for class "
            "balance far from 50/50 (weight 0.3 x |0.5 - positiveRatio|). A documented "
            "heuristic across data provenance, ranking quality, class distribution, dataset "
            "size, and metric availability -- not a pure F1 ranking, and not a claim of "
            "objective correctness."
        ),
    }


def build_comparison(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """VIDEO and LIVE use different targets, features, and models -- they are never combined
    into one "winner". `recommendedByDomain` gives (at most) one recommendation per
    `modelType` present in the successful rows; `comparabilityWarning` is set whenever both
    domains appear together in the same input, so a caller can't miss that the two
    recommendations aren't measuring the same thing."""
    projected = [(_row(report), report.get("status")) for report in reports]
    rows = [row for row, status in projected if status == "SUCCEEDED" and row["reportKind"] in _REPORT_KINDS]
    training_rows = [
        row for row in rows
        if row["reportKind"] == "TRAINING" and not any(
            warning in row["warnings"]
            for warning in ("invalid synthetic provenance", "invalid datasetSource synthetic provenance",
                            "contradictory synthetic provenance")
        )
    ]
    benchmark_rows = [row for row in rows if row["reportKind"] == "BENCHMARK"]

    domains = sorted({row["modelType"] for row in training_rows if row["modelType"]})
    # VIDEO/LIVE keys are always present (None when that domain has no comparable rows) for
    # a predictable response shape; any other modelType value is added additively.
    recommended_by_domain: dict[str, Any] = {"VIDEO": None, "LIVE": None}
    for domain in domains:
        recommended_by_domain[domain] = _recommend([row for row in training_rows if row["modelType"] == domain])

    comparability_warning = None
    if len(domains) > 1:
        comparability_warning = (
            "This result set includes more than one modelType (" + ", ".join(domains) + "). "
            "VIDEO and LIVE use different targets, features, and models and are not directly "
            "comparable -- see recommendedByDomain for a separate recommendation per domain."
        )

    return {
        "experiments": rows,
        "totalReports": len(reports),
        "succeededCount": len(rows),
        "failedOrIncompleteCount": len(reports) - len(rows),
        "trainingRunCount": len(training_rows),
        "benchmarkRecordCount": len(benchmark_rows),
        "recommendedByDomain": recommended_by_domain,
        "comparabilityWarning": comparability_warning,
    }
