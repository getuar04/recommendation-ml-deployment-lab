"""Compare stored experiment run reports (spec §47). Read-only: never mutates a report,
never trains anything.

Usage:
    python -m scripts.compare_experiments
    python -m scripts.compare_experiments --experiment-dir ./experiments --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", default=None, help="Where reports are read from (defaults to EXPERIMENT_DIR).")
    parser.add_argument("--json", action="store_true", help="Print the full comparison as JSON instead of a summary table.")
    args = parser.parse_args(argv)

    from app.experiments.comparison import build_comparison
    from app.experiments.report_store import list_reports

    if args.experiment_dir is not None:
        experiment_dir = Path(args.experiment_dir)
    else:
        from app.core.config import EXPERIMENT_DIR
        experiment_dir = EXPERIMENT_DIR

    reports = list_reports(experiment_dir)
    comparison = build_comparison(reports)

    if args.json:
        print(json.dumps(comparison, indent=2))
        return 0

    print(f"Reports found: {comparison['totalReports']} (succeeded: {comparison['succeededCount']}, "
          f"failed/incomplete: {comparison['failedOrIncompleteCount']})")
    print(f"Unique trusted training runs: {comparison['trainingRunCount']}; "
          f"benchmark records: {comparison['benchmarkRecordCount']}")
    print()
    for row in comparison["experiments"]:
        kind = row["reportKind"] or "INVALID"
        linked = f", source {row['sourceRunId']}" if row["sourceRunId"] else ""
        print(f"- [{kind}] {row['experimentId']} / {row['modelType']} (run {row['runId']}{linked})")
        print(f"    selectedModel={row['selectedModel']}  prAuc={row['prAuc']}  rocAuc={row['rocAuc']}  f1Score={row['f1Score']}")
        print(f"    precisionAt5={row['precisionAt5']}  recallAt10={row['recallAt10']}  ndcgAt10={row['ndcgAt10']}")
        print(f"    datasetSize={row['datasetSize']}  uniqueUsers={row['uniqueUsers']}  positiveRatio={row['positiveRatio']}")
        print(f"    synthetic={row['synthetic']}  trainingDurationSeconds={row['trainingDurationSeconds']}")
        if row["warnings"]:
            print(f"    warnings: {'; '.join(row['warnings'])}")
    print()
    if comparison.get("comparabilityWarning"):
        print(f"WARNING: {comparison['comparabilityWarning']}")
        print()
    for domain, recommended in comparison["recommendedByDomain"].items():
        if recommended:
            print(f"Recommended {domain} (heuristic, not an objective ranking): "
                  f"{recommended['experimentId']} (run {recommended['runId']})")
            print(f"  reason: {recommended['reason']}")
        else:
            print(f"No {domain} recommendation: nothing comparable in the current report set for this domain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
