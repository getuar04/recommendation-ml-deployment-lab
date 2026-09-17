"""Validate a Content Understanding dataset JSONL file and print a concise summary.

This is a validation/reporting tool only -- it does not train, score, or promote anything,
and it does not validate primary_category/subcategory against any taxonomy list (none is
product-approved yet; see app.ml.content_dataset_schema's module docstring).

Usage:
    python -m scripts.validate_content_dataset <path/to/dataset.jsonl>
    python -m scripts.validate_content_dataset <path/to/dataset.jsonl> --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.ml.content_dataset_eligibility import (
    select_gold_evaluation_set,
    select_training_set,
)
from app.ml.content_dataset_manifest import build_manifest
from app.ml.content_dataset_validator import validate_dataset_file


def _summarize(path: Path) -> dict:
    report = validate_dataset_file(path)
    manifest = build_manifest(report.records, dataset_version=path.stem)
    return {
        "path": str(path),
        "valid": report.is_valid,
        "recordCount": len(report.records),
        "errorCount": len(report.errors),
        "warningCount": len(report.issues) - len(report.errors),
        "trainingEligibleCount": len(select_training_set(report.records)),
        "goldEvaluationEligibleCount": len(select_gold_evaluation_set(report.records)),
        "manifest": manifest.to_dict(),
        "issues": [
            {"severity": issue.severity, "exampleId": issue.example_id, "message": issue.message}
            for issue in report.issues
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a dataset JSONL file")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text")
    args = parser.parse_args(argv)

    summary = _summarize(args.path)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Dataset: {summary['path']}")
        print(f"Valid: {summary['valid']}")
        print(f"Records: {summary['recordCount']}")
        print(f"Errors: {summary['errorCount']}  Warnings: {summary['warningCount']}")
        print(f"Training-eligible: {summary['trainingEligibleCount']}")
        print(f"Gold-evaluation-eligible: {summary['goldEvaluationEligibleCount']}")
        print(f"Taxonomy version: {summary['manifest']['taxonomyVersion']}")
        print(f"Source composition: {summary['manifest']['sourceComposition']}")
        print(f"Language composition: {summary['manifest']['languageComposition']}")
        print(f"Review status composition: {summary['manifest']['reviewStatusComposition']}")
        if summary["issues"]:
            print("\nIssues:")
            for issue in summary["issues"]:
                prefix = f"[{issue['severity']}]"
                where = f" ({issue['exampleId']})" if issue["exampleId"] else ""
                print(f"  {prefix}{where} {issue['message']}")

    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
