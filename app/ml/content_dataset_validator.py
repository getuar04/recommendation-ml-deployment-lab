"""Cross-record validation for the Content Understanding dataset foundation (Part 13).

Per-record structural checks (missing text signal, blank label with a taxonomy version
missing, invalid review-state combinations, invalid language, malformed hashtags/topics,
malformed timestamps) already happen in `app.ml.content_dataset_schema.DatasetRecord` itself
-- a record that violates them cannot even be constructed. This module adds the checks that
require seeing the WHOLE dataset at once: duplicate example IDs, and exact-fingerprint
duplicates that carry conflicting labels.

Deliberately does NOT validate primary_category/subcategory against any proposed taxonomy
list -- no final taxonomy exists (see the task's non-goals); validation here is structural/
provenance-based only.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.ml.content_dataset_fingerprint import compute_content_fingerprint
from app.ml.content_dataset_io import LineError, load_jsonl
from app.ml.content_dataset_schema import DatasetRecord


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "ERROR" or "WARNING"
    message: str
    example_id: str | None = None


@dataclass
class ValidationReport:
    records: list[DatasetRecord] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "ERROR"]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _line_error_issue(error: LineError) -> ValidationIssue:
    return ValidationIssue(severity="ERROR", message=f"line {error.line_number}: {error.message}")


def validate_records(records: list[DatasetRecord]) -> list[ValidationIssue]:
    """Cross-record checks over an already-constructed (so already per-record-valid) list of
    records: duplicate example IDs, and exact-fingerprint duplicates with conflicting labels."""
    issues: list[ValidationIssue] = []

    seen_ids: dict[str, int] = defaultdict(int)
    for record in records:
        seen_ids[record.example_id] += 1
    for example_id, count in seen_ids.items():
        if count > 1:
            issues.append(ValidationIssue(
                severity="ERROR", example_id=example_id,
                message=f"duplicate example_id {example_id!r} appears {count} times.",
            ))

    by_fingerprint: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        fingerprint = compute_content_fingerprint(record.title, record.hashtags)
        by_fingerprint[fingerprint].append(record)

    for fingerprint, group in by_fingerprint.items():
        if len(group) < 2:
            continue
        labels = {(r.primary_category, r.subcategory, r.taxonomy_version) for r in group}
        if len(labels) > 1:
            ids = ", ".join(r.example_id for r in group)
            issues.append(ValidationIssue(
                severity="ERROR",
                message=(
                    f"records [{ids}] share fingerprint {fingerprint} (exact normalized duplicate "
                    "content) but carry conflicting primary_category/subcategory/taxonomy_version labels."
                ),
            ))
        else:
            ids = ", ".join(r.example_id for r in group)
            issues.append(ValidationIssue(
                severity="WARNING",
                message=f"records [{ids}] share fingerprint {fingerprint} (exact normalized duplicate content).",
            ))

    return issues


def validate_dataset_file(path: Path) -> ValidationReport:
    """Loads `path` (tolerating and reporting per-line JSON/schema errors, see
    `app.ml.content_dataset_io.load_jsonl`) and runs cross-record checks on whatever records
    did parse. A missing or empty file is valid with zero records (Part 17)."""
    load_result = load_jsonl(path)
    issues = [_line_error_issue(error) for error in load_result.errors]
    issues.extend(validate_records(load_result.records))
    return ValidationReport(records=load_result.records, issues=issues)
