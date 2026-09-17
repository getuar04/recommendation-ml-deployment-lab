"""Dataset release manifest for the Content Understanding dataset foundation (Part 19).

Counts are always computed from the actual record list passed in -- never hardcoded, never
estimated. An empty dataset produces a manifest with recordCount=0 and empty composition
maps, which is the honest, expected manifest for the currently-empty real dataset (Part 17).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.ml.content_dataset_fingerprint import compute_content_fingerprint
from app.ml.content_dataset_schema import DatasetRecord

_MIXED_TAXONOMY_VERSION = "MIXED"


@dataclass(frozen=True)
class DatasetManifest:
    dataset_version: str
    taxonomy_version: str | None
    created_at: str
    record_count: int
    source_composition: dict[str, int]
    language_composition: dict[str, int]
    review_status_composition: dict[str, int]
    duplicate_fingerprint_count: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasetVersion": self.dataset_version,
            "taxonomyVersion": self.taxonomy_version,
            "createdAt": self.created_at,
            "recordCount": self.record_count,
            "sourceComposition": dict(self.source_composition),
            "languageComposition": dict(self.language_composition),
            "reviewStatusComposition": dict(self.review_status_composition),
            "duplicateFingerprintCount": self.duplicate_fingerprint_count,
            "description": self.description,
        }


def _resolve_taxonomy_version(records: list[DatasetRecord]) -> str | None:
    """Never fabricates a version: None if no record carries one, the single shared value if
    every labeled record agrees, or the literal "MIXED" sentinel (not a real version string)
    if labeled records disagree -- a mixed dataset must never be silently reported as
    belonging to one taxonomy version."""
    versions = {record.taxonomy_version for record in records if record.taxonomy_version}
    if not versions:
        return None
    if len(versions) == 1:
        return next(iter(versions))
    return _MIXED_TAXONOMY_VERSION


def build_manifest(
    records: list[DatasetRecord], *, dataset_version: str, description: str = "", created_at: datetime | None = None,
) -> DatasetManifest:
    fingerprints = [compute_content_fingerprint(record.title, record.hashtags) for record in records]
    fingerprint_counts = Counter(fingerprints)
    duplicate_fingerprint_count = sum(1 for count in fingerprint_counts.values() if count > 1)

    return DatasetManifest(
        dataset_version=dataset_version,
        taxonomy_version=_resolve_taxonomy_version(records),
        created_at=(created_at or datetime.now(timezone.utc)).isoformat(),
        record_count=len(records),
        source_composition=dict(Counter(record.label_source.value for record in records)),
        language_composition=dict(Counter(record.language or "UNSPECIFIED" for record in records)),
        review_status_composition=dict(Counter(record.review_status.value for record in records)),
        duplicate_fingerprint_count=duplicate_fingerprint_count,
        description=description,
    )
