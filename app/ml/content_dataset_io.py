"""JSONL read/write for the Content Understanding dataset foundation.

Format choice (Part 12 of the dataset-foundation task): JSONL -- one `DatasetRecord` per
line. Chosen over CSV (hashtags/topics/source_metadata are nested, CSV would need lossy
flattening or an embedded-JSON-in-a-cell hack) and over Parquet (not human-diffable in a git
PR, and this project has no existing Parquet dependency); JSONL is `git diff`-friendly line by
line, trivially streamable, and needs nothing beyond the stdlib `json` module already used
throughout this repository (e.g. `app.api.content_routes`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.ml.content_dataset_schema import DatasetRecord


@dataclass(frozen=True)
class LineError:
    line_number: int
    message: str


@dataclass(frozen=True)
class LoadResult:
    """Partial-failure-tolerant load result: a malformed line does not abort loading the rest
    of the file -- the CLI validator (Part 20) needs to report every problem in one pass, not
    just the first."""

    records: list[DatasetRecord]
    errors: list[LineError]


def write_jsonl(path: Path, records: list[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [record.model_dump_json(by_alias=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_jsonl(path: Path) -> LoadResult:
    """Empty file (or a file that does not exist) loads as zero records, zero errors -- an
    empty real dataset is a valid dataset (Part 17), not an error."""
    if not path.exists():
        return LoadResult(records=[], errors=[])

    records: list[DatasetRecord] = []
    errors: list[LineError] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(LineError(line_number, f"invalid JSON: {exc}"))
            continue
        try:
            records.append(DatasetRecord.model_validate(payload))
        except ValidationError as exc:
            errors.append(LineError(line_number, str(exc)))
    return LoadResult(records=records, errors=errors)
