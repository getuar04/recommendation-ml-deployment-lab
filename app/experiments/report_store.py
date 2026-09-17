"""Atomic, immutable experiment run report persistence (spec §10).

Layout: `<EXPERIMENT_DIR>/<experimentId>/<modelType>-<runId>.json` -- one file per run,
never overwritten (a colliding filename is a hard error, not silently replaced). Written via
tmp-file-then-`os.replace` (same pattern as `app.ml.model_store.save`), so a reader never
observes a partially-written report. Report bodies never contain absolute filesystem paths
or secrets; `list_reports()` skips a corrupt or malformed file with a logged warning instead
of raising, so one bad file can never break listing or comparison for the rest.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from app.core.logging import logger
from app.experiments.validation import validate_identifier

__all__ = ["list_reports", "write_report"]


def _run_dir(experiment_dir: Path, experiment_id: str) -> Path:
    validate_identifier(experiment_id, field="experimentId")
    run_dir = Path(experiment_dir) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_report(experiment_dir: Path, report: dict[str, Any]) -> Path:
    """Write one immutable report. Requires `report["experimentId"]`, `report["modelType"]`,
    and `report["runId"]` (a collision-resistant id -- callers should use `uuid4().hex`).
    Raises `FileExistsError` if that exact (experimentId, modelType, runId) was already
    written -- reports are never overwritten."""
    experiment_id = validate_identifier(str(report["experimentId"]), field="experimentId")
    model_type = validate_identifier(str(report["modelType"]), field="modelType")
    run_id = validate_identifier(str(report["runId"]), field="runId")

    run_dir = _run_dir(experiment_dir, experiment_id)
    filename = f"{model_type.lower()}-{run_id}.json"
    target_path = run_dir / filename
    if target_path.exists():
        raise FileExistsError(f"Experiment report {filename} already exists for {experiment_id}; run IDs must be unique.")

    # Operation-specific temp filename: deliberately short and NOT derived from `filename`
    # (it only needs to be unique within run_dir, not related to the final name) -- both to
    # avoid Windows MAX_PATH (260 chars) when combined with a deeply-nested experiment_dir,
    # and so two writers can never collide on the same temp file even in the
    # (should-never-happen, given runId's collision resistance) case of two calls targeting
    # the same report. This also means cleanup below can only ever remove *this* call's own
    # temp file.
    tmp_path = run_dir / f".{uuid.uuid4().hex[:8]}.tmp"
    try:
        tmp_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_path, target_path)  # atomic: tmp file is on the same filesystem/directory
    except Exception:
        # Best-effort cleanup of only this operation's own temp file. Never touches
        # target_path (untouched either way: os.replace either fully succeeded, moving this
        # exact tmp file, or never ran) and never touches any other file in run_dir.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target_path


def list_reports(experiment_dir: Path) -> list[dict[str, Any]]:
    """Every readable, well-formed report under `experiment_dir`, in deterministic
    (filename-sorted) order. A corrupt JSON file or one missing `experimentId` is skipped
    with a warning, never raised."""
    root = Path(experiment_dir)
    if not root.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping corrupt experiment report %s: %s", path.name, exc)
            continue
        if not isinstance(data, dict) or not isinstance(data.get("experimentId"), str):
            logger.warning("Skipping malformed experiment report %s: missing or invalid experimentId", path.name)
            continue
        results.append(data)
    return results
