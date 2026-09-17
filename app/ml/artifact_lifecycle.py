"""Candidate -> active -> previous artifact promotion lifecycle.

`MODEL_PATH`/`METADATA_PATH` (and the LIVE equivalents) remain the existing *active*
artifact paths that `app.ml.model_cache`/`app.ml.predictor` serve from and that many
existing tests monkeypatch directly -- their names and meaning are unchanged. Candidate and
previous are additive sibling paths computed from whatever the active path currently is
(see `sibling_paths()`), so a monkeypatched active path still gets correctly-located
candidate/previous files, and no test needs to change to know about them.

A trained model is never served directly from the candidate path: it is saved there first
(`app.ml.model_store.save`), then `promote()` validates it (schema/feature/checksum via
`model_store.load_validated`, plus a minimal inference smoke test) before ever touching the
active path. If validation fails, the active artifact is left completely untouched. On
success, the current active pair (if any) is moved to `previous` before the candidate is
swapped onto the active path, so a rollback target is always available after at least one
successful promotion.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.logging import logger
from app.ml import model_store

__all__ = ["PromotionValidationError", "promote", "rollback", "sibling_paths"]


class PromotionValidationError(Exception):
    """The candidate (or, during rollback, the previous) artifact failed validation and
    was NOT promoted; the active artifact is guaranteed unchanged."""


def sibling_paths(model_path: Path, metadata_path: Path) -> dict[str, Path]:
    """Candidate/previous paths as siblings of the given active paths, in the same
    directory -- so they follow a monkeypatched active path automatically."""
    return {
        "candidate_model": model_path.with_name(f"candidate_{model_path.name}"),
        "candidate_metadata": metadata_path.with_name(f"candidate_{metadata_path.name}"),
        "previous_model": model_path.with_name(f"previous_{model_path.name}"),
        "previous_metadata": metadata_path.with_name(f"previous_{metadata_path.name}"),
    }


def _smoke_test(model: Any, feature_names: list[str], categorical_features: list[str] | None = None) -> None:
    """A minimal inference call on a placeholder row: catches a candidate that deserializes
    fine but cannot actually score (e.g. a pipeline saved with a mismatched preprocessing
    step). Not a substitute for the real evaluation already performed during training.

    Categorical columns get a placeholder string rather than 0.0 -- the fitted
    `OneHotEncoder(handle_unknown="ignore")` accepts any string (even one never seen during
    training, degrading it to an all-zero encoding), but errors on a numeric value passed
    into a column it fit as string-typed.
    """
    categorical = set(categorical_features or [])
    row = pd.DataFrame([{name: ("__smoke_test__" if name in categorical else 0.0) for name in feature_names}])
    model.predict_proba(row[feature_names])


def _validate_or_raise(
    model_path: Path, metadata_path: Path, feature_names: list[str], categorical_features: list[str] | None,
) -> tuple[Any, dict[str, Any]]:
    """Runs `model_store.load_validated()` (letting its `ArtifactError` subclasses
    propagate as-is) plus the inference smoke test (raising `PromotionValidationError`)."""
    model, metadata = model_store.load_validated(feature_names, model_path=model_path, metadata_path=metadata_path)
    try:
        _smoke_test(model, feature_names, categorical_features)
    except Exception as exc:
        raise PromotionValidationError(f"Artifact at {model_path} failed the inference smoke test: {exc}") from exc
    return model, metadata


def promote(
    *,
    candidate_model_path: Path,
    candidate_metadata_path: Path,
    active_model_path: Path,
    active_metadata_path: Path,
    previous_model_path: Path,
    previous_metadata_path: Path,
    feature_names: list[str],
    categorical_features: list[str] | None = None,
) -> dict[str, Any]:
    """Validate the candidate, retire the current active artifact to `previous` (if one
    exists), then atomically swap the candidate onto the active path.

    Raises `PromotionValidationError` (leaving `active_*` completely untouched) if the
    candidate is corrupted, incompatible, or fails the smoke test; propagates
    `model_store.ArtifactNotFoundError` if no candidate exists at all.
    """
    try:
        _, metadata = _validate_or_raise(candidate_model_path, candidate_metadata_path, feature_names, categorical_features)
    except (model_store.ArtifactIncompatibleError, model_store.ArtifactCorruptedError) as exc:
        raise PromotionValidationError(f"Candidate artifact failed validation and was not promoted: {exc}") from exc

    if active_model_path.exists() and active_metadata_path.exists():
        previous_model_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(active_model_path, previous_model_path)
        os.replace(active_metadata_path, previous_metadata_path)
        logger.info("Retired previous active artifact to %s", previous_model_path)

    promoted_metadata = {**metadata, "promotedAt": datetime.now(timezone.utc).isoformat()}
    active_model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_metadata = active_metadata_path.with_name(active_metadata_path.name + ".tmp")
    tmp_metadata.write_text(json.dumps(promoted_metadata, indent=2), encoding="utf-8")

    os.replace(candidate_model_path, active_model_path)
    os.replace(tmp_metadata, active_metadata_path)
    candidate_metadata_path.unlink(missing_ok=True)
    logger.info("Promoted candidate artifact %s to active %s", candidate_model_path, active_model_path)
    return promoted_metadata


def rollback(
    *,
    active_model_path: Path,
    active_metadata_path: Path,
    previous_model_path: Path,
    previous_metadata_path: Path,
    feature_names: list[str],
    categorical_features: list[str] | None = None,
) -> dict[str, Any]:
    """Promote `previous` back onto `active`. Re-validates the previous artifact first
    (same checks as a normal promotion), so a corrupted `previous` can never be served
    either. Raises `model_store.ArtifactNotFoundError` if there is no previous artifact."""
    try:
        _, metadata = _validate_or_raise(previous_model_path, previous_metadata_path, feature_names, categorical_features)
    except (model_store.ArtifactIncompatibleError, model_store.ArtifactCorruptedError) as exc:
        raise PromotionValidationError(f"Previous artifact failed validation and rollback was aborted: {exc}") from exc

    rolled_back_metadata = {**metadata, "rolledBackAt": datetime.now(timezone.utc).isoformat()}
    active_model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_metadata = active_metadata_path.with_name(active_metadata_path.name + ".tmp")
    tmp_metadata.write_text(json.dumps(rolled_back_metadata, indent=2), encoding="utf-8")

    os.replace(previous_model_path, active_model_path)
    os.replace(tmp_metadata, active_metadata_path)
    previous_metadata_path.unlink(missing_ok=True)
    logger.info("Rolled back active artifact %s to previous version", active_model_path)
    return rolled_back_metadata
