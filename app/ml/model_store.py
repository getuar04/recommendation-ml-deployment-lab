"""Model artifact persistence: atomic writes plus strict load-time validation.

`save()` / `load_model()` / `load_metadata()` are the low-level, permissive primitives
used by simple call sites and tests (they intentionally accept partial metadata, matching
prior behavior, and are the functions existing tests monkeypatch module-level paths around).
`load_validated()` is the strict contract the serving path should use: it enforces
schema/feature/checksum compatibility and raises a specific `ArtifactError` subclass
instead of ever silently loading a stale or corrupted artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import joblib
import sklearn

from app.core.config import METADATA_PATH, MODEL_DIR, MODEL_PATH

SCHEMA_VERSION = "2.1.0"
SKLEARN_VERSION = sklearn.__version__
PYTHON_VERSION = platform.python_version()
CHECKSUM_CHUNK_SIZE = 1 << 20  # 1 MiB; stream the file rather than loading it whole to hash it.

# The contract a `load_validated()` caller can rely on -- now family-aware (Task: XGBRanker
# production promotion). Anything missing means the artifact was produced by older/incompatible
# training code and must not be served silently. `artifactChecksum` is required here (not just
# checked when present): strict loading must reject a schema-compatible-looking artifact that
# simply omits its integrity checksum.
#
# `modelFamily` itself is intentionally NOT in COMMON_REQUIRED_METADATA_FIELDS: every artifact
# saved before this task has no `modelFamily` key at all, and `_model_family_of()` below treats
# that absence as "classifier" -- the exact set of fields a pre-existing classifier artifact
# already satisfies (see REQUIRED_METADATA_FIELDS, unchanged in content/order from before this
# task, for 100% backward compatibility with every artifact already on disk).
COMMON_REQUIRED_METADATA_FIELDS = (
    "schemaVersion", "modelVersion", "modelType", "selectedModel", "featureNames",
    "sklearnVersion", "pythonVersion", "randomSeed", "trainingDurationSeconds", "trainedAt",
    "metrics", "datasetSource", "artifactChecksum",
)
CLASSIFIER_REQUIRED_METADATA_FIELDS = (
    "featureDefinitions", "targetDefinition", "splitStrategy", "selectionCriterion",
    "calibration", "decisionThreshold", "trainingSamples", "modelSelectionSamples",
    "calibrationSamples", "thresholdTuningSamples", "testSamples", "classDistribution",
    "modelComparison",
)
# A ranker never calibrates, never thresholds, and trains on ranking groups rather than a
# train/modelSelection/calibration/thresholdTuning/test split -- forcing it to also satisfy
# CLASSIFIER_REQUIRED_METADATA_FIELDS would mean inventing meaningless placeholder values for
# fields that describe a training process a ranker never runs (Task spec: "do not just stuff
# meaningless fake classifier values into ranker metadata").
RANKER_REQUIRED_METADATA_FIELDS = (
    "objective", "scoreSemantics", "normalizationStrategy", "normalizationScale", "rankingGroupVersion",
)
# Preserved name/content for backward compatibility: every existing caller/test that imports
# `REQUIRED_METADATA_FIELDS` directly (e.g. to check a legacy artifact is missing something)
# keeps seeing exactly the classifier field set this name has always meant.
REQUIRED_METADATA_FIELDS = COMMON_REQUIRED_METADATA_FIELDS + CLASSIFIER_REQUIRED_METADATA_FIELDS


def _model_family_of(metadata: dict[str, Any]) -> str:
    """Absence of `modelFamily` means the artifact predates this concept -- always a
    classifier, since every ranker-family artifact is only ever saved by code introduced in
    this task, which always sets `modelFamily` explicitly."""
    return metadata.get("modelFamily") or "classifier"


def _required_fields_for(model_family: str) -> tuple[str, ...]:
    if model_family == "ranker":
        return COMMON_REQUIRED_METADATA_FIELDS + RANKER_REQUIRED_METADATA_FIELDS
    return COMMON_REQUIRED_METADATA_FIELDS + CLASSIFIER_REQUIRED_METADATA_FIELDS


class ArtifactError(Exception):
    """Base class for model artifact load failures."""


class ArtifactNotFoundError(ArtifactError):
    """No model/metadata file exists at the expected path."""


class ArtifactCorruptedError(ArtifactError):
    """The artifact exists but cannot be trusted: unreadable, checksum mismatch, or undeserializable."""


class ArtifactIncompatibleError(ArtifactError):
    """The artifact is readable but describes a different schema/feature contract than the running code."""


class ArtifactBusyError(ArtifactError):
    """The artifact kept changing (a retrain is in progress) and a stable, fully-validated
    read could not be obtained within the caller's retry budget. Transient by nature --
    a retry shortly after should succeed once the in-progress write settles."""


def _checksum(path: Path) -> str:
    """SHA-256 of `path`, streamed in fixed-size chunks so large artifacts are never
    loaded into memory in full just to be hashed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHECKSUM_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(
    model: Any,
    metadata: dict[str, Any],
    *,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist model + metadata atomically: both are written, then swapped into place.

    Uses tmp-file-then-`os.replace` for each file, with the temp file created directly
    alongside its target (same directory, hence guaranteed same filesystem, which is what
    makes `os.replace` atomic) so a concurrent reader either sees the complete previous
    file or the complete new one -- never a partial write. Parent directories are created
    for the *actual* target paths (not just `model_dir`), so explicitly-supplied
    model/metadata paths outside `model_dir` still work.

    The model is swapped in before the metadata, so on a partial failure (e.g. the process
    is killed between the two `os.replace` calls) the checksum stored in the (old) metadata
    will not match the (new) model file, and `load_validated()` will raise
    `ArtifactCorruptedError` rather than silently serving a mismatched pair.
    """
    model_path = model_path or MODEL_PATH
    metadata_path = metadata_path or METADATA_PATH
    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_model_path = model_path.with_name(model_path.name + ".tmp")
    tmp_metadata_path = metadata_path.with_name(metadata_path.name + ".tmp")
    full_metadata = {**metadata, "schemaVersion": metadata.get("schemaVersion", SCHEMA_VERSION)}
    try:
        joblib.dump(model, tmp_model_path)
        full_metadata["artifactChecksum"] = _checksum(tmp_model_path)
        tmp_metadata_path.write_text(json.dumps(full_metadata, indent=2), encoding="utf-8")
        os.replace(tmp_model_path, model_path)
        os.replace(tmp_metadata_path, metadata_path)
    finally:
        tmp_model_path.unlink(missing_ok=True)
        tmp_metadata_path.unlink(missing_ok=True)
    return full_metadata


def load_model(*, model_path: Path | None = None) -> Any:
    return joblib.load(model_path or MODEL_PATH)


def load_metadata(*, metadata_path: Path | None = None) -> dict[str, Any]:
    return json.loads((metadata_path or METADATA_PATH).read_text(encoding="utf-8"))


def _validate_metadata_and_checksum(
    expected_features: list[str] | None, model_path: Path, metadata_path: Path,
) -> dict[str, Any]:
    """Shared validation core for `check_artifact_compatibility()` and `load_validated()`:
    existence, schema, feature contract, and checksum -- everything except deserializing
    the model itself. Raises a specific `ArtifactError` subclass; never lets a raw
    JSON/filesystem exception escape."""
    if not model_path.exists() or not metadata_path.exists():
        raise ArtifactNotFoundError(f"No trained model artifact at {model_path} / {metadata_path}.")

    try:
        metadata = load_metadata(metadata_path=metadata_path)
    except (json.JSONDecodeError, OSError) as exc:
        raise ArtifactCorruptedError(f"Model metadata at {metadata_path} is unreadable: {exc}") from exc

    model_family = _model_family_of(metadata)
    required_fields = _required_fields_for(model_family)
    missing = [field for field in required_fields if field not in metadata]
    if missing:
        raise ArtifactIncompatibleError(
            f"Model metadata at {metadata_path} (modelFamily={model_family!r}) is missing required "
            f"fields {missing}; it was likely produced by an older, incompatible training run. "
            "Retrain to fix this."
        )
    if metadata["schemaVersion"] != SCHEMA_VERSION:
        raise ArtifactIncompatibleError(
            f"Model metadata schemaVersion {metadata['schemaVersion']!r} is incompatible with "
            f"runtime schemaVersion {SCHEMA_VERSION!r}. Retrain to produce a compatible artifact."
        )
    if expected_features is not None and metadata.get("featureNames") != list(expected_features):
        raise ArtifactIncompatibleError(
            f"Model metadata featureNames do not match the current feature contract at {metadata_path}. "
            "Retrain to produce a compatible artifact."
        )

    # `artifactChecksum` is in REQUIRED_METADATA_FIELDS, so its presence is already
    # guaranteed by the missing-fields check above; a mismatch here means the file itself
    # is corrupted, truncated, or was partially written by a failed/concurrent save.
    try:
        actual_checksum = _checksum(model_path)
    except OSError as exc:
        raise ArtifactCorruptedError(f"Model artifact at {model_path} could not be read: {exc}") from exc
    if metadata["artifactChecksum"] != actual_checksum:
        raise ArtifactCorruptedError(
            f"Model artifact at {model_path} does not match its recorded checksum; "
            "the file may be corrupted, truncated, or partially written."
        )
    return metadata


def check_artifact_compatibility(
    expected_features: list[str] | None = None,
    *,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Validate schema/feature/checksum compatibility and return metadata -- without
    deserializing the model itself.

    This is the "lightweight but strict" check for callers (health, metrics) that only
    need to know whether an artifact is ready, not to actually run inference with it: it
    pays for reading + hashing the model file (bounded, streamed) but skips the
    (comparatively much more expensive) `joblib.load`. Raises the same `ArtifactError`
    subclasses as `load_validated()`.
    """
    model_path = model_path or MODEL_PATH
    metadata_path = metadata_path or METADATA_PATH
    return _validate_metadata_and_checksum(expected_features, model_path, metadata_path)


def load_validated(
    expected_features: list[str] | None = None,
    *,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load model + metadata together, validating schema/feature/checksum compatibility.

    Raises a specific `ArtifactError` subclass instead of ever silently serving a missing,
    stale, incompatible, or corrupted artifact. This is the function the serving path
    (`app.ml.model_cache`) uses; the plain `load_model`/`load_metadata` above stay
    permissive for simple persistence tests.
    """
    model_path = model_path or MODEL_PATH
    metadata_path = metadata_path or METADATA_PATH
    metadata = _validate_metadata_and_checksum(expected_features, model_path, metadata_path)

    try:
        model = load_model(model_path=model_path)
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactCorruptedError(f"Model artifact at {model_path} could not be deserialized: {exc}") from exc

    return model, metadata
