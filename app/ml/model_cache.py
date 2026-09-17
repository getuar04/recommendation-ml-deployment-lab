"""In-process cache for trained model + metadata, avoiding a disk load on every request.

Keyed by (model_path, metadata_path) so one cache instance safely serves both the VIDEO
and LIVE artifacts (or arbitrary test paths) without cross-invalidating each other.
Freshness is determined by a signature combining *both* files' (mtime, size): a retrain
replaces both atomically (`app.ml.model_store.save`), and either file changing -- including
metadata alone, e.g. a hand-edited or corrupted metadata.json with the model left untouched
-- changes the combined signature and triggers a reload plus a fresh compatibility
validation (`model_store.load_validated`) on the very next request.

Stable-read protocol: `get()` records the file signature both before and after the
(lock-free) `load_validated()` call. A model is only ever cached, or returned, under the
signature that was actually validated; if the files changed mid-load (a retrain racing a
reader), the result is discarded and the read is retried a bounded number of times before
raising `model_store.ArtifactBusyError`. A caller therefore only ever observes: the
previous valid pair (served from cache, unaffected by a write in progress), the new valid
pair (once the write has settled and a fresh load stabilizes), or a clear, bounded busy
error -- never a torn or partially-validated pair.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from app.ml import model_store

_FileSignature = tuple[int, int] | None
_Signature = tuple[_FileSignature, _FileSignature]

MAX_LOAD_ATTEMPTS = 3


class ModelCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], tuple[_Signature, Any, dict[str, Any]]] = {}

    @staticmethod
    def _file_signature(path: Path) -> _FileSignature:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    @classmethod
    def _signature(cls, model_path: Path, metadata_path: Path) -> _Signature:
        return (cls._file_signature(model_path), cls._file_signature(metadata_path))

    def get(
        self, model_path: Path, metadata_path: Path, *, expected_features: list[str] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return (model, metadata), reloading + revalidating if either file changed since
        the last call. See module docstring for the stable-read retry protocol."""
        key = (str(model_path), str(metadata_path))
        last_error: model_store.ArtifactError | None = None

        for _ in range(MAX_LOAD_ATTEMPTS):
            signature_before = self._signature(model_path, metadata_path)
            if signature_before[0] is None or signature_before[1] is None:
                with self._lock:
                    self._entries.pop(key, None)
                raise model_store.ArtifactNotFoundError(f"No trained model artifact at {model_path} / {metadata_path}.")

            with self._lock:
                cached = self._entries.get(key)
                if cached is not None and cached[0] == signature_before:
                    return cached[1], cached[2]

            # Deliberately outside the lock: joblib deserialization can be slow, and other
            # threads/requests must still be able to read a stable cached entry meanwhile.
            result: tuple[Any, dict[str, Any]] | None = None
            error: model_store.ArtifactError | None = None
            try:
                result = model_store.load_validated(expected_features, model_path=model_path, metadata_path=metadata_path)
            except model_store.ArtifactError as exc:
                error = exc

            signature_after = self._signature(model_path, metadata_path)
            stable = signature_after == signature_before and signature_after[0] is not None and signature_after[1] is not None
            if not stable:
                # Files changed while loading/validating (a retrain racing this read):
                # discard whatever we got -- success or failure -- and retry.
                last_error = error
                continue

            if error is not None:
                raise error  # a stable read that still failed is a real, non-transient error
            assert result is not None  # the only other path out of the try/except above

            with self._lock:
                self._entries[key] = (signature_after, *result)
            return result

        raise model_store.ArtifactBusyError(
            f"Model artifact at {model_path} / {metadata_path} kept changing while being loaded "
            f"(exceeded {MAX_LOAD_ATTEMPTS} attempts); this usually means a retrain is in progress. "
            "Retry shortly."
        ) from last_error

    def peek_status(
        self, model_path: Path, metadata_path: Path, *, expected_features: list[str] | None = None,
    ) -> str:
        """Lightweight readiness check for callers (health) that must not pay to deserialize
        a large model just to report status, and must not participate in `get()`'s retry
        loop. Returns one of "MISSING" / "READY" / "INCOMPATIBLE" / "CORRUPTED".

        If an already-cached, still-fresh entry exists, this is a pure in-memory check (two
        `stat()` calls, no further I/O) -- the common case once anything has actually served
        a request. Otherwise it falls back to `model_store.check_artifact_compatibility()`,
        which validates schema/feature/checksum without ever deserializing the model, so a
        health probe never forces a full model load purely to answer "is it ready".
        """
        key = (str(model_path), str(metadata_path))
        signature = self._signature(model_path, metadata_path)
        if signature[0] is None or signature[1] is None:
            return "MISSING"

        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and cached[0] == signature:
                return "READY"

        try:
            model_store.check_artifact_compatibility(expected_features, model_path=model_path, metadata_path=metadata_path)
            return "READY"
        except model_store.ArtifactNotFoundError:
            return "MISSING"
        except model_store.ArtifactIncompatibleError:
            return "INCOMPATIBLE"
        except model_store.ArtifactCorruptedError:
            return "CORRUPTED"

    def invalidate(self, model_path: Path | None = None, metadata_path: Path | None = None) -> None:
        """Drop cached entries. With no arguments, clears every entry in this cache."""
        with self._lock:
            if model_path is None and metadata_path is None:
                self._entries.clear()
                return
            self._entries.pop((str(model_path), str(metadata_path)), None)


video_cache = ModelCache()
live_cache = ModelCache()
