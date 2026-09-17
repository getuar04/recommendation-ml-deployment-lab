"""In-process cache for the loaded Two-Tower artifact + its precomputed content retrieval
index, mirroring `app.ml.model_cache`'s file-signature-based freshness contract (see that
module's docstring) -- the same already-established pattern this project uses for the
production RandomForest artifact, not a new invention.

Root cause this fixes (confirmed directly from code before any change here):
`app.services.two_tower_shadow_service.two_tower_retrieval_candidates()` previously called
`load_two_tower()` (disk read + `torch.load` + state_dict reconstruction) AND
`build_content_index()` (a full content-tower forward pass over the ENTIRE active catalog) on
EVERY SINGLE request -- unlike the production RandomForest path, which has always gone through
`app.ml.model_cache`. See the production-readiness report for measured before/after throughput.

Cache key: the Two-Tower artifact file's own (mtime, size) signature -- exactly like
`ModelCache` -- PLUS a catalog fingerprint (active-content count, max(updated_at)) so the
content index also rebuilds automatically whenever the catalog changes, without any explicit
invalidation call from content-mutation code paths. RandomForest model weights are never
touched by this module (separate cache, separate artifact, separate file) -- see module
docstring of `app.ml.two_tower.artifact_store` for that existing separation.

Thread-safe via a lock around cache-dict access only; the actual rebuild (disk load + full-
catalog embed) happens OUTSIDE the lock, matching `ModelCache`'s own concurrency design, so a
slow rebuild never blocks other threads from reading a still-valid cached entry. Under
concurrent load against a cold/just-invalidated cache, more than one thread can end up
recomputing the same rebuild -- exactly what `ModelCache` already accepts for the RandomForest
path today -- never a correctness issue, only possible duplicate work.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Content
from app.ml.two_tower import artifact_store
from app.ml.two_tower.features import build_content_vector
from app.ml.two_tower.retrieval import ContentIndex, build_content_index

_FileSignature = tuple[int, int] | None
_CatalogSignature = tuple[int, str | None]


def _file_signature(path: Path) -> _FileSignature:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _catalog_signature(db: Session) -> _CatalogSignature:
    """One cheap aggregate query (count + max(updated_at) of active VIDEO content) -- never a
    per-row scan -- used purely as a change-detection fingerprint, its values are never read
    for anything else. Scoped to `content_type == "VIDEO"` to match the exact population
    `get_or_build` below actually embeds -- a LIVE-only content change must not trigger a
    VIDEO index rebuild, and the fingerprint must track the same rows the index is built from."""
    count, max_updated = db.execute(
        select(func.count(Content.id), func.max(Content.updated_at)).where(
            Content.is_active.is_(True), Content.content_type == "VIDEO",
        )
    ).one()
    return (int(count or 0), max_updated.isoformat() if max_updated is not None else None)


class TwoTowerIndexCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[_FileSignature, _CatalogSignature, Any, dict[str, Any], ContentIndex]] = {}

    def get_or_build(self, db: Session, artifact_path: Path | None = None) -> tuple[Any, dict[str, Any], ContentIndex]:
        """Returns (model, metadata, content_index) for `artifact_path`'s CURRENT catalog,
        rebuilding only when the artifact file or the catalog fingerprint has changed since
        the last call. Raises whatever `load_two_tower()` raises on a missing/corrupt artifact
        -- never fabricates a fallback here; the caller
        (`two_tower_shadow_service.two_tower_retrieval_candidates`) already owns the
        fail-closed/fallback policy, unchanged by this cache."""
        # Read the module attribute dynamically (not imported by value) so a test/deployment
        # that monkeypatches `artifact_store.TWO_TOWER_ARTIFACT_PATH` (the established
        # convention -- see tests/test_two_tower_guarded_integration.py) is respected here too.
        resolved_path = artifact_path or artifact_store.TWO_TOWER_ARTIFACT_PATH
        key = str(resolved_path)
        artifact_sig = _file_signature(resolved_path)
        catalog_sig = _catalog_signature(db)

        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and cached[0] == artifact_sig and cached[1] == catalog_sig:
                return cached[2], cached[3], cached[4]

        model, metadata = artifact_store.load_two_tower(resolved_path)
        categories: list[str] = metadata["categories"]
        # VIDEO-only retrieval pool: `Content` has no VIDEO/LIVE separation enforced at
        # ingestion (event/content APIs accept either domain into the same table), so this
        # Two-Tower index -- exclusively used by VIDEO recommendation serving
        # (app.services.two_tower_shadow_service, gated by TWO_TOWER_RETRIEVAL_ENABLED) --
        # must not embed or retrieve LIVE content just because its category happens to match
        # one of this model's known categories. This module has no LIVE caller of its own to
        # preserve; the filter is local to this VIDEO-only cache.
        catalog = db.scalars(
            select(Content).where(Content.is_active.is_(True), Content.content_type == "VIDEO")
        ).all()
        catalog = [c for c in catalog if (c.category or "").upper() in categories]
        content_ids = [c.content_id for c in catalog]
        if content_ids:
            content_vectors = np.stack([build_content_vector(c, categories) for c in catalog])
            index = build_content_index(model, content_ids, content_vectors)
        else:
            index = ContentIndex(content_ids=[], embeddings=np.empty((0, metadata["embeddingDim"]), dtype=np.float32))

        with self._lock:
            self._entries[key] = (artifact_sig, catalog_sig, model, metadata, index)
        return model, metadata, index

    def invalidate(self, artifact_path: Path | None = None) -> None:
        """Drop cached entries. With no arguments, clears every entry in this cache. Never
        touches the RandomForest cache (app.ml.model_cache) -- separate cache instance
        entirely."""
        with self._lock:
            if artifact_path is None:
                self._entries.clear()
                return
            self._entries.pop(str(artifact_path), None)


two_tower_index_cache = TwoTowerIndexCache()
