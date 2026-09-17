"""Phase 3.2 (shadow proof) + Phase 3.4 (guarded integration) -- Two-Tower retrieval.

`two_tower_retrieval_candidates()` is the ONLY function `app.services.recommendation_service`
imports from this module, and only via a LAZY, function-local import guarded by the
`TWO_TOWER_RETRIEVAL_ENABLED` flag (default false) -- so this module's `torch`/two-tower
dependency is never loaded, and this file has zero effect on `POST /recommendations`, unless
the flag is explicitly turned on. `retrieve_and_rank_with_two_tower()` (the full shadow
pipeline, including its own scoring/reranking) remains for tests/offline scripts
(`scripts/run_two_tower_shadow_comparison.py`) only and is still never called from the request
path -- `recommend()` does its OWN scoring/reranking (its existing, unmodified code) on top of
whatever candidate list it receives, current or Two-Tower-sourced, so that step is never
duplicated between this module and `recommendation_service.py`.

    user -> Two-Tower retrieval Top-K -> EXISTING candidate schema -> EXISTING RandomForest
         -> EXISTING reranker -> Top-N

Responsibility boundary (kept explicit, not blurred):
    Two-Tower    -- retrieval / candidate generation ONLY. Narrows the full catalog to a
                    shortlist. Carries no per-candidate personalised scoring and no business/
                    contextual/diversity logic.
    RandomForest -- the SAME fine-grained personalised scoring model production uses, loaded
                    via the SAME `app.ml.model_cache`/`app.ml.model_store` contract, at the
                    SAME active `MODEL_PATH` -- this module never trains, retrains, or promotes
                    a ranker, and never reads a different model artifact than production does.
    reranker     -- the SAME business/contextual/diversity adjustments production uses
                    (`app.ml.reranker.rerank`/`explanation`), called exactly as
                    `recommendation_service.recommend()` calls them.
Nothing here reimplements ranker features, reranker rules, or explanation logic -- every step
from "existing candidate schema" onward is the literal, unmodified production code, imported
and called, not copied.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select

from app.core.config import MODEL_PATH, RECOMMENDATION_HISTORY_MAX_INTERACTIONS
from app.db.models import Content
from app.db.repositories import recent_interactions_for_ranking
from app.ml import model_cache, model_store
from app.ml.dataset_builder import FEATURES, MAX_CONTENT_AGE_HOURS, history_from_rows
from app.ml.predictor import probabilities
from app.ml.reranker import explanation, rerank
from app.ml.two_tower.features import build_user_vector
from app.ml.two_tower.index_cache import two_tower_index_cache
from app.ml.two_tower.retrieval import embed_user, retrieve_top_k
from app.schemas.recommendation_schemas import Candidate
from app.services.recommendation_service import ModelArtifactInvalid, ModelNotTrained

DEFAULT_RETRIEVAL_K = 30


def content_to_candidate(content: Content, now: datetime) -> Candidate | None:
    """Converts a DB `Content` row into the EXACT `Candidate` structure the existing ranker/
    reranker consume -- no shim class, no schema hacks (Step 2.4/Step 2.5). Returns None (never
    raises) for a row that fails the schema's own validation, mirroring
    `recommendation_service.recommend()`'s existing "skip an invalid/blank candidate rather than
    fail the whole batch" convention, applied here at the Two-Tower -> ranker boundary."""
    # content.created_at can come back offset-naive from SQLite even though it was written
    # aware (see app.experiments.dataset_generation's identical fix) -- assume UTC, matching
    # how it was written (DateTime(timezone=True), default=datetime.now(timezone.utc)).
    created_at = content.created_at
    if created_at is not None and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_hours = (now - created_at).total_seconds() / 3600.0 if created_at else 0.0
    age_hours = max(0.0, min(age_hours, MAX_CONTENT_AGE_HOURS))
    popularity = max(0.0, min(1.0, float(content.popularity_score or 0.0)))
    try:
        return Candidate(
            contentId=content.content_id, creatorId=content.creator_id, category=content.category,
            contentPopularityScore=popularity, contentAgeHours=age_hours,
            creatorFollowed=False, alreadySeen=False,  # already-seen exclusion happens at retrieval time (below)
            title=content.title, hashtags=content.hashtags, topics=content.topics,
            entities=content.entities, subgenres=content.subgenres,
            # Explicit, matching the field defaults exactly (None) -- mypy's dataclass_transform
            # synthesis of Candidate.__init__ treats aliased optional fields as required
            # keyword arguments despite their runtime default; this is a static-typing-only
            # quirk, not a real behavior change (see app.schemas.recommendation_schemas.Candidate).
            language=None, candidateSource=None, socialContext=None, localBucketSource=None,
        )
    except Exception:  # noqa: BLE001 -- returns None for an invalid row rather than raise (see docstring's documented convention)
        return None


def _empty_response(user_id: str, model_version: str, interaction_count: int) -> dict[str, Any]:
    return {
        "userId": user_id, "modelVersion": model_version, "source": "TWO_TOWER_SHADOW",
        "interactionCount": interaction_count, "twoTowerRetrievedCount": 0, "recommendations": [],
    }


def two_tower_retrieval_candidates(
    db, history, user_id: str, seen_content_ids: set[str], *, retrieval_k: int, now: datetime,
    artifact_path: Path | None = None,
) -> list[Candidate]:
    """Retrieval / candidate generation ONLY (Phase 3.4 Step 3, items 1-4): load the trained
    artifact, embed the user from an ALREADY-BUILT `history` (the same `FeatureHistory`
    `recommendation_service.recommend()` already constructed -- from a caller-supplied
    userProfile or from DB rows, whichever path that request took -- so this never re-derives
    or duplicates that logic), retrieve Top-K over the active catalog, convert to the existing
    `Candidate` schema. Scoring/reranking/response-building are the CALLER's job (recommend()'s
    own existing code) -- never done here, so that logic is never duplicated.

    Artifact load and content-index build are CACHED per request
    (`app.ml.two_tower.index_cache.two_tower_index_cache`, mirroring `app.ml.model_cache`'s
    file-signature freshness contract) -- neither the trained artifact nor the full catalog is
    re-embedded on every call; only this one user's embedding (cheap, user-specific) is
    computed per request. Only the top-K retrieved content rows are re-fetched from the DB
    (never the full catalog), so a stale/detached ORM object from a previous session is never
    served.

    Raises on any failure (artifact missing/corrupt, no matching catalog, etc.) -- the caller
    (`recommendation_service._try_two_tower_candidates`) is responsible for catching and
    falling back; this function never fabricates a result."""
    two_tower_model, tt_metadata, content_index = two_tower_index_cache.get_or_build(db, artifact_path)
    categories: list[str] = tt_metadata["categories"]

    user_vector = build_user_vector(history, user_id, categories)
    user_embedding = embed_user(two_tower_model, user_vector)

    if not content_index.content_ids:
        return []

    # Brute-force NumPy/PyTorch cosine similarity (app.ml.two_tower.retrieval) against the
    # CACHED index -- no FAISS/vector DB, matching the PoC's own Step 6 design (still
    # proof-of-concept scale); no per-request re-embedding of the catalog either way.
    retrieved = retrieve_top_k(user_embedding, content_index, retrieval_k, exclude_content_ids=seen_content_ids)
    if not retrieved:
        return []

    retrieved_ids = [content_id for content_id, _similarity in retrieved]
    catalog_by_id = {c.content_id: c for c in db.scalars(select(Content).where(Content.content_id.in_(retrieved_ids))).all()}
    return [
        candidate for content_id, _similarity in retrieved
        if (row := catalog_by_id.get(content_id)) is not None
        and (candidate := content_to_candidate(row, now)) is not None
    ]


def retrieve_and_rank_with_two_tower(
    db, user_id: str, *, limit: int = 10, retrieval_k: int = DEFAULT_RETRIEVAL_K,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Phase 3.2 shadow pipeline. See module docstring for the full flow/boundary.

    Failure behavior (Step 8, deliberately fail-closed):
      - Two-Tower artifact missing/corrupt -> raises
        `app.ml.two_tower.artifact_store.TwoTowerArtifactError` (a subclass). Never falls back
        to fabricated candidates, never trains a replacement at request time.
      - Production RandomForest artifact missing/corrupt -> raises the SAME
        `ModelNotTrained`/`ModelArtifactInvalid` `recommend()` raises for the same conditions.
    Either failure is confined to THIS function: it is never called from the public
    `POST /recommendations` path, so it cannot affect production behavior.
    """
    # Same production RandomForest loading contract recommend() uses -- same path, same
    # in-process cache, same compatibility validation, same exceptions on failure.
    try:
        rf_model, rf_metadata = model_cache.video_cache.get(
            MODEL_PATH, model_store.METADATA_PATH, expected_features=FEATURES,
        )
    except model_store.ArtifactNotFoundError as exc:
        raise ModelNotTrained() from exc
    except model_store.ArtifactIncompatibleError as exc:
        raise ModelArtifactInvalid(str(exc), reason="INCOMPATIBLE") from exc
    except model_store.ArtifactCorruptedError as exc:
        raise ModelArtifactInvalid(str(exc), reason="CORRUPTED") from exc
    except model_store.ArtifactBusyError as exc:
        raise ModelArtifactInvalid(str(exc), reason="BUSY") from exc

    now = datetime.now(timezone.utc)
    rows = recent_interactions_for_ranking(db, user_id, limit=RECOMMENDATION_HISTORY_MAX_INTERACTIONS)
    seen_content_ids = {row.content_id for row in rows}
    history_content_by_id = (
        {item.content_id: item for item in db.scalars(select(Content).where(Content.content_id.in_(seen_content_ids))).all()}
        if seen_content_ids else {}
    )
    history = history_from_rows(rows, history_content_by_id)
    interaction_count = len(rows)
    cold_start = not rows

    # (a)+(b) RUNTIME-ONLY artifact load + retrieval / candidate generation ONLY -- no scoring,
    # no reranking here; shared with the Phase 3.4 guarded-integration path
    # (two_tower_retrieval_candidates), never duplicated between the two callers.
    candidates = two_tower_retrieval_candidates(
        db, history, user_id, seen_content_ids, retrieval_k=retrieval_k, now=now, artifact_path=artifact_path,
    )
    if not candidates:
        return _empty_response(user_id, rf_metadata["modelVersion"], interaction_count)

    # (c) FROM HERE ON: the EXISTING production ranking code, called unmodified -- not
    # reimplemented, not duplicated.
    feature_rows = [
        history.features(
            user_id=user_id, category=c.category, creator_id=c.creator_id, content_id=c.content_id,
            timestamp=now, content_popularity_score=c.content_popularity_score,
            content_created_at=now - timedelta(hours=c.content_age_hours),
            creator_followed=c.creator_followed, already_seen=c.already_seen,
            hashtags=c.hashtags, topics=c.topics, entities=c.entities, subgenres=c.subgenres, title=c.title,
        )
        for c in candidates
    ]
    if not feature_rows:
        return _empty_response(user_id, rf_metadata["modelVersion"], interaction_count)

    model_scores = probabilities(rf_model, feature_rows)  # SAME production RandomForest call
    scored = [
        {"candidate": c, "features": f, "model_score": float(s), "reason": explanation(f, c, cold_start)}
        for c, f, s in zip(candidates, feature_rows, model_scores)
    ]
    def _model_score(item: dict[str, object]) -> float:
        # `scored`'s dict values are heterogeneous (Candidate/features/float/str), so mypy
        # infers `object` for every key -- cast is safe here: "model_score" is always the
        # `float(s)` set immediately above, never anything else.
        return cast(float, item["model_score"])

    scored.sort(key=_model_score, reverse=True)
    chosen = rerank(scored, limit, cold_start=cold_start)  # SAME production reranker call

    return {
        "userId": user_id, "modelVersion": rf_metadata["modelVersion"], "source": "TWO_TOWER_SHADOW",
        "interactionCount": interaction_count, "twoTowerRetrievedCount": len(candidates),
        "recommendations": [
            {"contentId": item["candidate"].content_id, "category": item["candidate"].category,
             "score": round(item["adjusted_score"], 6), "rank": rank, "reason": item["reason"]}
            for rank, item in enumerate(chosen, 1)
        ],
    }
