"""Content-ingestion-time enrichment boundary (see `app.ml.content_classifier`'s module
docstring for the classification approach itself). Runs ONCE, when content enters the local
Content projection with no caller-supplied category -- never on the `/candidates/generate`
hot path, and never re-run by it (see `app.api.candidate_routes.generate`, which only ever
reads `Content.category`/`category_confidence`/`category_source` back).

Reusable by a future real Content/LIVE Service Kafka consumer without any change: this
module is the local-projection enrichment boundary, exactly like
`app.services.event_service.store_event` already is for interaction events
(`app.services.kafka_behavior_consumer` reuses that function unchanged) and
`app.services.providers.live_candidate_provider` is for the LIVE local content projection.
No such consumer exists yet -- that integration gap is deliberately left unfilled here (no
upstream Kafka topic/schema has been agreed for content lifecycle events either), only the
reusable enrichment function itself.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import (
    CONTENT_CATEGORY_CREATOR_PRIOR_WEIGHT,
    CONTENT_CATEGORY_CREATOR_PROFILE_MAX_ROWS,
    CONTENT_CATEGORY_CREATOR_PROFILE_MIN_SAMPLES,
    CONTENT_CATEGORY_HIGH_CONFIDENCE_THRESHOLD,
    CONTENT_CATEGORY_LOW_CONFIDENCE_THRESHOLD,
    CONTENT_CLASSIFIER_METADATA_PATH,
    CONTENT_CLASSIFIER_MODEL_PATH,
)
from app.core.logging import logger
from app.db.models import Content
from app.ml.content_classifier import (
    UNKNOWN_CATEGORY,
    ContentClassification,
    classify_text,
    combine_with_creator_prior,
    load_classifier,
)

__all__ = ["ClassifierNotTrained", "infer_category", "infer_live_category_from_creator_history"]

# Process-local cache, keyed by the resolved model path and invalidated by the artifact's own
# mtime -- mirrors app.ml.model_cache's freshness contract at a much smaller scale (a single
# small sklearn Pipeline, not a hot-swappable production model with promote/rollback).
_cache: dict[str, tuple[object, dict, int]] = {}


class ClassifierNotTrained(Exception):
    """No content-classifier artifact exists yet at CONTENT_CLASSIFIER_MODEL_PATH -- run
    scripts/train_content_classifier.py. Distinct from the VIDEO/LIVE ModelNotTrained
    exceptions on purpose: this is a different artifact with no promotion/rollback lifecycle."""


def _cached_classifier():
    model_path = CONTENT_CLASSIFIER_MODEL_PATH
    metadata_path = CONTENT_CLASSIFIER_METADATA_PATH
    if not model_path.exists() or not metadata_path.exists():
        raise ClassifierNotTrained(
            f"No content-classifier artifact at {model_path} -- run "
            "scripts/train_content_classifier.py first."
        )
    key = str(model_path)
    current_mtime = model_path.stat().st_mtime_ns
    cached = _cache.get(key)
    if cached is not None and cached[2] == current_mtime:
        return cached[0], cached[1]
    pipeline, metadata = load_classifier(model_path=model_path, metadata_path=metadata_path)
    _cache[key] = (pipeline, metadata, current_mtime)
    return pipeline, metadata


def _creator_category_counts(db: Session, creator_id: str) -> dict[str, int]:
    """Bounded (CONTENT_CATEGORY_CREATOR_PROFILE_MAX_ROWS), VIDEO-only (LIVE category
    inference is out of scope here and untouched), derived entirely from this service's own
    local `Content` rows -- no synchronous Content/Follow/UBS/Candidate Service call.
    Excludes nothing else: the content currently being classified does not exist as a
    `Content` row yet at creation time, so self-leakage cannot occur through this query."""
    rows = db.scalars(
        select(Content.category)
        .where(Content.creator_id == creator_id, Content.content_type == "VIDEO")
        .order_by(Content.created_at.desc())
        .limit(CONTENT_CATEGORY_CREATOR_PROFILE_MAX_ROWS)
    ).all()
    counts: dict[str, int] = {}
    for category in rows:
        counts[category] = counts.get(category, 0) + 1
    return counts


def infer_category(db: Session, *, creator_id: str, title: str | None, hashtags: list[str] | None) -> ContentClassification:
    """Never makes a synchronous external call (Content/Follow/UBS/Candidate Service) --
    reads only the local classifier artifact and this service's own local `Content` table.
    Raises `ClassifierNotTrained` only when the artifact genuinely does not exist yet."""
    pipeline, _ = _cached_classifier()
    current_probs = classify_text(pipeline, title, hashtags)
    creator_counts = _creator_category_counts(db, creator_id)
    result = combine_with_creator_prior(
        current_probs, creator_counts,
        low_confidence_threshold=CONTENT_CATEGORY_LOW_CONFIDENCE_THRESHOLD,
        high_confidence_threshold=CONTENT_CATEGORY_HIGH_CONFIDENCE_THRESHOLD,
        creator_prior_weight=CONTENT_CATEGORY_CREATOR_PRIOR_WEIGHT,
        creator_profile_min_samples=CONTENT_CATEGORY_CREATOR_PROFILE_MIN_SAMPLES,
    )
    logger.debug(
        "content category inferred creatorId=%s category=%s confidence=%.3f source=%s",
        creator_id, result.category, result.confidence, result.source,
    )
    return result


def infer_live_category_from_creator_history(db: Session, *, creator_id: str) -> ContentClassification:
    """LIVE-specific category fallback (LIVE ingestion contract audit) for when the caller
    (a real Live Service) has no category to supply at all -- deliberately NOT a call into
    `infer_category`/`combine_with_creator_prior`: that function's own precedence rule #1
    ("no usable current-content evidence at all -> UNKNOWN, regardless of creator history: a
    creator prior alone is never enough to assert what a SPECIFIC piece of content is about")
    would always return UNKNOWN here, since a LIVE stream's raw ingestion payload
    (contentId/creatorId/title/lifecycle) never reliably carries usable classifiable text, and
    LIVE text classification is explicitly out of this classifier's scope regardless (see
    `infer_category`'s own module docstring).

    Instead this is an honest, simpler, deterministic signal: this creator's own historical
    VIDEO category distribution (`_creator_category_counts`, unchanged, reused verbatim), used
    ONLY when there are at least `CONTENT_CATEGORY_CREATOR_PROFILE_MIN_SAMPLES` such rows --
    the SAME threshold VIDEO's own creator-prior blend already requires before trusting a
    creator prior at all, not a new invented one. Below that, or with no VIDEO history at all,
    returns UNKNOWN (`source="CREATOR_HISTORY"`, `confidence=0.0`) rather than a fabricated
    guess. Never touches the content classifier artifact/model at all.
    """
    counts = _creator_category_counts(db, creator_id)
    total = sum(counts.values())
    if total < CONTENT_CATEGORY_CREATOR_PROFILE_MIN_SAMPLES:
        return ContentClassification(UNKNOWN_CATEGORY, 0.0, "CREATOR_HISTORY")
    top_category = max(counts, key=lambda category: counts[category])
    confidence = counts[top_category] / total
    return ContentClassification(top_category, confidence, "CREATOR_HISTORY")
