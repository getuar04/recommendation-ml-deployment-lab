"""Spec §5: content lifecycle. A content item carries enough information for feature
building and ranking (id, creator, type, category, popularity, created_at). Validation
happens here; interaction ingestion (app.services.event_service) additionally requires the
referenced content to exist and be active before an event is accepted."""
import json
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Content, Interaction
from app.ml.feature_builder import content_popularity_score
from app.ml.semantic_tokens import normalize_title, normalize_tokens
from app.services.content_enrichment_service import (
    ClassifierNotTrained,
    infer_category,
    infer_live_category_from_creator_history,
)

# Phase A §9: tagged as demo scaffolding, not the production Content Service contract.
router = APIRouter(tags=["demo-content-service"])


class ContentType(str, Enum):
    VIDEO = "VIDEO"
    LIVE = "LIVE"


class ContentCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    content_id: str = Field(alias="contentId", min_length=1, max_length=128)
    creator_id: str = Field(alias="creatorId", min_length=1, max_length=128)
    content_type: ContentType = Field(ContentType.VIDEO, alias="contentType")
    # Optional (content-understanding foundation): the real Content Service contract for a
    # VIDEO supplies only contentId/creatorId/title/hashtags -- never category (see
    # app.ml.content_classifier's module docstring). Omitting it triggers local category
    # inference at ingestion time; explicitly supplying it (every pre-existing caller/test)
    # is trusted outright, byte-identical to before this field became optional -- an explicit
    # caller-supplied category is never second-guessed by the classifier.
    category: str | None = Field(None, min_length=1, max_length=64)
    duration_seconds: float | None = Field(None, alias="durationSeconds", gt=0)
    created_at: datetime | None = Field(None, alias="createdAt")
    # Semantic metadata (VIDEO only): optional, additive -- a caller sending only the fields
    # above keeps working unchanged. Normalized the exact same way app.schemas.recommendation_
    # schemas.Candidate normalizes request-time semantic fields (app.ml.semantic_tokens),
    # which is what the training/inference normalization-parity requirement depends on.
    title: str | None = Field(None)
    hashtags: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    subgenres: list[str] = Field(default_factory=list)
    # `primaryCategory`/`subcategory`/`taxonomyVersion` are deliberately NOT declared here --
    # ownership-audit finding (Task: correct the public Content ingestion write contract):
    # these are RMS-OWNED canonical semantic fields (their DB columns/read-serialization
    # remain, see app.db.models.Content and _serialize below), never caller-writable, exactly
    # like `category_confidence`/`category_source` have never been declared ContentCreate
    # fields either. `model_config` below does not set `extra="forbid"` (unchanged, and
    # deliberately not broadened by this fix -- see this module's own audit-driven decision
    # not to reject arbitrary unknown fields generally), so a caller that sends these
    # reserved names anyway gets pydantic's existing default "ignore" behavior -- the exact
    # same behavior a caller already gets today for `categoryConfidence`/`categorySource`.
    # No caller input can reach `primary_category`/`subcategory`/`taxonomy_version` on the
    # `Content` row through this endpoint; see create_content's own comment for where their
    # values now come from instead.
    #
    # `popularityScore` is deliberately NOT declared here either -- popularity-ownership-audit
    # finding (Task: deep popularityScore audit): it used to be a plain `Field(0.5, ...)`
    # default a caller could freely override with any value in [0, 1] at creation time, and
    # nothing ever updated it afterward -- a real, currently-forgeable, permanently-static
    # signal fed directly into the active VIDEO 34-feature model. It is RMS-owned now, exactly
    # like the taxonomy fields above: derived locally from real interaction evidence (see
    # app.ml.feature_builder.content_popularity_score, recomputed by
    # app.services.event_service.store_event after every new real VIDEO interaction), never
    # caller-writable. A caller that still sends `popularityScore` gets the same silent-ignore
    # behavior as the taxonomy fields; see create_content's own comment for where the initial
    # value comes from instead.

    @field_validator("content_id", "creator_id")
    @classmethod
    def _no_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("category")
    @classmethod
    def _no_blank_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("title")
    @classmethod
    def _normalize_title(cls, value: str | None) -> str | None:
        return normalize_title(value)

    @field_validator("hashtags", "topics", "entities", "subgenres", mode="before")
    @classmethod
    def _normalize_field_tokens(cls, value: object, info) -> list[str]:
        return normalize_tokens(value, field=info.field_name)


def _serialize(row: Content) -> dict:
    return {
        "contentId": row.content_id, "creatorId": row.creator_id,
        "contentType": row.content_type, "category": row.category,
        "durationSeconds": row.duration_seconds, "popularityScore": row.popularity_score,
        "isActive": row.is_active, "status": "CREATED",
        "title": row.title, "hashtags": row.hashtags, "topics": row.topics,
        "entities": row.entities, "subgenres": row.subgenres,
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "updatedAt": row.updated_at.isoformat() if row.updated_at else None,
        # Content-understanding debug/verification fields (None for every row whose category
        # was explicitly caller-supplied, or that predates this feature -- see
        # app.db.models.Content's own comment). Internal/debugging visibility only, per the
        # content-understanding audit's own instruction not to over-expose internal fields;
        # kept in this same demo-scaffolding response rather than a second, parallel
        # representation since none currently exists.
        "categoryConfidence": row.category_confidence, "categorySource": row.category_source,
        # Future canonical taxonomy foundation (additive, storage-only -- see app.db.models.
        # Content's own comment): None for every row today, since nothing in this task
        # populates a final taxonomy. Exposed the same way categoryConfidence/categorySource
        # already are, not as a new internal-only concept.
        "primaryCategory": row.primary_category, "subcategory": row.subcategory,
        "taxonomyVersion": row.taxonomy_version,
    }


def _resolve_pending_interactions(db: Session, content: Content) -> None:
    """Eventual-consistency reconciliation (Task: content_pending evidence-leak audit): the
    other half of app.services.event_service.store_event's content_pending mechanism -- an
    interaction that arrived before this Content row existed was durably stored with
    content_pending=True and excluded from every evidence path (app.db.repositories, plus the
    LIVE providers that query Interaction directly), but nothing ever cleared that flag once
    Content caught up. Called once, right after this Content row is committed, so a pending
    interaction becomes eligible for evidence at most one HTTP request after its content
    arrives, never requiring a background scheduler.

    Domain-safe: only resolves a pending row whose own event_type domain (VIDEO vs LIVE, via
    the same `event_type.startswith("LIVE_")` convention app.services.event_service.store_event
    already uses for a content that WAS known at ingestion time) agrees with this Content's own
    content_type. A domain-mismatched pending row (e.g. a LIVE_* event for a content_id that
    turns out to be VIDEO) is left content_pending=True permanently -- exactly the outcome
    store_event itself would have produced (ContentTypeMismatchError, rejected outright) had
    Content existed at ingestion time; there is no way to retroactively reject an already-
    stored row, so it simply never becomes evidence, matching the "Content never arrives"
    durability guarantee rather than being silently activated as if it had matched.

    Idempotent by construction: the WHERE clause only ever matches rows still pending, so a
    second reconciliation pass for the same content_id (there is none today -- content_id is
    unique, a repeat POST /contents fails with 409 before this is called again -- but this
    holds regardless) touches zero rows and commits nothing new."""
    pending_rows = db.scalars(
        select(Interaction).where(
            Interaction.content_id == content.content_id, Interaction.content_pending.is_(True),
        )
    ).all()
    if not pending_rows:
        return
    is_live_content = content.content_type == "LIVE"
    resolved_any = False
    for row in pending_rows:
        if row.event_type.startswith("LIVE_") == is_live_content:
            row.content_pending = False
            resolved_any = True
    if resolved_any:
        db.commit()


_CREATE_CONTENT_EXAMPLE = {"contentId": "video-101", "creatorId": "creator-20", "contentType": "VIDEO",
                          "category": "SPORT", "durationSeconds": 60, "popularityScore": 0.5, "isActive": True,
                          "status": "CREATED", "createdAt": "2026-07-30T09:05:00+00:00", "updatedAt": "2026-07-30T09:05:00+00:00",
                          "categoryConfidence": None, "categorySource": None,
                          "primaryCategory": None, "subcategory": None, "taxonomyVersion": None}


@router.post("/contents", status_code=status.HTTP_201_CREATED,
             responses={201: {"content": {"application/json": {"example": _CREATE_CONTENT_EXAMPLE}}},
                        409: {"content": {"application/json": {"example": {"error": "CONTENT_ALREADY_EXISTS", "message": "Content 'video-101' already exists.", "requestId": "..."}}}},
                        422: {"content": {"application/json": {"example": {"error": "VALIDATION_ERROR", "message": "Request validation failed.", "requestId": "..."}}}}})
def create_content(payload: ContentCreate, db: Session = Depends(get_db)):
    """Example values above are illustrative only, not measured results."""
    created = payload.created_at or datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)

    category = payload.category
    category_confidence: float | None = None
    category_source: str | None = None
    # taxonomy_version is RMS-owned provenance, never caller-controlled (ownership-audit
    # finding): it starts NULL -- an explicit caller-supplied `category` carries no RMS
    # classification evidence at all, so there is no taxonomy RMS can honestly claim this row
    # belongs to (must not pretend a caller-supplied legacy category belongs to v1-bootstrap).
    # It is stamped ONLY below, from `classification.taxonomy_version`, and only on the
    # branch where RMS's own classifier actually ran.
    taxonomy_version: str | None = None
    if category is not None:
        category = category.upper()
    elif payload.content_type is ContentType.LIVE:
        # LIVE ingestion contract audit: a real Live Service reliably knows lifecycle facts
        # (streamId/creatorId/title/start-end) but not category/hashtags/semantic topics --
        # requiring the caller to fabricate one to pass validation would be worse than not
        # having it. Deliberately NOT the VIDEO text classifier (out of scope, and a LIVE
        # ingestion payload rarely has classifiable text anyway) -- this creator's own
        # historical VIDEO category distribution instead, or a transparent UNKNOWN when that
        # doesn't exist/isn't enough evidence either. See infer_live_category_from_creator_
        # history's own docstring for exactly why this is not routed through infer_category.
        classification = infer_live_category_from_creator_history(db, creator_id=payload.creator_id)
        category = classification.category
        category_confidence = classification.confidence
        category_source = classification.source
        taxonomy_version = classification.taxonomy_version
    else:
        try:
            classification = infer_category(db, creator_id=payload.creator_id, title=payload.title, hashtags=payload.hashtags)
        except ClassifierNotTrained as exc:
            raise HTTPException(503, detail={"error": "CONTENT_CLASSIFIER_NOT_TRAINED", "message": str(exc)})
        # UNKNOWN is a real, storable category value here, never rejected: OneHotEncoder(
        # handle_unknown="ignore") in app.ml.pipeline_builder already makes any category
        # value the active VIDEO model has never seen -- UNKNOWN included -- safe to store
        # and serve (zero-vector for that one feature, never a crash); see
        # app.ml.content_classifier's module docstring for the full safety argument.
        category = classification.category
        category_confidence = classification.confidence
        category_source = "INFERRED"
        taxonomy_version = classification.taxonomy_version

    row = Content(content_id=payload.content_id, creator_id=payload.creator_id,
                  category=category, content_type=payload.content_type.value,
                  duration_seconds=payload.duration_seconds,
                  # No real interaction evidence exists yet for brand-new content -- the same
                  # neutral "no evidence" prior app.services.event_service.store_event later
                  # recomputes this from once real engagement accrues (see
                  # ContentCreate's own comment above for why this is no longer caller input).
                  popularity_score=content_popularity_score([]),
                  is_active=True, created_at=created, updated_at=created,
                  title=payload.title,
                  hashtags_json=json.dumps(payload.hashtags) if payload.hashtags else None,
                  topics_json=json.dumps(payload.topics) if payload.topics else None,
                  entities_json=json.dumps(payload.entities) if payload.entities else None,
                  subgenres_json=json.dumps(payload.subgenres) if payload.subgenres else None,
                  category_confidence=category_confidence, category_source=category_source,
                  # primary_category/subcategory are deliberately omitted here (left at their
                  # column default of NULL) -- no caller input can reach them (see
                  # ContentCreate's own comment) and RMS derives nothing for them yet; this is
                  # intentional, not an oversight (see app.db.models.Content's own comment).
                  taxonomy_version=taxonomy_version)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, detail={"error": "CONTENT_ALREADY_EXISTS",
                                         "message": f"Content '{payload.content_id}' already exists."})
    db.refresh(row)
    _resolve_pending_interactions(db, row)
    return _serialize(row)


@router.get("/contents/{content_id}")
def get_content(content_id: str, db: Session = Depends(get_db)):
    row = db.scalar(select(Content).where(Content.content_id == content_id))
    if row is None:
        raise HTTPException(404, detail={"error": "CONTENT_NOT_FOUND",
                                         "message": f"Content '{content_id}' does not exist."})
    return _serialize(row)


class ContentStatusUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    is_active: bool = Field(alias="isActive")


_DEACTIVATED_CONTENT_EXAMPLE = {**_CREATE_CONTENT_EXAMPLE, "isActive": False}


@router.patch("/contents/{content_id}", responses={
    200: {"content": {"application/json": {"example": _DEACTIVATED_CONTENT_EXAMPLE}}},
    404: {"content": {"application/json": {"example": {"error": "CONTENT_NOT_FOUND", "message": "Content 'video-101' does not exist.", "requestId": "..."}}}}})
def update_content_status(content_id: str, payload: ContentStatusUpdate, db: Session = Depends(get_db)):
    """Spec §5/§24: the only lifecycle mutation exposed -- activate/deactivate. No hard
    delete. Deactivated content is excluded from candidate generation (app.api.candidate_routes)
    and from recommendation scoring (app.services.recommendation_service), and stops
    accepting new interaction events (app.services.event_service.ContentInactiveError);
    already-stored interactions/reports referencing it are untouched. Example values above
    are illustrative only, not measured results."""
    row = db.scalar(select(Content).where(Content.content_id == content_id))
    if row is None:
        raise HTTPException(404, detail={"error": "CONTENT_NOT_FOUND",
                                         "message": f"Content '{content_id}' does not exist."})
    row.is_active = payload.is_active
    db.commit()
    db.refresh(row)
    return _serialize(row)
