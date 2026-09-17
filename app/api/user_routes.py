from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.db.repositories import interactions
from app.schemas.recommendation_schemas import SearchIntent, UserContext
from app.services.session_intent_provider import provider as session_intent_provider
from app.services.user_context_provider import persist_user_context
from app.services.user_profile_service import get_behaviour_profile, get_profile

# Phase A §9: tagged as demo scaffolding, not the production User Behavior Service contract.
router=APIRouter(tags=["demo-user-behavior"])


class UserCreate(BaseModel):
    # Reject unknown fields (matches app.schemas.event_schemas.EventCreate's own convention)
    # so a caller sending onboarding fields at the wrong nesting level (e.g. a flat "age"/
    # "region"/"interests" instead of inside "userContext") gets a 422, not a silent no-op
    # that looks like success while quietly discarding the data.
    model_config=ConfigDict(populate_by_name=True, extra="forbid")
    user_id:str=Field(alias="userId",min_length=1,max_length=128)
    # Optional (backward compatible: every pre-existing caller/test sends only userId and
    # keeps working byte-identically). Persisted once at creation time -- see
    # app.services.user_context_provider and app.services.recommendation_service's
    # effective_user_context for how a later recommendation request automatically resolves
    # this without needing to resend it.
    user_context: UserContext | None = Field(None, alias="userContext")

    @field_validator("user_id")
    @classmethod
    def _no_blank(cls,value:str)->str:
        value=value.strip()
        if not value: raise ValueError("userId must not be blank")
        return value


_CREATE_USER_EXAMPLE = {"userId": "user-1001", "status": "CREATED", "behaviourStatus": "NO_INTERACTIONS",
                        "recommendationStatus": "COLD_START", "createdAt": "2026-07-30T09:00:00+00:00"}


@router.post("/users",status_code=status.HTTP_201_CREATED,
             responses={201: {"content": {"application/json": {"example": _CREATE_USER_EXAMPLE}}},
                        409: {"content": {"application/json": {"example": {"error": "USER_ALREADY_EXISTS", "message": "User 'user-1001' already exists.", "requestId": "..."}}}}})
def create_user(payload:UserCreate,db:Session=Depends(get_db)):
    """Spec §4: creating a user assigns no interests, trains nothing, fabricates nothing.
    A brand-new user is a cold-start user until real interactions arrive. An optional
    `userContext` is persisted (never fabricated when absent) in the SAME transaction as the
    user row itself -- either both are committed or neither is, so a user can never exist
    with a missing/partial context row.
    Example values above are illustrative only, not measured results."""
    row=User(user_id=payload.user_id,status="ACTIVE",created_at=datetime.now(timezone.utc))
    db.add(row)
    if payload.user_context is not None:
        persist_user_context(db,payload.user_id,payload.user_context)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409,detail={"error":"USER_ALREADY_EXISTS","message":f"User '{payload.user_id}' already exists."})
    db.refresh(row)
    return {"userId":row.user_id,"status":"CREATED","behaviourStatus":"NO_INTERACTIONS",
            "recommendationStatus":"COLD_START","createdAt":row.created_at.isoformat()}


@router.get("/users/{user_id}")
def get_user(user_id:str,db:Session=Depends(get_db)):
    row=db.scalar(select(User).where(User.user_id==user_id))
    if row is None:
        raise HTTPException(404,detail={"error":"USER_NOT_FOUND","message":f"User '{user_id}' does not exist."})
    # Intentionally NOT VIDEO-filtered: `interactions(db, user_id)` (app.db.repositories) counts
    # every stored interaction regardless of content_type. This demo-user-behavior endpoint
    # stands in for the real User Behavior Service, which README.md ("consumes events and
    # maintains point-in-time user, category, creator and LIVE behavior features") documents as
    # spanning both VIDEO and LIVE -- unlike app.services.recommendation_service/training_service/
    # cohort_aggregation_service, which deliberately re-derive their OWN VIDEO-only interaction
    # count for scoring/strategy/training. A caller wanting the VIDEO-only count that drives
    # recommendation strategy should read RecommendationResponse.interactionCount instead.
    count=len(interactions(db,user_id))
    return {"userId":row.user_id,"status":row.status,"interactionCount":count,
            "behaviourStatus":"NO_INTERACTIONS" if count==0 else "HAS_INTERACTIONS",
            "createdAt":row.created_at.isoformat() if row.created_at else None}


_COLD_START_PROFILE_EXAMPLE = {"userId": "user-2000", "status": "COLD_START", "interactionCount": 0,
                               "categoryAffinities": [], "creatorAffinities": [],
                               "message": "Not enough user interactions to build a behaviour profile."}
_ACTIVE_PROFILE_EXAMPLE = {"userId": "user-1001", "status": "ACTIVE", "interactionCount": 12,
                          "categoryAffinities": [{"category": "SPORT", "score": 0.78, "interactionCount": 7}],
                          "creatorAffinities": [{"creatorId": "creator-20", "score": 0.71, "interactionCount": 4}],
                          "lastInteractionAt": "2026-07-30T10:30:00+00:00"}


@router.get("/users/{user_id}/behaviour-profile",
            responses={200: {"content": {"application/json": {"examples": {
                "coldStart": {"summary": "User with no interactions", "value": _COLD_START_PROFILE_EXAMPLE},
                "active": {"summary": "User with recorded interactions", "value": _ACTIVE_PROFILE_EXAMPLE},
            }}}}})
def behaviour_profile(user_id:str,db:Session=Depends(get_db)):
    """Spec §11: behaviour calculated from recorded interactions; a user without
    interactions gets an explicit COLD_START answer, never fabricated preferences.
    Example values above are illustrative only, not measured results."""
    return get_behaviour_profile(db,user_id)
@router.get("/users/{user_id}/profile", deprecated=True)
def profile(user_id:str,db:Session=Depends(get_db)):
    cats=get_profile(db,user_id)
    if not cats: raise HTTPException(404,detail={"error":"USER_NOT_FOUND","message":"No interactions found for this user."})
    return {"userId":user_id,"categories":cats}


_SEARCH_INTENT_RECORDED_EXAMPLE = {"userId": "user-1001", "normalizedQuery": "coldplay live concert",
                                   "category": None, "topics": ["live performance"], "entities": ["coldplay"],
                                   "subgenres": ["concert"], "confidence": 1.0,
                                   "recordedAt": "2026-08-18T10:00:00+00:00", "expiresAt": "2026-08-18T10:30:00+00:00",
                                   "status": "ACTIVE"}


@router.post("/users/{user_id}/search-intent", status_code=status.HTTP_201_CREATED,
             responses={201: {"content": {"application/json": {"example": _SEARCH_INTENT_RECORDED_EXAMPLE}}},
                        422: {"content": {"application/json": {"example": {"error": "EMPTY_SEARCH_INTENT", "message": "At least one of query/category/topics/entities/subgenres is required.", "requestId": "..."}}}}})
def record_search_intent(user_id: str, payload: SearchIntent, db: Session = Depends(get_db)):
    """Records an explicit user search as this user's CURRENT/session intent (finalization
    spec "Search -> Persistent Session Intent"). Reuses the exact same `SearchIntent` shape
    and normalization/tokenization already used for a request-scoped `searchIntent` on
    POST /recommendations -- generic token-overlap matching, no per-entity logic.

    CURRENT DEMO: stored in RMS's own local, in-memory, TTL-bounded Session Intent Provider
    (app.services.session_intent_provider) -- NOT the production User Behavior Service
    integration. A subsequent POST /recommendations for this same userId with NO
    searchIntent in the body will automatically pick up this intent (decayed by elapsed
    time) until it expires or a new search replaces it.

    PRODUCTION TARGET: this action is expected to originate from Event Tracking Service,
    flow through Kafka, and be owned by User Behavior Service; RMS would then read the
    same resolved intent from UBS instead of this local store, with zero change to how
    reranking uses it. Example values above are illustrative only, not measured results."""
    if not payload.query and not payload.category and not payload.topics and not payload.entities and not payload.subgenres:
        raise HTTPException(422, detail={"error": "EMPTY_SEARCH_INTENT",
                                         "message": "At least one of query/category/topics/entities/subgenres is required."})
    record = session_intent_provider.record_search(db, user_id, payload)
    return {"userId": user_id, "normalizedQuery": payload.query, "category": payload.category,
            "topics": payload.topics, "entities": payload.entities, "subgenres": payload.subgenres,
            "confidence": payload.confidence, "recordedAt": record.recorded_at.isoformat(),
            "expiresAt": record.expires_at.isoformat(), "status": "ACTIVE"}


_SEARCH_INTENT_ACTIVE_EXAMPLE = {"userId": "user-1001", "status": "ACTIVE", "query": "coldplay live concert",
                                 "category": None, "topics": ["live performance"], "entities": ["coldplay"],
                                 "subgenres": ["concert"], "originalConfidence": 1.0, "currentConfidence": 0.62,
                                 "recordedAt": "2026-08-18T10:00:00+00:00", "expiresAt": "2026-08-18T10:30:00+00:00"}
_SEARCH_INTENT_NONE_EXAMPLE = {"userId": "user-1001", "status": "NO_ACTIVE_SEARCH_INTENT"}


@router.get("/users/{user_id}/search-intent",
            responses={200: {"content": {"application/json": {"examples": {
                "active": {"summary": "User has an unexpired search intent", "value": _SEARCH_INTENT_ACTIVE_EXAMPLE},
                "none": {"summary": "No search recorded, or it has fully expired", "value": _SEARCH_INTENT_NONE_EXAMPLE},
            }}}}})
def get_search_intent(user_id: str, db: Session = Depends(get_db)):
    """Read-only debug/demo visibility into what POST /recommendations would currently
    resolve for this user if the request body omits searchIntent -- not part of the
    production contract. `currentConfidence` reflects TTL decay already applied;
    `originalConfidence` is the value as recorded. Example values above are illustrative
    only, not measured results."""
    record = session_intent_provider.peek(db, user_id)
    if record is None:
        return {"userId": user_id, "status": "NO_ACTIVE_SEARCH_INTENT"}
    active = session_intent_provider.get_active_intent(db, user_id)
    return {"userId": user_id, "status": "ACTIVE", "query": record.intent.query, "category": record.intent.category,
            "topics": record.intent.topics, "entities": record.intent.entities, "subgenres": record.intent.subgenres,
            "originalConfidence": record.intent.confidence, "currentConfidence": active.confidence if active else 0.0,
            "recordedAt": record.recorded_at.isoformat(), "expiresAt": record.expires_at.isoformat()}

