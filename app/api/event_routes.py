from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.event_schemas import EventCreate, EventResponse
from app.services.event_service import (
    ContentInactiveError,
    ContentTypeMismatchError,
    store_event,
)

# Phase A §9: tagged as demo scaffolding, not the production Event Tracking Service contract.
router=APIRouter(tags=["demo-event-tracking"])
_CREATE_EVENT_EXAMPLE = {"eventId": "event-001", "stored": True, "watchPercentage": 86.67}


# Eventual-consistency audit (Task: local-existence blocking validation): a locally-unknown
# contentId is no longer a rejection (app.services.event_service.store_event stores the
# interaction with Interaction.content_pending=True instead) -- RMS does not own Content, so
# a projection that simply hasn't arrived yet must not be treated as an invalid event. There
# is therefore no CONTENT_NOT_FOUND response from this endpoint any more.
_EVENT_RESPONSES: dict[int | str, dict[str, Any]] = {201: {"content": {"application/json": {"example": _CREATE_EVENT_EXAMPLE}}},
                    409: {"content": {"application/json": {"example": {"error": "CONTENT_INACTIVE", "message": "Content 'video-101' is not active.", "requestId": "..."}}}},
                    422: {"content": {"application/json": {"example": {"error": "CONTENT_TYPE_MISMATCH", "message": "eventType 'LIVE_JOINED' does not match this content's type.", "requestId": "..."}}}}}


def _create_event(event: EventCreate, db: Session) -> dict:
    """Watch percentage and the training label are always server-derived, never accepted
    from the client. A duplicate eventId returns the original result idempotently
    (stored=false) rather than creating a second interaction."""
    try:
        row,stored=store_event(db,event)
    except ContentInactiveError:
        raise HTTPException(409,detail={"error":"CONTENT_INACTIVE","message":f"Content '{event.content_id}' is not active."})
    except ContentTypeMismatchError as exc:
        raise HTTPException(422,detail={"error":"CONTENT_TYPE_MISMATCH","message":str(exc)})
    return {"eventId":row.event_id,"stored":stored,"watchPercentage":row.watch_percentage}


@router.post("/events",response_model=EventResponse,status_code=status.HTTP_201_CREATED,responses=_EVENT_RESPONSES)
def create_event(event:EventCreate,db:Session=Depends(get_db)):
    """Example values above are illustrative only, not measured results."""
    return _create_event(event, db)


@router.post("/interactions",response_model=EventResponse,status_code=status.HTTP_201_CREATED,responses=_EVENT_RESPONSES,deprecated=True)
def create_interaction(event:EventCreate,db:Session=Depends(get_db)):
    """Spec §6/§63: backward-compatible alias for POST /events using the exact same
    validation, idempotency, and label-derivation logic (app.services.event_service) --
    `/events` is preserved unchanged for existing callers. Example values above are
    illustrative only, not measured results."""
    return _create_event(event, db)

