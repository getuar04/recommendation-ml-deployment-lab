from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Content, Interaction, User
from app.db.repositories import find_event, interactions_for_content
from app.ml.feature_builder import (
    content_popularity_score,
    interaction_signals,
    watch_percentage,
)


class ContentInactiveError(Exception):
    """The referenced content exists but has been deactivated and must not accept new
    interactions."""

class ContentTypeMismatchError(Exception):
    """The event's own domain (LIVE_* eventType vs. every other eventType -- VIDEO_*/CONTENT_*/
    CREATOR_FOLLOWED, the existing VIDEO-domain vocabulary) does not match the referenced
    Content row's `content_type` (Task: real LIVE event-integration audit, Part 13 -- a
    malformed/misclassified event must never contaminate the wrong pipeline's history/dynamic
    state). `Content.content_type` is authoritative; this rejects at ingestion instead of
    silently storing an Interaction row a downstream LIVE/VIDEO-only consumer would either have
    to filter out itself or, worse, never filters (see app.services.providers.
    live_dynamic_state_provider, which trusts its caller's stream_ids and has no defense of its
    own against a stray wrong-domain row already sharing that content_id)."""

def _ensure_user(db: Session, user_id: str) -> None:
    """Spec §7 post-condition ('user exists') kept additive: an event for an unknown user
    registers the user row instead of rejecting, so pre-existing ingestion flows and tests
    keep working while the users table stays consistent with observed activity."""
    if db.scalar(select(User.id).where(User.user_id == user_id)) is None:
        db.add(User(user_id=user_id, status="ACTIVE"))

def store_event(db: Session, event):
    # Idempotency first (application-level fast path): a duplicate delivery of an
    # already-stored event returns the existing row unconditionally, even if the
    # referenced content has since been deactivated -- content-existence/active
    # validation only gates *new* interactions. This check alone is race-prone (two
    # concurrent requests for the same eventId can both pass it before either commits);
    # the database-level fallback below is what actually makes idempotency safe.
    old=find_event(db,event.event_id)
    if old: return old,False
    # Eventual-consistency audit (Task: local-existence blocking validation): RMS does not
    # own Content -- a locally-missing Content row can simply mean that projection hasn't
    # arrived yet (out-of-order delivery relative to content.created), not that the event is
    # invalid. A genuinely new interaction is therefore never rejected for this reason alone;
    # see Interaction.content_pending's own comment (app.db.models) for what this sets and
    # why. The active/content-type checks below require an authoritative Content row to
    # compare against, so they only run -- unweakened -- when one is actually present; they
    # are never assumed to pass when it isn't.
    content=db.scalar(select(Content).where(Content.content_id==event.content_id))
    content_pending = content is None
    if content is not None:
        if not content.is_active:
            raise ContentInactiveError(event.content_id)
        is_live_event = event.event_type.value.startswith("LIVE_")
        is_live_content = content.content_type == "LIVE"
        if is_live_event != is_live_content:
            raise ContentTypeMismatchError(
                f"eventType={event.event_type.value!r} does not match content_type={content.content_type!r} "
                f"for contentId={event.content_id!r}"
            )
    data=event.model_dump(); data.pop("live_watch_time_seconds",None); data["event_type"]=event.event_type.value
    signals = interaction_signals(event)
    for field in signals._fields:
        data[field] = getattr(signals, field)
    # Server-derived, never accepted from the caller: watch_percentage is not a field on
    # EventCreate at all, so there is nothing here for a client to override.
    data["watch_percentage"]=watch_percentage(event.watch_time_seconds,event.content_duration_seconds)
    data["content_pending"]=content_pending
    _ensure_user(db, event.user_id)
    row=Interaction(**data); db.add(row)
    try:
        db.commit()
    except IntegrityError:
        # Lost a race: another concurrent request inserted this eventId (or the same
        # brand-new userId) first and committed between our idempotency check and our
        # own commit. Roll back everything from this attempt -- including the pending
        # User row from _ensure_user() above, so a losing request never leaves a
        # partial/duplicate user behind -- then fall back to the now-persisted row
        # instead of surfacing a raw database error as an internal 500.
        db.rollback()
        existing=find_event(db,event.event_id)
        if existing is not None:
            return existing,False
        # Not an eventId collision -- the only conflict was the auto-created User row (two
        # concurrent events for the same brand-new userId). The winner's commit already
        # created that user, so this is a genuinely new interaction, not a duplicate: retry
        # the Interaction insert alone (User already exists, no need to add it again) instead
        # of surfacing the loser's raw IntegrityError as an internal 500.
        row=Interaction(**data); db.add(row)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing=find_event(db,event.event_id)
            if existing is not None:
                return existing,False
            raise
    if content is not None and content.content_type == "VIDEO":
        # Real-engagement popularity ownership (Task: popularityScore audit): recomputed
        # from the full current real-interaction aggregate for this content_id after every
        # new VIDEO interaction -- never for LIVE (its 17-feature contract has no
        # popularity signal at all; see app.ml.feature_builder.content_popularity_score's
        # own docstring). A duplicate delivery (the `old` early-return above) never reaches
        # here, matching "recompute only on genuinely new evidence". Skipped when content is
        # None (content_pending=True on this row) -- there is no local Content row to update;
        # the real Content Service owns this content's popularity signal until its projection
        # arrives, at which point the next resolved interaction recomputes normally.
        content.popularity_score = content_popularity_score(interactions_for_content(db, content.content_id))
        db.commit()
    db.refresh(row); return row,True
