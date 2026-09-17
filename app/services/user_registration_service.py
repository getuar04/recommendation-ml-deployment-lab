"""Applies one validated `user.registered` event (app.schemas.user_events.UserRegisteredEvent)
to RMS's existing local user + onboarding-context projection (app.db.models.User,
UserOnboardingContext) -- no parallel/second user-profile model is introduced.

Minimum accepted contract (Task: user.registered contract audit): the only fields required to
accept a registration are the event envelope/identity fields already enforced by
app.schemas.user_events.UserRegisteredEvent (type, eventId, target.id == data.userId,
non-blank userId, a timezone-aware occurredAt). Every recommendation-enrichment field below
-- birthday, region, interests -- is OPTIONAL: missing, null, or (for interests) an empty
list must never by themselves cause the registration to be rejected, and a malformed
`birthday` degrades to "no age" rather than failing the whole event (see
app.schemas.user_events._UserRegisteredData._tolerate_malformed_birthday).

Data minimization: User Service's event carries several PII fields (name, nickName, email,
phoneNumber, gender) that the existing onboarding/cold-start path (app.services.
user_context_provider, app.services.recommendation_service.recommend's
`effective_user_context` resolution, app.services.cohort_preference_provider) has no use for
today -- UserOnboardingContext has no columns for any of them, and none is read anywhere in
this recommendation path. They are validated as present-but-unused by
app.schemas.user_events._UserRegisteredData (extra="ignore") and never reach this function at
all, let alone get persisted. Only `birthday` (converted to the existing `age` field the
cold-start cohort system already uses, app.core.cohort_context.age_bucket_for), `region`
(see _safe_region's own docstring), and `interests` (mapped 1:1 onto UserOnboardingContext.
interests_json, the same column POST /users's own optional `userContext.interests` already
writes) are persisted.

`language` is never populated from this event (User Service's `user.registered` payload has
no language field at all) -- left untouched on both create and update.

Idempotency / redelivery / update semantics: `user_id`/`UserOnboardingContext.user_id` are
the same identifiers app.services.event_service._ensure_user / app.api.user_routes.
create_user already key on -- reused directly rather than a new identity concept. A duplicate
delivery of the same event (or a legitimate concurrent create from a different ingestion
path, e.g. an interaction event's own _ensure_user racing on the same brand-new userId) is
idempotent by construction: create-or-update, never a unique-constraint crash, never a
duplicate row.

Re-audit finding: an update MUST NOT erase previously-persisted enrichment merely because a
LATER event (a redelivery, or a genuinely different subsequent user.registered-shaped event
for the same user) happens to omit that one optional field -- age/region/interests are each
only overwritten when the INCOMING event actually supplies a non-empty value for that field;
an omitted/empty field on the incoming event always preserves whatever is already stored
(see _apply_enrichment, used on both the plain-update and the lost-race-recovery paths). This
does not compare occurredAt against any previously-stored value to reject a stale/out-of-order
redelivery -- the most recently PROCESSED delivery's SUPPLIED fields always win (last-write-
wins per field, the same policy UserDemographicContext already documents for its own single-
row-per-user semantics). For `user.registered` specifically (expected to be emitted once per
user) this is an accepted simplification, not a general solution for a hypothetical future
user.updated event stream with its own, possibly conflicting, update ordering.
"""
from __future__ import annotations

import json
from datetime import date, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.cohort_context import normalize_region
from app.db.models import UserOnboardingContext
from app.schemas.user_events import UserRegisteredEvent
from app.services.event_service import _ensure_user

MIN_ONBOARDING_AGE = 0
MAX_ONBOARDING_AGE = 120
# Matches UserOnboardingContext.region's own real column bound (app.db.models, String(8)) --
# the same short-code contract app.schemas.recommendation_schemas.UserContext.region already
# enforces for every other caller (e.g. POST /users's own userContext.region).
_REGION_MAX_LENGTH = 8


def _age_from_birthday(birthday: date, *, as_of: datetime) -> int | None:
    """Derives the same `age` shape app.schemas.recommendation_schemas.UserContext.age already
    validates (0-120) from a birthdate, evaluated as of the event's own occurredAt (the
    platform's record of when registration happened) rather than wall-clock "now" at
    processing time. Returns None (never raises, never fabricates) for a birthday in the
    future relative to occurredAt, or a resulting age outside the existing bound -- exactly
    like app.core.cohort_context.age_bucket_for's own "no bucket" degradation, this is treated
    as "no age context available", not a reason to reject the whole registration."""
    as_of_date = as_of.date()
    if birthday > as_of_date:
        return None
    age = as_of_date.year - birthday.year - ((as_of_date.month, as_of_date.day) < (birthday.month, birthday.day))
    if not (MIN_ONBOARDING_AGE <= age <= MAX_ONBOARDING_AGE):
        return None
    return age


def _safe_region(raw_region: str | None) -> str | None:
    """Re-audit finding (Task: user.registered contract audit): persists `region` through the
    exact same normalize_region() every other region-accepting caller in this codebase
    already uses (app.core.cohort_context -- also what POST /users's own userContext.region
    goes through), rather than unconditionally discarding it. Only persisted when the
    NORMALIZED value actually fits UserOnboardingContext.region's real column bound
    (_REGION_MAX_LENGTH); a longer value -- e.g. a genuine free-form address rather than a
    short region/country code, the exact incompatibility this module previously flagged as
    unconfirmed with User Service -- is safely treated as absent (None) rather than
    truncated or allowed to fail at the database level. Never guesses/derives a short code
    from a longer value -- an incompatible region is dropped, never invented."""
    normalized = normalize_region(raw_region)
    if normalized is None or len(normalized) > _REGION_MAX_LENGTH:
        return None
    return normalized


def _apply_enrichment(
    context: UserOnboardingContext, *, age: int | None, region: str | None, interests_json: str | None,
) -> None:
    """Applies this event's SUPPLIED enrichment fields onto `context`, preserving whatever is
    already stored for any field this event did not supply (see this module's own docstring,
    "Re-audit finding"). Never erases existing age/region/interests merely because a later
    event omits that one field."""
    if age is not None:
        context.age = age
    if region is not None:
        context.region = region
    if interests_json is not None:
        context.interests_json = interests_json


def apply_user_registered(db: Session, event: UserRegisteredEvent) -> None:
    """Idempotent create-or-update of the local User + UserOnboardingContext rows for one
    user.registered event. Never raises for "the user/context already exists" -- only a
    genuine, unexpected persistence failure propagates (see app.services.kafka_user_consumer,
    which relies on that to avoid committing a Kafka offset for a failed write)."""
    user_id = event.data.user_id
    age = _age_from_birthday(event.data.birthday, as_of=event.data.occurred_at) if event.data.birthday else None
    region = _safe_region(event.data.region)
    interests_json = json.dumps(event.data.interests) if event.data.interests else None

    _ensure_user(db, user_id)
    try:
        db.commit()
    except IntegrityError:
        # Lost a race with a concurrent creator (e.g. an interaction event's own _ensure_user,
        # app.services.event_service, processing on a different thread/consumer for the same
        # brand-new userId) -- the user exists either way; nothing left to do for this step.
        db.rollback()

    existing_context = db.get(UserOnboardingContext, user_id)
    if existing_context is None:
        # A brand-new context: fields this event did not supply simply stay NULL (there is
        # nothing prior to preserve) -- identical end state to _apply_enrichment's own
        # preserve-if-absent rule, just with an empty starting point.
        db.add(UserOnboardingContext(user_id=user_id, age=age, region=region, language=None, interests_json=interests_json))
        try:
            db.commit()
        except IntegrityError:
            # Lost a race with a concurrent context write for the same brand-new userId (e.g.
            # a simultaneous POST /users onboarding call) -- the winner's row already exists;
            # apply this event's SUPPLIED values on top of it instead of crashing.
            db.rollback()
            existing_context = db.get(UserOnboardingContext, user_id)
            if existing_context is not None:
                _apply_enrichment(existing_context, age=age, region=region, interests_json=interests_json)
                db.commit()
    else:
        _apply_enrichment(existing_context, age=age, region=region, interests_json=interests_json)
        db.commit()
