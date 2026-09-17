"""User Service Kafka event contract: `user.registered` only (Task: continuous user-projection
ingestion). Envelope shape matches the real, confirmed `ranking-service` JSON convention
already documented in app.core.config (`{type, eventId, target:{type,id}, data:{...}}`,
CloudEvents-shaped) -- reused here rather than inventing a second envelope shape.

Scope: NEW user registrations only, arriving continuously after integration begins. Users
that already existed in User Service before this integration are a SEPARATE, NOT-YET-
implemented historical snapshot/backfill task (see README "Initial bootstrap / backfill") --
this module does not attempt to solve that.

RMS is not the source of truth for User Service's own PII fields -- see
app.services.user_registration_service's own docstring for exactly which `data` fields are
persisted (data-minimization: only what the existing onboarding/cold-start path can actually
use) and which are intentionally never stored (name/nickName/email/phoneNumber/gender -- see
that module for why).
"""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.ml.semantic_tokens import normalize_tokens

EVENT_TYPE = "user.registered"


class _EventTarget(BaseModel):
    # extra="ignore": tolerate additional platform envelope fields RMS doesn't need, exactly
    # like every other inbound schema in this project (EventCreate is the one deliberate
    # extra="forbid" exception, for a different reason -- see its own docstring).
    model_config = ConfigDict(extra="ignore")
    type: str
    id: str = Field(min_length=1)

    @field_validator("type")
    @classmethod
    def _target_type_must_be_user(cls, value: str) -> str:
        if value != "user":
            raise ValueError("target.type must equal 'user'")
        return value


class _UserRegisteredData(BaseModel):
    """Only the fields RMS validates/reads are typed strictly; every other field User Service
    sends (name/nickName/email/phoneNumber/gender/...) is accepted (extra="ignore") but never
    read by this schema or persisted downstream -- see user_registration_service.py."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    user_id: str = Field(alias="userId", min_length=1)
    # birthday/region/interests are the only User Service fields the existing onboarding/
    # cold-start path can use (age bucket, cohort region, onboarding-interest retrieval) --
    # see user_registration_service.py for the exact persistence mapping. `region`'s real
    # format is still unconfirmed with User Service (an early example payload showed a
    # free-form address, not the short normalized code app.core.cohort_context.
    # normalize_region/UserOnboardingContext.region (String(8)) expect) -- validated here as
    # a bounded string only; user_registration_service.py persists it ONLY when the
    # normalized value actually fits the existing region contract, safely dropping (never
    # truncating) anything that doesn't.
    birthday: date | None = None
    region: str | None = Field(None, max_length=256)
    interests: list[str] = Field(default_factory=list)
    occurred_at: datetime = Field(alias="occurredAt")

    @field_validator("interests", mode="before")
    @classmethod
    def _normalize_interests(cls, value: object) -> list[str]:
        # Reuses the exact same normalizer app.schemas.recommendation_schemas.UserContext.
        # interests already applies -- one list/string-contract rule for "interests" repo-wide.
        return normalize_tokens(value, field="interests")

    @field_validator("birthday", mode="before")
    @classmethod
    def _tolerate_malformed_birthday(cls, value: object) -> object:
        # Audit finding (Task: user.registered contract audit): `birthday` is an OPTIONAL
        # recommendation-enrichment field, not identity -- a value User Service cannot parse
        # as a real date must not reject the whole registration (it used to: pydantic's own
        # strict `date` coercion would fail the entire event). Degrades to None (treated
        # identically to "birthday omitted") for anything that isn't already a `date` or a
        # valid ISO-8601 date string; a syntactically valid but implausible date (future,
        # or yielding an out-of-range age) is intentionally NOT handled here -- it still
        # reaches app.services.user_registration_service._age_from_birthday's own separate,
        # existing bounds check unchanged.
        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None

    @field_validator("occurred_at")
    @classmethod
    def _occurred_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("data.occurredAt must include timezone information (e.g. a 'Z' suffix or an explicit +HH:MM/-HH:MM offset)")
        return value


class UserRegisteredEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    event_id: str = Field(alias="eventId", min_length=1)
    target: _EventTarget
    data: _UserRegisteredData

    @field_validator("type")
    @classmethod
    def _type_must_be_user_registered(cls, value: str) -> str:
        if value != EVENT_TYPE:
            raise ValueError(f"type must equal {EVENT_TYPE!r}")
        return value

    @model_validator(mode="after")
    def _target_id_must_match_data_user_id(self) -> UserRegisteredEvent:
        if self.target.id != self.data.user_id:
            raise ValueError("target.id must equal data.userId")
        return self
