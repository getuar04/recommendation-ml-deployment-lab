"""Persisted onboarding-context read/write, backing `app.db.models.UserOnboardingContext`.

Write side: `persist_user_context` is called exactly once, from `app.api.user_routes.
create_user`, when `POST /users` supplies a `userContext`. It only stages the row
(`db.add`) -- the caller commits it in the SAME transaction as the `User` row insert, so a
user is never created with a partially-written context (or vice versa).

Read side: `load_user_context` is called from `app.services.recommendation_service.recommend`
as the fallback when a recommendation request omits its own `userContext` -- see that
function's `effective_user_context` for the exact precedence (explicit request value always
wins; this is only consulted when the request has none). Mirrors
`app.services.session_intent_provider.DatabaseSessionIntentProvider.peek`'s own read-path
safety posture exactly: any failure (including `db` legitimately being `None` for an offline
`recommend(None, request)` caller) degrades to "no persisted context" rather than breaking a
recommendation response, and rolls back a real `db` session first so a failed read here can't
poison a later statement in the same request.
"""
from __future__ import annotations

import json

from app.core.logging import logger
from app.db.models import UserOnboardingContext
from app.schemas.recommendation_schemas import UserContext


def persist_user_context(db, user_id: str, context: UserContext) -> None:
    """Stages a new onboarding-context row for `user_id`. `context` is already validated/
    normalized by `UserContext` itself (region upper-cased, language lower-cased, interests
    token-normalized) -- stored as-is, no second normalization pass."""
    db.add(UserOnboardingContext(
        user_id=user_id,
        age=context.age,
        region=context.region,
        language=context.language,
        interests_json=json.dumps(context.interests) if context.interests else None,
    ))


def load_user_context(db, user_id: str) -> UserContext | None:
    """Returns this user's persisted onboarding context as a `UserContext`, or `None` if the
    user was never created with one (or predates this feature, or doesn't exist at all)."""
    try:
        row = db.get(UserOnboardingContext, user_id)
    except Exception as exc:  # noqa: BLE001 -- read path must never break a recommendation response
        if db is not None:
            db.rollback()
        logger.info("user context read failed userId=%s failureType=%s", user_id, type(exc).__name__)
        return None
    if row is None:
        return None
    return UserContext(age=row.age, region=row.region, language=row.language, interests=row.interests)
