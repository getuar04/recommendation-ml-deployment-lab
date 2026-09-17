from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import (
    auth_routes,
    candidate_routes,
    content_routes,
    drift_routes,
    event_routes,
    experiment_routes,
    health_routes,
    recommendation_routes,
    training_routes,
    user_routes,
)
from app.core import request_context
from app.core.config import API_V1_PREFIX, DATABASE_URL
from app.core.logging import logger
from app.core.request_context import RequestIDMiddleware, generate_request_id
from app.db.database import Base, engine
from app.services import kafka_behavior_consumer, kafka_user_consumer

# Group A: production (PostgreSQL) schema changes go through Alembic migrations
# (`alembic upgrade head`), not automatic mutation at startup. create_all() is kept only
# for the SQLite-isolated test/dev path, where tests already rely on it (see
# tests/conftest.py's clean_db fixture) and there is no persistent schema to protect.
if DATABASE_URL.startswith("sqlite"):
    Base.metadata.create_all(engine)

# No-op unless KAFKA_BEHAVIOR_ENABLED=true (default false) -- see
# app.services.kafka_behavior_consumer. Any connection failure is caught inside that module
# and only updates its own STATUS; it never raises here, so local/demo/test startup and every
# existing test are completely unaffected.
kafka_behavior_consumer.start_consumer_in_background()
# Same no-op-unless-enabled posture, independent thread/topic/consumer group (Task: continuous
# user-projection ingestion) -- see app.services.kafka_user_consumer.
kafka_user_consumer.start_consumer_in_background()

app=FastAPI(title="Recommendation ML Service",version="1.0.0")
app.add_middleware(RequestIDMiddleware)
for r in (event_routes,training_routes,recommendation_routes,user_routes,health_routes,candidate_routes,content_routes,experiment_routes,drift_routes,auth_routes): app.include_router(r.router,prefix=API_V1_PREFIX)


def _request_id(request: Request) -> str:
    # The middleware always sets this; the getattr fallback only matters for a response
    # that somehow never went through it (e.g. a handler invoked directly in a unit test).
    return getattr(request.state, "request_id", None) or generate_request_id()


def _error_body(code: str, message: str, request_id: str, *, details: dict | None = None, extra: dict | None = None) -> dict:
    """The one central place every error response shape is assembled. `error`/`message`
    are the legacy flat fields every route has always returned (kept byte-for-byte
    unchanged for backward compatibility); `errorDetails` is the additive, canonical
    nested form -- `errorDetails.code`/`errorDetails.message` always mirror the legacy
    fields exactly, and `errorDetails.details` is always an object (never omitted, never
    null), even when there is nothing to report. `extra` carries any additional safe,
    route-provided top-level fields (e.g. the existing `"reason"` on artifact-invalid
    responses) -- preserved at the top level exactly as before, and also folded into
    `errorDetails.details` so the structured form carries the same information."""
    details = dict(details or {})
    if extra:
        details.update(extra)
    body: dict[str, Any] = {"error": code, "message": message}
    if extra:
        body.update(extra)
    body["errorDetails"] = {"code": code, "message": message, "details": details}
    body["requestId"] = request_id
    return body


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    # Central mapping, single place: every HTTPException raised anywhere in the app already
    # carries a structured {"error": "CODE", "message": "..."} detail (the convention every
    # route already follows); this handler's only job is to guarantee status/shape
    # consistency and attach the request ID -- it does not invent new error codes.
    request_id = _request_id(request)
    extra: dict[str, Any]
    if isinstance(exc.detail, dict):
        code = str(exc.detail.get("error", "HTTP_ERROR"))
        message = str(exc.detail.get("message", ""))
        extra = {k: v for k, v in exc.detail.items() if k not in ("error", "message")}
    else:
        code, message, extra = "HTTP_ERROR", str(exc.detail), {}
    content = _error_body(code, message, request_id, extra=extra)
    # Preserve any safe header the raise site explicitly set on the exception itself (e.g.
    # WWW-Authenticate: Bearer from app.core.jwt_auth's 401s) -- this used to be silently
    # discarded here, since this handler always built a brand-new headers dict containing
    # only the request-id header. Generic, not auth-specific: any current or future
    # HTTPException(..., headers=...) call site is covered. The request-id header is set
    # last so it always wins over anything (unexpectedly) duplicated in exc.headers.
    headers = dict(exc.headers or {})
    headers[request_context.REQUEST_ID_HEADER] = request_id
    return JSONResponse(status_code=exc.status_code, content=content, headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Safe field/location info only -- never the raw `input` value from a Pydantic error
    # dict (it can echo back arbitrary client-supplied data) and never a Python repr.
    fields = [
        {"location": [str(part) for part in error.get("loc", ())], "message": error.get("msg", ""), "type": error.get("type", "")}
        for error in exc.errors()
    ]
    request_id = _request_id(request)
    message = "Request validation failed."
    content = _error_body("VALIDATION_ERROR", message, request_id, details={"fields": fields})
    content["details"] = {"fields": fields}  # legacy top-level placement, unchanged from Session 3
    return JSONResponse(status_code=422, content=content, headers={request_context.REQUEST_ID_HEADER: request_id})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    request_id = _request_id(request)
    # Full exception detail (including traceback) stays server-side, tied to the request ID
    # so it can be correlated with the client-visible error -- the response body itself never
    # gets str(exc), a traceback, a filesystem path, or a database URL.
    logger.exception("unhandled exception requestId=%s path=%s", request_id, request.url.path)
    content = _error_body("INTERNAL_SERVER_ERROR", "An unexpected error occurred.", request_id)
    return JSONResponse(status_code=500, content=content, headers={request_context.REQUEST_ID_HEADER: request_id})
