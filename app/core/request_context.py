"""Central request-ID handling: one middleware assigns every request a request ID (accepted
from the configured header if it looks safe, generated otherwise), makes it available to
route handlers/exception handlers via `request.state.request_id`, and echoes it back on the
configured response header for both successful and failed responses.

The ID is never used for authorization -- it is purely a correlation identifier for logs and
client-visible error bodies.
"""
from __future__ import annotations

import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.config import REQUEST_ID_HEADER
from app.core.logging import logger

# Conservative allowlist: ASCII letters/digits plus a few common correlation-id separators.
# Excludes control characters and anything outside a safe, short, log-friendly charset by
# construction (a whitelist regex, not a blacklist of "bad" characters). 128 is a generous
# but bounded maximum -- long enough for a UUID or a typical upstream trace ID, short enough
# that a malicious/broken client can't smuggle an oversized value into logs or headers.
MAX_REQUEST_ID_LENGTH = 128
_VALID_REQUEST_ID = re.compile(rf"^[A-Za-z0-9._-]{{1,{MAX_REQUEST_ID_LENGTH}}}$")


def generate_request_id() -> str:
    return uuid.uuid4().hex


def normalize_request_id(candidate: str | None) -> str:
    """Return `candidate` if it's a safe, reasonably-sized token; otherwise generate one.
    A client-supplied value is never trusted for anything beyond correlation -- an invalid
    one is silently replaced, not rejected with an error (a malformed correlation header
    should never be able to fail an otherwise-valid request)."""
    if candidate and _VALID_REQUEST_ID.match(candidate):
        return candidate
    return generate_request_id()


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = normalize_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id

        started = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request completed method=%s path=%s status=%d latency_ms=%.2f requestId=%s",
            request.method, request.url.path, response.status_code, latency_ms, request_id,
        )
        return response
