"""Shared outbound-HTTP conventions for real-service adapters (User Behavior Service,
Candidate Service): one bounded call, no retry loop, structured/safe logging, and the
project's existing X-Internal-API-Key service-to-service auth convention
(app.core.security.API_KEY_HEADER_NAME, app.core.config.INTERNAL_API_KEY) reused for the
OUTBOUND direction rather than inventing a second auth scheme -- this is unrelated to and
unaffected by the removal of this service's own INBOUND X-Internal-API-Key gate (there is no
longer a require_internal_api_key dependency; see app.core.security's own docstring). A
failed call raises `UpstreamServiceError` exactly once -- the caller
(app.services.providers.*) decides its own fallback policy; this module never retries or
swallows failures silently.
"""
from __future__ import annotations

from typing import Any

import httpx

from app.core.config import INTERNAL_API_KEY
from app.core.logging import logger
from app.core.security import API_KEY_HEADER_NAME


class UpstreamServiceError(Exception):
    """A real-service call failed. `service` is a short label ("UBS"/"CandidateService");
    `reason` is one of NOT_CONFIGURED/TIMEOUT/UNREACHABLE/HTTP_ERROR/MALFORMED_RESPONSE --
    safe, structured, never a raw stack trace, URL, or response body."""

    def __init__(self, service: str, reason: str, message: str) -> None:
        super().__init__(message)
        self.service = service
        self.reason = reason


def call_json(
    *, service: str, base_url: str, path: str, timeout_ms: int,
    method: str = "GET", params: dict[str, Any] | None = None, json_body: dict[str, Any] | None = None,
) -> Any:
    """Performs one outbound call and returns the parsed JSON body. Never logs request/response
    bodies (may carry user behavior data) -- only the outcome (service/path/status/error type),
    matching this project's existing "no embeddings/profiles/tokens in logs" convention."""
    headers = {API_KEY_HEADER_NAME: INTERNAL_API_KEY} if INTERNAL_API_KEY else {}
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = httpx.request(
            method, url, params=params, json=json_body, headers=headers, timeout=timeout_ms / 1000,
        )
    except httpx.TimeoutException as exc:
        logger.info("upstream call timed out service=%s path=%s timeoutMs=%d", service, path, timeout_ms)
        raise UpstreamServiceError(service, "TIMEOUT", f"{service} request timed out") from exc
    except httpx.HTTPError as exc:
        logger.info("upstream call failed service=%s path=%s error=%s", service, path, type(exc).__name__)
        raise UpstreamServiceError(service, "UNREACHABLE", f"{service} is unreachable") from exc
    if response.status_code >= 400:
        logger.info("upstream call returned error service=%s path=%s status=%d", service, path, response.status_code)
        raise UpstreamServiceError(service, "HTTP_ERROR", f"{service} returned HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        logger.info("upstream call returned malformed body service=%s path=%s", service, path)
        raise UpstreamServiceError(service, "MALFORMED_RESPONSE", f"{service} returned a non-JSON body") from exc
