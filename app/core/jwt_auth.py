"""Bearer JWT verification for GET /auth/me (Task: first isolated JWT/JWKS authentication
vertical slice). Platform contract confirmed via Follow Service's own real, independently-
audited Keycloak integration: RS256 only, JWT header carries `kid`, the signing key is
resolved through the issuer's JWKS, `exp` is validated with a small configurable clock-skew
allowance (default 10s, JWT_CLOCK_SKEW_SECONDS), audience validation is conditional/
configurable (JWT_AUDIENCE), and authenticated identity is EXACTLY the verified `sub` claim
-- never email/preferred_username/nickname/name/request-supplied userId, and never an
unverified-but-decoded claim used for anything but selecting an already-allow-listed issuer
(see verify_bearer_token's own comment on that one deliberate exception).

Scope (deliberate -- see the task this module was added for): ONLY GET /auth/me
(app.api.auth_routes) depends on this today. No other route (recommendations, candidates,
events, users, training, drift, health) gained this dependency in this task -- a separate
follow-up task propagates verified `sub` into user-facing routes once /auth/me is confirmed
working against a real platform token in a deployed environment.

Fails closed: an empty/misconfigured JWT_ALLOWED_ISSUERS or JWT_ISSUER_JWKS_URIS means every
request through get_verified_user_id gets 401 -- never an unauthenticated pass-through and
never a startup crash (Keycloak/JWKS reachability is checked per-request, on demand, never
at import/startup time -- see app.api.health_routes, unaffected by this module entirely).

JWKS retrieval/caching is NOT hand-rolled: PyJWT's own `PyJWKClient` owns the HTTP fetch,
timeout, and kid-based key cache (never hand-rolled RSA/JWK parsing here). One PyJWKClient
per configured issuer, built once and reused for the life of the process -- a request for an
already-cached kid never makes a network call; a request for an unseen kid relies on
PyJWKClient's own documented refetch-on-unknown-kid behavior (kid-rotation support) rather
than a second, hand-rolled cache-invalidation layer here.

Never logs a raw token or decoded claim -- every log line below carries only a short, safe,
enum-like failure category (see _unauthorized's own comment).
"""
from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, PyJWKClientError

from app.core.config import (
    JWT_ALLOWED_ISSUERS,
    JWT_AUDIENCE,
    JWT_CLOCK_SKEW_SECONDS,
    JWT_ISSUER_JWKS_URIS,
    JWT_JWKS_CACHE_TTL_SECONDS,
    JWT_JWKS_HTTP_TIMEOUT_SECONDS,
)
from app.core.logging import logger

__all__ = ["get_verified_user_id", "verify_bearer_token"]

ALLOWED_ALGORITHMS = ["RS256"]
_WWW_AUTHENTICATE = {"WWW-Authenticate": "Bearer"}

# auto_error=False: a missing/malformed Authorization header must produce this module's own
# structured 401 body (via get_verified_user_id below), not FastAPI/Starlette's default
# HTTPBearer error shape -- same convention this project's now-removed X-Internal-API-Key
# gate used to follow (app.core.security's own former require_internal_api_key). Also what
# makes FastAPI advertise a `bearerAuth` OpenAPI security scheme on ONLY the route(s) that
# depend on this, never globally.
_bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth")

# One PyJWKClient per configured issuer, built lazily and cached for the life of the process
# -- see this module's own docstring for why the HTTP fetch/cache itself is never hand-rolled.
_jwks_clients: dict[str, PyJWKClient] = {}


def _unauthorized(reason: str) -> HTTPException:
    # `reason` is a short, safe, enum-like category only (e.g. "EXPIRED_TOKEN",
    # "UNKNOWN_KID") -- never a raw token, decoded claim, or a cryptographic library's own
    # exception message. Logged and returned identically for every failure mode: the client
    # never learns WHY beyond "invalid/missing credentials" (STEP 9's no-detail-leak
    # requirement); only this server-side log line's category differs.
    logger.info("auth/me rejected reason=%s", reason)
    return HTTPException(
        401, detail={"error": "UNAUTHORIZED", "message": "A valid Bearer token is required."},
        headers=_WWW_AUTHENTICATE,
    )


def _jwks_client_for_issuer(issuer: str) -> PyJWKClient | None:
    jwks_uri = JWT_ISSUER_JWKS_URIS.get(issuer)
    if not jwks_uri:
        return None
    client = _jwks_clients.get(issuer)
    if client is None:
        client = PyJWKClient(
            jwks_uri, cache_keys=True, lifespan=JWT_JWKS_CACHE_TTL_SECONDS,
            timeout=JWT_JWKS_HTTP_TIMEOUT_SECONDS,
        )
        _jwks_clients[issuer] = client
    return client


def verify_bearer_token(token: str) -> str:
    """Returns the verified, non-empty `sub` claim for `token`, or raises
    HTTPException(401). Every failure mode -- structurally invalid token, wrong/missing
    `alg`, missing/unknown `kid`, disallowed/unconfigured issuer, unreachable JWKS, bad
    signature, expired token, disallowed audience, missing `sub` -- fails closed with the
    same safe, generic 401; only the server-side log category differs."""
    try:
        header = jwt.get_unverified_header(token)
    except Exception:  # noqa: BLE001 -- any structurally-invalid token is 401, never a 500
        raise _unauthorized("MALFORMED_TOKEN")

    if header.get("alg") != "RS256":
        raise _unauthorized("DISALLOWED_ALGORITHM")
    kid = header.get("kid")
    if not kid:
        raise _unauthorized("MISSING_KID")

    # The token's own `iss` claim is read here WITHOUT signature verification -- it is used
    # for exactly one purpose, selecting among the strict, operator-configured
    # JWT_ALLOWED_ISSUERS/JWT_ISSUER_JWKS_URIS entries below, and is NEVER concatenated into
    # a URL or trusted for anything else before the real, signature-verified decode further
    # down. Every other verify_* option is explicitly disabled too (not merely left to
    # PyJWT's own signature-off cascade default) so this call can never be mistaken for a
    # trusted decode.
    try:
        unverified_claims = jwt.decode(
            token, options={"verify_signature": False, "verify_exp": False, "verify_aud": False, "verify_iss": False},
        )
    except Exception:  # noqa: BLE001
        raise _unauthorized("MALFORMED_TOKEN")
    issuer = unverified_claims.get("iss")
    if not issuer or issuer not in JWT_ALLOWED_ISSUERS:
        raise _unauthorized("ISSUER_NOT_ALLOWED")

    jwks_client = _jwks_client_for_issuer(issuer)
    if jwks_client is None:
        # Allow-listed issuer with no configured JWKS URI -- a configuration gap, treated
        # identically to "issuer not allowed": fail closed, never a 500, never a detail leak.
        raise _unauthorized("ISSUER_NOT_ALLOWED")

    try:
        signing_key = jwks_client.get_signing_key(kid)
    except PyJWKClientError:
        # Covers "kid not present in the (possibly just-refreshed) JWKS" -- PyJWKClient's
        # own documented behavior already retries a fresh fetch before giving up here (kid-
        # rotation support), so no second, hand-rolled refresh layer is added in this module.
        raise _unauthorized("UNKNOWN_KID")
    except Exception:  # noqa: BLE001 -- JWKS network/HTTP failure must fail closed, never 500
        raise _unauthorized("JWKS_UNAVAILABLE")

    decode_kwargs: dict = {"algorithms": ALLOWED_ALGORITHMS, "issuer": issuer, "leeway": JWT_CLOCK_SKEW_SECONDS}
    if JWT_AUDIENCE:
        decode_kwargs["audience"] = JWT_AUDIENCE
    else:
        decode_kwargs["options"] = {"verify_aud": False}

    try:
        claims = jwt.decode(token, signing_key.key, **decode_kwargs)
    except jwt.ExpiredSignatureError:
        raise _unauthorized("EXPIRED_TOKEN")
    except jwt.InvalidAudienceError:
        raise _unauthorized("AUDIENCE_NOT_ALLOWED")
    except jwt.InvalidTokenError:
        # The broad PyJWT base class for "signature/issuer/structurally invalid" -- includes
        # a bad signature, a tampered payload, and a wrong (but well-formed) issuer claim
        # slipping past the unverified pre-check under a race with JWT_ALLOWED_ISSUERS being
        # reloaded (not possible today -- config is loaded once at import -- kept as defense
        # in depth regardless).
        raise _unauthorized("INVALID_SIGNATURE")
    except Exception:  # noqa: BLE001 -- any other verification failure is still a clean 401
        raise _unauthorized("VERIFICATION_FAILED")

    sub = claims.get("sub")
    if not sub or not isinstance(sub, str) or not sub.strip():
        raise _unauthorized("MISSING_SUB")
    return sub


def get_verified_user_id(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme)) -> str:
    """FastAPI dependency: the ONLY authenticated-identity source this task introduces.
    Returns exactly the verified JWT `sub` claim -- never a request-supplied userId, never
    email/preferred_username/nickname/name. `credentials` is None for both a missing
    Authorization header and a non-Bearer scheme (Starlette's own HTTPBearer already
    rejects/blanks a mismatched scheme before this function ever runs)."""
    if credentials is None:
        raise _unauthorized("MISSING_OR_INVALID_AUTHORIZATION")
    return verify_bearer_token(credentials.credentials)
