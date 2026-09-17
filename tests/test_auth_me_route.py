"""HTTP-level contract tests for GET /auth/me (Task: first isolated JWT/JWKS authentication
vertical slice). Complements tests/test_jwt_auth.py's strict, unmocked cryptographic
verification tests with: the real route wiring end to end, the safe-error/no-detail-leak
contract, confirmation that no OTHER route accidentally gained this dependency, and that
OpenAPI scopes the Bearer security scheme to this one route only.
"""
from __future__ import annotations

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt import PyJWKClient
from jwt.algorithms import RSAAlgorithm

from app.core import jwt_auth

AUTH_ME_URL = "/api/v1/recommendation-ml-service/auth/me"
ISSUER = "https://keycloak.test/realms/test-realm"
JWKS_URI = "https://keycloak.test/realms/test-realm/protocol/openid-connect/certs"
KID = "test-key-1"


def _public_jwk_dict(public_key) -> dict:
    result = RSAAlgorithm(RSAAlgorithm.SHA256).to_jwk(public_key)
    return json.loads(result) if isinstance(result, str) else dict(result)


_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_JWK = _public_jwk_dict(_PRIVATE_KEY.public_key())
_PUBLIC_JWK.update(kid=KID, use="sig", alg="RS256")
JWKS_DOCUMENT = {"keys": [_PUBLIC_JWK]}


def _token(*, sub="3fa85f64-5717-4562-b3fc-2c963f66afa6", exp_delta=3600):
    now = int(time.time())
    claims = {"iss": ISSUER, "sub": sub, "iat": now, "exp": now + exp_delta}
    return jwt.encode(claims, _PRIVATE_KEY, algorithm="RS256", headers={"kid": KID})


@pytest.fixture(autouse=True)
def _reset_jwks_client_cache():
    jwt_auth._jwks_clients.clear()
    yield
    jwt_auth._jwks_clients.clear()


@pytest.fixture(autouse=True)
def _configure_jwt(monkeypatch):
    monkeypatch.setattr(jwt_auth, "JWT_ALLOWED_ISSUERS", [ISSUER])
    monkeypatch.setattr(jwt_auth, "JWT_ISSUER_JWKS_URIS", {ISSUER: JWKS_URI})
    monkeypatch.setattr(jwt_auth, "JWT_AUDIENCE", None)
    monkeypatch.setattr(jwt_auth, "JWT_CLOCK_SKEW_SECONDS", 10)


def _patch_jwks_fetch(monkeypatch):
    monkeypatch.setattr(PyJWKClient, "fetch_data", lambda self: JWKS_DOCUMENT)


# --------------------------------------------------------------------- contract failures


def test_auth_me_without_authorization_header_is_401(client):
    response = client.get(AUTH_ME_URL)
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert response.json()["error"] == "UNAUTHORIZED"


def test_auth_me_wrong_auth_scheme_is_401(client):
    response = client.get(AUTH_ME_URL, headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_me_malformed_jwt_is_401(client, monkeypatch):
    _patch_jwks_fetch(monkeypatch)
    response = client.get(AUTH_ME_URL, headers={"Authorization": "Bearer not-a-jwt-at-all"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


# --------------------------------------------------------------------- WWW-Authenticate consistency
#
# Regression coverage: app/main.py's shared HTTPException handler used to build its
# JSONResponse headers from scratch (only the request-id header), silently discarding any
# header the raise site itself set on the exception (exc.headers) -- exposed by
# app.core.jwt_auth being the first code in this repository to ever set
# HTTPException(..., headers=...). Fixed generically in the shared handler, not per-route;
# these tests exercise several DIFFERENT underlying jwt_auth.py raise sites (not just the
# one originally reported) to confirm the fix is not accidentally tied to one specific path.


def test_auth_me_expired_token_preserves_www_authenticate_header(client, monkeypatch):
    _patch_jwks_fetch(monkeypatch)
    token = _token(exp_delta=-3600)
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_me_unknown_kid_preserves_www_authenticate_header(client, monkeypatch):
    _patch_jwks_fetch(monkeypatch)
    now = int(time.time())
    token = jwt.encode({"iss": ISSUER, "sub": "user-1", "iat": now, "exp": now + 3600}, _PRIVATE_KEY,
                        algorithm="RS256", headers={"kid": "never-issued-kid"})
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_me_issuer_not_allowed_preserves_www_authenticate_header(client, monkeypatch):
    _patch_jwks_fetch(monkeypatch)
    now = int(time.time())
    token = jwt.encode({"iss": "https://not-allowed.example.com", "sub": "user-1", "iat": now, "exp": now + 3600},
                        _PRIVATE_KEY, algorithm="RS256", headers={"kid": KID})
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_me_invalid_signature_preserves_www_authenticate_header(client, monkeypatch):
    _patch_jwks_fetch(monkeypatch)
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = int(time.time())
    # Same kid as the real JWKS entry, but signed with a DIFFERENT private key.
    token = jwt.encode({"iss": ISSUER, "sub": "user-1", "iat": now, "exp": now + 3600}, other_key,
                        algorithm="RS256", headers={"kid": KID})
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_me_jwks_failure_preserves_www_authenticate_header(client, monkeypatch):
    monkeypatch.setattr(PyJWKClient, "fetch_data", lambda self: (_ for _ in ()).throw(ConnectionError("no route")))
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {_token()}"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_me_wrong_audience_preserves_www_authenticate_header(client, monkeypatch):
    monkeypatch.setattr(jwt_auth, "JWT_AUDIENCE", "recommendation-ml-service")
    _patch_jwks_fetch(monkeypatch)
    now = int(time.time())
    token = jwt.encode({"iss": ISSUER, "sub": "user-1", "aud": "some-other-service", "iat": now, "exp": now + 3600},
                        _PRIVATE_KEY, algorithm="RS256", headers={"kid": KID})
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_me_missing_sub_preserves_www_authenticate_header(client, monkeypatch):
    _patch_jwks_fetch(monkeypatch)
    now = int(time.time())
    token = jwt.encode({"iss": ISSUER, "iat": now, "exp": now + 3600}, _PRIVATE_KEY,
                        algorithm="RS256", headers={"kid": KID})
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_me_fails_closed_when_no_issuer_is_configured(client, monkeypatch):
    """Missing required auth configuration must fail closed, never silently accept."""
    monkeypatch.setattr(jwt_auth, "JWT_ALLOWED_ISSUERS", [])
    monkeypatch.setattr(jwt_auth, "JWT_ISSUER_JWKS_URIS", {})
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {_token()}"})
    assert response.status_code == 401


# --------------------------------------------------------------------- happy path + response shape


def test_auth_me_valid_token_returns_authenticated_and_exact_sub(client, monkeypatch):
    _patch_jwks_fetch(monkeypatch)
    token = _token(sub="3fa85f64-5717-4562-b3fc-2c963f66afa6")
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    body = response.json()
    assert body == {"authenticated": True, "userId": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}


def test_auth_me_response_never_contains_the_raw_token(client, monkeypatch):
    _patch_jwks_fetch(monkeypatch)
    token = _token()
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {token}"})
    assert token not in response.text

    unauthorized = client.get(AUTH_ME_URL, headers={"Authorization": "Bearer garbage-token-value"})
    assert "garbage-token-value" not in unauthorized.text


def test_auth_me_response_has_no_extra_fields(client, monkeypatch):
    """Intentionally minimal -- no raw token, full claims, email, roles, or JWKS details."""
    _patch_jwks_fetch(monkeypatch)
    response = client.get(AUTH_ME_URL, headers={"Authorization": f"Bearer {_token()}"})
    assert set(response.json().keys()) == {"authenticated", "userId"}


# --------------------------------------------------------------------- blast-radius / isolation


def test_health_remains_public(client):
    response = client.get("/api/v1/recommendation-ml-service/health")
    assert response.status_code in (200, 503)


def test_existing_recommendation_endpoints_do_not_require_jwt(client):
    """CRITICAL per this task: no user-facing route gained the JWT dependency yet."""
    response = client.post("/api/v1/recommendation-ml-service/recommendations", json={"userId": "u", "limit": 5, "candidates": []})
    assert response.status_code != 401

    response = client.post("/api/v1/recommendation-ml-service/recommendations/live", json={"userId": "u", "limit": 5})
    assert response.status_code != 401


def test_existing_routes_have_not_accidentally_gained_the_auth_dependency():
    """Inspects the actual FastAPI dependency graph (not just observed status codes) for
    every OTHER router -- proves get_verified_user_id is wired onto GET /auth/me alone."""
    from app.api import (
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
    from app.core.jwt_auth import get_verified_user_id

    for module in (
        candidate_routes, content_routes, drift_routes, event_routes, experiment_routes,
        health_routes, recommendation_routes, training_routes, user_routes,
    ):
        for route in module.router.routes:
            dependant_calls = [dep.call for dep in route.dependant.dependencies]
            assert get_verified_user_id not in dependant_calls, f"{module.__name__}:{route.path} unexpectedly requires JWT auth"


def test_openapi_exposes_bearer_auth_for_auth_me_only(client):
    spec = client.get("/openapi.json").json()
    assert spec["paths"][AUTH_ME_URL]["get"].get("security"), "GET /auth/me must declare a security requirement"
    assert "BearerAuth" in spec.get("components", {}).get("securitySchemes", {})
    assert spec["components"]["securitySchemes"]["BearerAuth"]["type"] == "http"
    assert spec["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"

    other_paths = (
        ("/api/v1/recommendation-ml-service/recommendations", "post"),
        ("/api/v1/recommendation-ml-service/recommendations/live", "post"),
        ("/api/v1/recommendation-ml-service/health", "get"),
        ("/api/v1/recommendation-ml-service/events", "post"),
        ("/api/v1/recommendation-ml-service/model/train", "post"),
    )
    for path, method in other_paths:
        assert not spec["paths"][path][method].get("security"), f"{method.upper()} {path} must not require auth in this task"
