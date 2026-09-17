"""Strict, lower-level cryptographic verification coverage for app.core.jwt_auth
(Task: first isolated JWT/JWKS authentication vertical slice). Real generated RSA keys, real
PyJWT signing/verification, real PyJWKClient JWK-set parsing/kid-matching/caching -- only the
JWKS HTTP fetch itself (PyJWKClient.fetch_data) is faked, so this never touches the real
internet/Keycloak while still exercising genuine cryptographic verification end to end. No
dependency overrides here (see tests/test_auth_me_route.py for the simpler HTTP-contract-
level tests, which may use them) -- this file must never be weakened to make a test pass.
"""
from __future__ import annotations

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from jwt import PyJWKClient
from jwt.algorithms import RSAAlgorithm

from app.core import jwt_auth

ISSUER = "https://keycloak.test/realms/test-realm"
JWKS_URI = "https://keycloak.test/realms/test-realm/protocol/openid-connect/certs"
KID = "test-key-1"
OTHER_KID = "test-key-2"


def _public_jwk_dict(public_key) -> dict:
    # PyJWT's RSAAlgorithm.to_jwk returns a dict on newer releases and a JSON string on
    # older ones (as_dict was added later) -- handle either without pinning to one exact
    # PyJWT version's return type.
    result = RSAAlgorithm(RSAAlgorithm.SHA256).to_jwk(public_key)
    return json.loads(result) if isinstance(result, str) else dict(result)


_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_JWK = _public_jwk_dict(_PRIVATE_KEY.public_key())
_PUBLIC_JWK.update(kid=KID, use="sig", alg="RS256")
JWKS_DOCUMENT = {"keys": [_PUBLIC_JWK]}

_OTHER_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_OTHER_PUBLIC_JWK = _public_jwk_dict(_OTHER_PRIVATE_KEY.public_key())
_OTHER_PUBLIC_JWK.update(kid=OTHER_KID, use="sig", alg="RS256")
JWKS_DOCUMENT_WITH_ROTATED_KEY = {"keys": [_PUBLIC_JWK, _OTHER_PUBLIC_JWK]}
JWKS_DOCUMENT_MISSING_KID = {"keys": [_OTHER_PUBLIC_JWK]}


def _token(*, kid=KID, alg="RS256", issuer=ISSUER, sub="user-123", exp_delta=3600,
           aud=None, signing_key=None, no_kid=False, no_sub=False, blank_sub=False):
    now = int(time.time())
    claims = {"iss": issuer, "iat": now, "exp": now + exp_delta}
    if not no_sub:
        claims["sub"] = "" if blank_sub else sub
    if aud is not None:
        claims["aud"] = aud
    headers = {} if no_kid else {"kid": kid}
    key = signing_key if signing_key is not None else _PRIVATE_KEY
    return jwt.encode(claims, key, algorithm=alg, headers=headers)


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


def _patch_jwks_fetch(monkeypatch, responses):
    """`responses`: a list of JWKS dicts/Exceptions returned on successive fetch_data()
    calls (the last entry repeats once exhausted). Returns a mutable call counter."""
    calls = {"n": 0}

    def fake_fetch_data(self):
        idx = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        response = responses[idx]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(PyJWKClient, "fetch_data", fake_fetch_data)
    return calls


# --------------------------------------------------------------------- valid token


def test_valid_rs256_token_returns_exact_verified_sub(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(sub="3fa85f64-5717-4562-b3fc-2c963f66afa6")
    assert jwt_auth.verify_bearer_token(token) == "3fa85f64-5717-4562-b3fc-2c963f66afa6"


# --------------------------------------------------------------------- structural/algorithm rejections


def test_malformed_jwt_is_rejected(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token("not-a-jwt-at-all")
    assert exc_info.value.status_code == 401


def test_token_with_valid_header_but_corrupt_payload_is_rejected(monkeypatch):
    """Distinct from test_malformed_jwt_is_rejected: that token fails to parse at the very
    first step (jwt.get_unverified_header, which only touches the header segment). This one
    has a structurally valid, parseable header (so it passes that first check) but a
    corrupted payload segment, which only fails at the second decode call (the unverified
    `iss` peek, which parses header AND payload) -- covers jwt_auth.py's own second
    malformed-token except branch, not merely re-exercising the first one."""
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    valid_token = _token()
    header_segment, _payload_segment, signature_segment = valid_token.split(".")
    corrupt_token = f"{header_segment}.not-valid-base64-json!!!.{signature_segment}"
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(corrupt_token)
    assert exc_info.value.status_code == 401


def test_non_rs256_algorithm_is_rejected(monkeypatch):
    """HS256, signed with a plain string secret -- structurally valid, but the wrong
    algorithm family entirely; must be rejected purely from the header, before any JWKS
    lookup is even attempted."""
    fetch_calls = _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = jwt.encode({"iss": ISSUER, "sub": "user-1", "exp": int(time.time()) + 3600}, "some-secret",
                        algorithm="HS256", headers={"kid": KID})
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401
    assert fetch_calls["n"] == 0  # rejected before any JWKS network call


def test_missing_kid_is_rejected(monkeypatch):
    fetch_calls = _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(no_kid=True)
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401
    assert fetch_calls["n"] == 0  # rejected before any JWKS network call


# --------------------------------------------------------------------- issuer allow-list


def test_issuer_not_in_allow_list_is_rejected(monkeypatch):
    fetch_calls = _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(issuer="https://not-allowed.example.com")
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401
    assert fetch_calls["n"] == 0  # never reaches JWKS for a disallowed issuer


def test_allow_listed_issuer_with_no_configured_jwks_uri_fails_closed(monkeypatch):
    monkeypatch.setattr(jwt_auth, "JWT_ALLOWED_ISSUERS", [ISSUER])
    monkeypatch.setattr(jwt_auth, "JWT_ISSUER_JWKS_URIS", {})  # configuration gap
    token = _token()
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401


# --------------------------------------------------------------------- JWKS / kid resolution


def test_unknown_kid_is_rejected(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(kid="never-issued-kid")
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401


def test_unknown_kid_triggers_a_bounded_jwks_refresh_for_key_rotation(monkeypatch):
    """The first fetch returns a JWKS that does not yet contain the signing key (simulating
    this process's cache predating a rotation on the IdP side); PyJWKClient's own documented
    refresh-on-unknown-kid behavior must fetch again and succeed once the second response
    contains it -- never a hand-rolled refresh loop here."""
    fetch_calls = _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT_MISSING_KID, JWKS_DOCUMENT_WITH_ROTATED_KEY])
    token = _token(kid=KID, sub="rotated-user")
    assert jwt_auth.verify_bearer_token(token) == "rotated-user"
    # At least one refresh beyond the initial (stale-cache-missing-the-kid) fetch, and
    # bounded -- not an unbounded/repeated retry loop for a single verification call.
    assert 1 < fetch_calls["n"] <= 3


def test_jwks_cache_is_reused_across_calls_without_a_second_network_call(monkeypatch):
    fetch_calls = _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    jwt_auth.verify_bearer_token(_token(sub="user-a"))
    jwt_auth.verify_bearer_token(_token(sub="user-b"))
    assert fetch_calls["n"] == 1  # second call reused the cached JWKS, no new fetch


def test_jwks_network_failure_is_a_safe_401_not_a_500(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [ConnectionError("no route to host")])
    token = _token()
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401
    # Never a raw exception message/internal detail in the client-facing body.
    assert "no route to host" not in str(exc_info.value.detail)


# --------------------------------------------------------------------- signature / expiration


def test_invalid_signature_is_rejected(monkeypatch):
    """Same kid as the real key (so JWKS resolution succeeds) but signed with a DIFFERENT
    private key -- the cryptographic signature check itself must fail."""
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(signing_key=_OTHER_PRIVATE_KEY)  # kid=KID, but wrong key material
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401


def test_expired_token_is_rejected(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(exp_delta=-3600)
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401


def test_token_within_configured_clock_skew_is_accepted(monkeypatch):
    """Expired 5 seconds ago, well inside the default 10s leeway -- must still verify."""
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(exp_delta=-5)
    assert jwt_auth.verify_bearer_token(token) == "user-123"


# --------------------------------------------------------------------- sub


def test_missing_sub_is_rejected(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(no_sub=True)
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401


def test_blank_sub_is_rejected(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(blank_sub=True)
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401


# --------------------------------------------------------------------- audience (conditional/configurable)


def test_audience_not_configured_skips_audience_validation(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(aud="anything-or-nothing")
    assert jwt_auth.verify_bearer_token(token) == "user-123"


def test_configured_audience_is_accepted_when_present(monkeypatch):
    monkeypatch.setattr(jwt_auth, "JWT_AUDIENCE", "recommendation-ml-service")
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(aud="recommendation-ml-service")
    assert jwt_auth.verify_bearer_token(token) == "user-123"


def test_wrong_configured_audience_is_rejected(monkeypatch):
    monkeypatch.setattr(jwt_auth, "JWT_AUDIENCE", "recommendation-ml-service")
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(aud="some-other-service")
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert exc_info.value.status_code == 401


# --------------------------------------------------------------------- safe error contract (unit level)


def test_unauthorized_response_never_contains_the_raw_token(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    token = _token(exp_delta=-3600)
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token(token)
    assert token not in str(exc_info.value.detail)


def test_unauthorized_response_has_www_authenticate_bearer_header(monkeypatch):
    _patch_jwks_fetch(monkeypatch, [JWKS_DOCUMENT])
    with pytest.raises(HTTPException) as exc_info:
        jwt_auth.verify_bearer_token("not-a-jwt-at-all")
    assert exc_info.value.headers.get("WWW-Authenticate") == "Bearer"
