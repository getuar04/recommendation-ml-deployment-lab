"""Post-deployment smoke test: read-only / minimally-mutating verification that a freshly
deployed instance is actually serving traffic correctly. Intended to run once, right after
`alembic upgrade head` + app startup, from CI/CD or an operator's shell -- see
docs/DEPLOYMENT_RUNBOOK.md "Smoke verification".

Checks, in order:
  1. GET /health -- process is alive and reachable.
  2. Database connectivity -- read off the same /health response (`dependencies.postgres`),
     not a second connection implementation.
  3. GET /model/status -- READY/MISSING/INCOMPATIBLE/CORRUPTED. MISSING is a normal, expected
     state on a genuinely fresh deployment (no artifact provisioned yet) and is reported as
     such, not treated as a failure -- see "Model artifact provisioning" in the runbook.
  4. If READY: selectedModel/modelFamily/modelVersion match --expected-* when supplied.
  5. If READY: one cold-start POST /recommendations smoke request with inline, caller-supplied
     candidates (no DB content is read or written -- see README "Candidate Service /
     Recommendation Service boundary": /recommendations trusts its candidates array as-is).
     Verifies every returned score is finite and ranks are contiguous (1..N, no gaps/dupes).

Never trains, never seeds data, never writes to the database, never touches a model artifact.
The only network calls are outbound reads (plus one stateless-scoring POST that persists
nothing) against the target service.

Usage:
    python -m scripts.production_smoke_test --base-url http://localhost:3500
    python -m scripts.production_smoke_test --base-url http://localhost:3500 \\
        --expected-model-family classifier --expected-model-version recommendation-prod-20260824135926

INTERNAL_API_KEY (or --api-key) is required only if the target deployment has one configured
(see app/core/security.py) -- GET /health never requires it; GET /model/status and
POST /recommendations do whenever the target's INTERNAL_API_KEY is set.

Exit code 0 on success (including a MISSING model on a fresh deployment), non-zero on any
unexpected failure -- see the printed report for exactly which check failed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any

import httpx

_SMOKE_USER_ID = "smoke-test-cold-start-user"
_SMOKE_CANDIDATES = [
    {
        "contentId": "smoke-test-candidate-1",
        "creatorId": "smoke-test-creator",
        "category": "SPORT",
        "contentPopularityScore": 0.5,
        "contentAgeHours": 1.0,
        "creatorFollowed": False,
        "alreadySeen": False,
    },
    {
        "contentId": "smoke-test-candidate-2",
        "creatorId": "smoke-test-creator",
        "category": "MUSIC",
        "contentPopularityScore": 0.75,
        "contentAgeHours": 5.0,
        "creatorFollowed": False,
        "alreadySeen": False,
    },
    {
        "contentId": "smoke-test-candidate-3",
        "creatorId": "smoke-test-creator",
        "category": "ENTERTAINMENT",
        "contentPopularityScore": 0.25,
        "contentAgeHours": 12.0,
        "creatorFollowed": False,
        "alreadySeen": False,
    },
]


class SmokeTestFailure(Exception):
    """A check failed in a way that should stop the smoke test and exit non-zero."""


def _headers(api_key: str | None) -> dict[str, str]:
    return {"X-Internal-API-Key": api_key} if api_key else {}


def check_health(client: httpx.Client) -> dict[str, Any]:
    response = client.get("/api/v1/recommendation-ml-service/health")
    if response.status_code not in (200, 503):
        raise SmokeTestFailure(f"GET /health returned unexpected status {response.status_code}: {response.text}")
    body = response.json()
    if response.status_code == 503:
        raise SmokeTestFailure(f"GET /health reports degraded: {body}")
    postgres_status = body.get("dependencies", {}).get("postgres")
    if postgres_status != "connected":
        raise SmokeTestFailure(f"GET /health status=200 but dependencies.postgres={postgres_status!r}")
    print(f"  OK: service={body.get('service')} version={body.get('version')} "
          f"commit={body.get('commit')} environment={body.get('environment')} postgres=connected")
    return body


def check_model_status(
    client: httpx.Client, headers: dict[str, str], *, expected_family: str | None, expected_version: str | None,
) -> dict[str, Any]:
    response = client.get("/api/v1/recommendation-ml-service/model/status", headers=headers)
    if response.status_code == 401:
        raise SmokeTestFailure("GET /model/status returned 401 UNAUTHORIZED -- pass --api-key / set INTERNAL_API_KEY.")
    if response.status_code != 200:
        raise SmokeTestFailure(f"GET /model/status returned unexpected status {response.status_code}: {response.text}")
    body = response.json()
    status = body.get("status")

    if status == "MISSING":
        print("  OK (expected on a fresh deployment): status=MISSING -- no model artifact provisioned yet.")
        return body
    if status in ("INCOMPATIBLE", "CORRUPTED"):
        raise SmokeTestFailure(f"GET /model/status reports {status}: {body.get('message')}")
    if status != "READY":
        raise SmokeTestFailure(f"GET /model/status returned unrecognized status {status!r}: {body}")

    selected_model = body.get("selectedModel")
    model_family = body.get("modelFamily")
    model_version = body.get("modelVersion")
    print(f"  OK: status=READY selectedModel={selected_model} modelFamily={model_family} "
          f"modelVersion={model_version} trainedAt={body.get('trainedAt')}")

    if expected_family and model_family != expected_family:
        raise SmokeTestFailure(f"modelFamily={model_family!r} does not match --expected-model-family={expected_family!r}")
    if expected_version and model_version != expected_version:
        raise SmokeTestFailure(f"modelVersion={model_version!r} does not match --expected-model-version={expected_version!r}")
    return body


def check_cold_start_recommendation(client: httpx.Client, headers: dict[str, str]) -> None:
    payload = {"userId": _SMOKE_USER_ID, "candidates": _SMOKE_CANDIDATES, "limit": len(_SMOKE_CANDIDATES)}
    response = client.post("/api/v1/recommendation-ml-service/recommendations", headers=headers, json=payload)
    if response.status_code == 401:
        raise SmokeTestFailure("POST /recommendations returned 401 UNAUTHORIZED -- pass --api-key / set INTERNAL_API_KEY.")
    if response.status_code == 503:
        raise SmokeTestFailure(f"POST /recommendations returned 503 despite GET /model/status=READY: {response.text}")
    if response.status_code != 200:
        raise SmokeTestFailure(f"POST /recommendations returned unexpected status {response.status_code}: {response.text}")

    body = response.json()
    recommendations = body.get("recommendations", [])
    if not recommendations:
        raise SmokeTestFailure(f"POST /recommendations returned zero recommendations for {len(_SMOKE_CANDIDATES)} candidates: {body}")

    scores = [item["score"] for item in recommendations]
    for score in scores:
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise SmokeTestFailure(f"POST /recommendations returned a non-finite score: {scores}")

    ranks = sorted(item["rank"] for item in recommendations)
    expected_ranks = list(range(1, len(recommendations) + 1))
    if ranks != expected_ranks:
        raise SmokeTestFailure(f"POST /recommendations returned non-contiguous ranks {ranks}, expected {expected_ranks}")

    print(f"  OK: strategy={body.get('strategy')} interactionCount={body.get('interactionCount')} "
          f"{len(recommendations)} recommendation(s), scores finite, ranks contiguous 1..{len(recommendations)}")


def run(base_url: str, api_key: str | None, expected_family: str | None, expected_version: str | None) -> int:
    headers = _headers(api_key)
    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        try:
            print("1/3 GET /health")
            check_health(client)

            print("2/3 GET /model/status")
            model_status = check_model_status(
                client, headers, expected_family=expected_family, expected_version=expected_version,
            )

            print("3/3 POST /recommendations (cold-start smoke)")
            if model_status.get("status") == "READY":
                check_cold_start_recommendation(client, headers)
            else:
                print(f"  SKIPPED: model status is {model_status.get('status')}, not READY.")
        except SmokeTestFailure as exc:
            print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
            return 1
        except httpx.HTTPError as exc:
            print(f"\nSMOKE TEST FAILED: could not reach {base_url}: {exc}", file=sys.stderr)
            return 1

    print("\nSMOKE TEST PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("SMOKE_TEST_BASE_URL", "http://localhost:3500"),
                         help="Target service base URL (default: %(default)s, or SMOKE_TEST_BASE_URL).")
    parser.add_argument("--api-key", default=os.getenv("INTERNAL_API_KEY"),
                         help="X-Internal-API-Key value; only required if the target has INTERNAL_API_KEY configured.")
    parser.add_argument("--expected-model-family", default=os.getenv("SMOKE_TEST_EXPECTED_MODEL_FAMILY"),
                         help="Fail if GET /model/status's modelFamily does not match (e.g. classifier).")
    parser.add_argument("--expected-model-version", default=os.getenv("SMOKE_TEST_EXPECTED_MODEL_VERSION"),
                         help="Fail if GET /model/status's modelVersion does not match.")
    args = parser.parse_args(argv)

    return run(args.base_url, args.api_key, args.expected_model_family, args.expected_model_version)


if __name__ == "__main__":
    raise SystemExit(main())
