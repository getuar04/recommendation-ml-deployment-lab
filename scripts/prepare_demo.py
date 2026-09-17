"""DEMO/LOCAL-only operational script: prepares the running service for tomorrow's
presentation against a REAL server (Docker or `uvicorn app.main:app` locally) -- never an
in-process TestClient, never a direct DB write.

Default flow (every step through the real HTTP API, nothing bypassed):
    1. GET  /api/v1/recommendation-ml-service/health                       -- verify the service/DB are up
    2. seed deterministic demo content/users/interactions (scripts.seed_demo_users, idempotent)
    3. GET  /api/v1/recommendation-ml-service/model/status                  -- verify the ALREADY-ACTIVE artifact is the
       frozen model (see EXPECTED_MODEL below) -- does NOT retrain by default (see next)
    4. GET  /api/v1/recommendation-ml-service/health                        -- confirm the service is still healthy

Retraining is opt-in (`--retrain`), NOT the default, and for a specific, evidenced reason:
`POST /model/train`'s real production path (training_service.train ->
train_and_select_cross_family) PROMOTES its winner to the active artifact BEFORE this script
gets a chance to verify it -- and a same-day investigation proved XGBRanker's HARD
eligibility gates can flip between platforms (byte-identical code/data/seed, Windows
consistently ineligible, this Linux/Docker deployment consistently eligible; see
DEMO_RUNBOOK.md's "XGBoost cross-platform eligibility" note). A live rehearsal on this exact
container reproduced it: XGBRanker won and was promoted, replacing the known-good
LogisticRegression artifact, before any check could catch it. The already-active artifact
(`recommendation-prod-20260824135926`, LogisticRegression) is already the validated,
rehearsed model for tomorrow -- there is no need to retrain at all for the demo to work.

If `--retrain` is passed anyway, this script now closes that gap itself: on any mismatch
(wrong selectedModel/modelFamily, or eligibleSelection != True) it immediately calls
POST /model/rollback to restore the previous artifact BEFORE raising, so a bad promotion
never survives past this script's own exit -- not merely "fails loudly after the fact".

Usage:
    python -m scripts.prepare_demo --base-url http://localhost:3500
    python -m scripts.prepare_demo --base-url http://localhost:3500 --retrain   # opt-in, see above

Safe to re-run: seeding is idempotent (uuid5 ids, duplicate eventIds are no-ops); by default
this script never writes to the active model artifact at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from scripts.seed_demo_users import seed, seed_candidate_pool

EXPECTED_MODEL = "LogisticRegression"
EXPECTED_FAMILY = "classifier"
TRAIN_POLL_INTERVAL_SECONDS = 1.0
TRAIN_POLL_TIMEOUT_SECONDS = 180


class PrepareDemoError(RuntimeError):
    pass


def _get(base_url: str, path: str, api_key: str | None = None) -> tuple[int, dict]:
    headers = {"X-Internal-API-Key": api_key} if api_key else {}
    req = urllib.request.Request(f"{base_url}{path}", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(base_url: str, path: str, api_key: str | None = None) -> tuple[int, dict]:
    headers = {"X-Internal-API-Key": api_key} if api_key else {}
    req = urllib.request.Request(f"{base_url}{path}", data=b"", headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def check_health(base_url: str) -> dict:
    status, body = _get(base_url, "/api/v1/recommendation-ml-service/health")
    if status != 200 or body.get("status") != "ok":
        raise PrepareDemoError(f"Service is not healthy: HTTP {status} {body}")
    print(f"  status={body['status']} dependencies={body.get('dependencies')}")
    return body


def _rollback(base_url: str, api_key: str | None, *, reason: str) -> None:
    print(f"  SAFETY ROLLBACK: {reason}")
    status, body = _post(base_url, "/api/v1/recommendation-ml-service/model/rollback", api_key)
    if status != 200:
        print(f"  ROLLBACK ALSO FAILED (HTTP {status} {body}) -- the active artifact may still "
              f"be wrong. Do not present without manually checking GET /api/v1/recommendation-ml-service/model/status.")
        return
    print(f"  rolled back: active is now modelVersion={body.get('modelVersion')}")


def train_and_verify(base_url: str, api_key: str | None) -> dict:
    status, body = _post(base_url, "/api/v1/recommendation-ml-service/model/train", api_key)
    if status != 202:
        raise PrepareDemoError(f"POST /model/train failed: HTTP {status} {body}")
    job_id = body["jobId"]
    print(f"  training job accepted: jobId={job_id}")

    deadline = time.monotonic() + TRAIN_POLL_TIMEOUT_SECONDS
    job = None
    while time.monotonic() < deadline:
        status, job = _get(base_url, f"/api/v1/recommendation-ml-service/model/train/jobs/{job_id}", api_key)
        if status != 200:
            raise PrepareDemoError(f"GET job status failed: HTTP {status} {job}")
        if job["status"] in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(TRAIN_POLL_INTERVAL_SECONDS)
    else:
        raise PrepareDemoError(f"Training job {job_id} did not finish within {TRAIN_POLL_TIMEOUT_SECONDS}s")

    if job["status"] != "SUCCEEDED":
        raise PrepareDemoError(f"Training job {job_id} FAILED: {job.get('error')}")

    result = job["result"]
    selected = result.get("selectedModel")
    family = result.get("modelFamily")
    eligible = result.get("eligibleSelection")
    print(f"  training SUCCEEDED (and already PROMOTED to active): selectedModel={selected} "
          f"modelFamily={family} eligibleSelection={eligible} modelVersion={result.get('modelVersion')}")

    if selected != EXPECTED_MODEL or family != EXPECTED_FAMILY or eligible is not True:
        # The wrong model is already live at this point -- training_service.train() promotes
        # before returning. Restore safety immediately, THEN report the failure, rather than
        # leaving a bad artifact active while this exception propagates.
        _rollback(base_url, api_key, reason="production selection did not match the expected frozen model")
        raise PrepareDemoError(
            f"PRODUCTION SELECTION DID NOT MATCH THE EXPECTED FROZEN MODEL (now rolled back).\n"
            f"  expected: selectedModel={EXPECTED_MODEL} modelFamily={EXPECTED_FAMILY} eligibleSelection=True\n"
            f"  actual:   selectedModel={selected} modelFamily={family} eligibleSelection={eligible}\n"
            f"This is NOT auto-corrected by re-promoting a fake result -- investigate before presenting. "
            f"See DEMO_RUNBOOK.md's XGBoost cross-platform eligibility note. Full result:\n"
            f"{json.dumps(result, indent=2, default=str)}"
        )
    return result


def verify_active_model(base_url: str) -> dict:
    status, body = _get(base_url, "/api/v1/recommendation-ml-service/model/status")
    if status != 200 or body.get("status") != "READY":
        raise PrepareDemoError(f"Model status is not READY: HTTP {status} {body}")
    selected = body.get("selectedModel")
    family = body.get("modelFamily")
    print(f"  status={body['status']} selectedModel={selected} modelFamily={family} "
          f"modelVersion={body.get('modelVersion')} trainedAt={body.get('trainedAt')}")
    if selected != EXPECTED_MODEL or family != EXPECTED_FAMILY:
        raise PrepareDemoError(
            f"ACTIVE MODEL IS NOT THE EXPECTED FROZEN MODEL: expected {EXPECTED_MODEL}/{EXPECTED_FAMILY}, "
            f"got {selected}/{family}. Not auto-corrected. If this happened without you passing "
            f"--retrain, something else (another script, another operator) changed the active "
            f"artifact -- investigate with GET /api/v1/recommendation-ml-service/model/versions before presenting."
        )
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:3500")
    parser.add_argument("--api-key", default=None, help="X-Internal-API-Key, if INTERNAL_API_KEY is configured.")
    parser.add_argument("--skip-seed", action="store_true", help="Skip re-seeding (DB already has demo data).")
    parser.add_argument("--retrain", action="store_true",
                         help="Opt-in: retrain via the real production path before verifying. "
                              "See the module docstring for why this is NOT the default.")
    args = parser.parse_args()

    try:
        _section("1/4 HEALTH CHECK")
        check_health(args.base_url)

        if not args.skip_seed:
            _section("2/4 SEED DETERMINISTIC DEMO DATA (users A-E + shared candidate pool)")
            seed(args.base_url, args.api_key)
            seed_candidate_pool(args.base_url, args.api_key)
        else:
            _section("2/4 SEED DETERMINISTIC DEMO DATA -- SKIPPED (--skip-seed)")

        if args.retrain:
            _section("3/4 RETRAIN VIA THE REAL PRODUCTION PATH (POST /api/v1/recommendation-ml-service/model/train) -- opt-in, see docstring")
            train_and_verify(args.base_url, args.api_key)
        else:
            _section("3/4 VERIFY THE ALREADY-ACTIVE ARTIFACT (no retrain -- pass --retrain to opt in)")
        verify_active_model(args.base_url)

        _section("4/4 FINAL HEALTH CHECK")
        check_health(args.base_url)

    except PrepareDemoError as exc:
        print(f"\nPREPARE_DEMO FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nDEMO ENVIRONMENT READY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
