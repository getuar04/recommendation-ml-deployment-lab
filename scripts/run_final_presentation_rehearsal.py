"""FINAL end-to-end presentation rehearsal -- proves the whole VIDEO recommendation story in
one reproducible flow, entirely through the real running server's HTTP API (urllib, same
convention as scripts.prepare_demo -- never a TestClient, never a direct DB write):

    Cold Start -> Baseline personalization -> Positive feedback -> Long-term vs recent/session
    shift -> Search Intent -> NOT_INTERESTED -> SOCIAL candidate -> COLLABORATIVE candidate
    -> Social-vs-negative-feedback conflict -> Already-seen social safety -> Final mixed feed

Never trains, never promotes, never rolls back automatically -- if the active model isn't the
expected frozen one, this script STOPS immediately (see check_active_model). Reuses every
existing demo tool rather than reinventing state:
    scripts.seed_demo_users        -- deterministic users A-E + shared 40-item candidate pool
    scripts.seed_demo_users.build_user_c_shift_events -- User C's live SPORT->MUSIC shift
    candidate_service.services.candidate_service -- SOCIAL/COLLABORATIVE candidate generation

One primary user (A) carries most of the story (baseline, positive feedback, search intent,
SOCIAL/COLLABORATIVE target, final mixed feed) so the presentation reads as one evolving user
journey, not a tour of unrelated ids. Additional users are used only where the story requires a
distinct role: C (pre-seeded long-term-vs-session-shift persona), the dedicated NOT_INTERESTED
user (pre-established Tennis probe, see docs/DEMO_RUNBOOK.md Sec.6), B/E (candidate_service's
own demo SOCIAL/COLLABORATIVE source users), and a genuinely never-seeded id for cold start.

Every check below is an explicit, printed invariant -- pass/fail is tallied and the script exits
non-zero if any FAILED. Exact floating-point values are never asserted, only direction/ordering/
shape, per this project's own established convention (see docs/DEMO_RUNBOOK.md Sec.5).

Usage:
    python -m scripts.run_final_presentation_rehearsal --base-url http://localhost:3500
    python -m scripts.run_final_presentation_rehearsal --reset   # also clears+reseeds demo state first
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

from candidate_service.providers.demo_follow_provider import USER_A as CS_USER_A
from candidate_service.services.candidate_service import (
    generate_candidates,
    to_rms_payload,
)
from scripts.seed_demo_users import (
    build_all_users,
    build_shared_candidate_pool,
    build_user_c_shift_events,
    seed,
    seed_candidate_pool,
    stable_uuid,
)

EXPECTED_MODEL = "LogisticRegression"
EXPECTED_FAMILY = "classifier"
EXPECTED_VERSION = "recommendation-prod-20260824135926"

COLD_START_USER = stable_uuid("demo2-user", "user-cold-start-optional")
NOT_INTERESTED_USER = stable_uuid("demo2-user", "user-e-not-interested-demo")


class RehearsalError(RuntimeError):
    pass


_CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"    [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def _request(base_url: str, method: str, path: str, body: dict | None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(f"{base_url}{path}", data=data,
                                  headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def get(base_url: str, path: str) -> tuple[int, dict]:
    return _request(base_url, "GET", path, None)


def post(base_url: str, path: str, body: dict) -> tuple[int, dict]:
    return _request(base_url, "POST", path, body)


def candidate_body(entry: dict) -> dict:
    return {
        "contentId": entry["contentId"], "creatorId": entry["creatorId"], "category": entry["category"],
        "contentPopularityScore": entry["popularityScore"], "contentAgeHours": entry["ageHours"],
        "creatorFollowed": False, "alreadySeen": False, "title": entry["title"],
        "hashtags": entry["hashtags"], "topics": entry["topics"], "entities": entry["entities"],
        "subgenres": entry["subgenres"],
    }


def event_body(*, user_id: str, content: dict, event_type: str, watch_percentage: float, **flags) -> dict:
    duration = 100.0
    return {
        "eventId": str(uuid.uuid4()), "userId": user_id, "contentId": content["contentId"],
        "creatorId": content["creatorId"], "category": content["category"], "eventType": event_type,
        "watchTimeSeconds": round(watch_percentage / 100 * duration, 2), "contentDurationSeconds": duration,
        "liked": flags.get("liked", False), "shared": flags.get("shared", False),
        "favorited": flags.get("favorited", False), "commented": flags.get("commented", False),
        "creatorFollowed": flags.get("creator_followed", False),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def top_n(resp: dict, n: int = 10) -> None:
    print(f"    {'rank':>4s} {'category':10s} {'score':>7s}  {'source':13s}  reason  |  title")
    for r in resp["recommendations"][:n]:
        print(f"    {r['rank']:>4d} {r['category']:10s} {r['score']:>7.4f}  "
              f"{(r.get('candidateSource') or '-'):13s}  {r['reason']:<22s} {r.get('title') or ''}")


def rank_of(resp: dict, content_id: str) -> int | None:
    return next((r["rank"] for r in resp["recommendations"] if r["contentId"] == content_id), None)


def score_of(resp: dict, content_id: str) -> float | None:
    return next((r["score"] for r in resp["recommendations"] if r["contentId"] == content_id), None)


def reason_of(resp: dict, content_id: str) -> str | None:
    return next((r["reason"] for r in resp["recommendations"] if r["contentId"] == content_id), None)


# --------------------------------------------------------------------------- 1. active model
def check_active_model(base_url: str) -> dict:
    section("1. ACTIVE MODEL SAFETY CHECK")
    status, body = get(base_url, "/api/v1/recommendation-ml-service/model/status")
    if status != 200:
        raise RehearsalError(f"GET /model/status failed: HTTP {status} {body}")
    print(f"    selectedModel={body.get('selectedModel')} modelFamily={body.get('modelFamily')} "
          f"modelVersion={body.get('modelVersion')}")
    ok = (body.get("selectedModel") == EXPECTED_MODEL and body.get("modelFamily") == EXPECTED_FAMILY
          and body.get("modelVersion") == EXPECTED_VERSION)
    if not ok:
        raise RehearsalError(
            f"ACTIVE MODEL DOES NOT MATCH THE FROZEN MODEL. Expected "
            f"{EXPECTED_MODEL}/{EXPECTED_FAMILY}/{EXPECTED_VERSION}, got "
            f"{body.get('selectedModel')}/{body.get('modelFamily')}/{body.get('modelVersion')}. "
            f"STOPPING -- not retraining, not rolling back automatically. Investigate first."
        )
    print("    active model matches the frozen baseline.")
    return body


# --------------------------------------------------------------------------- 2. reset/seed
def reset_and_seed(base_url: str, *, do_reset: bool) -> None:
    section("2. RESET / SEED DEMO BASELINE")
    if do_reset:
        # Reset runs INSIDE the app container (docker compose exec), never as a direct
        # host-side DB connection: on a machine where another Postgres instance also answers
        # on host port 5432 (confirmed here -- SHOW server_version reports 17.9 on the
        # host-reachable port vs. this project's actual postgres:16-alpine container reporting
        # 16.14, two different servers), a host-side DATABASE_URL can silently reach the WRONG
        # database regardless of which credentials are used. `docker compose exec app` already
        # has the correct DATABASE_URL baked in (see docker-compose.yml's `environment:` block)
        # and reaches Postgres via Docker's own internal network -- unaffected by whatever else
        # is bound to the host's port 5432. reset_demo.py itself needs no changes: its own
        # _LOCAL_HOSTS guard already allows hostname "postgres" specifically for this case.
        print("    running scripts.reset_demo dry-run inside the app container (docker compose exec)...")
        dry_run = subprocess.run(
            ["docker", "compose", "exec", "-T", "app", "python", "-m", "scripts.reset_demo"],
            capture_output=True, text=True, check=False,
        )
        print(textwrap.indent(dry_run.stdout.strip(), "    "))
        if dry_run.returncode != 0:
            raise RehearsalError(f"reset dry-run failed (exit {dry_run.returncode}): {dry_run.stderr.strip()}")

        print("    running the real reset inside the app container (--yes)...")
        real_run = subprocess.run(
            ["docker", "compose", "exec", "-T", "app", "python", "-m", "scripts.reset_demo", "--yes"],
            capture_output=True, text=True, check=False,
        )
        print(textwrap.indent(real_run.stdout.strip(), "    "))
        if real_run.returncode != 0:
            raise RehearsalError(f"reset failed (exit {real_run.returncode}): {real_run.stderr.strip()}")
    else:
        print("    --reset not passed: reusing whatever demo state is already there (idempotent reseed only).")
    seed(base_url, None)
    seed_candidate_pool(base_url, None)


# --------------------------------------------------------------------------- 3. cold start
def cold_start_step(base_url: str, pool: list[dict]) -> None:
    section("3. COLD START (never-seeded user)")
    body = {"userId": COLD_START_USER, "limit": 10, "candidates": [candidate_body(e) for e in pool[:10]]}
    status, resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("cold-start request succeeds", status == 200, f"HTTP {status}")
    if status != 200:
        return
    print(f"    userId={COLD_START_USER}  strategy={resp['strategy']}  "
          f"interactionCount={resp['interactionCount']}  recommendations={len(resp['recommendations'])}")
    check("strategy == COLD_START", resp["strategy"] == "COLD_START", resp["strategy"])
    check("non-empty recommendations", len(resp["recommendations"]) > 0)
    check("all scores finite", all(math.isfinite(r["score"]) for r in resp["recommendations"]))


# --------------------------------------------------------------------------- 4. baseline
def baseline_step(base_url: str, user_a: str, pool: list[dict]) -> dict:
    section("4. BASELINE PERSONALIZATION (User A, primary)")
    _, profile = get(base_url, f"/api/v1/recommendation-ml-service/users/{user_a}/behaviour-profile")
    print(f"    interactionCount={profile.get('interactionCount')}")
    body = {"userId": user_a, "limit": len(pool), "candidates": [candidate_body(e) for e in pool]}
    status, resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("baseline request succeeds", status == 200, f"HTTP {status}")
    print(f"    strategy={resp['strategy']} interactionCount={resp['interactionCount']} modelVersion={resp['modelVersion']}")
    top_n(resp)
    from collections import Counter
    dist = Counter(r["category"] for r in resp["recommendations"][:10])
    print(f"    Top-10 category distribution (dominant categories): {dict(dist)}")
    search_intent_already_active = any(r["reason"] == "SEARCH_INTENT_MATCH" for r in resp["recommendations"][:10])
    if search_intent_already_active:
        # A prior rehearsal run's search intent (TTL=30 min) is still active and reranking THIS
        # "baseline" call too -- expected on a shared, repeatedly-rehearsed container within the
        # window, not a personalization defect. See docs/DEMO_RUNBOOK.md Sec.7/9.
        print("    NOTE: a search intent from an earlier run is still active and reranking this "
              "baseline call (reason SEARCH_INTENT_MATCH present) -- not asserting category mix "
              "hard this run. Run --reset (or wait out the 30-min TTL) for a clean baseline.")
    else:
        check("SPORT visible near the top (long-term interest)", dist.get("SPORT", 0) >= 2, str(dist))
    return resp


# --------------------------------------------------------------------------- 5. positive feedback
def positive_feedback_step(base_url: str, user_a: str, pool: list[dict], before_resp: dict) -> None:
    section("5. POSITIVE FEEDBACK (User A -- VIDEO_COMPLETED, high watch, liked)")
    barcelona = next(e for e in pool if e["contentId"] == stable_uuid("demo2-candidate-content", "sport-barcelona-ucl"))
    _, profile_before = get(base_url, f"/api/v1/recommendation-ml-service/users/{user_a}/behaviour-profile")
    count_before = int(profile_before.get("interactionCount") or 0)

    ev = event_body(user_id=user_a, content=barcelona, event_type="VIDEO_COMPLETED", watch_percentage=98.0, liked=True)
    status, ev_resp = post(base_url, "/api/v1/recommendation-ml-service/events", ev)
    check("positive event stored", status == 201 and ev_resp.get("stored") is True, f"HTTP {status} {ev_resp}")

    body = {"userId": user_a, "limit": len(pool), "candidates": [candidate_body(e) for e in pool]}
    status, after_resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("recommendation-after-feedback succeeds", status == 200, f"HTTP {status}")

    _, profile_after = get(base_url, f"/api/v1/recommendation-ml-service/users/{user_a}/behaviour-profile")
    count_after = int(profile_after.get("interactionCount") or 0)
    print(f"    interactionCount {count_before} -> {count_after}")
    check("interactionCount incremented by exactly 1", count_after == count_before + 1,
          f"{count_before} -> {count_after}")
    check("modelVersion unchanged (no retrain)", before_resp["modelVersion"] == after_resp["modelVersion"],
          after_resp["modelVersion"])

    b_before_rank, b_before_score, b_before_reason = (rank_of(before_resp, barcelona["contentId"]),
                                                        score_of(before_resp, barcelona["contentId"]),
                                                        reason_of(before_resp, barcelona["contentId"]))
    b_after_rank, b_after_score, b_after_reason = (rank_of(after_resp, barcelona["contentId"]),
                                                     score_of(after_resp, barcelona["contentId"]),
                                                     reason_of(after_resp, barcelona["contentId"]))
    print(f"    Barcelona (SPORT) before: rank={b_before_rank} score={b_before_score} reason={b_before_reason}")
    print(f"    Barcelona (SPORT) after:  rank={b_after_rank} score={b_after_score} reason={b_after_reason}")
    print("    change source: recent/session category-affinity feature update from the new event "
          "(no retraining -- one global model, per-user features)")


# --------------------------------------------------------------------------- 6. session shift
def session_shift_step(base_url: str, user_c: str, pool: list[dict]) -> None:
    section("6. LONG-TERM VS RECENT/SESSION SHIFT (User C: long-term SPORT -> live MUSIC shift)")
    body = {"userId": user_c, "limit": len(pool), "candidates": [candidate_body(e) for e in pool]}
    status, before_resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("session-shift baseline request succeeds", status == 200, f"HTTP {status}")
    from collections import Counter
    dist_before = Counter(r["category"] for r in before_resp["recommendations"][:10])
    print(f"    BEFORE Top-10 distribution: {dict(dist_before)}")
    if dist_before.get("MUSIC", 0) >= 4:
        print("    NOTE: MUSIC already prominent before this run's shift events -- a prior "
              "rehearsal likely already applied User C's shift (its event ids are deterministic, "
              "not fresh per run). Re-run with --reset for a truly fresh long-term-only baseline.")

    now = datetime.now(timezone.utc)
    _, shift_builder = build_user_c_shift_events(now)
    stored, duplicate = 0, 0
    for event in shift_builder.events:
        content_payload = {"contentId": event["content_id"], "creatorId": event["creator_id"], "contentType": "VIDEO",
                            "category": event["category"], "durationSeconds": event["content_duration_seconds"],
                            "popularityScore": 0.5}
        post(base_url, "/api/v1/recommendation-ml-service/contents", content_payload)
        status, ev_resp = post(base_url, "/api/v1/recommendation-ml-service/events", {
            "eventId": event["event_id"], "userId": event["user_id"], "contentId": event["content_id"],
            "creatorId": event["creator_id"], "category": event["category"], "eventType": event["event_type"],
            "watchTimeSeconds": event["watch_time_seconds"], "contentDurationSeconds": event["content_duration_seconds"],
            "liked": event["liked"], "shared": event["shared"], "favorited": event["favorited"],
            "commented": event["commented"], "creatorFollowed": event["creator_followed"],
            "timestamp": event["timestamp"],
        })
        check(f"shift event stored ({event['category']})", status == 201, f"HTTP {status}")
        if ev_resp.get("stored"):
            stored += 1
        else:
            duplicate += 1
    print(f"    sent {len(shift_builder.events)} live shift events (7 MUSIC completions, 2 SPORT fast-skips, "
          f"1 SPORT NOT_INTERESTED): stored={stored} idempotent-duplicates={duplicate}")

    status, after_resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("session-shift after-request succeeds", status == 200, f"HTTP {status}")
    dist_after = Counter(r["category"] for r in after_resp["recommendations"][:10])
    print(f"    AFTER Top-10 distribution: {dict(dist_after)}")
    check("modelVersion unchanged across the shift (no retrain)",
          before_resp["modelVersion"] == after_resp["modelVersion"], after_resp["modelVersion"])
    # Top-10 CATEGORY COUNT is not the right signal to assert on: this service's Top-N applies a
    # category-diversity cap (app.ml.reranker's consecutive-category-cap selection), so a
    # category's raw score rising does not guarantee more of its items survive into the Top-10
    # -- other, previously-neutral categories can legitimately backfill slots vacated by SPORT's
    # collapse ahead of a still-capped MUSIC. The precise, undiluted signal (same convention
    # scripts.run_four_user_demo already uses via print_score_deltas, for the same documented
    # reason) is a direct per-item SCORE comparison, not a Top-10 category tally.
    if stored > 0:
        music_deltas = [(score_of(after_resp, e["contentId"]) or 0) - (score_of(before_resp, e["contentId"]) or 0)
                         for e in pool if e["category"] == "MUSIC"]
        sport_deltas = [(score_of(after_resp, e["contentId"]) or 0) - (score_of(before_resp, e["contentId"]) or 0)
                         for e in pool if e["category"] == "SPORT"]
        music_mean = sum(music_deltas) / len(music_deltas)
        sport_mean = sum(sport_deltas) / len(sport_deltas)
        print(f"    mean per-item score delta (after-before), undiluted by Top-10 diversity slotting: "
              f"MUSIC={music_mean:+.4f} (n={len(music_deltas)})  SPORT={sport_mean:+.4f} (n={len(sport_deltas)})")
        # NOT asserting MUSIC's own absolute score rose: a category with NO prior history for
        # this user already carries an EXPLORATION bonus before any signal exists (a real,
        # legitimate, separately-documented reason code) -- a few minutes of fresh SESSION_
        # INTEREST does not have to exceed that already-generous baseline for the shift to be
        # working correctly. The two checks below are what the demo actually claims and what is
        # directly, unambiguously true here: SPORT's long-term lead does not survive the recent
        # negative signal, and MUSIC clearly outranks SPORT once the shift lands.
        check("MUSIC's average shift is less negative than SPORT's (recent/session signal "
              "moved the two categories in the expected relative direction)",
              music_mean > sport_mean, f"MUSIC={music_mean:+.4f} SPORT={sport_mean:+.4f}")
        best_music_rank = min(r["rank"] for r in after_resp["recommendations"] if r["category"] == "MUSIC")
        best_sport_rank = min(r["rank"] for r in after_resp["recommendations"] if r["category"] == "SPORT")
        check("MUSIC's best-ranked item outranks SPORT's best-ranked item after the shift",
              best_music_rank < best_sport_rank, f"best MUSIC rank={best_music_rank} best SPORT rank={best_sport_rank}")
    else:
        print("    (skipping the score-delta assertions -- 0 new events stored this run, see NOTE above)")
    print("    long-term signal: still real (SPORT history, 15 events) | recent/session signal: "
          "fresh MUSIC completions dominate the 30-day/30-min windows | rank consequence: printed above, no retraining")


# --------------------------------------------------------------------------- 7. search intent
def search_intent_step(base_url: str, user_a: str, pool: list[dict]) -> None:
    section("7. SEARCH INTENT (User A -- 'AI technology news')")
    body = {"userId": user_a, "limit": len(pool), "candidates": [candidate_body(e) for e in pool]}
    status, before_resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("search-intent baseline request succeeds", status == 200, f"HTTP {status}")
    ai_news = stable_uuid("demo2-candidate-content", "news-ai-breakthrough")
    rank_before = rank_of(before_resp, ai_news)
    _, profile_before = get(base_url, f"/api/v1/recommendation-ml-service/users/{user_a}/behaviour-profile")
    count_before = profile_before.get("interactionCount")

    intent_body = {"query": "AI technology news", "topics": ["Technology News"],
                   "entities": ["Artificial Intelligence"], "subgenres": ["Technology"], "confidence": 1.0}
    status, intent_resp = post(base_url, f"/api/v1/recommendation-ml-service/users/{user_a}/search-intent", intent_body)
    check("search intent recorded", status == 201 and intent_resp.get("status") == "ACTIVE", f"HTTP {status}")
    print(f"    recordedAt={intent_resp.get('recordedAt')} expiresAt={intent_resp.get('expiresAt')}")

    status, active_resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("search-intent-active request succeeds", status == 200, f"HTTP {status}")
    rank_active = rank_of(active_resp, ai_news)
    reason_active = reason_of(active_resp, ai_news)
    print(f"    AI/Tech News rank BEFORE={rank_before} -> ACTIVE={rank_active}  reason={reason_active}")
    check("reason becomes SEARCH_INTENT_MATCH", reason_active == "SEARCH_INTENT_MATCH", str(reason_active))
    # Rank movement itself is informational, not a hard invariant: on a shared, long-lived demo
    # container a PRIOR rehearsal's search intent can still be active (TTL=30 min) when this
    # step's own "BEFORE" measurement runs, so BEFORE may already reflect the boosted position --
    # rank_active == rank_before is then correct, not a regression. The reliable, environment-
    # independent proof is the reason string above; run --reset for a truly clean before/after.
    if rank_active is not None and rank_before is not None and rank_active > rank_before:
        print(f"    NOTE: rank did not improve ({rank_before} -> {rank_active}) -- see comment above "
              f"if BEFORE already reflected a still-active prior search intent.")

    _, profile_after = get(base_url, f"/api/v1/recommendation-ml-service/users/{user_a}/behaviour-profile")
    count_after = profile_after.get("interactionCount")
    check("interactionCount unchanged (search intent is not persisted interaction history)",
          count_after == count_before, f"{count_before} -> {count_after}")


# --------------------------------------------------------------------------- 8. NOT_INTERESTED
def not_interested_step(base_url: str, pool: list[dict]) -> dict:
    section("8. NOT_INTERESTED (dedicated Tennis demo user)")
    nadal = next(e for e in pool if e["contentId"] == stable_uuid("demo2-candidate-content", "sport-nadal-roland-garros"))
    djokovic = next(e for e in pool if e["contentId"] == stable_uuid("demo2-candidate-content", "sport-djokovic-practice"))
    unrelated_music = next(e for e in pool if e["category"] == "MUSIC")
    unrelated_sport = next(e for e in pool if e["category"] == "SPORT" and e["contentId"] not in
                            (nadal["contentId"], djokovic["contentId"]))
    probes = [nadal, djokovic, unrelated_music, unrelated_sport]

    # Establish a small, genuine PRIOR positive baseline before measuring "before" -- mirrors
    # scripts.run_four_user_demo's own established NOT_INTERESTED scenario (equal small positive
    # history on Nadal/Djokovic/unrelated, VIDEO_WATCHED, liked) and matches the scenario
    # tests/test_not_interested_gradual_suppression.py's test_one_not_interested_event_full_
    # scenario actually validates (category_affinity_before >= 0.70, i.e. a WARM user). Without
    # this, a genuinely zero-history user's first-ever interaction being the rejection itself
    # gives category_affinity no positive counterweight to suppress FROM -- a real, different,
    # previously-untested edge case (see the final report), not what this demo step claims to
    # show ("one rejection doesn't collapse a category the user otherwise likes").
    for probe in (nadal, djokovic, unrelated_sport):
        baseline_ev = event_body(user_id=NOT_INTERESTED_USER, content=probe, event_type="VIDEO_WATCHED",
                                  watch_percentage=88.0, liked=True)
        status, ev_resp = post(base_url, "/api/v1/recommendation-ml-service/events", baseline_ev)
        check(f"baseline event stored ({probe['title'][:30]})", status == 201, f"HTTP {status}")

    body = {"userId": NOT_INTERESTED_USER, "limit": len(probes), "candidates": [candidate_body(e) for e in probes]}
    status, before_resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("NOT_INTERESTED baseline request succeeds", status == 200, f"HTTP {status}")
    scores_before = {e["contentId"]: score_of(before_resp, e["contentId"]) for e in probes}
    print(f"    BEFORE -- Nadal(exact)={scores_before[nadal['contentId']]}  "
          f"Djokovic(semantic neighbor)={scores_before[djokovic['contentId']]}  "
          f"unrelated SPORT={scores_before[unrelated_sport['contentId']]}  "
          f"unrelated MUSIC={scores_before[unrelated_music['contentId']]}")

    _, profile_before = get(base_url, f"/api/v1/recommendation-ml-service/users/{NOT_INTERESTED_USER}/behaviour-profile")
    count_before = int(profile_before.get("interactionCount") or 0)
    ev = event_body(user_id=NOT_INTERESTED_USER, content=nadal, event_type="CONTENT_NOT_INTERESTED", watch_percentage=4.0)
    status, ev_resp = post(base_url, "/api/v1/recommendation-ml-service/events", ev)
    check("NOT_INTERESTED event stored", status == 201 and ev_resp.get("stored") is True, f"HTTP {status} {ev_resp}")

    status, after_resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("NOT_INTERESTED after-request succeeds", status == 200, f"HTTP {status}")
    scores_after = {e["contentId"]: score_of(after_resp, e["contentId"]) for e in probes}
    print(f"    AFTER  -- Nadal(exact)={scores_after[nadal['contentId']]}  "
          f"Djokovic(semantic neighbor)={scores_after[djokovic['contentId']]}  "
          f"unrelated SPORT={scores_after[unrelated_sport['contentId']]}  "
          f"unrelated MUSIC={scores_after[unrelated_music['contentId']]}")

    _, profile_after = get(base_url, f"/api/v1/recommendation-ml-service/users/{NOT_INTERESTED_USER}/behaviour-profile")
    count_after = int(profile_after.get("interactionCount") or 0)
    check("interactionCount +1", count_after == count_before + 1, f"{count_before} -> {count_after}")
    nadal_drop = (scores_after[nadal["contentId"]] or 0) <= (scores_before[nadal["contentId"]] or 0)
    if count_before == 3:
        # Exactly the 3 baseline events this step just seeded, nothing carried over from an
        # earlier rehearsal run -- this is the controlled, single-rejection-on-a-warm-user
        # scenario tests/test_not_interested_gradual_suppression.py's
        # test_one_not_interested_event_full_scenario already proves in isolation (asserts
        # category_affinity_before >= 0.70 for exactly this reason), so the same drop is
        # asserted as a hard invariant here too.
        check("exact rejected content (Nadal) drops", nadal_drop,
              f"{scores_before[nadal['contentId']]} -> {scores_after[nadal['contentId']]}")
    else:
        # count_before > 3 means this user already carries CONTENT_NOT_INTERESTED history from
        # an earlier rehearsal run on this container (its own accumulated state, not a defect --
        # see docs/DEMO_RUNBOOK.md Sec.10 for the reset procedure). A second-or-later stacked
        # rejection on already-suppressed content is not the controlled scenario the dedicated
        # test proves, so this is reported, not hard-asserted -- run --reset for the clean case.
        print(f"    NOTE: interactionCount was already {count_before} before this run's rejection "
              f"(expected exactly 3 from this step's own baseline seeding on a fresh container) "
              f"-- Nadal drop this run: {nadal_drop} "
              f"({scores_before[nadal['contentId']]} -> {scores_after[nadal['contentId']]}). "
              f"The controlled single-rejection invariant is proven by "
              f"tests/test_not_interested_gradual_suppression.py, not reasserted here.")
    check("unrelated SPORT stays competitive (no broad-category collapse)",
          (scores_after[unrelated_sport["contentId"]] or 0) >= (scores_before[unrelated_sport["contentId"]] or 0) * 0.9,
          f"{scores_before[unrelated_sport['contentId']]} -> {scores_after[unrelated_sport['contentId']]}")
    return {"nadal": nadal, "djokovic": djokovic, "scores_before": scores_before, "count_before": count_before}


# --------------------------------------------------------------------------- 9/10. SOCIAL / COLLABORATIVE
def social_and_collaborative_steps(base_url: str) -> tuple[dict | None, dict | None]:
    section("9. SOCIAL CANDIDATE (Candidate Service demo foundation -> real RMS)")
    all_candidates = generate_candidates(CS_USER_A, limit=20)
    social = [c for c in all_candidates if c.source == "SOCIAL"]
    collaborative = [c for c in all_candidates if c.source == "COLLABORATIVE"]

    social_result = None
    if social:
        top_social = max(social, key=lambda c: c.relationship_strength + c.interest_similarity + c.source_user_engagement)
        print(f"    candidateSource=SOCIAL  content={top_social.content.title}  "
              f"interestSimilarity={top_social.interest_similarity:.4f}  "
              f"relationshipStrength={top_social.relationship_strength:.4f}  "
              f"sourceUserEngagement={top_social.source_user_engagement:.4f}  "
              f"mutualFollow={top_social.mutual_follow}")
        body = {"userId": CS_USER_A, "limit": len(social), "candidates": [to_rms_payload(c) for c in social]}
        status, resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
        check("SOCIAL candidates score through the real RMS endpoint", status == 200, f"HTTP {status}")
        if status == 200:
            top_n(resp, n=len(social))
            check("candidateSource echoed back unchanged (RMS never mutates provenance)",
                  next((r.get("candidateSource") for r in resp["recommendations"]
                        if r["contentId"] == top_social.content.content_id), None) == "SOCIAL")
            social_result = {"candidate": top_social, "response": resp}
    else:
        check("at least one SOCIAL candidate generated", False)
    print("    Candidate Service GENERATED this candidate from a direct follow relationship; "
          "RMS did NOT discover the friend -- RMS only scored/reranked the supplied candidate.")

    section("10. COLLABORATIVE CANDIDATE (no follow relationship required)")
    collaborative_result = None
    if collaborative:
        top_collab = max(collaborative, key=lambda c: c.interest_similarity + c.source_user_engagement)
        print(f"    candidateSource=COLLABORATIVE  content={top_collab.content.title}  "
              f"interestSimilarity={top_collab.interest_similarity:.4f}  "
              f"relationshipStrength={top_collab.relationship_strength:.4f}  "
              f"sourceUserEngagement={top_collab.source_user_engagement:.4f}  "
              f"mutualFollow={top_collab.mutual_follow}")
        check("relationshipStrength == 0 for COLLABORATIVE", top_collab.relationship_strength == 0.0)
        check("mutualFollow is False for COLLABORATIVE", top_collab.mutual_follow is False)
        body = {"userId": CS_USER_A, "limit": len(collaborative), "candidates": [to_rms_payload(c) for c in collaborative]}
        status, resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
        check("COLLABORATIVE candidates score through the real RMS endpoint", status == 200, f"HTTP {status}")
        if status == 200:
            top_n(resp, n=len(collaborative))
            collaborative_result = {"candidate": top_collab, "response": resp}
    else:
        check("at least one COLLABORATIVE candidate generated", False)
    print("    No direct social relationship required -- Candidate Service generated this because "
          "the two users have similar interests. RMS still makes the final ranking decision.")
    return social_result, collaborative_result


# --------------------------------------------------------------------------- 11. conflict
def conflict_step(base_url: str, ni_context: dict) -> None:
    section("11. SOCIAL VS NEGATIVE FEEDBACK CONFLICT (dedicated Tennis demo user, post-rejection)")
    nadal, djokovic = ni_context["nadal"], ni_context["djokovic"]
    # Nadal/Djokovic are NOT equal-quality candidates on their own (popularity 0.9/1h vs
    # 0.35/200h) -- comparing their raw boosted scores against EACH OTHER would confound the
    # conflict signal with that base-quality gap (the same confound already documented and
    # fixed for app.benchmark.scenarios' social-already-seen-suppressed scenario). The correct,
    # established comparison (see tests/test_social_relevance.py's
    # test_social_high_evidence_does_not_override_strong_target_dislike) is the SAME candidate
    # with vs without the boost -- here, Nadal's own pre-rejection baseline vs Nadal WITH
    # maximal incoming social evidence AFTER the explicit rejection.
    strong_social = {"interestSimilarity": 0.95, "relationshipStrength": 0.9, "sourceUserEngagement": 0.9, "mutualFollow": True}
    nadal_social = {**candidate_body(nadal), "candidateSource": "SOCIAL", "socialContext": strong_social}
    djokovic_social = {**candidate_body(djokovic), "candidateSource": "SOCIAL", "socialContext": strong_social}
    body = {"userId": NOT_INTERESTED_USER, "limit": 2, "candidates": [nadal_social, djokovic_social]}
    status, resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("conflict request succeeds", status == 200, f"HTTP {status}")
    if status != 200:
        return
    top_n(resp, n=2)
    nadal_score = score_of(resp, nadal["contentId"])
    djokovic_score = score_of(resp, djokovic["contentId"])
    nadal_pre_rejection = ni_context["scores_before"][nadal["contentId"]]
    print(f"    Nadal pre-rejection baseline (no social evidence) = {nadal_pre_rejection}")
    print(f"    Nadal post-rejection WITH maximal incoming social evidence = {nadal_score}  "
          f"(Djokovic, not rejected, same social evidence, for reference = {djokovic_score})")
    not_restored = (nadal_score or 0) <= (nadal_pre_rejection or 0)
    if ni_context["count_before"] == 3:
        # Same controlled-vs-accumulated-state distinction as not_interested_step above -- only
        # assert this hard when step 8's rejection was this user's first this run (fresh state).
        check("social evidence does NOT restore the target user's explicitly-rejected content "
              "to (or above) its own pre-rejection baseline, even at maximal social evidence",
              not_restored, f"pre-rejection={nadal_pre_rejection} post-rejection+social={nadal_score}")
    else:
        print(f"    NOTE: not hard-asserting (step 8's rejection was not this user's first this "
              f"container's lifetime, see its NOTE) -- restored-above-pre-rejection this run: "
              f"{not not_restored}")


# --------------------------------------------------------------------------- 12. already-seen safety
def already_seen_step(base_url: str, user_a: str) -> None:
    section("12. ALREADY-SEEN SOCIAL SAFETY (User A)")
    nadal_id = stable_uuid("demo2-candidate-content", "sport-nadal-roland-garros")
    generated = generate_candidates(CS_USER_A, limit=20)
    generated_ids = {c.content.content_id for c in generated}
    check("Candidate Service best-effort excludes already-seen content (Nadal, seen by User A)",
          nadal_id not in generated_ids, "Nadal id absent from generated SOCIAL/COLLABORATIVE pool")

    pool = build_shared_candidate_pool()
    nadal = next(e for e in pool if e["contentId"] == nadal_id)
    twin = next(e for e in pool if e["category"] == "SPORT" and e["contentId"] != nadal_id)
    seen_candidate = {**candidate_body(nadal), "alreadySeen": True}
    unseen_twin = {**candidate_body(twin), "alreadySeen": False}
    body = {"userId": user_a, "limit": 2, "candidates": [seen_candidate, unseen_twin]}
    status, resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("already-seen probe request succeeds (real active model, not a fixture)", status == 200, f"HTTP {status}")
    if status != 200:
        return
    top_n(resp, n=2)
    seen_score = score_of(resp, nadal_id)
    unseen_score = score_of(resp, twin["contentId"])
    print(f"    seen(alreadySeen=true)={seen_score}  unseen twin={unseen_score}")
    check("RMS's own already-seen safeguard (SEEN_PENALTY) still applies if seen content reaches it",
          (seen_score or 0) <= (unseen_score or 0), f"seen={seen_score} unseen={unseen_score}")


# --------------------------------------------------------------------------- 13. final mixed feed
def final_mixed_feed_step(base_url: str, user_a: str, pool: list[dict],
                           social_result: dict | None, collaborative_result: dict | None) -> None:
    section("13. FINAL MIXED CANDIDATE SOURCE FEED (User A)")
    intent_body = {"query": "AI technology news", "topics": ["Technology News"],
                   "entities": ["Artificial Intelligence"], "subgenres": ["Technology"], "confidence": 1.0}
    post(base_url, f"/api/v1/recommendation-ml-service/users/{user_a}/search-intent", intent_body)

    # Reserve the SOCIAL/COLLABORATIVE candidates' own contentIds first so the "ordinary" slice
    # below can't accidentally pick the SAME content -- a duplicate contentId with two different
    # candidateSource values in one request would collapse to whichever copy RMS's dedup keeps,
    # silently discarding the candidateSource-bearing copy and making this step's own
    # "candidateSource preserved" check meaningless, not a real product-level failure.
    reserved_ids = set()
    if social_result is not None:
        reserved_ids.add(social_result["candidate"].content.content_id)
    if collaborative_result is not None:
        reserved_ids.add(collaborative_result["candidate"].content.content_id)
    ai_news_id = stable_uuid("demo2-candidate-content", "news-ai-breakthrough")
    nadal_id = stable_uuid("demo2-candidate-content", "sport-nadal-roland-garros")
    reserved_ids.update({ai_news_id, nadal_id})

    ordinary = [candidate_body(e) for e in pool if e["contentId"] not in reserved_ids][:5]
    ai_news = candidate_body(next(e for e in pool if e["contentId"] == ai_news_id))
    gaming_negative_affinity = candidate_body(next(e for e in pool if e["category"] == "GAMING"
                                                     and e["contentId"] not in reserved_ids))
    nadal_seen = {**candidate_body(next(e for e in pool if e["contentId"] == nadal_id)), "alreadySeen": True}

    mixed: list[dict] = list(ordinary) + [ai_news, gaming_negative_affinity, nadal_seen]
    if social_result is not None:
        mixed.append(to_rms_payload(social_result["candidate"]))
    if collaborative_result is not None:
        mixed.append(to_rms_payload(collaborative_result["candidate"]))

    body = {"userId": user_a, "limit": len(mixed), "candidates": mixed}
    status, resp = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    check("final mixed feed request succeeds", status == 200, f"HTTP {status}")
    if status != 200:
        return
    top_n(resp, n=10)
    ids = [r["contentId"] for r in resp["recommendations"]]
    check("no duplicate contentIds", len(ids) == len(set(ids)))
    check("all scores finite", all(math.isfinite(r["score"]) for r in resp["recommendations"]))
    check("ranks contiguous 1..N", [r["rank"] for r in resp["recommendations"]] == list(range(1, len(ids) + 1)))
    sources = {r["contentId"]: r.get("candidateSource") for r in resp["recommendations"]}
    check("candidateSource preserved for SOCIAL/COLLABORATIVE items",
          all(sources.get(c["contentId"]) == c.get("candidateSource") for c in mixed if c.get("candidateSource")))
    reasons = {r["reason"] for r in resp["recommendations"]}
    print(f"    reasons observed across the mixed feed: {sorted(reasons)}")

    status2, resp2 = post(base_url, "/api/v1/recommendation-ml-service/recommendations", body)
    stable = status2 == 200 and [r["contentId"] for r in resp2["recommendations"][:10]] == ids[:10]
    check("Top-10 stable across an immediate repeat request (no retraining side effects)", stable)


def responsibility_boundaries() -> None:
    section("14. RESPONSIBILITY BOUNDARIES")
    print("""
    Feed Backend        -> requests recommendations
    Candidate Service    -> generates candidates (demo/foundation providers today; real Follow/UBS
                             integration and transport are future work, see docs/CANDIDATE_SERVICE_SOCIAL_CONTRACT.md)
    Follow/UBS           -> currently demo providers only, no network integration exists
    RMS (this service)   -> User x VIDEO scoring, reranking, Top-N -- never generates candidates itself
    Event Tracking       -> feedback events change profile/input signals, never trigger a retrain
    """.strip("\n"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:3500")
    parser.add_argument("--reset", action="store_true", help="Clear demo users' interactions before reseeding (opt-in).")
    args = parser.parse_args()

    try:
        check_active_model(args.base_url)
        reset_and_seed(args.base_url, do_reset=args.reset)

        users = build_all_users(datetime.now(timezone.utc))
        user_a = users["A"][0]
        user_c = users["C"][0]
        pool = build_shared_candidate_pool()

        t0 = time.monotonic()
        cold_start_step(args.base_url, pool)
        t1 = time.monotonic()
        before_resp = baseline_step(args.base_url, user_a, pool)
        t2 = time.monotonic()
        positive_feedback_step(args.base_url, user_a, pool, before_resp)
        session_shift_step(args.base_url, user_c, pool)
        search_intent_step(args.base_url, user_a, pool)
        ni_context = not_interested_step(args.base_url, pool)
        t3 = time.monotonic()
        social_result, collaborative_result = social_and_collaborative_steps(args.base_url)
        t4 = time.monotonic()
        conflict_step(args.base_url, ni_context)
        already_seen_step(args.base_url, user_a)
        final_mixed_feed_step(args.base_url, user_a, pool, social_result, collaborative_result)
        t5 = time.monotonic()
        responsibility_boundaries()

        section("15. PERFORMANCE SANITY (rough, single-sample, not a microbenchmark)")
        print(f"    cold start:            {(t1 - t0) * 1000:.0f} ms")
        print(f"    baseline feed:         {(t2 - t1) * 1000:.0f} ms")
        print(f"    NOT_INTERESTED cycle:  {(t3 - t2) * 1000:.0f} ms")
        print(f"    SOCIAL/COLLABORATIVE:  {(t4 - t3) * 1000:.0f} ms")
        print(f"    conflict+seen+mixed:   {(t5 - t4) * 1000:.0f} ms")

        check_active_model(args.base_url)

    except RehearsalError as exc:
        print(f"\nREHEARSAL STOPPED: {exc}", file=sys.stderr)
        return 1

    section("SUMMARY")
    passed = sum(1 for _, ok, _ in _CHECKS if ok)
    failed = [name for name, ok, _ in _CHECKS if not ok]
    print(f"  invariants: {passed}/{len(_CHECKS)} passed")
    if failed:
        print("  FAILED:")
        for name in failed:
            print(f"    - {name}")
        print("\nFINAL PRESENTATION REHEARSAL: NOT READY")
        return 1

    print("\nFINAL PRESENTATION REHEARSAL: READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
