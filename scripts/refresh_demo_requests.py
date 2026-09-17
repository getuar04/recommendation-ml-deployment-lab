"""Regenerates the demo request bodies whose timestamps go stale between rehearsals --
consolidates what used to be two ad-hoc inline Python snippets in DEMO_RUNBOOK.md into one
reusable, tested script. Never changes user/content/creator/category IDs -- only inserts
fresh UTC timestamps (and, for events, a fresh unique eventId) so the presenter never has to
hand-edit a JSON file before presenting.

Writes, ready to POST as-is:
    docs/demo-user-c-shift-events.json   -- User C's 10 SPORT->MUSIC shift events, camelCase,
                                             timestamped so all 10 land inside the 30-minute
                                             SESSION_WINDOW relative to run time.
    docs/not-interested-event.json       -- one fresh CONTENT_NOT_INTERESTED event (Nadal,
                                             the dedicated NOT_INTERESTED demo user), unique
                                             eventId + current timestamp every run.
    docs/positive-interaction-event.json -- one fresh VIDEO_COMPLETED event (User A, the
                                             Barcelona SPORT candidate already in her baseline
                                             pool), unique eventId + current timestamp.

Usage:
    python -m scripts.refresh_demo_requests
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.seed_demo_users import (
    build_shared_candidate_pool,
    build_user_c_shift_events,
    stable_uuid,
)

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"

NOT_INTERESTED_DEMO_USER = stable_uuid("demo2-user", "user-e-not-interested-demo")
USER_A = stable_uuid("demo2-user", "user-a-longterm-sport")


def _to_event_body(event: dict) -> dict:
    """snake_case EventBuilder dict -> camelCase POST /api/v1/recommendation-ml-service/events body."""
    return {
        "eventId": event["event_id"], "userId": event["user_id"], "contentId": event["content_id"],
        "creatorId": event["creator_id"], "category": event["category"], "eventType": event["event_type"],
        "watchTimeSeconds": event["watch_time_seconds"], "contentDurationSeconds": event["content_duration_seconds"],
        "liked": event["liked"], "shared": event["shared"], "favorited": event["favorited"],
        "commented": event["commented"], "creatorFollowed": event["creator_followed"],
        "timestamp": event["timestamp"],
    }


def refresh_user_c_shift_events(now: datetime) -> list[dict]:
    _, builder = build_user_c_shift_events(now)
    bodies = [_to_event_body(e) for e in builder.events]
    path = DOCS_DIR / "demo-user-c-shift-events.json"
    path.write_text(json.dumps(bodies, indent=2), encoding="utf-8")
    print(f"wrote {path} ({len(bodies)} events, ready to POST /api/v1/recommendation-ml-service/events as-is)")
    return bodies


def refresh_not_interested_event(now: datetime) -> dict:
    pool = build_shared_candidate_pool()
    nadal = next(e for e in pool if e["contentId"] == stable_uuid("demo2-candidate-content", "sport-nadal-roland-garros"))
    body = {
        "eventId": str(uuid.uuid4()), "userId": NOT_INTERESTED_DEMO_USER,
        "contentId": nadal["contentId"], "creatorId": nadal["creatorId"], "category": "SPORT",
        "eventType": "CONTENT_NOT_INTERESTED", "watchTimeSeconds": 4.0, "contentDurationSeconds": 100.0,
        "liked": False, "shared": False, "favorited": False, "commented": False, "creatorFollowed": False,
        "timestamp": now.isoformat(),
    }
    path = DOCS_DIR / "not-interested-event.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"wrote {path} (fresh eventId={body['eventId']}, timestamp={body['timestamp']})")
    return body


def refresh_positive_interaction_event(now: datetime) -> dict:
    pool = build_shared_candidate_pool()
    barcelona = next(e for e in pool if e["contentId"] == stable_uuid("demo2-candidate-content", "sport-barcelona-ucl"))
    body = {
        "eventId": str(uuid.uuid4()), "userId": USER_A,
        "contentId": barcelona["contentId"], "creatorId": barcelona["creatorId"], "category": "SPORT",
        "eventType": "VIDEO_COMPLETED", "watchTimeSeconds": 98.0, "contentDurationSeconds": 100.0,
        "liked": True, "shared": False, "favorited": False, "commented": False, "creatorFollowed": False,
        "timestamp": now.isoformat(),
    }
    path = DOCS_DIR / "positive-interaction-event.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"wrote {path} (fresh eventId={body['eventId']}, timestamp={body['timestamp']})")
    return body


def main() -> None:
    now = datetime.now(timezone.utc)
    refresh_user_c_shift_events(now)
    refresh_not_interested_event(now)
    refresh_positive_interaction_event(now)


if __name__ == "__main__":
    main()
