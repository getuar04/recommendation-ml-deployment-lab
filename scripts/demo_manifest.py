"""Single source of truth for the demo's important user/content IDs -- prints them, never
duplicates them. Every ID below is read directly from scripts.seed_demo_users (the actual
seed definitions), not retyped, so this can never silently drift from what gets seeded.

Usage:
    python -m scripts.demo_manifest
"""
from __future__ import annotations

from datetime import datetime, timezone

from scripts.seed_demo_users import (
    build_all_users,
    build_shared_candidate_pool,
    stable_uuid,
)

# Slugs of the candidate-pool items the demo talks about by name -- see
# scripts.seed_demo_users.CANDIDATE_POOL_SPEC for the full 40-item pool this is drawn from.
NAMED_CANDIDATE_SLUGS = {
    "Nadal / Tennis": "sport-nadal-roland-garros",
    "Djokovic / Tennis": "sport-djokovic-practice",
    "Football (Barcelona)": "sport-barcelona-ucl",
    "Football (Real Madrid)": "sport-real-madrid-laliga",
    "Basketball (Lakers)": "sport-lakers-buzzer",
    "Basketball (Warriors)": "sport-warriors-celtics",
    "Global/World News": "news-g20-summit",
    "AI/Tech News": "news-ai-breakthrough",
    "Music example (Dua Lipa)": "music-dualipa-single",
    "Music example (Tomorrowland)": "music-tomorrowland-mainstage",
}


def main() -> None:
    now = datetime.now(timezone.utc)
    users = build_all_users(now)
    pool = build_shared_candidate_pool()
    pool_by_id = {c["contentId"]: c for c in pool}

    print("=" * 88)
    print("DEMO USERS")
    print("=" * 88)
    for label, (user_id, builder) in users.items():
        print(f"  {label}: {user_id}  ({len(builder.events)} seeded interactions)")

    print("\n" + "=" * 88)
    print("NAMED CANDIDATE-POOL CONTENT (shared 40-item pool, all users)")
    print("=" * 88)
    for name, slug in NAMED_CANDIDATE_SLUGS.items():
        content_id = stable_uuid("demo2-candidate-content", slug)
        entry = pool_by_id.get(content_id)
        category = entry["category"] if entry else "?"
        title = entry["title"] if entry else "(not found -- slug drifted from CANDIDATE_POOL_SPEC)"
        print(f"  {name:30s} [{category:6s}] {content_id}  {title}")


if __name__ == "__main__":
    main()
