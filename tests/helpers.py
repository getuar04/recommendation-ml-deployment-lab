from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def _synthetic_labeled_frame(n=600, seed=0):
    rng = np.random.RandomState(seed)
    categories = ["FOOD", "SPORT", "MUSIC"]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event_types = ["VIDEO_WATCHED", "VIDEO_COMPLETED", "VIDEO_SKIPPED", "CONTENT_NOT_INTERESTED"]
    rows = []
    for i in range(n):
        category = categories[i % 3]
        target = 1 if (category == "SPORT" and rng.rand() < 0.8) or (category != "SPORT" and rng.rand() < 0.2) else 0
        rows.append({
            "category": category, "category_affinity": rng.rand(), "recent_category_affinity": rng.rand(),
            "has_category_history": rng.randint(0, 2),
            "average_category_watch_percentage": rng.rand() * 100, "recent_category_watch_percentage": rng.rand() * 100,
            "category_completion_rate": rng.rand(), "recent_category_completion_rate": rng.rand(),
            "category_positive_count": rng.rand() * 3, "category_negative_count": rng.rand() * 3,
            "category_interaction_count": rng.rand() * 3, "has_creator_history": rng.randint(0, 2),
            "creator_interaction_count": rng.rand() * 3, "creator_completion_rate": rng.rand(),
            "creator_followed": rng.randint(0, 2),
            "hashtag_affinity": 0.5, "topic_affinity": 0.5, "entity_affinity": 0.5,
            "subgenre_affinity": 0.5, "title_affinity": 0.5,
            "semantic_positive_match_count": 0, "semantic_negative_match_count": 0,
            "has_semantic_history": 0, "strongest_semantic_affinity": 0.5, "average_semantic_affinity": 0.5,
            "user_total_interaction_count": rng.rand() * 3,
            "content_popularity_score": rng.rand(), "content_age_hours": rng.rand() * 100,
            "already_seen": rng.randint(0, 2), "hour_of_day": rng.randint(0, 24),
            "days_since_last_category_interaction": rng.rand() * 30,
            "session_category_affinity": rng.rand(), "has_session_activity": rng.randint(0, 2),
            "session_average_watch_percentage": rng.rand() * 100,
            "session_positive_interaction_count": rng.rand() * 3, "session_negative_interaction_count": rng.rand() * 3,
            "session_category_streak_valence_matched": rng.randint(-5, 6),
            "last_interaction_category_match": rng.randint(0, 2),
            "session_intent_confidence": rng.rand(),
            "target": target, "user_id": f"user-{i % 40}", "creator_id": f"creator-{i % 10}",
            "content_id": f"content-{i}", "candidate_group": f"user-{i % 40}:day-{i // 20}",
            "timestamp": start + timedelta(minutes=i),
            "event_type": event_types[i % len(event_types)] if target == 0 else "VIDEO_COMPLETED",
            "event_watch_percentage": 5.0 if target == 0 else 95.0,
            "event_liked": bool(target == 1 and i % 2 == 0), "event_shared": False, "event_favorited": False,
            "event_creator_followed": bool(target == 1 and i % 5 == 0),
        })
    return pd.DataFrame(rows)
