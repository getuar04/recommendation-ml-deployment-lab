from collections import defaultdict

from app.db.repositories import interactions
from app.ml.feature_builder import (
    COMPLETION_WATCH_PERCENTAGE_THRESHOLD,
    affinity_score,
    build_profiles,
    interaction_signals,
)


def get_behaviour_profile(db, user_id):
    """Spec §11: bounded, mathematically explainable scores from actual interactions only.
    Category affinity reuses the exact sigmoid affinity from `app.ml.feature_builder`;
    creator affinity applies the same bounded sigmoid over completion/follow signals.

    Intentionally NOT VIDEO-filtered (unlike app.services.recommendation_service/
    training_service/cohort_aggregation_service, which each derive their own VIDEO-only view):
    `interactions(db, user_id)` counts every stored interaction regardless of content_type,
    matching README.md's description of the real User Behavior Service this demo endpoint
    stands in for ("maintains point-in-time user, category, creator and LIVE behavior
    features"). This is a general cross-content-type activity profile, not the VIDEO
    recommendation engine's own (separately computed) interaction count/strategy."""
    rows = interactions(db, user_id)
    if not rows:
        return {"userId": user_id, "status": "COLD_START", "interactionCount": 0,
                "categoryAffinities": [], "creatorAffinities": [],
                "message": "Not enough user interactions to build a behaviour profile."}
    categories = get_profile(db, user_id)
    creators: dict[str, dict[str, float]] = defaultdict(lambda: {"interactions": 0, "completed": 0, "followed": 0})
    last = None
    for row in rows:
        signals = interaction_signals(row)
        c = creators[row.creator_id]
        c["interactions"] += 1
        c["completed"] += int(row.event_type == "VIDEO_COMPLETED" or (row.watch_percentage or 0) >= COMPLETION_WATCH_PERCENTAGE_THRESHOLD)
        c["followed"] += int(signals.creator_followed)
        ts = row.timestamp
        last = ts if last is None or ts > last else last
    creator_affinities = sorted(
        ({"creatorId": creator_id, "score": affinity_score(c["completed"] * 4 + c["followed"] * 6),
          "interactionCount": int(c["interactions"])} for creator_id, c in creators.items()),
        key=lambda item: item["score"], reverse=True)
    return {"userId": user_id, "status": "ACTIVE", "interactionCount": len(rows),
            "categoryAffinities": [{"category": c["category"], "score": c["affinityScore"],
                                    "interactionCount": c["interactionCount"]} for c in categories],
            "creatorAffinities": creator_affinities,
            "lastInteractionAt": last.isoformat() if last else None}

def get_profile(db,user_id):
    cats=build_profiles(interactions(db,user_id)).get(user_id,{})
    result=[]
    for category,d in cats.items():
        result.append({"category":category,"affinityScore":d["affinity_score"],"interactionCount":int(d["interaction_count"]),"impressionCount":int(d["impression_count"]),"watchCount":int(d["watch_count"]),"averageWatchPercentage":round(d["average_watch_percentage"],2),"completionCount":int(d["completion_count"]),"skipCount":int(d["skip_count"]),"fastSkipCount":int(d["fast_skip_count"]),"likeCount":int(d["like_count"]),"shareCount":int(d["share_count"]),"favoriteCount":int(d["favorite_count"]),"commentCount":int(d["comment_count"]),"creatorFollowCount":int(d["creator_follow_count"]),"notInterestedCount":int(d["not_interested_count"]),"recentInteractionCount":int(d["recent_interaction_count"]),"lastInteractionTimestamp":d["last_interaction_timestamp"].isoformat()})
    return sorted(result,key=lambda x:x["affinityScore"],reverse=True)

