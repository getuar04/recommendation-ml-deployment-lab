"""Single feature and target contract for the LIVE recommendation domain."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

LIVE_CATEGORICAL=["category","region","language"]
LIVE_NUMERIC=["live_category_affinity","creator_affinity","creator_followed","previous_live_interaction_count","previous_live_watch_time","average_live_watch_time_for_category","recent_live_category_activity","current_viewer_count","viewer_growth_rate","live_age_minutes","region_match","language_match","hour_of_day","already_joined"]
LIVE_FEATURES=LIVE_CATEGORICAL+LIVE_NUMERIC

LIVE_FEATURE_DEFINITIONS={
    "live_category_affinity":"Prior LIVE preference for the candidate category in [0,1].",
    "creator_affinity":"Prior LIVE preference for the candidate creator in [0,1].",
    "creator_followed":"Whether the creator was followed before scoring.",
    "previous_live_interaction_count":"Prior interactions with this creator/live context.",
    "previous_live_watch_time":"Total prior LIVE watch seconds for this creator/live context.",
    "average_live_watch_time_for_category":"Mean prior LIVE watch seconds for the category.",
    "recent_live_category_activity":"Recent LIVE category interaction count.",
    "current_viewer_count":"Current active viewer count at scoring time.",
    "viewer_growth_rate":"Recent normalized viewer growth supplied by Live Service.",
    "live_age_minutes":"Minutes since the stream started.",
    "region_match":"Whether stream region matches the requesting user context.",
    "language_match":"Whether stream language matches the requesting user context.",
    "hour_of_day":"Scoring-time hour.",
    "already_joined":"Whether the user recently joined this stream.",
    "category":"LIVE stream category.","region":"Stream region.","language":"Stream language.",
}

# Named so app.ml.live_dataset_builder's session-level label policy (live_session_target)
# can share the exact same numbers instead of re-hardcoding them -- both apply the same
# "meaningful stay"/"too short to count" semantics, just to a single raw event here vs. a
# reconstructed multi-event session there.
LIVE_MEANINGFUL_STAY_SECONDS = 60
LIVE_SHORT_STAY_SECONDS = 10

def live_target(event: Any)->int|None:
    positive_action=any((getattr(event,"liked",False),getattr(event,"shared",False),getattr(event,"commented",False),getattr(event,"gift_sent",False),getattr(event,"creator_followed",False)))
    joined=getattr(event,"joined",False); watch=float(getattr(event,"watch_time_seconds",0) or 0)
    if (joined and watch>=LIVE_MEANINGFUL_STAY_SECONDS) or positive_action:return 1
    if (getattr(event,"impression",False) and not joined) or (joined and watch<LIVE_SHORT_STAY_SECONDS and not positive_action):return 0
    return None

def derive_live_affinities(previous_live_interactions: int, previous_live_watch_time: float) -> dict[str, float]:
    """Single source of truth for deriving affinity features from prior LIVE history.

    Shared by synthetic training data generation and online serving so both
    compute live_category_affinity/creator_affinity identically and avoid
    train/serve skew.
    """
    return {
        "live_category_affinity": min(1.0, 0.5 + previous_live_interactions * 0.04),
        "creator_affinity": min(1.0, 0.5 + previous_live_interactions * 0.05),
        "average_live_watch_time_for_category": previous_live_watch_time / max(1, previous_live_interactions),
        "recent_live_category_activity": previous_live_interactions,
    }

def live_feature_row(candidate: Any, *, category_affinity:float=0.5, creator_affinity:float=0.5,
                     average_category_watch_time:float=0, recent_category_activity:int=0,
                     scoring_time:datetime|None=None, previous_interaction_count:float|None=None,
                     previous_watch_time:float|None=None, creator_followed:bool|None=None)->dict[str,Any]:
    """`previous_interaction_count`/`previous_watch_time`/`creator_followed` default to the
    candidate's own (caller-supplied) fields when omitted -- unchanged behavior for every
    existing caller. Passing them explicitly lets a caller with real local LIVE history
    (app.services.providers.live_history_provider) override the caller-supplied values with
    authoritative locally-stored ones (see app.services.live_recommendation_service)."""
    now=scoring_time or datetime.now(timezone.utc)
    resolved_interaction_count = candidate.previous_live_interactions if previous_interaction_count is None else previous_interaction_count
    resolved_watch_time = candidate.previous_live_watch_time if previous_watch_time is None else previous_watch_time
    resolved_creator_followed = candidate.creator_followed if creator_followed is None else creator_followed
    return {
        "category":candidate.category.upper(),"region":candidate.region.lower(),"language":candidate.language.lower(),
        "live_category_affinity":max(0,min(1,category_affinity)),"creator_affinity":max(0,min(1,creator_affinity)),
        "creator_followed":int(resolved_creator_followed),"previous_live_interaction_count":resolved_interaction_count,
        "previous_live_watch_time":resolved_watch_time,"average_live_watch_time_for_category":average_category_watch_time,
        "recent_live_category_activity":recent_category_activity,"current_viewer_count":candidate.current_viewer_count,
        "viewer_growth_rate":candidate.viewer_growth_rate,"live_age_minutes":candidate.live_age_minutes,
        "region_match":int(candidate.region_match),"language_match":int(candidate.language_match),"hour_of_day":now.hour,
        "already_joined":int(candidate.already_joined),
    }
