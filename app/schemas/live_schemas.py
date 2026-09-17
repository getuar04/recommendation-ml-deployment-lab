from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import RECOMMENDATION_MAX_CANDIDATES
from app.schemas.limits import (
    LIVE_RECOMMENDATION_LIMIT_DEFAULT,
    LIVE_RECOMMENDATION_LIMIT_MAX,
    LIVE_RECOMMENDATION_LIMIT_MIN,
)


class LiveStatus(str, Enum):
    ACTIVE="ACTIVE"
    ENDED="ENDED"

# Phase A §7: viewerGrowthRate bound (-1..10) is generous headroom above the reranker's own
# 0.15 "trending" threshold (app/ml/live_reranker.py); currentViewerCount/liveAgeMinutes
# ceilings reject clearly-malformed input far above any realistic value, not a tightened
# realistic range.
class LiveCandidate(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    stream_id: str=Field(alias="streamId",min_length=1,max_length=128)
    creator_id: str=Field(alias="creatorId",min_length=1,max_length=128)
    category: str=Field(min_length=1,max_length=64)
    status: LiveStatus
    current_viewer_count: int=Field(alias="currentViewerCount",ge=0,le=10_000_000)
    viewer_growth_rate: float=Field(alias="viewerGrowthRate",ge=-1.0,le=10.0)
    started_at: datetime|None=Field(None,alias="startedAt")
    live_age_minutes: float=Field(alias="liveAgeMinutes",ge=0,le=1440)
    creator_followed: bool=Field(False,alias="creatorFollowed")
    previous_live_interactions: int=Field(0,alias="previousLiveInteractions",ge=0)
    previous_live_watch_time: float=Field(0,alias="previousLiveWatchTime",ge=0)
    region: str=Field(max_length=32)
    language: str=Field(max_length=32)
    region_match: bool=Field(False,alias="regionMatch")
    language_match: bool=Field(False,alias="languageMatch")
    already_joined: bool=Field(False,alias="alreadyJoined")

class LiveRecommendationRequest(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    user_id: str=Field(alias="userId",min_length=1,max_length=128)
    limit: int=Field(LIVE_RECOMMENDATION_LIMIT_DEFAULT,ge=LIVE_RECOMMENDATION_LIMIT_MIN,le=LIVE_RECOMMENDATION_LIMIT_MAX)
    # Optional (local-candidate-foundation addition): omitting/emptying candidates no longer
    # means "score nothing" -- app.services.live_recommendation_service.recommend_live falls
    # back to its own local LIVE candidate pool (app.services.providers.live_candidate_provider)
    # when this is empty. Explicit candidates, when supplied, are still used unchanged and take
    # precedence -- this is purely additive for every existing caller.
    candidates: list[LiveCandidate]=Field(default_factory=list,max_length=RECOMMENDATION_MAX_CANDIDATES)
