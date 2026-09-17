from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.candidate_schemas import (
    CandidateGenerationRequest,
    LiveCandidate,
    LiveCandidateGenerationRequest,
)
from app.services.providers.live_candidate_provider import load_active_live_candidates
from app.services.providers.video_candidate_provider import generate_video_candidates

# Phase A §9: tagged as demo scaffolding, not the production Candidate Service contract --
# see README "Boundary with the real platform". Kept, not removed, per explicit instruction.
router = APIRouter(tags=["demo-candidate-service"])


@router.post("/candidates/generate")
def generate(payload: CandidateGenerationRequest, db: Session = Depends(get_db)):
    """Thin route: all VIDEO candidate-generation business logic lives in
    app.services.providers.video_candidate_provider.generate_video_candidates, shared byte-
    for-byte with app.services.recommendation_service.recommend() (via
    app.services.providers.candidate_provider.resolve) -- never duplicated, never called over
    HTTP by that service. This route only serializes the shared function's output into the
    pre-existing response shape (including the per-candidate `source` bucket label, which is
    route-response-only metadata, never part of the `Candidate` schema itself)."""
    results = generate_video_candidates(db, payload.user_id, limit=payload.limit)
    candidates = [
        {
            "contentId": candidate.content_id, "creatorId": candidate.creator_id, "category": candidate.category,
            "contentPopularityScore": candidate.content_popularity_score, "contentAgeHours": candidate.content_age_hours,
            "creatorFollowed": candidate.creator_followed, "alreadySeen": candidate.already_seen, "source": source,
            "title": candidate.title, "hashtags": candidate.hashtags, "topics": candidate.topics,
            "entities": candidate.entities, "subgenres": candidate.subgenres,
        }
        for candidate, source in results
    ]
    return {"userId": payload.user_id, "candidates": candidates}

LIVE_SOURCES=("FOLLOWED_CREATOR","PREFERRED_LIVE_CATEGORY","TRENDING_LIVE","REGION_MATCH","NEW_LIVE","EXPLORATION")
@router.post("/candidates/generate/live")
def generate_live(payload:LiveCandidateGenerationRequest,db:Session=Depends(get_db)):
    user_id=payload.user_id;limit=payload.limit;parsed=[]
    # Local-candidate-foundation addition: an empty/omitted streams[] falls back to this
    # service's own local LIVE candidate pool (app.services.providers.live_candidate_provider)
    # instead of trivially returning zero candidates -- explicit streams, when supplied,
    # are used unchanged and take precedence (byte-identical to before this fallback existed).
    streams=payload.streams or load_active_live_candidates(db,user_id,limit=limit)
    for stream in streams:
        if stream.status.value=="ACTIVE":parsed.append(stream)
    preferred={item.category for item in sorted(parsed,key=lambda x:x.previous_live_interactions,reverse=True)[:2] if item.previous_live_interactions}
    pools:dict[str,list[LiveCandidate]]={source:[] for source in LIVE_SOURCES}
    for stream in parsed:
        if stream.creator_followed:pools["FOLLOWED_CREATOR"].append(stream)
        if stream.category in preferred:pools["PREFERRED_LIVE_CATEGORY"].append(stream)
        if stream.viewer_growth_rate>=.15 or stream.current_viewer_count>=500:pools["TRENDING_LIVE"].append(stream)
        if stream.region_match:pools["REGION_MATCH"].append(stream)
        if stream.live_age_minutes<=20:pools["NEW_LIVE"].append(stream)
        pools["EXPLORATION"].append(stream)
    quota=max(1,limit//len(LIVE_SOURCES));result=[];ids=set()
    for source in LIVE_SOURCES:
        added=0
        for stream in sorted(pools[source],key=lambda x:x.current_viewer_count,reverse=True):
            if stream.stream_id in ids:continue
            result.append({**stream.model_dump(by_alias=True,mode="json"),"source":source});ids.add(stream.stream_id);added+=1
            if added>=quota:break
    for stream in parsed:
        if len(result)>=limit:break
        if stream.stream_id not in ids:result.append({**stream.model_dump(by_alias=True,mode="json"),"source":"EXPLORATION"});ids.add(stream.stream_id)
    return {"userId":user_id,"contentType":"LIVE","candidates":result[:limit]}
