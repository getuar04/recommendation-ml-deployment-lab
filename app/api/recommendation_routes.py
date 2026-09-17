from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.live_schemas import LiveRecommendationRequest
from app.schemas.recommendation_schemas import RecommendationRequest
from app.schemas.response_models import (
    ErrorResponse,
    LiveRecommendationResponse,
    RecommendationResponse,
)
from app.services.live_recommendation_service import (
    LiveModelArtifactInvalid,
    LiveModelNotTrained,
    recommend_live,
)
from app.services.recommendation_service import (
    CandidateSourceUnavailable,
    ModelArtifactInvalid,
    ModelNotTrained,
    UserBehaviorSourceUnavailable,
    recommend,
)

router=APIRouter(tags=["production-ranking"])

_COLD_START_RECOMMENDATION_EXAMPLE = {"userId": "user-2000", "modelVersion": "recommendation-prod-20260730143000",
                                     "strategy": "COLD_START", "interactionCount": 0,
                                     "recommendations": [{"contentId": "video-501", "category": "MUSIC", "score": 0.55, "rank": 1, "reason": "Popular in MUSIC"}]}
_HYBRID_RECOMMENDATION_EXAMPLE = {"userId": "user-tiered", "modelVersion": "recommendation-prod-20260730143000",
                                 "strategy": "HYBRID", "interactionCount": 3,
                                 "recommendations": [{"contentId": "video-221", "category": "SPORT", "score": 0.71, "rank": 1, "reason": "Matches recent SPORT activity"}]}
_PERSONALISED_RECOMMENDATION_EXAMPLE = {"userId": "user-1001", "modelVersion": "recommendation-prod-20260730143000",
                                       "strategy": "PERSONALISED_ML", "interactionCount": 14,
                                       "recommendations": [{"contentId": "video-221", "category": "SPORT", "score": 0.91, "rank": 1, "reason": "High category affinity and creator followed"}]}
_MODEL_NOT_TRAINED_EXAMPLE = {"error": "MODEL_NOT_TRAINED", "message": "Train the recommendation model before requesting predictions.", "requestId": "..."}
_MODEL_ARTIFACT_INVALID_EXAMPLE = {"error": "MODEL_ARTIFACT_INVALID", "reason": "INCOMPATIBLE", "message": "Model metadata featureNames do not match the current feature contract. Retrain to produce a compatible artifact.", "requestId": "..."}


@router.post("/recommendations", response_model=RecommendationResponse, responses={
    200: {"content": {"application/json": {"examples": {
        "coldStart": {"summary": "New user, no interaction history", "value": _COLD_START_RECOMMENDATION_EXAMPLE},
        "hybrid": {"summary": "Some history, below the personalisation threshold", "value": _HYBRID_RECOMMENDATION_EXAMPLE},
        "personalised": {"summary": "Enough history for full personalisation", "value": _PERSONALISED_RECOMMENDATION_EXAMPLE},
    }}}},
    503: {"model": ErrorResponse, "content": {"application/json": {"examples": {
        "notTrained": {"summary": "No artifact exists yet", "value": _MODEL_NOT_TRAINED_EXAMPLE},
        "invalid": {"summary": "Artifact exists but is corrupted or incompatible", "value": _MODEL_ARTIFACT_INVALID_EXAMPLE},
    }}}},
})
def recommendations(req:RecommendationRequest,db:Session=Depends(get_db)):
    """`strategy` is COLD_START/HYBRID/PERSONALISED_ML depending on the user's prior
    interaction count (PERSONALISED_RECOMMENDATION_MIN_INTERACTIONS, configurable) -- the
    model still scores every request either way. `score` is the post-reranking adjusted
    score. Explicit, non-empty `candidates` in the request body are treated as caller-supplied
    (not re-verified against the content catalog in this service) and take precedence; an
    empty/omitted `candidates` falls back to this service's own local VIDEO candidate
    generation (app.services.providers.video_candidate_provider, the same logic
    POST /candidates/generate uses) instead of scoring nothing. Example values above are
    illustrative only, not measured results."""
    try:return recommend(db,req,exclude_already_seen=True)
    except ModelNotTrained: raise HTTPException(503,detail={"error":"MODEL_NOT_TRAINED","message":"Train the recommendation model before requesting predictions."})
    except ModelArtifactInvalid as e: raise HTTPException(503,detail={"error":"MODEL_ARTIFACT_INVALID","reason":e.reason,"message":str(e)})
    # Dual-mode REAL-service dependency failures (app.services.providers.*): a clear,
    # structured 503 -- never a generic 500 -- when RECOMMENDATION_DATA_MODE=REAL and a
    # required upstream (UBS/Candidate Service) is unavailable with no configured fallback.
    except UserBehaviorSourceUnavailable as e: raise HTTPException(503,detail={"error":"USER_BEHAVIOR_SERVICE_UNAVAILABLE","message":str(e)})
    except CandidateSourceUnavailable as e: raise HTTPException(503,detail={"error":"CANDIDATE_SERVICE_UNAVAILABLE","message":str(e)})


_LIVE_RECOMMENDATION_EXAMPLE = {"userId": "user-1001", "modelVersion": "live-recommendation-prod-20260730143000",
                               "strategy": "HYBRID", "interactionCount": 4,
                               "recommendations": [{"streamId": "stream-42", "category": "GAMING", "score": 0.81, "rank": 1, "reason": "High viewer growth and category affinity"}]}


@router.post("/recommendations/live", response_model=LiveRecommendationResponse, responses={
    200: {"content": {"application/json": {"example": _LIVE_RECOMMENDATION_EXAMPLE}}},
    503: {"model": ErrorResponse, "content": {"application/json": {"example": {"error": "LIVE_MODEL_NOT_TRAINED", "message": "Train the LIVE recommendation model before requesting predictions.", "requestId": "..."}}}},
})
def live_recommendations(req:LiveRecommendationRequest,db:Session=Depends(get_db)):
    """The active LIVE model is currently trained on synthetic data only (no real LIVE
    behavior log existed yet at training time) -- see GET /model/metrics/live's
    datasetSource field. Serving itself now prefers this user's own real, locally stored
    LIVE history over caller-supplied history fields when any exists (see
    app.services.live_recommendation_service's module docstring for the precedence rule).
    Example values above are illustrative only, not measured results."""
    try:return recommend_live(req,db)
    except LiveModelNotTrained:raise HTTPException(503,detail={"error":"LIVE_MODEL_NOT_TRAINED","message":"Train the LIVE recommendation model before requesting predictions."})
    except LiveModelArtifactInvalid as e: raise HTTPException(503,detail={"error":"LIVE_MODEL_ARTIFACT_INVALID","reason":e.reason,"message":str(e)})
