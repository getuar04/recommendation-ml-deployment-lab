"""Phase A §6: typed response models mirroring the exact dict shapes the routes already
return -- these are metadata only. No route's actual return statement changes, and no field
is renamed, added, or removed from the wire format. Each model exists so `app.openapi()`
generates a real, named schema instead of an empty `{}` (confirmed via `app.openapi()` before
this change), so a TypeScript/Java client generator gets real types instead of `any`/`Object`.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ------------------------------------------------------------------------------- VIDEO / LIVE

class RecommendationItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    content_id: str = Field(alias="contentId")
    category: str
    # Documentation-only change (no wire-format/behavior change -- see module docstring):
    # `score` is the model + business-adjustment (boost/penalty) relevance score computed
    # BEFORE the diversity-constrained greedy selection in app.ml.reranker.rerank() picks the
    # final slate order. That selection deliberately relaxes strict score order slot-by-slot
    # (MAX_CONSECUTIVE_CATEGORY / MAX_CREATOR_IN_TOP_10, see rerank()'s docstring/Tier 1-3
    # comments) so a lower-scored candidate can be placed ahead of a higher-scored one of the
    # same category/creator to satisfy diversity. `score` is therefore NOT guaranteed to be
    # globally descending across `rank` when diversity constraints are active -- it reflects
    # relevance, not final selection position.
    score: float = Field(
        description=(
            "Model + business-adjustment relevance score for this item, computed before "
            "diversity-constrained slate selection. Not guaranteed to be globally descending "
            "by rank: diversity constraints (category/creator caps) can place a lower-scored "
            "item ahead of a higher-scored one of the same category or creator."
        ),
    )
    rank: int
    # CLAUDE-P1-002 remediation: documentation-only change (no wire-format/behavior change --
    # see module docstring). `reason` is produced by app.ml.reranker.explanation(), whose own
    # docstring already says "Transparent heuristic explanation; this is not model attribution
    # or SHAP" -- a first-matching flag-presence rule, not score attribution. That was true
    # internally but undocumented on the actual OpenAPI-facing field, so a consumer had no way
    # to know `reason` can name a signal that had zero effect on this item's score/rank (or
    # omit one that did). Exposing that honestly here, not redesigning the heuristic itself.
    reason: str = Field(
        description=(
            "Heuristic display label naming a plausible contributing signal for this "
            "recommendation (e.g. FOLLOWED_CREATOR, POPULAR_CONTENT, EXPLORATION) -- NOT "
            "causal model attribution. It is chosen by a fixed-priority rule over the "
            "candidate's flags/features and can name a signal that had no measurable effect "
            "on this item's score or rank, or omit one that did. Do not present it to end "
            "users as a proof of why an item ranked where it did."
        ),
    )
    # Response enrichment (pass-through only): the exact same candidate-supplied fields the
    # request already carried, echoed back so a caller doesn't have to re-join its own
    # candidate list against the ranked response by contentId. This is NOT the semantic
    # scoring mechanism itself -- scoring already happened before this dict is built (see
    # app.services.recommendation_service.recommend()); these fields never feed back into
    # ranking, they only make the response self-describing.
    creator_id: str = Field(alias="creatorId")
    content_popularity_score: float = Field(alias="contentPopularityScore")
    content_age_hours: float = Field(alias="contentAgeHours")
    creator_followed: bool = Field(alias="creatorFollowed")
    already_seen: bool = Field(alias="alreadySeen")
    title: str | None = None
    hashtags: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    subgenres: list[str] = Field(default_factory=list)
    # Finalization spec Decision 16: same pass-through convention as the fields above -- these
    # never feed scoring/ranking (see app.ml.reranker's search_relevance/social_relevance/
    # cold_start_context_relevance, which read them off the request-supplied candidate object
    # directly), they only make the response self-describing.
    language: str | None = None
    regions: list[str] = Field(default_factory=list)
    candidate_source: str | None = Field(None, alias="candidateSource")
    # candidateSource/reason observability audit: RMS's own local bucket label
    # (TRENDING/NEW_CONTENT/PREFERRED_CATEGORY/etc.) for a locally-generated candidate --
    # see app.schemas.recommendation_schemas.Candidate.local_bucket_source's own docstring
    # for exactly how this differs from candidateSource above. None for explicit/REAL-mode/
    # Two-Tower candidates, which were never sourced from a local SOURCE_ORDER bucket.
    local_bucket_source: str | None = Field(None, alias="localBucketSource")


class RecommendationResponse(BaseModel):
    """RMS real-data integration prep: this response represents RANKED only -- RMS scored and
    ordered these candidates. It is NOT a record that the caller actually SERVED any of them in
    a feed slate (a Feed Backend decision, made after this response is returned -- position,
    truncation, and slate composition are entirely out of RMS's hands), and it is NOT proof of
    an IMPRESSED event (a client-confirmed exposure, further downstream still, and outside RMS
    entirely). Likewise, the candidates scored here were RETRIEVED by whatever supplied
    `request.candidates` (or, when enabled, Two-Tower retrieval) -- a step RMS consumes, not
    one it owns. RANKED != SERVED != IMPRESSED != RETRIEVED: never label this response, or any
    persisted record of it, as a confirmed impression."""
    model_config = ConfigDict(populate_by_name=True)
    user_id: str = Field(alias="userId")
    model_version: str = Field(alias="modelVersion")
    strategy: str
    interaction_count: int = Field(alias="interactionCount")
    recommendations: list[RecommendationItem]


class LiveRecommendationItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    stream_id: str = Field(alias="streamId")
    category: str
    score: float
    rank: int
    # See RecommendationItem.reason -- same heuristic-not-causal contract, same explanation().
    reason: str = Field(
        description=(
            "Heuristic display label naming a plausible contributing signal for this "
            "recommendation -- NOT causal model attribution. See RecommendationItem.reason."
        ),
    )


class LiveRecommendationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    user_id: str = Field(alias="userId")
    model_version: str = Field(alias="modelVersion")
    # Lifecycle/strategy observability (LIVE lifecycle audit): same COLD_START/HYBRID/
    # PERSONALISED_ML tiers and threshold as RecommendationResponse.strategy (reused verbatim,
    # never a second LIVE-specific rule) -- see app.services.live_recommendation_service.
    # recommend_live's own comment for exactly how the evidence count is derived (real LIVE
    # session count, falling back to real VIDEO interaction evidence when this user has none).
    strategy: str
    interaction_count: int = Field(alias="interactionCount")
    recommendations: list[LiveRecommendationItem]


# ------------------------------------------------------------------------------- model control

class ModelStatusResponse(BaseModel):
    """Wired with response_model_exclude_none=True on its routes (see training_routes.py) to
    reproduce the existing behavior exactly: MISSING returns only {"status": "MISSING"} with
    no other keys at all, not null-valued ones."""
    model_config = ConfigDict(populate_by_name=True)
    status: str
    model_version: str | None = Field(None, alias="modelVersion")
    selected_model: str | None = Field(None, alias="selectedModel")
    model_family: str | None = Field(None, alias="modelFamily")
    trained_at: str | None = Field(None, alias="trainedAt")
    error_code: str | None = Field(None, alias="errorCode")
    message: str | None = None


class ModelMetricsSummaryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    model_version: str | None = Field(None, alias="modelVersion")
    selected_model: str | None = Field(None, alias="selectedModel")
    # "classifier" or "ranker" (app.ml.model_store._model_family_of) -- tells a caller whether
    # `metrics` below is classifier-shaped (prAuc/rocAuc/f1Score/...) or ranker-shaped
    # (ndcgByDifficulty/criticalPassRate/...); see app.ml.artifact_metadata.
    model_family: str | None = Field(None, alias="modelFamily")
    trained_at: str | None = Field(None, alias="trainedAt")
    training_samples: int | None = Field(None, alias="trainingSamples")
    test_samples: int | None = Field(None, alias="testSamples")
    decision_threshold: float | None = Field(None, alias="decisionThreshold")
    training_duration_seconds: float | None = Field(None, alias="trainingDurationSeconds")
    metrics: dict = Field(default_factory=dict)


class ModelVersionSlot(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    model_version: str | None = Field(None, alias="modelVersion")
    selected_model: str | None = Field(None, alias="selectedModel")
    model_family: str | None = Field(None, alias="modelFamily")
    trained_at: str | None = Field(None, alias="trainedAt")
    promoted_at: str | None = Field(None, alias="promotedAt")
    rolled_back_at: str | None = Field(None, alias="rolledBackAt")


class ModelVersionsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    active: ModelVersionSlot | None = None
    previous: ModelVersionSlot | None = None
    candidate: ModelVersionSlot | None = None


class TrainingJobError(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    error: str | None = None
    message: str | None = None


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    job_id: str = Field(alias="jobId")
    job_type: str = Field(alias="jobType")
    status: str
    result: dict | None = None
    error: TrainingJobError | None = None
    created_at: str | None = Field(None, alias="createdAt")
    updated_at: str | None = Field(None, alias="updatedAt")
    requested_at: str | None = Field(None, alias="requestedAt")
    started_at: str | None = Field(None, alias="startedAt")
    finished_at: str | None = Field(None, alias="finishedAt")


# ------------------------------------------------------------------------------------- errors

class ErrorDetails(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Matches app.main._error_body()'s output exactly: `error`/`message` are the legacy
    flat fields every route has always returned; `errorDetails` mirrors them in the additive
    nested form; `requestId` is always present."""
    model_config = ConfigDict(populate_by_name=True)
    error: str
    message: str
    error_details: ErrorDetails = Field(alias="errorDetails")
    request_id: str = Field(alias="requestId")
