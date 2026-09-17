from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import RECOMMENDATION_MAX_CANDIDATES
from app.schemas.limits import (
    CANDIDATE_LIMIT_DEFAULT,
    CANDIDATE_LIMIT_MAX,
    CANDIDATE_LIMIT_MIN,
    LIVE_CANDIDATE_LIMIT_DEFAULT,
    LIVE_CANDIDATE_LIMIT_MAX,
    LIVE_CANDIDATE_LIMIT_MIN,
)
from app.schemas.live_schemas import LiveCandidate


def _reject_blank_user_id(value: str) -> str:
    if not value.strip():
        raise ValueError("userId must not be blank")
    return value


class CandidateGenerationRequest(BaseModel):
    """POST /candidates/generate. userId is required and non-empty: anonymous/cold-start
    candidate generation is not currently supported by this endpoint."""
    model_config = ConfigDict(populate_by_name=True)
    user_id: str = Field(alias="userId", min_length=1)
    limit: int = Field(CANDIDATE_LIMIT_DEFAULT, ge=CANDIDATE_LIMIT_MIN, le=CANDIDATE_LIMIT_MAX)

    _validate_user_id = field_validator("user_id")(_reject_blank_user_id)


class LiveCandidateGenerationRequest(BaseModel):
    """POST /candidates/generate/live. userId is required and non-empty."""
    model_config = ConfigDict(populate_by_name=True)
    user_id: str = Field(alias="userId", min_length=1)
    limit: int = Field(LIVE_CANDIDATE_LIMIT_DEFAULT, ge=LIVE_CANDIDATE_LIMIT_MIN, le=LIVE_CANDIDATE_LIMIT_MAX)
    streams: list[LiveCandidate] = Field(default_factory=list, max_length=RECOMMENDATION_MAX_CANDIDATES)

    _validate_user_id = field_validator("user_id")(_reject_blank_user_id)
