"""Candidate data-adapter boundary (dual-mode integration).

LOCAL mode (default): explicit `request.candidates`, when the caller supplies any, go
straight to the ranker unchanged. An empty/omitted `request.candidates` no longer means
"score nothing" -- it now falls back to this service's own local VIDEO candidate generation
(`app.services.providers.video_candidate_provider.generate_video_candidates`, the same
function `POST /candidates/generate` uses), never an HTTP self-call and never a second,
duplicated implementation of that logic (see "userId -> internal candidates" integration).

REAL mode's pre-existing behavior is UNCHANGED: when `CANDIDATE_SERVICE_BASE_URL` is
configured, `recommend()` still sources candidates from a real Candidate Service first
(regardless of whether the caller also supplied any), normalizing its response into the SAME
`Candidate` schema (`app.schemas.recommendation_schemas.Candidate`); on failure it falls back
to `request.candidates` if the caller supplied any, else raises `CandidateSourceUnavailable`
-- exactly as before this module gained a local-generation fallback. Local generation is
reached only when REAL mode's Candidate Service branch does not apply at all (LOCAL mode, or
REAL mode with no Candidate Service configured) -- local generation is never itself a
fallback FROM a failed Candidate Service call, and never calls one.
"""
from __future__ import annotations

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import (
    CANDIDATE_SERVICE_BASE_URL,
    CANDIDATE_SERVICE_GENERATE_PATH,
    CANDIDATE_SERVICE_TIMEOUT_MS,
    RECOMMENDATION_DATA_MODE,
)
from app.core.logging import logger
from app.schemas.recommendation_schemas import Candidate
from app.services.providers.video_candidate_provider import generate_video_candidates
from app.services.service_clients import UpstreamServiceError, call_json

__all__ = ["CandidateSourceUnavailable", "resolve"]


class CandidateSourceUnavailable(Exception):
    """Raised only when REAL mode's Candidate Service call fails AND the caller supplied no
    request.candidates to fall back to -- a clear, observable dependency failure rather than
    silently scoring zero candidates."""


def _parse_candidates(body: object) -> list[Candidate]:
    raw = body.get("candidates") if isinstance(body, dict) else None
    if not isinstance(raw, list):
        raise UpstreamServiceError(
            "CandidateService", "MALFORMED_RESPONSE", "Candidate Service response missing a 'candidates' list",
        )
    candidates: list[Candidate] = []
    for item in raw:
        try:
            candidates.append(Candidate.model_validate(item))
        except ValidationError:
            # One malformed candidate must not fail the whole batch -- consistent with
            # recommend()'s own per-candidate tolerance for blank content_id/creator_id/category.
            continue
    return candidates


def _fetch_from_service(user_id: str, limit: int) -> list[Candidate]:
    """Targets `POST {CANDIDATE_SERVICE_BASE_URL}/api/v1/candidates/generate` with the same
    request/response shape this project's own local candidate-generation stand-in
    (`app.api.candidate_routes.generate`) already uses -- the one Candidate Service contract
    already established in this repository.

    KNOWN GAP -- confirmed, not assumed: a real-contract search of every sibling repository in
    the local Soft Dome workspace found NO "Candidate Service" repository, OpenAPI doc, or
    Postman collection anywhere either. This is this project's own placeholder request/
    response shape, not a verified external API.

    VIDEO scope (this service's current production scope, unchanged): deliberately targets
    `/candidates/generate` -- this repo's VIDEO-only local stand-in -- never
    `/candidates/generate/live` (the separate LIVE stand-in, app.schemas.live_schemas). The
    explicit `contentType: "VIDEO"` request field below is metadata on OUR OWN outgoing
    request (safe for an unrecognized real service to ignore), not a claim about a real
    Candidate Service response field -- if a real Candidate Service exposes multiple feed
    types, its actual filter parameter name should replace/extend this once confirmed."""
    assert CANDIDATE_SERVICE_BASE_URL is not None  # only called once resolve() has already checked this
    body = call_json(
        service="CandidateService", base_url=CANDIDATE_SERVICE_BASE_URL, path=CANDIDATE_SERVICE_GENERATE_PATH,
        timeout_ms=CANDIDATE_SERVICE_TIMEOUT_MS, method="POST",
        json_body={"userId": user_id, "limit": limit, "contentType": "VIDEO"},
    )
    return _parse_candidates(body)


def resolve(db: Session, request, *, seen_content_ids: frozenset[str] | None = None) -> tuple[list, str]:
    """Returns (candidates, candidateSourceLabel); label is one of
    REQUEST/CANDIDATE_SERVICE/FALLBACK/LOCAL_GENERATION.

    REAL mode with a configured Candidate Service: UNCHANGED from before this module gained
    local generation -- always tries the Candidate Service first (even when the caller also
    supplied candidates), falls back to `request.candidates` on failure if the caller
    supplied any (label FALLBACK), else raises `CandidateSourceUnavailable`. `db` is not used
    on this branch.

    Otherwise (LOCAL mode, the default; or REAL mode with no Candidate Service configured):
    explicit non-empty `request.candidates` wins outright (label REQUEST); an empty/omitted
    list generates from local RMS state instead (label LOCAL_GENERATION) -- never an outbound
    call, never Candidate Service as a fallback.

    `seen_content_ids` (default None, additive): only meaningful on the LOCAL_GENERATION
    branch. When the caller (`recommend()`) passes its own already-resolved authoritative
    already-seen set, local generation excludes those ids from the eligible pool up front
    (`app.services.providers.video_candidate_provider.generate_video_candidates`'s
    `exclude_content_ids`) instead of generating exactly `request.limit` candidates that may
    later be dropped post-hoc with nothing to backfill them. Every existing caller that omits
    this parameter (e.g. `POST /candidates/generate`'s debug route, indirectly, via its own
    direct `generate_video_candidates` call; and any `resolve()` caller that never passes it)
    is unaffected.

    `db=None` (the established offline/demo calling convention -- e.g.
    `scripts/demo_production_recommendation_trace.py`'s `recommend(None, request)`) cannot
    query local state at all: local generation is skipped and `request.candidates` (possibly
    empty) is returned as-is, byte-identical to this module's behavior before local
    generation existed, rather than raising for lack of a session."""
    if RECOMMENDATION_DATA_MODE == "REAL" and CANDIDATE_SERVICE_BASE_URL:
        try:
            candidates = _fetch_from_service(request.user_id, request.limit)
            return candidates, "CANDIDATE_SERVICE"
        except UpstreamServiceError as exc:
            logger.info("Candidate Service unavailable userId=%s reason=%s hasRequestCandidates=%s",
                        request.user_id, exc.reason, bool(request.candidates))
            if request.candidates:
                return request.candidates, "FALLBACK"
            raise CandidateSourceUnavailable(str(exc)) from exc

    if request.candidates or db is None:
        return request.candidates, "REQUEST"
    local_candidates = generate_video_candidates(
        db, request.user_id, limit=request.limit, exclude_content_ids=seen_content_ids,
    )
    return [candidate for candidate, _source in local_candidates], "LOCAL_GENERATION"
