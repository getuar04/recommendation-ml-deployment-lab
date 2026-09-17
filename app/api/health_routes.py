"""Lightweight liveness/readiness probe: is this process alive and can it reach its
database. Deliberately does NOT report model readiness -- that is a separate concern with
its own dedicated, already-comprehensive endpoints (GET /model/status, GET /model/status/live,
see app.api.training_routes) that distinguish READY/MISSING/INCOMPATIBLE/CORRUPTED/BUSY per
model type. Coupling model artifact state into /health would make a standard infrastructure
liveness probe (load balancer, container orchestrator) restart/evict a perfectly healthy
process just because a model has not been trained yet, or fail training-recovery workflows
that specifically need the service to stay reachable while a bad artifact is being replaced.
"""
import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import (
    APP_ENV,
    CANDIDATE_SERVICE_BASE_URL,
    CANDIDATE_SERVICE_TIMEOUT_MS,
    KAFKA_BEHAVIOR_ENABLED,
    RECOMMENDATION_DATA_MODE,
    SERVICE_BUILD_TIME,
    SERVICE_COMMIT,
    SERVICE_NAME,
    SERVICE_VERSION,
    UBS_BASE_URL,
    UBS_TIMEOUT_MS,
)
from app.db.database import engine
from app.services import kafka_behavior_consumer

router = APIRouter(tags=["health"])


_HEALTH_EXAMPLE = {
    "status": "ok", "service": "recommendation-ml-service", "version": "1.0.0",
    "environment": "local", "commit": "unknown", "buildTime": "unknown",
    "message": "Health check completed.",
    "dependencies": {"postgres": "connected"},
}
_HEALTH_DEGRADED_EXAMPLE = {
    **_HEALTH_EXAMPLE, "status": "degraded", "dependencies": {"postgres": "disconnected"},
}

# Dual-mode REAL-service dependency visibility (see app/services/providers/). Deliberately
# NOT folded into overall `status`/HTTP code -- a REAL-mode dependency being unavailable is
# reported here, in `dependencies`, exactly like postgres already is, but it never flips this
# liveness probe to 503 on its own (see this module's own docstring: /health stays a cheap
# "is this process alive and can it reach ITS OWN database" check, not a full dependency-graph
# probe -- an orchestrator must not restart/evict an otherwise-healthy process just because an
# optional real-service integration is degraded). LOCAL mode (the default) reports NONE of
# these keys at all -- "dependencies" stays exactly {"postgres": ...}, unchanged from before
# this dual-mode integration existed, so LOCAL mode never looks unhealthy (or even different)
# merely because REAL-mode dependencies are disabled.
_REAL_DEPENDENCY_HEALTH_TIMEOUT_S = 0.5


def _real_dependency_status(base_url: str | None, timeout_ms: int) -> str:
    """DISABLED (not REAL mode, or not configured) / HEALTHY / DEGRADED (reachable, non-2xx)
    / UNAVAILABLE (unreachable/timeout). A short, hard-capped timeout regardless of the
    configured client timeout -- this is a liveness probe, not a request path, and must never
    make GET /health itself slow."""
    if RECOMMENDATION_DATA_MODE != "REAL" or not base_url:
        return "DISABLED"
    try:
        timeout_s = min(timeout_ms / 1000, _REAL_DEPENDENCY_HEALTH_TIMEOUT_S)
        response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=timeout_s)
        return "HEALTHY" if response.status_code < 400 else "DEGRADED"
    except Exception:  # noqa: BLE001 -- liveness probe: any network failure (timeout/connect/etc.) means UNAVAILABLE, never a raised exception
        return "UNAVAILABLE"


def _kafka_status() -> str:
    if not KAFKA_BEHAVIOR_ENABLED:
        return "DISABLED"
    return "HEALTHY" if kafka_behavior_consumer.STATUS.get("connected") else "UNAVAILABLE"


@router.get("/health", responses={
    200: {"content": {"application/json": {"example": _HEALTH_EXAMPLE}}},
    503: {"content": {"application/json": {"example": _HEALTH_DEGRADED_EXAMPLE}}},
})
def health():
    """`status` is "degraded" (HTTP 503) only if PostgreSQL is unreachable; "ok" (HTTP 200)
    otherwise. The database check is a single real `SELECT 1` against the existing shared
    engine (app.db.database.engine) -- no second connection implementation, no expensive
    query, no model loading/training/inference/drift evaluation. Never exposes a database
    URL, model path, or checksum. Example values above are illustrative only, not measured
    results."""
    postgres_connected = True
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 -- liveness probe: any DB connectivity failure means disconnected, never a raised exception
        postgres_connected = False

    dependencies = {"postgres": "connected" if postgres_connected else "disconnected"}
    # LOCAL mode (default): none of these keys are added at all -- see module-level comment
    # above `_real_dependency_status`. Overall `status`/HTTP code is governed by postgres
    # alone, unchanged -- a degraded/unavailable REAL-mode dependency is visible here but
    # never flips this liveness probe to 503 on its own.
    ubs_status = _real_dependency_status(UBS_BASE_URL, UBS_TIMEOUT_MS)
    if ubs_status != "DISABLED":
        dependencies["ubs"] = ubs_status
    candidate_service_status = _real_dependency_status(CANDIDATE_SERVICE_BASE_URL, CANDIDATE_SERVICE_TIMEOUT_MS)
    if candidate_service_status != "DISABLED":
        dependencies["candidateService"] = candidate_service_status
    kafka_status = _kafka_status()
    if kafka_status != "DISABLED":
        dependencies["kafka"] = kafka_status

    payload = {
        "status": "ok" if postgres_connected else "degraded",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "environment": APP_ENV,
        "commit": SERVICE_COMMIT,
        "buildTime": SERVICE_BUILD_TIME,
        "message": "Health check completed.",
        "dependencies": dependencies,
    }
    return JSONResponse(status_code=200 if postgres_connected else 503, content=payload)
