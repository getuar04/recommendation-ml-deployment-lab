"""GET /auth/me -- Task: first isolated JWT/JWKS authentication vertical slice. Proves the
full Bearer-token -> RS256/JWKS verification -> authenticated-identity pipeline end to end
via one deliberately minimal diagnostic endpoint, before any user-facing recommendation
route depends on it -- see app.core.jwt_auth's own module docstring for the full scope/
rollout rationale. No other router in this service gained this dependency in this task.
"""
from fastapi import APIRouter, Depends

from app.core.jwt_auth import get_verified_user_id

router = APIRouter(tags=["auth"])

_AUTH_ME_EXAMPLE = {"authenticated": True, "userId": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
_UNAUTHORIZED_EXAMPLE = {"error": "UNAUTHORIZED", "message": "A valid Bearer token is required.", "requestId": "..."}


@router.get("/auth/me", responses={
    200: {"content": {"application/json": {"example": _AUTH_ME_EXAMPLE}}},
    401: {"content": {"application/json": {"example": _UNAUTHORIZED_EXAMPLE}}},
})
def auth_me(user_id: str = Depends(get_verified_user_id)):
    """Intentionally minimal: never returns the raw token, full JWT claims, email, roles, or
    JWKS details -- see app.core.jwt_auth's own module docstring. `userId` is EXACTLY the
    verified JWT `sub` claim, never a request-supplied/fallback identity source."""
    return {"authenticated": True, "userId": user_id}
