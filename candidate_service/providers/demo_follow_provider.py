"""Local, deterministic, no-network stand-in for a real Follow Service. Reuses the existing
demo user identities (scripts.seed_demo_users) so this package's demo aligns with the rest of
the project's demo ecosystem rather than inventing a second set of user IDs.

Fixed relationships, target = demo User A:
    A <-> B  mutual follow
    A  -> C  one-way (A follows C; C does not follow back)
    A  -- E  no relationship at all (used for the COLLABORATIVE, similarity-only scenario)
    A  -- F  no relationship at all (used for the low-similarity exclusion scenario)
"""
from __future__ import annotations

from datetime import datetime, timezone

from candidate_service.domain.models import FollowRelation
from scripts.seed_demo_users import build_all_users

_NOW = datetime.now(timezone.utc)
_USERS = build_all_users(_NOW)
# Only the deterministic uuid5 user id (index 0) is read from build_all_users' result -- the
# per-user EventBuilder (index 1, whose event timestamps genuinely are `_NOW`-relative) is
# never used here, so this module's own identities stay deterministic across runs despite the
# wall-clock `_NOW` above.

USER_A = _USERS["A"][0]  # target
USER_B = _USERS["B"][0]  # mutual follow
USER_C = _USERS["C"][0]  # one-way follow (A follows C)
USER_E = _USERS["E"][0]  # no relationship -- collaborative candidate

_RELATIONS_BY_TARGET: dict[str, list[FollowRelation]] = {
    USER_A: [
        FollowRelation(user_id=USER_B, follows_target=True, followed_by_target=True),
        FollowRelation(user_id=USER_C, follows_target=False, followed_by_target=True),
    ],
}


class DemoFollowRelationsProvider:
    def related_users(self, user_id: str) -> list[FollowRelation]:
        return list(_RELATIONS_BY_TARGET.get(user_id, []))
