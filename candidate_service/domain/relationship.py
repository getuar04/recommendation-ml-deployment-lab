"""Demo/V1 relationship-strength policy -- NOT the final production formula. A real Follow
Service-backed implementation may weight this very differently (follow duration, DM-free
interaction frequency signals Follow Service itself exposes, etc.); this is deliberately the
simplest defensible placeholder for a demo: mutual follow is materially stronger evidence than
one-way, no relationship is zero. No message/DM signals -- none exist anywhere in this
codebase and none are invented here.
"""
from __future__ import annotations

MUTUAL_FOLLOW_STRENGTH = 0.9
ONE_WAY_FOLLOW_STRENGTH = 0.5
NO_RELATIONSHIP_STRENGTH = 0.0


def relationship_strength(*, follows_target: bool, followed_by_target: bool) -> float:
    """`follows_target`: the related user follows the target user. `followed_by_target`: the
    target user follows the related user. Both true = mutual follow."""
    if follows_target and followed_by_target:
        return MUTUAL_FOLLOW_STRENGTH
    if follows_target or followed_by_target:
        return ONE_WAY_FOLLOW_STRENGTH
    return NO_RELATIONSHIP_STRENGTH
