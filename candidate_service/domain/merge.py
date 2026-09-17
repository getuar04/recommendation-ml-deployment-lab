"""Deduplicates candidates reached through more than one related user/source (e.g. content X
strongly engaged by both a SOCIAL friend and a COLLABORATIVE similar user). Demo/V1 policy:
SOCIAL wins over COLLABORATIVE outright (a real relationship is stronger evidence than pure
behavioral similarity); within the same source, the candidate with the strongest bounded
relationship_strength + interest_similarity evidence wins. No unbounded summing across
duplicate sources -- each content_id keeps exactly one SocialCandidate."""
from __future__ import annotations

from candidate_service.domain.models import SocialCandidate


def merge_candidates(candidates: list[SocialCandidate]) -> list[SocialCandidate]:
    best: dict[str, SocialCandidate] = {}
    for candidate in candidates:
        current = best.get(candidate.content.content_id)
        if current is None or _stronger(candidate, current):
            best[candidate.content.content_id] = candidate
    return list(best.values())


def _stronger(candidate: SocialCandidate, current: SocialCandidate) -> bool:
    if candidate.source != current.source:
        return candidate.source == "SOCIAL"
    return (
        candidate.relationship_strength + candidate.interest_similarity
        > current.relationship_strength + current.interest_similarity
    )
