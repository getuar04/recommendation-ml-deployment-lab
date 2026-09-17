"""Interest similarity between two users' long-term/recent category-affinity profiles.
Deliberately not session-level: a single session is noisy and short-lived, whereas long-term/
recent affinity is stable enough to compare two different users against each other."""
from __future__ import annotations

import math


def cosine_similarity(vector_a: dict[str, float], vector_b: dict[str, float]) -> float:
    """[0,1]: both inputs are non-negative affinity scores (category -> [0,1]), so cosine
    similarity over them is already bounded in [0,1] without an extra clamp needed for the
    normal case -- the min/max below is defensive only. Zero-vector safe (returns 0.0, never
    divides by zero) and NaN-safe by construction."""
    keys = vector_a.keys() | vector_b.keys()
    if not keys:
        return 0.0
    dot = sum(vector_a.get(key, 0.0) * vector_b.get(key, 0.0) for key in keys)
    norm_a = math.sqrt(sum(value * value for value in vector_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vector_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))
