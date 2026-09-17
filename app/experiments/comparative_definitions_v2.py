"""Version 2 of the comparative-experiment definitions (spec: rigorous "same algorithm,
different dataset" comparison). Deliberately separate from `app.experiments.comparative_definitions`
(v1), which stays completely unmodified and importable so v1's already-persisted volumes and
reports remain fully reproducible from source: nothing in this module (or its downstream
callers) ever mutates v1's identifiers, and no v1 experiment_id is reused here.

Every field below is *identical* to the corresponding v1 definition (same seed, same
users/contents/creators/interactions, same category_weights shape) -- only `experiment_id`/
`dataset_version` change, to `*-v2`. This is intentional: v2 exists to fix v1's experimental-
validity issues (fixed generation timestamp, deterministic post-generation class-balance
stratification, a symmetric primary candidate scenario -- see comparative_dataset_generation.py
and fixed_scenario.py) without changing what the "declared training-dataset category
distribution" treatment actually is.
"""
from __future__ import annotations

from app.experiments.comparative_definitions import (
    COMPARATIVE_BALANCED,
    COMPARATIVE_ENTERTAINMENT,
    COMPARATIVE_MUSIC,
    COMPARATIVE_SEED,
    COMPARATIVE_SPORT,
    LOCKED_ALGORITHM,
    ComparativeExperimentDefinition,
)

COMPARATIVE_SPORT_V2 = ComparativeExperimentDefinition(
    experiment_id="comparative-sport-v2",
    dataset_version="comparative-sport-v2",
    description=COMPARATIVE_SPORT.description + " (v2: fixed generation timestamp, deterministic class-balance stratification, symmetric primary candidate scenario.)",
    synthetic=True,
    seed=COMPARATIVE_SPORT.seed,
    algorithm=COMPARATIVE_SPORT.algorithm,
    users=COMPARATIVE_SPORT.users,
    contents=COMPARATIVE_SPORT.contents,
    creators=COMPARATIVE_SPORT.creators,
    interactions=COMPARATIVE_SPORT.interactions,
    dominant_category=COMPARATIVE_SPORT.dominant_category,
    category_weights=dict(COMPARATIVE_SPORT.category_weights),
    imbalance_note=COMPARATIVE_SPORT.imbalance_note,
    category_mapping_note=COMPARATIVE_SPORT.category_mapping_note,
)

COMPARATIVE_ENTERTAINMENT_V2 = ComparativeExperimentDefinition(
    experiment_id="comparative-entertainment-v2",
    dataset_version="comparative-entertainment-v2",
    description=COMPARATIVE_ENTERTAINMENT.description + " (v2: fixed generation timestamp, deterministic class-balance stratification, symmetric primary candidate scenario.)",
    synthetic=True,
    seed=COMPARATIVE_ENTERTAINMENT.seed,
    algorithm=COMPARATIVE_ENTERTAINMENT.algorithm,
    users=COMPARATIVE_ENTERTAINMENT.users,
    contents=COMPARATIVE_ENTERTAINMENT.contents,
    creators=COMPARATIVE_ENTERTAINMENT.creators,
    interactions=COMPARATIVE_ENTERTAINMENT.interactions,
    dominant_category=COMPARATIVE_ENTERTAINMENT.dominant_category,
    category_weights=dict(COMPARATIVE_ENTERTAINMENT.category_weights),
    imbalance_note=COMPARATIVE_ENTERTAINMENT.imbalance_note,
    category_mapping_note=COMPARATIVE_ENTERTAINMENT.category_mapping_note,
)

COMPARATIVE_MUSIC_V2 = ComparativeExperimentDefinition(
    experiment_id="comparative-music-v2",
    dataset_version="comparative-music-v2",
    description=COMPARATIVE_MUSIC.description + " (v2: fixed generation timestamp, deterministic class-balance stratification, symmetric primary candidate scenario.)",
    synthetic=True,
    seed=COMPARATIVE_MUSIC.seed,
    algorithm=COMPARATIVE_MUSIC.algorithm,
    users=COMPARATIVE_MUSIC.users,
    contents=COMPARATIVE_MUSIC.contents,
    creators=COMPARATIVE_MUSIC.creators,
    interactions=COMPARATIVE_MUSIC.interactions,
    dominant_category=COMPARATIVE_MUSIC.dominant_category,
    category_weights=dict(COMPARATIVE_MUSIC.category_weights),
    imbalance_note=COMPARATIVE_MUSIC.imbalance_note,
    category_mapping_note=COMPARATIVE_MUSIC.category_mapping_note,
)

COMPARATIVE_BALANCED_V2 = ComparativeExperimentDefinition(
    experiment_id="comparative-balanced-v2",
    dataset_version="comparative-balanced-v2",
    description=COMPARATIVE_BALANCED.description + " (v2: fixed generation timestamp, deterministic class-balance stratification, symmetric primary candidate scenario.)",
    synthetic=True,
    seed=COMPARATIVE_BALANCED.seed,
    algorithm=COMPARATIVE_BALANCED.algorithm,
    users=COMPARATIVE_BALANCED.users,
    contents=COMPARATIVE_BALANCED.contents,
    creators=COMPARATIVE_BALANCED.creators,
    interactions=COMPARATIVE_BALANCED.interactions,
    dominant_category=COMPARATIVE_BALANCED.dominant_category,
    category_weights=dict(COMPARATIVE_BALANCED.category_weights),
    imbalance_note=COMPARATIVE_BALANCED.imbalance_note,
    category_mapping_note=COMPARATIVE_BALANCED.category_mapping_note,
)

COMPARATIVE_DEFINITIONS_V2: dict[str, ComparativeExperimentDefinition] = {
    "sport": COMPARATIVE_SPORT_V2,
    "entertainment": COMPARATIVE_ENTERTAINMENT_V2,
    "music": COMPARATIVE_MUSIC_V2,
    "balanced": COMPARATIVE_BALANCED_V2,
}

# Re-exported so v2 callers never need to reach into the v1 module directly for these shared,
# version-independent constants.
__all__ = [
    "COMPARATIVE_BALANCED_V2", "COMPARATIVE_DEFINITIONS_V2", "COMPARATIVE_ENTERTAINMENT_V2",
    "COMPARATIVE_MUSIC_V2", "COMPARATIVE_SEED", "COMPARATIVE_SPORT_V2", "LOCKED_ALGORITHM",
]
