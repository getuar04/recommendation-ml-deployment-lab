"""Versioned, explicit definitions for the "same algorithm, different dataset" comparative
experiment: four synthetic datasets with deliberately different category-interest
distributions, trained with the *exact same* algorithm, hyperparameters, random seed,
training pipeline, feature engineering, split strategy, calibration, and threshold-selection
logic -- see README "Comparative Experiments". The only intentional variable across the four
is `category_weights` (which drives `app.experiments.comparative_dataset_generation`).

Deliberately separate from `app.experiments.definitions` (the pre-existing small/medium/large
*scale* experiments): those exist to stress-test pipeline correctness at different data
volumes and explicitly allow automatic model selection to pick either candidate; these four
exist to demonstrate learned-behavior differences from data composition alone, which requires
the algorithm to be locked identically across all four (see `algorithm` below and
`app.ml.trainer._candidate_models`'s `only` parameter). Nothing here is a claim about real
user behavior -- every definition is entirely synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.experiments.definitions import DEFAULT_CATEGORIES
from app.experiments.validation import validate_identifier

# The one algorithm locked across every comparative experiment (spec: "Use exactly one fixed
# algorithm for all experiments... Do NOT allow automatic model selection to choose a
# different algorithm for each dataset"). Threaded into app.ml.trainer.train_models via
# app.core.config.TRAINING_ALGORITHM_LOCK (see .env.experiment-*.env), never hard-coded into
# the trainer itself -- normal (non-comparative) training keeps comparing both candidates.
LOCKED_ALGORITHM = "LogisticRegression"

# One fixed seed shared by every comparative definition -- the Core Experiment Rule requires
# the random seed to remain identical across every experiment; only category_weights differs.
COMPARATIVE_SEED = 20260805

_ALLOWED_ALGORITHMS = {"LogisticRegression", "RandomForestClassifier"}
_WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ComparativeExperimentDefinition:
    experiment_id: str
    dataset_version: str
    description: str
    synthetic: bool
    seed: int
    algorithm: str
    users: int
    contents: int
    creators: int
    interactions: int
    dominant_category: str
    category_weights: dict[str, float]
    imbalance_note: str
    category_mapping_note: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.experiment_id, field="experiment_id")
        validate_identifier(self.dataset_version, field="dataset_version")
        if not self.synthetic:
            raise ValueError("All comparative experiment definitions are synthetic; synthetic=False is not supported.")
        if self.algorithm not in _ALLOWED_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(_ALLOWED_ALGORITHMS)} (got {self.algorithm!r})")
        if self.users <= 0 or self.contents <= 0 or self.creators <= 0 or self.interactions <= 0:
            raise ValueError("users/contents/creators/interactions must all be positive.")
        if not self.category_weights:
            raise ValueError("category_weights must not be empty.")
        unsupported = set(self.category_weights) - set(DEFAULT_CATEGORIES)
        if unsupported:
            raise ValueError(f"category_weights uses unsupported categories: {sorted(unsupported)} (supported: {DEFAULT_CATEGORIES})")
        if any(weight <= 0 for weight in self.category_weights.values()):
            raise ValueError("category_weights values must all be positive.")
        total = sum(self.category_weights.values())
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"category_weights must sum to 1.0 (got {total}).")
        if self.dominant_category != "NONE" and self.dominant_category not in self.category_weights:
            raise ValueError(f"dominant_category {self.dominant_category!r} must be 'NONE' or a key of category_weights.")


COMPARATIVE_SPORT = ComparativeExperimentDefinition(
    experiment_id="comparative-sport-v1",
    dataset_version="comparative-sport-v1",
    description=(
        "Comparative experiment A: users with strong, concentrated interest in SPORT "
        "content. Entirely synthetic -- generation targets are targets, not guarantees; the "
        "realized distribution is always measured after generation."
    ),
    synthetic=True,
    seed=COMPARATIVE_SEED,
    algorithm=LOCKED_ALGORITHM,
    users=60,
    contents=100,
    creators=12,
    interactions=4000,
    dominant_category="SPORT",
    category_weights={"SPORT": 0.8, "FITNESS": 0.1, "TRAVEL": 0.1},
    imbalance_note="Generation targets ~80% SPORT / ~10% FITNESS / ~10% TRAVEL interaction share; the realized ratio is measured, not assumed.",
)

COMPARATIVE_ENTERTAINMENT = ComparativeExperimentDefinition(
    experiment_id="comparative-entertainment-v1",
    dataset_version="comparative-entertainment-v1",
    description=(
        "Comparative experiment B: users with strong, concentrated interest in movie/"
        "entertainment-style content. This repository's supported category set "
        "(app.experiments.definitions.DEFAULT_CATEGORIES) has no MOVIES/ENTERTAINMENT "
        "category -- COMEDY is used as the closest existing supported category rather than "
        "inventing an unsupported value; see `category_mapping_note`. Entirely synthetic."
    ),
    synthetic=True,
    seed=COMPARATIVE_SEED,
    algorithm=LOCKED_ALGORITHM,
    users=60,
    contents=100,
    creators=12,
    interactions=4000,
    dominant_category="COMEDY",
    category_weights={"COMEDY": 0.8, "MUSIC": 0.1, "NEWS": 0.1},
    imbalance_note="Generation targets ~80% COMEDY (entertainment proxy) / ~10% MUSIC / ~10% NEWS interaction share; the realized ratio is measured, not assumed.",
    category_mapping_note="MOVIES/ENTERTAINMENT is not a supported category in this repository; COMEDY is used as the closest existing supported category.",
)

COMPARATIVE_MUSIC = ComparativeExperimentDefinition(
    experiment_id="comparative-music-v1",
    dataset_version="comparative-music-v1",
    description=(
        "Comparative experiment C: users with strong, concentrated interest in MUSIC "
        "content. Entirely synthetic."
    ),
    synthetic=True,
    seed=COMPARATIVE_SEED,
    algorithm=LOCKED_ALGORITHM,
    users=60,
    contents=100,
    creators=12,
    interactions=4000,
    dominant_category="MUSIC",
    category_weights={"MUSIC": 0.8, "COMEDY": 0.1, "FASHION": 0.1},
    imbalance_note="Generation targets ~80% MUSIC / ~10% COMEDY / ~10% FASHION interaction share; the realized ratio is measured, not assumed.",
)

COMPARATIVE_BALANCED = ComparativeExperimentDefinition(
    experiment_id="comparative-balanced-v1",
    dataset_version="comparative-balanced-v1",
    description=(
        "Comparative experiment D: mixed user behaviour without one strongly dominant "
        "category -- five supported categories at an equal ~20% interaction share each. "
        "Entirely synthetic."
    ),
    synthetic=True,
    seed=COMPARATIVE_SEED,
    algorithm=LOCKED_ALGORITHM,
    users=60,
    contents=100,
    creators=12,
    interactions=4000,
    dominant_category="NONE",
    category_weights={"SPORT": 0.2, "MUSIC": 0.2, "TECH": 0.2, "FOOD": 0.2, "GAMING": 0.2},
    imbalance_note="Generation targets an equal ~20% interaction share across SPORT/MUSIC/TECH/FOOD/GAMING; the realized ratio is measured, not assumed.",
)


COMPARATIVE_DEFINITIONS: dict[str, ComparativeExperimentDefinition] = {
    "sport": COMPARATIVE_SPORT,
    "entertainment": COMPARATIVE_ENTERTAINMENT,
    "music": COMPARATIVE_MUSIC,
    "balanced": COMPARATIVE_BALANCED,
}
