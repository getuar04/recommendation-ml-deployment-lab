"""Versioned, explicit experiment definitions (spec §45). All three are entirely synthetic
-- `synthetic=True` on every definition, never omitted or defaulted -- and generation
parameters are *targets*, not guarantees: the realized class balance of a generated dataset
is measured after generation (see app.experiments.dataset_summary), never assumed from these
parameters. Nothing here is a claim about real user behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.experiments.validation import validate_identifier

DEFAULT_CATEGORIES = ["FOOD", "SPORT", "MUSIC", "TECH", "GAMING", "TRAVEL", "COMEDY", "NEWS", "FASHION", "FITNESS"]


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    dataset_version: str
    description: str
    synthetic: bool
    seed: int
    users: int
    contents: int
    creators: int
    interactions: int
    imbalance_note: str
    categories: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))

    def __post_init__(self) -> None:
        validate_identifier(self.experiment_id, field="experiment_id")
        validate_identifier(self.dataset_version, field="dataset_version")
        if not self.synthetic:
            raise ValueError("All experiment definitions in this repository are synthetic; synthetic=False is not supported.")
        if self.users <= 0 or self.contents <= 0 or self.creators <= 0 or self.interactions <= 0:
            raise ValueError("users/contents/creators/interactions must all be positive.")


SMALL_BALANCED = ExperimentDefinition(
    experiment_id="small-balanced-v1",
    dataset_version="small-balanced-v1",
    description=(
        "Fast, complete-pipeline verification. Small synthetic dataset, generation "
        "parameters tuned toward a roughly balanced label split. Entirely synthetic."
    ),
    synthetic=True,
    seed=20260801,
    users=25,
    contents=40,
    creators=8,
    # 1200 (-> 647 labeled rows after point-in-time labeling) was too marginal for
    # LogisticRegression to reliably clear every mandatory behavioral gate in
    # app.ml.eligibility -- confirmed by direct measurement, not assumed. 3000 (-> ~1483
    # labeled rows) is the smallest tested interaction count, at this same
    # users/contents/creators scale, where every gate passes; entity counts are left
    # unchanged so the scenario stays "small", only interaction density increased.
    interactions=3000,
    imbalance_note="Generation targets a roughly balanced positive/negative split; the realized ratio is measured, not assumed.",
    categories=["FOOD", "SPORT", "MUSIC", "TECH", "GAMING"],
)

MEDIUM_REALISTIC_SYNTHETIC = ExperimentDefinition(
    experiment_id="medium-realistic-synthetic-v1",
    dataset_version="medium-realistic-synthetic-v1",
    description=(
        "Normal PoC-scale synthetic behavior: more users, content, creators, and "
        "categories than the small scenario, with moderate class imbalance. Despite the "
        "name 'realistic', this is still entirely synthetic data, not real user behavior."
    ),
    synthetic=True,
    seed=20260802,
    users=150,
    contents=350,
    creators=25,
    interactions=9000,
    imbalance_note="Generation targets moderate imbalance (skewed toward negative/neutral outcomes); the realized ratio is measured, not assumed.",
)

LARGE_IMBALANCED_SYNTHETIC = ExperimentDefinition(
    experiment_id="large-imbalanced-synthetic-v1",
    dataset_version="large-imbalanced-synthetic-v1",
    description=(
        "Scale and class-imbalance stress test: larger interaction volume, longer "
        "synthetic history, more cold-start (low-interaction) users, and generation "
        "parameters tuned toward strong imbalance. Entirely synthetic."
    ),
    synthetic=True,
    seed=20260803,
    users=400,
    contents=800,
    creators=60,
    interactions=30000,
    imbalance_note="Generation targets strong imbalance and many cold-start users; the realized ratio is measured, not assumed.",
)

DEFINITIONS: dict[str, ExperimentDefinition] = {
    "small_balanced": SMALL_BALANCED,
    "medium_realistic_synthetic": MEDIUM_REALISTIC_SYNTHETIC,
    "large_imbalanced_synthetic": LARGE_IMBALANCED_SYNTHETIC,
}
