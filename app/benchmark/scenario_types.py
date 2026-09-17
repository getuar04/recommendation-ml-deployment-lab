"""Typed scenario/result definitions for the ranking benchmark (app.benchmark).

`BenchmarkCandidate.relevance` is a benchmark-only graded-relevance judgment (see
app.benchmark.metrics) -- a completely separate concept from the model's binary training
target (app.ml.feature_builder.target_for). It is never sent into `Candidate`'s ML-facing
fields and never reaches `FeatureHistory.features()`/the model's feature vector; it exists
purely so this module can score the model's/reranker's OUTPUT against a known-correct answer.
See tests/test_benchmark_leakage.py for the explicit protection test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

# SOCIAL: diagnostic-only, like TEMPORAL/LOCALIZATION -- never included in
# app.benchmark.scenarios.all_named_scenarios()'s return, and app.ml.candidate_evaluation.
# SELECTION_DIFFICULTIES (the tuple that actually gates model eligibility/selection) is its
# own independent literal, unaffected by this tuple. Adding a value here cannot change what a
# model is trained/selected on.
DIFFICULTIES = ("EASY", "MEDIUM", "HARD", "ADVERSARIAL", "TEMPORAL", "LOCALIZATION", "SOCIAL")
Difficulty = Literal["EASY", "MEDIUM", "HARD", "ADVERSARIAL", "TEMPORAL", "LOCALIZATION", "SOCIAL"]


@dataclass(frozen=True)
class HistoryEvent:
    """One raw interaction event used to seed a scenario's user history -- becomes a real
    `app.db.models.Interaction` row (plus, if it carries semantic metadata, a real `Content`
    row), so the benchmark builds history through the exact same point-in-time
    `FeatureHistory.update()` path production training/serving both use. `when` is a
    timedelta relative to the scenario's `reference_timestamp` (negative = in the past),
    never an absolute wall-clock time, so a scenario is reproducible regardless of when the
    benchmark actually runs.
    """
    content_id: str
    creator_id: str
    category: str
    event_type: str
    watch_percentage: float
    when: timedelta
    liked: bool = False
    shared: bool = False
    favorited: bool = False
    creator_followed: bool = False
    hashtags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    subgenres: list[str] = field(default_factory=list)
    title: str | None = None


@dataclass(frozen=True)
class BenchmarkCandidate:
    """One candidate in a scenario's pool. Every field except `relevance`/`note` maps 1:1 to
    `app.schemas.recommendation_schemas.Candidate` and is sent, verbatim, into the real
    request the benchmark scores through `recommendation_service.recommend()`.

    `relevance`: benchmark-only graded truth (see module docstring) -- 0-4, higher is better;
    see app.benchmark.metrics.RELEVANCE_SCALE for the exact meaning of each grade.
    """
    content_id: str
    category: str
    creator_id: str
    relevance: int
    content_popularity_score: float = 0.5
    content_age_hours: float = 5.0
    already_seen: bool = False
    creator_followed: bool = False
    hashtags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    subgenres: list[str] = field(default_factory=list)
    title: str | None = None
    note: str = ""
    # Optional related-user provenance (app.schemas.recommendation_schemas.Candidate's own
    # candidateSource/socialContext) -- None for every existing scenario (byte-identical
    # request as before this field existed); a SOCIAL/COLLABORATIVE benchmark scenario sets
    # both. `social_context`, when given, is the same plain dict shape the real Pydantic
    # SocialContext accepts (interestSimilarity/relationshipStrength/sourceUserEngagement/
    # mutualFollow) so this module never imports app.schemas directly (kept dependency-light,
    # matching every other field here being a plain type).
    candidate_source: str | None = None
    social_context: dict[str, float | bool] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.relevance <= 4:
            raise ValueError(f"relevance must be in [0, 4] (got {self.relevance!r}) for candidate {self.content_id!r}")


@dataclass(frozen=True)
class RelativeConstraint:
    """A single ordering requirement: candidate `higher` must outrank/outscore candidate
    `lower`. Never an exact expected score -- always a relative comparison, optionally with a
    minimum margin, matching how a real production ranking policy is actually validated.

    `critical`, when true, marks a constraint whose failure represents a real product-quality
    problem (e.g. "explicit rejection outranks a strong relevant match") that must be
    surfaced separately from the aggregate benchmark score -- see app.benchmark.runner.
    """
    name: str
    higher: str
    lower: str
    min_margin: float = 0.0
    critical: bool = False
    description: str = ""


@dataclass(frozen=True)
class BenchmarkScenario:
    scenario_id: str
    difficulty: Difficulty
    description: str
    user_id: str
    history: list[HistoryEvent]
    candidates: list[BenchmarkCandidate]
    constraints: list[RelativeConstraint]
    limit: int | None = None  # defaults to len(candidates) -- every candidate should be ranked.

    def __post_init__(self) -> None:
        ids = [c.content_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError(f"scenario {self.scenario_id!r}: duplicate candidate contentId in {ids}")
        known = set(ids)
        for constraint in self.constraints:
            for content_id in (constraint.higher, constraint.lower):
                if content_id not in known:
                    raise ValueError(
                        f"scenario {self.scenario_id!r}: constraint {constraint.name!r} references "
                        f"unknown candidate {content_id!r}"
                    )

    @property
    def effective_limit(self) -> int:
        return self.limit or len(self.candidates)


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    critical: bool
    description: str
    passed: bool
    margin: float
    left_score: float
    right_score: float
    stage: str  # "raw" | "reranked"


@dataclass(frozen=True)
class StageResult:
    """One ranking stage's (raw model, or final reranked) evaluation for one scenario."""
    ranking: list[str]  # content_id, best-first
    scores_by_content_id: dict[str, float]
    metrics: dict[str, float]
    constraint_results: list[ConstraintResult]

    @property
    def critical_constraints_total(self) -> int:
        return sum(1 for c in self.constraint_results if c.critical)

    @property
    def critical_constraints_passed(self) -> int:
        return sum(1 for c in self.constraint_results if c.critical and c.passed)

    @property
    def constraints_total(self) -> int:
        return len(self.constraint_results)

    @property
    def constraints_passed(self) -> int:
        return sum(1 for c in self.constraint_results if c.passed)


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    difficulty: Difficulty
    algorithm: str
    raw: StageResult
    reranked: StageResult
    reranker_improved: list[str]  # constraint names that failed raw but passed reranked
    reranker_degraded: list[str]  # constraint names that passed raw but failed reranked
    reranker_neutral: list[str]   # constraint names whose pass/fail status did not change
