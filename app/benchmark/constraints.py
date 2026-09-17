"""Relative ranking-constraint evaluation (app.benchmark.scenario_types.RelativeConstraint).

A constraint is always a relative comparison between two named candidates' scores in one
ranking stage -- never an exact expected score, and never a comparison against an arbitrary
row. `min_margin` (default 0.0) lets a scenario require the win to be more than infinitesimal
when that matters; a plain 0.0 only requires strict `>`.
"""
from __future__ import annotations

from app.benchmark.scenario_types import ConstraintResult, RelativeConstraint


def evaluate_constraint(
    constraint: RelativeConstraint, scores_by_content_id: dict[str, float], *, stage: str,
) -> ConstraintResult:
    if constraint.higher not in scores_by_content_id or constraint.lower not in scores_by_content_id:
        raise ValueError(
            f"constraint {constraint.name!r} references a candidate not present in this ranking's "
            f"scores ({constraint.higher!r}, {constraint.lower!r})"
        )
    left_score = scores_by_content_id[constraint.higher]
    right_score = scores_by_content_id[constraint.lower]
    margin = left_score - right_score
    passed = margin > constraint.min_margin if constraint.min_margin > 0 else margin > 0
    return ConstraintResult(
        name=constraint.name, critical=constraint.critical, description=constraint.description,
        passed=passed, margin=round(margin, 6), left_score=round(left_score, 6),
        right_score=round(right_score, 6), stage=stage,
    )


def evaluate_constraints(
    constraints: list[RelativeConstraint], scores_by_content_id: dict[str, float], *, stage: str,
) -> list[ConstraintResult]:
    return [evaluate_constraint(constraint, scores_by_content_id, stage=stage) for constraint in constraints]


def classify_reranker_effect(
    raw_results: list[ConstraintResult], reranked_results: list[ConstraintResult],
) -> tuple[list[str], list[str], list[str]]:
    """Compares each constraint's pass/fail status raw vs. reranked. Returns
    (improved, degraded, neutral) constraint names -- improved: failed raw, passed reranked;
    degraded: passed raw, failed reranked; neutral: unchanged either way."""
    raw_by_name = {r.name: r.passed for r in raw_results}
    reranked_by_name = {r.name: r.passed for r in reranked_results}
    improved, degraded, neutral = [], [], []
    for name, raw_passed in raw_by_name.items():
        reranked_passed = reranked_by_name.get(name)
        if reranked_passed is None:
            continue
        if not raw_passed and reranked_passed:
            improved.append(name)
        elif raw_passed and not reranked_passed:
            degraded.append(name)
        else:
            neutral.append(name)
    return improved, degraded, neutral
