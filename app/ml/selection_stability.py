"""Multi-seed selection-stability aggregation (Task 5 spec sections 16-17): a candidate that
wins spectacularly on one seed but fails eligibility on others must not automatically become
the default. Pure, dependency-free aggregation over per-seed results a caller already
collected (see scripts/run_selection_experiment.py) -- this module has no opinion on HOW those
results were produced, only how to summarize and judge them.
"""
from __future__ import annotations

from typing import Any


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return round(mean, 6), round(variance ** 0.5, 6)


def aggregate_algorithm(name: str, per_seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    """`per_seed_records`: one record per evaluated seed for ONE algorithm, each shaped like
    {"eligible": bool, "hardPassed": int, "hardTotal": int, "softPassed": int, "softTotal": int,
     "mediumNdcg": float, "hardNdcg": float, "adversarialNdcg": float,
     "criticalPassRate": float, "prAuc": float, "qualityScore": float, "isWinner": bool}.

    Returns exactly the columns Task 5's required multi-seed experiment table asks for:
    eligibility rate, mean hard/soft gates passed, mean/std of the headline ranking metrics,
    and winner count."""
    n = len(per_seed_records)
    eligible_count = sum(1 for r in per_seed_records if r["eligible"])
    winner_count = sum(1 for r in per_seed_records if r["isWinner"])

    hard_ndcg_mean, hard_ndcg_std = _mean_std([r["hardNdcg"] for r in per_seed_records])
    adversarial_ndcg_mean, _ = _mean_std([r["adversarialNdcg"] for r in per_seed_records])
    medium_ndcg_mean, _ = _mean_std([r["mediumNdcg"] for r in per_seed_records])
    pr_auc_mean, _ = _mean_std([r["prAuc"] for r in per_seed_records])
    critical_rate_mean, _ = _mean_std([r["criticalPassRate"] for r in per_seed_records])
    quality_mean, quality_std = _mean_std([r["qualityScore"] for r in per_seed_records])
    hard_passed_mean, _ = _mean_std([r["hardPassed"] for r in per_seed_records])
    soft_passed_mean, _ = _mean_std([r["softPassed"] for r in per_seed_records])

    return {
        "algorithm": name,
        "seeds": n,
        "eligibleSeeds": eligible_count,
        "eligibilityRate": round(eligible_count / n, 6) if n else 0.0,
        "hardGatesMeanPassed": hard_passed_mean,
        "hardGatesTotal": per_seed_records[0]["hardTotal"] if per_seed_records else 0,
        "softGatesMeanPassed": soft_passed_mean,
        "softGatesTotal": per_seed_records[0]["softTotal"] if per_seed_records else 0,
        "mediumNdcgMean": medium_ndcg_mean,
        "hardNdcgMean": hard_ndcg_mean,
        "hardNdcgStd": hard_ndcg_std,
        "adversarialNdcgMean": adversarial_ndcg_mean,
        "criticalPassRateMean": critical_rate_mean,
        "prAucMean": pr_auc_mean,
        "qualityScoreMean": quality_mean,
        "qualityScoreStd": quality_std,
        "winnerCount": winner_count,
    }


# Task 5 spec section 17: "Choose a reasonable policy based on measured data. Do not
# overcomplicate it." The simplest policy the spec explicitly offers -- a candidate must be
# eligible on EVERY evaluated seed to be considered a stable default -- is also the most
# conservative one available, appropriate for a production-safety decision (as opposed to a
# pure quality one, where averaging would be reasonable).
MINIMUM_ELIGIBILITY_RATE_FOR_STABLE_DEFAULT = 1.0


def is_stable_default_candidate(aggregate: dict[str, Any]) -> bool:
    """A candidate is stable enough to even be considered as a new default only if it was
    eligible on every evaluated seed."""
    return aggregate["eligibilityRate"] >= MINIMUM_ELIGIBILITY_RATE_FOR_STABLE_DEFAULT


def winner_counts(per_seed_selected: list[str]) -> dict[str, int]:
    """`per_seed_selected`: the winning algorithm name for each evaluated seed, in order.
    Returns {algorithm: number of seeds it won}, over only the algorithms that won at least
    once (an algorithm that never won is simply absent, not zero-padded, since the caller
    already knows the full candidate set from `aggregate_algorithm`)."""
    counts: dict[str, int] = {}
    for name in per_seed_selected:
        counts[name] = counts.get(name, 0) + 1
    return counts
