"""PHASE 1 -- eligibility policy (Task 5): is a candidate safe/behaviorally acceptable enough
to be considered for selection at all? Kept strictly separate from PHASE 2 quality ranking
(app.ml.quality_scorer) -- `decide()` answers "acceptable to deploy", never "best ranking".

A candidate is eligible only if ALL of:
  1. every HARD ModelBehavior gate passes (app.ml.gate_severity.HARD_GATES) -- negative/
     notInterested/recent/session/semantic. SOFT gate failures never affect eligibility, only
     quality (see app.ml.gate_severity's module docstring for the measured evidence behind
     this split).
  2. the HARD RerankerPolicy checks pass:
     a. an otherwise-identical unseen candidate still beats an already-seen one AFTER final
        reranking (app.ml.candidate_evaluation) -- the production home `alreadySeen` moved to
        (Task 5 spec; app.ml.reranker.SEEN_PENALTY is what actually guarantees this in
        production, so this checks the guarantee itself, not a raw-model proxy for it).
     b. a strong, real, long-term-preferred category still beats a genuinely never-seen one
        after final reranking (the "easy-fundamental-preference" scenario's
        `football_beats_music` critical constraint, forced in identically to (a) above --
        app.ml.candidate_evaluation.FUNDAMENTAL_PREFERENCE_SCENARIO_ID). Added after a real
        production XGBRanker candidate was found to pass every existing gate here while still
        failing this single most-basic personalization case in practice: app.ml.eligibility's
        generic ModelBehavior gates use placeholder category tokens that
        OneHotEncoder(handle_unknown="ignore") always encodes as all-zero, making them
        structurally unable to see a real, named-category-identity bias a candidate learned --
        this named-category check is what actually catches that failure mode.
  3. its END_TO_END critical-constraint pass rate (MEDIUM/HARD/ADVERSARIAL, reranked stage)
     meets MINIMUM_CRITICAL_PASS_RATE -- a candidate cannot win purely on a good average NDCG
     while systematically failing genuinely product-critical constraints (Task 5 spec section 12).
"""
from __future__ import annotations

from typing import Any

# Chosen from measured evidence (scripts/run_selection_experiment.py, seeds 42/101/2026): the
# only candidate that clears every HARD ModelBehavior gate (LogisticRegression) measured a
# critical pass rate comfortably above this line on every seed; every candidate that fails a
# HARD ModelBehavior gate is already rejected by requirement (1) regardless of this threshold.
# 0.75 rejects a candidate that fails 1 in 4 (or more) genuinely product-critical constraints --
# not a tolerance tuned to produce any particular winner (Task 5 explicitly forbids that).
MINIMUM_CRITICAL_PASS_RATE = 0.75


def decide(severity: dict[str, Any], candidate_eval: dict[str, Any]) -> dict[str, Any]:
    """`severity`: app.ml.gate_severity.severity_report(...)'s return value.
    `candidate_eval`: app.ml.candidate_evaluation.evaluate_candidate(...)'s return value.

    Returns {"eligible": bool, "rejectionReasons": [str, ...], "hardModelBehaviorOk": bool,
    "rerankerPolicyOk": bool, "criticalPassRate": float, "criticalPassed": int,
    "criticalTotal": int} -- enough to fully explain why a candidate was or was not eligible
    (Task 5 spec: eligibility must remain auditable, disagreement must never be hidden)."""
    hard_failed = list(severity["hardFailedGates"])
    reranker_ok = bool(candidate_eval["rerankerPolicy"]["unseenBeatsSeenPassed"])
    # `.get(..., True)`: safe default for any caller/fixture predating this check (matches this
    # codebase's established convention for additive fields, e.g. app.ml.dataset_builder's
    # `not_interested` RecentWatchEvent field) -- a real `evaluate_candidate()` result always
    # populates this key, so the default only ever applies to pre-existing test doubles.
    fundamental_ok = bool(candidate_eval["rerankerPolicy"].get("fundamentalPreferencePassed", True))
    critical_passed = candidate_eval["endToEnd"]["criticalPassed"]
    critical_total = candidate_eval["endToEnd"]["criticalTotal"]
    critical_rate = critical_passed / critical_total if critical_total else 0.0

    reasons: list[str] = []
    if hard_failed:
        reasons.append(f"failed HARD model-behavior gate(s): {', '.join(hard_failed)}")
    if not reranker_ok:
        reasons.append(
            "failed HARD reranker-policy check: an unseen candidate did not beat an "
            "equivalent already-seen candidate after final reranking"
        )
    if not fundamental_ok:
        reasons.append(
            "failed HARD reranker-policy check: a strong, real, long-term-preferred category "
            "did not beat a genuinely never-seen category after final reranking"
        )
    if critical_rate < MINIMUM_CRITICAL_PASS_RATE:
        reasons.append(
            f"end-to-end critical constraint pass rate {critical_rate:.2%} "
            f"({critical_passed}/{critical_total}) is below the minimum "
            f"{MINIMUM_CRITICAL_PASS_RATE:.0%}"
        )

    return {
        "eligible": not reasons,
        "rejectionReasons": reasons,
        "hardModelBehaviorOk": not hard_failed,
        "rerankerPolicyOk": reranker_ok and fundamental_ok,
        "criticalPassRate": round(critical_rate, 6),
        "criticalPassed": critical_passed,
        "criticalTotal": critical_total,
    }
