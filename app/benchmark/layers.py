"""Unified evaluation-layer classification and suites.

Three distinct responsibilities were previously conflated across app.ml.eligibility (raw-model
gates, called directly against `model.predict_proba`, never through reranking) and
app.benchmark (which always evaluates BOTH a raw and a final/reranked stage). This module
makes the distinction explicit and reusable, without duplicating either system's actual
scenario/gate logic:

    Layer A -- MODEL BEHAVIOR: can the raw ML probability respond correctly to features,
        with no reranking involved at all? This is what every app.ml.eligibility gate
        already tests, and what app.benchmark's "raw" stage constraints test.
    Layer B -- RERANKER POLICY: does app.ml.reranker.rerank() itself correctly apply seen-
        content deprioritization, diversity decay, search/social/freshness boosts? These are
        guaranteed by fixed, model-independent reranker mechanics (e.g. SEEN_PENALTY), not by
        what the raw model happens to learn.
    Layer C -- END-TO-END: does the FINAL, reranked ordering satisfy the constraint -- the
        combination of whatever the raw model produced and whatever reranking then did to it?
        This is what app.benchmark's "reranked" stage constraints test.

`GateClassification` records, for each existing production gate, which layer it actually
belongs to and why -- see CLASSIFICATIONS below. Nothing here changes gate behavior,
thresholds, or eligibility; this module is read-only with respect to app.ml.eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.benchmark.runner import run_scenario_isolated, use_model
from app.benchmark.scenarios import all_named_scenarios, dominant_category_scenario
from app.ml.eligibility import evaluate_eligibility

Layer = str  # "MODEL_BEHAVIOR" | "RERANKER_POLICY" | "END_TO_END"
GateStatus = str  # "KEEP_AS_MODEL_GATE" | "MOVE_TO_RERANKER_POLICY_TEST" | "MOVE_TO_END_TO_END_BENCHMARK" | "REDESIGN_GATE" | "REMOVE_AS_REDUNDANT"


@dataclass(frozen=True)
class GateClassification:
    gate: str
    layer: Layer
    recommended_status: GateStatus
    rationale: str


# Evidence-based classification -- see the accompanying diagnosis report for the measured
# margins/OOD checks/ablations behind each entry. Every gate here executes against the raw
# model only (app.ml.eligibility never calls app.ml.reranker); this classification identifies
# which of them SHOULD stay that way vs. which test something the reranker (or the full
# end-to-end path) already guarantees or is better positioned to guarantee.
#
# `alreadySeen` is deliberately NOT listed below: Task 4's audit recommended MOVE_TO_
# RERANKER_POLICY_TEST for it, and Task 5 already acted on that finding -- it no longer exists
# as a raw app.ml.eligibility gate at all (see that module's `_DIAGNOSTIC_GATES`), and its
# production RerankerPolicy home is now app.ml.eligibility_policy.decide's `rerankerPolicyOk`
# check. Keeping a stale "recommended" entry here for a gate that no longer exists would be
# documentation drift, not a live recommendation.
CLASSIFICATIONS: tuple[GateClassification, ...] = (
    GateClassification("longTerm", "MODEL_BEHAVIOR", "KEEP_AS_MODEL_GATE",
                        "Pure category-affinity response with no reranking involved; the model must learn this itself."),
    GateClassification("recent", "MODEL_BEHAVIOR", "KEEP_AS_MODEL_GATE",
                        "Recent-vs-stale is a raw-model learning question; reranker has no recency-vs-long-term logic."),
    GateClassification("session", "MODEL_BEHAVIOR", "KEEP_AS_MODEL_GATE",
                        "Feature-value sanity check on session features the model consumes directly."),
    GateClassification("negative", "MODEL_BEHAVIOR", "KEEP_AS_MODEL_GATE",
                        "Implicit-negative suppression is not enforced anywhere in the reranker; must be learned."),
    GateClassification("notInterested", "MODEL_BEHAVIOR", "KEEP_AS_MODEL_GATE",
                        "Explicit-rejection suppression is not enforced anywhere in the reranker; must be learned. "
                        "Root-cause diagnosis (this task) shows session/recent-category features can override this "
                        "signal in tree models -- REDESIGN candidate if that proves structural, not a reason to relax it."),
    GateClassification("semantic", "MODEL_BEHAVIOR", "KEEP_AS_MODEL_GATE",
                        "Token-affinity response is model-learned; app.ml.reranker's own semantic boost is a small, "
                        "bounded multiplier layered ON TOP of this, not a substitute for it."),
    GateClassification("creator", "MODEL_BEHAVIOR", "KEEP_AS_MODEL_GATE",
                        "Creator-affinity response is model-learned; reranker has no creator-preference logic "
                        "(only a same-creator REPETITION cap, a different concept)."),
    GateClassification("coldStart", "MODEL_BEHAVIOR", "KEEP_AS_MODEL_GATE",
                        "Popularity-monotonicity with zero history is a raw-model sanity check; reranker has no "
                        "popularity-specific cold-start logic of its own."),
    GateClassification("subthemeRejectionLocalization", "MODEL_BEHAVIOR", "REDESIGN_GATE",
                        "The underlying feature (category_affinity) is correctly ordered (same-subtheme-repeat > "
                        "diverse-subtheme-rejection), but this task's ablation shows the trained model's OUTPUT "
                        "does not reliably preserve that ordering once other feature groups vary realistically -- "
                        "the current gate's narrow, single-category-affinity-dominant probe passes on a margin "
                        "that does not generalize to the benchmark's more realistic candidates (see NOT_INTERESTED "
                        "diagnosis). Keep as a model-behavior concern (this IS something the model should learn), "
                        "but the probe itself needs a harder, more realistic redesign, not a raised tolerance."),
)


def production_gate_report(model: Any) -> dict[str, Any]:
    """Layer A (MODEL_BEHAVIOR) -- exactly today's app.ml.eligibility.evaluate_eligibility,
    unchanged. Included here so all three layers can be read from one place."""
    return evaluate_eligibility(model)


def reranker_policy_report(algorithm: str, model: Any, training_result: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
    """Layer B -- constraints whose pass/fail depends specifically on RERANKING, isolated by
    comparing a scenario's raw vs. reranked constraint results and keeping only the ones the
    reranker itself resolves (the dominant-category scenario's diversity diagnostics, plus
    every constraint the reranker flips from fail to pass or pass to fail -- see
    app.benchmark.constraints.classify_reranker_effect)."""
    with use_model(model, algorithm, training_result, tmp_dir / algorithm / "reranker_policy"):
        dominant = run_scenario_isolated(dominant_category_scenario(), algorithm=algorithm)
        scenario_results = [run_scenario_isolated(s, algorithm=algorithm) for s in all_named_scenarios()]

    reranker_touched = sum(len(r.reranker_improved) + len(r.reranker_degraded) for r in scenario_results)
    reranker_improved = sum(len(r.reranker_improved) for r in scenario_results)
    reranker_degraded = sum(len(r.reranker_degraded) for r in scenario_results)
    return {
        "dominantCategoryTopShare": dominant.reranked.metrics["topCategoryShare"],
        "dominantCategoryUniqueCount": dominant.reranked.metrics["uniqueCategories"],
        "constraintsTouchedByReranking": reranker_touched,
        "rerankerImproved": reranker_improved,
        "rerankerDegraded": reranker_degraded,
        "passed": reranker_degraded == 0,  # the reranker must never make a constraint worse.
        "total": 1,
    }


def end_to_end_report(algorithm: str, model: Any, training_result: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
    """Layer C -- the FINAL reranked-stage constraint pass rate across every named scenario
    (EASY/MEDIUM/HARD/ADVERSARIAL), critical and non-critical combined."""
    with use_model(model, algorithm, training_result, tmp_dir / algorithm / "end_to_end"):
        scenario_results = [run_scenario_isolated(s, algorithm=algorithm) for s in all_named_scenarios()]
    passed = sum(r.reranked.constraints_passed for r in scenario_results)
    total = sum(r.reranked.constraints_total for r in scenario_results)
    critical_passed = sum(r.reranked.critical_constraints_passed for r in scenario_results)
    critical_total = sum(r.reranked.critical_constraints_total for r in scenario_results)
    return {"passed": passed, "total": total, "criticalPassed": critical_passed, "criticalTotal": critical_total}


def unified_report(algorithm: str, model: Any, training_result: dict[str, Any], tmp_dir: Path) -> dict[str, Any]:
    gates = production_gate_report(model)
    gates_passed = sum(1 for g in gates["gates"].values() if g["pass"])
    return {
        "algorithm": algorithm,
        "modelBehavior": {"passed": gates_passed, "total": len(gates["gates"]), "eligible": gates["eligible"]},
        "rerankerPolicy": reranker_policy_report(algorithm, model, training_result, tmp_dir),
        "endToEnd": end_to_end_report(algorithm, model, training_result, tmp_dir),
    }
