"""Per-candidate RerankerPolicy + END_TO_END benchmark evaluation (Task 5: layered model
selection). Reuses app.benchmark end-to-end (never duplicates scenario/constraint logic) --
this module is purely orchestration: given one already-fitted (ideally already-calibrated)
candidate model, run it through the real benchmark pipeline and summarize the results into the
shape app.ml.eligibility_policy/app.ml.quality_scorer consume.

Imports app.benchmark lazily, inside the function body: app.benchmark.runner itself imports
app.ml.trainer at module load time (to reuse train_models for its own per-algorithm training),
so importing app.benchmark at THIS module's top level -- given app.ml.trainer imports this
module -- would create trainer -> candidate_evaluation -> benchmark.runner -> trainer, an
import cycle. Deferring the import to call time breaks it: by the time this function actually
runs, app.ml.trainer has already finished loading.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

# MEDIUM/HARD/ADVERSARIAL feed model selection (Task 5 spec); EASY stays diagnostic-only --
# it can overweight basic category preference that every candidate already learns easily (see
# the measured per-algorithm ModelBehavior gate margins app.ml.gate_severity documents).
SELECTION_DIFFICULTIES: tuple[str, ...] = ("MEDIUM", "HARD", "ADVERSARIAL")

# Reuses the existing "adversarial-seen-vs-unseen" scenario's critical `unseen_beats_seen`
# constraint (app.benchmark.scenarios) as the RerankerPolicy home for the alreadySeen property
# app.ml.eligibility used to gate the raw model on directly (Task 5 spec: alreadySeen "should
# primarily be RERANKER_POLICY responsibility" -- app.ml.reranker.SEEN_PENALTY already
# guarantees this downstream; this checks it holds after reranking, not before).
UNSEEN_BEATS_SEEN_SCENARIO_ID = "adversarial-seen-vs-unseen"
UNSEEN_BEATS_SEEN_CONSTRAINT = "unseen_beats_seen"

# Forces app.benchmark.scenarios.category_identity_preference_scenario()'s critical
# `sport_beats_music_by_identity_alone` constraint into every candidate evaluation, exactly like
# UNSEEN_BEATS_SEEN_SCENARIO_ID above -- same mechanism, different gap it closes. Root cause
# (2026-09-04 investigation): app.ml.eligibility's generic ModelBehavior gates deliberately probe
# with placeholder category tokens ("CATEGORY_A"/"CATEGORY_B", never a real category name -- see
# that module's own docstring), which OneHotEncoder(handle_unknown="ignore") encodes as all-zero
# for BOTH sides of every probe -- structurally blind to a real, named-category-identity bias a
# candidate may have learned (confirmed directly against a real trained XGBRanker production
# candidate: a neutral/no-history MUSIC probe scored materially higher than a maximal-affinity/
# strong-history SPORT probe with every OTHER feature held equal). easy_scenario()'s own
# "football_beats_music" does use real category names, but its SPORT candidate also carries
# topic/hashtag semantic metadata -- extra reranker-side reinforcement a bare, metadata-free
# candidate (the shape production candidate pools commonly send) never gets, which was
# empirically enough to mask this exact failure for the same XGBRanker candidate above. This
# scenario is deliberately excluded from all_named_scenarios()/SELECTION_DIFFICULTIES (never
# folded into the byDifficulty aggregates that assume "every candidate already learns basic
# category preference easily" -- true for the classifiers that assumption was measured against,
# not proven, and now disproven, for ranker-family candidates trained on an entirely separate
# synthetic pipeline, app.ml.ranking_groups) -- forcing just this one, single most-fundamental
# critical constraint here keeps the fix narrowly scoped to the actual gap.
FUNDAMENTAL_PREFERENCE_SCENARIO_ID = "fundamental-category-identity-preference"
FUNDAMENTAL_PREFERENCE_CONSTRAINT = "sport_beats_music_by_identity_alone"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_candidate(
    algorithm: str, model: Any, training_result_stub: dict[str, Any], tmp_dir: Path, *,
    difficulties: tuple[str, ...] = SELECTION_DIFFICULTIES,
) -> dict[str, Any]:
    """Runs the real app.benchmark pipeline for one candidate model and returns:

        {"rerankerPolicy": {"unseenBeatsSeenPassed": bool, "fundamentalPreferencePassed": bool},
         "endToEnd": {
             "byDifficulty": {difficulty: {"rawNdcgAt10", "finalNdcgAt10",
                                            "criticalPassed", "criticalTotal",
                                            "totalPassed", "totalTotal"}, ...},
             "criticalPassed": int, "criticalTotal": int,
             "totalPassed": int, "totalTotal": int}}

    `training_result_stub` only needs the fields app.benchmark.runner.use_model's artifact
    metadata reads (splitLifecycleDescription/decisionThreshold/splitSizes/classDistribution/
    metrics/modelComparison) -- it does not need to be a fully calibrated+test-evaluated
    result, since this benchmark artifact is never promoted/served for real."""
    from app.benchmark.runner import run_scenario_isolated, use_model
    from app.benchmark.scenarios import (
        all_named_scenarios,
        category_identity_preference_scenario,
    )

    all_scenarios = all_named_scenarios()
    selected_scenarios = [s for s in all_scenarios if s.difficulty in difficulties]
    need_unseen_probe_separately = not any(s.scenario_id == UNSEEN_BEATS_SEEN_SCENARIO_ID for s in selected_scenarios)
    # category_identity_preference_scenario() is deliberately never returned by
    # all_named_scenarios() (see FUNDAMENTAL_PREFERENCE_SCENARIO_ID's comment above) -- unlike
    # UNSEEN_BEATS_SEEN_SCENARIO_ID, it is always evaluated separately, never found already
    # present in `selected_scenarios`.
    fundamental_scenario = category_identity_preference_scenario()

    with use_model(model, algorithm, training_result_stub, tmp_dir / algorithm):
        results = {s.scenario_id: run_scenario_isolated(s, algorithm=algorithm) for s in selected_scenarios}
        if need_unseen_probe_separately:
            unseen_scenario = next(s for s in all_scenarios if s.scenario_id == UNSEEN_BEATS_SEEN_SCENARIO_ID)
            results[unseen_scenario.scenario_id] = run_scenario_isolated(unseen_scenario, algorithm=algorithm)
        results[fundamental_scenario.scenario_id] = run_scenario_isolated(fundamental_scenario, algorithm=algorithm)

    by_difficulty: dict[str, Any] = {}
    critical_passed = critical_total = total_passed = total_total = 0
    for difficulty in difficulties:
        matching = [r for r in results.values() if r.difficulty == difficulty]
        d_critical_passed = sum(r.reranked.critical_constraints_passed for r in matching)
        d_critical_total = sum(r.reranked.critical_constraints_total for r in matching)
        d_total_passed = sum(r.reranked.constraints_passed for r in matching)
        d_total_total = sum(r.reranked.constraints_total for r in matching)
        by_difficulty[difficulty] = {
            "rawNdcgAt10": round(_mean(r.raw.metrics["ndcgAt10"] for r in matching), 6),
            "finalNdcgAt10": round(_mean(r.reranked.metrics["ndcgAt10"] for r in matching), 6),
            "criticalPassed": d_critical_passed, "criticalTotal": d_critical_total,
            "totalPassed": d_total_passed, "totalTotal": d_total_total,
        }
        critical_passed += d_critical_passed
        critical_total += d_critical_total
        total_passed += d_total_passed
        total_total += d_total_total

    unseen_result = results[UNSEEN_BEATS_SEEN_SCENARIO_ID]
    unseen_beats_seen_passed = any(
        c.name == UNSEEN_BEATS_SEEN_CONSTRAINT and c.passed for c in unseen_result.reranked.constraint_results
    )

    fundamental_result = results[FUNDAMENTAL_PREFERENCE_SCENARIO_ID]
    fundamental_preference_passed = any(
        c.name == FUNDAMENTAL_PREFERENCE_CONSTRAINT and c.passed for c in fundamental_result.reranked.constraint_results
    )

    return {
        "rerankerPolicy": {
            "unseenBeatsSeenPassed": unseen_beats_seen_passed,
            "fundamentalPreferencePassed": fundamental_preference_passed,
        },
        "endToEnd": {
            "byDifficulty": by_difficulty,
            "criticalPassed": critical_passed, "criticalTotal": critical_total,
            "totalPassed": total_passed, "totalTotal": total_total,
        },
    }
