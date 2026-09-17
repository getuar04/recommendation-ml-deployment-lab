"""Aggregate the four comparative-experiment reports (produced by
`scripts.run_comparative_experiment`, one per Docker environment) into one final JSON and
Markdown comparison report. Runs on the host, outside any single experiment's container --
by design, no two comparative experiment volumes are ever mounted into one container at the
same path (see README "Comparative Experiments"), so aggregation happens after each
experiment's report JSON has been captured to a host-side directory (redirect
`scripts.run_comparative_experiment`'s stdout, or `docker compose cp` the persisted report
out of that dataset's own `recommendation_reports_<name>` volume).

Every number in the output is read from the four input reports (which are themselves real,
executed training/recommendation results -- see `scripts.run_comparative_experiment`); this
script never fabricates or assumes an outcome. Where the data does not show the hypothesized
effect, that is reported honestly, with a list of possible causes, not silently omitted.

`--experiment-version` defaults to `v1`, so the command below -- the exact command documented
before the v2 rigor pass existed -- is unchanged and produces exactly the v1-style comparison
it always has:

    python -m scripts.build_comparative_report --input-dir comparative_reports --output-dir comparative_reports

Pass `--experiment-version v2` for the rigor-pass comparison (against `comparative_reports_v2`
by convention): every cross-report invariant (spec section 5/7) is *enforced* -- computed and
checked, not merely computed and displayed -- before comparison.json/comparison.md are
written; on any failure, the process exits non-zero, names exactly which invariant(s) failed,
and writes nothing (an earlier valid report in the output directory is never overwritten by a
failed run). The v2 verdict is built only from the symmetric primary scenario's raw model
probabilities (`app.experiments.comparative_scoring`) -- the asymmetric secondary
(business-reranking) scenario is reported separately and never influences it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_EXPECTED_KEYS = ("sport", "entertainment", "music", "balanced")
_METRIC_KEYS = ("accuracy", "precision", "recall", "f1Score", "prAuc", "rocAuc")
_DIAGNOSTIC_CAUSES = (
    ("weak dataset separation (the dominant category's interactions may not differ enough "
     "from the others in the underlying features)"),
    "ineffective features (the feature set may not capture category preference strongly enough)",
    ("regularization (LogisticRegression's default L2 regularization may be pulling category "
     "effects toward zero)"),
    ("class imbalance (a heavily skewed positive/negative ratio can dominate the learned "
     "decision boundary over category-level signal)"),
    ("candidate metadata (the fixed candidates' popularity/age/creator-follow values may "
     "outweigh category in the reranking business logic)"),
    ("feature construction (category is one-hot encoded with no interaction terms against "
     "user history, and the fixed test user is cold-start with no history to interact with)"),
    ("insufficient sample size (a small dataset gives the model less signal to learn a "
     "category-specific pattern from)"),
    ("recommendation fallback or reranking logic (post-model business reranking -- seen/"
     "diversity adjustments -- can reorder results independent of the raw model score)"),
    ("model limitations (a linear model may not capture the category effect as sharply as a "
     "more flexible one would)"),
)


def _load_reports(input_dir: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for key in _EXPECTED_KEYS:
        path = input_dir / f"{key}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing comparative report for '{key}': expected {path}. Run "
                f"scripts.run_comparative_experiment for every one of {_EXPECTED_KEYS} first "
                f"and capture its stdout there (see README)."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "SUCCEEDED":
            raise ValueError(f"Report for '{key}' has status={data.get('status')!r}, not SUCCEEDED -- cannot build a comparison from a failed run.")
        reports[key] = data
    return reports


# ===================================================================================
# v1 (unchanged -- see module docstring: the documented v1 command must keep working
# exactly as it always has).
# ===================================================================================

def _verify_same_algorithm_and_config(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    algorithms = {key: report["algorithm"] for key, report in reports.items()}
    hyperparams = {key: report["algorithmHyperparameters"] for key, report in reports.items()}
    dataset_seeds = {key: report["datasetGenerationSeed"] for key, report in reports.items()}
    model_seeds = {key: report["modelRandomState"] for key, report in reports.items()}
    scenarios = {key: report["fixedScenario"] for key, report in reports.items()}

    def _all_equal(values: dict[str, Any]) -> bool:
        distinct = list(values.values())
        return all(value == distinct[0] for value in distinct)

    from app.experiments.fixed_scenario import (
        FIXED_CANDIDATES,
        FIXED_RECOMMENDATION_LIMIT,
        FIXED_TEST_USER_ID,
    )

    return {
        "algorithmIdenticalAcrossExperiments": _all_equal(algorithms),
        "algorithms": algorithms,
        "hyperparametersIdenticalAcrossExperiments": _all_equal(hyperparams),
        "hyperparameters": hyperparams,
        "datasetGenerationSeedIdenticalAcrossExperiments": _all_equal(dataset_seeds),
        "datasetGenerationSeeds": dataset_seeds,
        "modelRandomStateIdenticalAcrossExperiments": _all_equal(model_seeds),
        "modelRandomStates": model_seeds,
        "fixedScenarioIdenticalAcrossExperiments": _all_equal(scenarios),
        "fixedScenarios": scenarios,
        "fixedCandidateListSource": (
            "All four experiments' /recommendations requests were built from the single "
            "shared app.experiments.fixed_scenario.FIXED_CANDIDATES constant -- identical by "
            "construction, not merely by comparison."
        ),
        "fixedCandidateCount": len(FIXED_CANDIDATES),
        "fixedTestUserId": FIXED_TEST_USER_ID,
        "fixedRecommendationLimit": FIXED_RECOMMENDATION_LIMIT,
    }


def _top_categories(report: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for item in report.get("top10", []):
        category = item.get("category")
        if category and category not in seen:
            seen.append(category)
    return seen


def _category_distribution_of_top10(report: dict[str, Any]) -> dict[str, float]:
    top10 = report.get("top10", [])
    if not top10:
        return {}
    counts: dict[str, int] = {}
    for item in top10:
        category = item.get("category")
        if category:
            counts[category] = counts.get(category, 0) + 1
    total = len(top10)
    return {category: round(count / total, 4) for category, count in sorted(counts.items(), key=lambda kv: -kv[1])}


def _comparison_table(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, report in reports.items():
        metrics = report.get("metrics") or {}
        rows.append({
            "experiment": report["experimentId"],
            "dataset": key,
            "algorithm": report["algorithm"],
            "dominantCategory": report["dominantCategory"],
            "f1Score": metrics.get("f1Score"),
            "prAuc": metrics.get("prAuc"),
            "rocAuc": metrics.get("rocAuc"),
            "topRecommendedCategories": _top_categories(report),
        })
    return rows


def _candidate_score_matrix(reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every candidate's rank/score in each experiment's Top-10 (None if it fell outside the
    Top 10 in that experiment -- still a real, honest result, not an error)."""
    all_candidate_ids: list[str] = []
    for report in reports.values():
        for item in report.get("top10", []):
            if item["contentId"] not in all_candidate_ids:
                all_candidate_ids.append(item["contentId"])

    matrix: dict[str, dict[str, Any]] = {}
    for candidate_id in sorted(all_candidate_ids):
        per_experiment = {}
        for key, report in reports.items():
            match = next((item for item in report.get("top10", []) if item["contentId"] == candidate_id), None)
            per_experiment[key] = {"rank": match["rank"], "score": match["score"]} if match else {"rank": None, "score": None}
        scores = [entry["score"] for entry in per_experiment.values() if entry["score"] is not None]
        matrix[candidate_id] = {
            "perExperiment": per_experiment,
            "scoreSpread": round(max(scores) - min(scores), 6) if len(scores) >= 2 else None,
        }
    return matrix


def _dominant_candidate_id(dominant_category: str) -> str | None:
    from app.experiments.fixed_scenario import FIXED_CANDIDATES
    for candidate in FIXED_CANDIDATES:
        if candidate["category"] == dominant_category:
            return candidate["contentId"]
    return None


def _learning_explanation(reports: dict[str, dict[str, Any]], matrix: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """For each non-balanced experiment, whether its own dominant category's fixed candidate
    ranked/scored higher than in the balanced experiment -- computed from the actual matrix,
    never assumed. Reports the honest result either way."""
    findings = []
    balanced_report = reports.get("balanced")
    for key, report in reports.items():
        if key == "balanced":
            continue
        dominant_category = report["dominantCategory"]
        candidate_id = _dominant_candidate_id(dominant_category)
        if candidate_id is None or candidate_id not in matrix:
            findings.append({
                "experiment": report["experimentId"], "dominantCategory": dominant_category,
                "observation": f"No fixed candidate in category {dominant_category!r} reached any experiment's Top 10; cannot compare.",
                "supportsHypothesis": None,
            })
            continue
        this_entry = matrix[candidate_id]["perExperiment"][key]
        balanced_entry = matrix[candidate_id]["perExperiment"]["balanced"] if balanced_report else {"rank": None, "score": None}
        this_score, balanced_score = this_entry["score"], balanced_entry["score"]
        if this_score is None and balanced_score is None:
            observation = f"Candidate {candidate_id!r} (category {dominant_category}) did not reach the Top 10 in either the {key} or balanced experiment."
            supports = None
        elif this_score is None:
            observation = f"Candidate {candidate_id!r} reached the Top 10 in balanced (score {balanced_score}) but NOT in {key} -- opposite of the hypothesis."
            supports = False
        elif balanced_score is None:
            observation = f"Candidate {candidate_id!r} reached the Top 10 in {key} (rank {this_entry['rank']}, score {this_score}) but not in balanced -- consistent with the hypothesis."
            supports = True
        else:
            delta = round(this_score - balanced_score, 6)
            supports = delta > 0
            observation = (
                f"Candidate {candidate_id!r} (category {dominant_category}) scored {this_score} "
                f"(rank {this_entry['rank']}) in {key} vs {balanced_score} (rank {balanced_entry['rank']}) "
                f"in balanced -- a delta of {delta:+.6f}, {'consistent with' if supports else 'NOT consistent with'} "
                f"the hypothesis that {dominant_category}-dominant training increases this candidate's score."
            )
        findings.append({
            "experiment": report["experimentId"], "dominantCategory": dominant_category,
            "candidateId": candidate_id, "observation": observation, "supportsHypothesis": supports,
        })
    return findings


def _conclusion(findings: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
    supporting = [f for f in findings if f["supportsHypothesis"] is True]
    contradicting = [f for f in findings if f["supportsHypothesis"] is False]
    inconclusive = [f for f in findings if f["supportsHypothesis"] is None]

    top_categories_differ = len({tuple(row["topRecommendedCategories"][:3]) for row in comparison_rows}) > 1

    if supporting and not contradicting:
        verdict = "YES"
        statement = (
            f"The same locked algorithm and configuration produced meaningfully different "
            f"recommendations when trained on different datasets: all {len(supporting)} "
            f"dominant-category experiment(s) with a comparable fixed candidate scored/ranked "
            f"that candidate higher than the balanced-training run did."
        )
    elif supporting and contradicting:
        verdict = "PARTIALLY"
        statement = (
            f"The same locked algorithm and configuration produced some measurable "
            f"differences: {len(supporting)} experiment(s) showed the hypothesized "
            f"dominant-category score/rank increase, but {len(contradicting)} did not -- "
            f"see the diagnostic causes list for why the effect was inconsistent."
        )
    elif not supporting and not contradicting:
        verdict = "INCONCLUSIVE"
        statement = "No dominant-category fixed candidate reached any experiment's Top 10, so this specific hypothesis could not be evaluated from the Top-10 lists alone; see topRecommendedCategoriesDiffer and the per-experiment metrics/scores for a broader comparison."
    else:
        verdict = "NO"
        statement = (
            f"The same locked algorithm and configuration did NOT produce the hypothesized "
            f"per-candidate score/rank increase in any of the {len(contradicting)} "
            f"dominant-category experiment(s) tested against balanced -- see "
            f"diagnosticPossibleCauses for why, and topRecommendedCategoriesDiffer/"
            f"metricComparison for whether any other difference is visible."
        )
    return {
        "verdict": verdict, "statement": statement,
        "topRecommendedCategoriesDifferAcrossExperiments": top_categories_differ,
        "supportingFindings": len(supporting), "contradictingFindings": len(contradicting), "inconclusiveFindings": len(inconclusive),
        "diagnosticPossibleCauses": list(_DIAGNOSTIC_CAUSES) if (contradicting or inconclusive) else [],
    }


def build_comparison(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    verification = _verify_same_algorithm_and_config(reports)
    comparison_rows = _comparison_table(reports)
    matrix = _candidate_score_matrix(reports)
    top10_category_distribution = {key: _category_distribution_of_top10(report) for key, report in reports.items()}
    metric_comparison = {
        key: {metric: (report.get("metrics") or {}).get(metric) for metric in _METRIC_KEYS}
        for key, report in reports.items()
    }
    findings = _learning_explanation(reports, matrix)
    conclusion = _conclusion(findings, comparison_rows)

    return {
        "generatedFrom": {key: report["experimentId"] for key, report in reports.items()},
        "sameAlgorithmVerification": verification,
        "comparisonTable": comparison_rows,
        "metricComparison": metric_comparison,
        "candidateScoreMatrix": matrix,
        "top10CategoryDistribution": top10_category_distribution,
        "recommendationLearningFindings": findings,
        "businessInterpretation": (
            "A recommendation service that genuinely learns from behavioral data should surface "
            "more of a user's likely-preferred category when that category dominates the training "
            "signal, without any change to the algorithm, code, or serving logic -- this is what "
            "operators should expect when retraining against a shifted content-consumption pattern "
            "(e.g. a seasonal spike in one content category). See recommendationLearningFindings "
            "for whether this run's data actually shows that; a synthetic-data PoC result here is "
            "evidence about pipeline behavior, not a production guarantee."
        ),
        "syntheticDataLimitations": (
            "All four datasets are entirely synthetic (app.experiments.comparative_dataset_generation), "
            "generated from simple weighted-category heuristics, not real user behavior -- category "
            "preference is injected directly into the generation process rather than emerging from "
            "organic engagement patterns, which can make the learned signal cleaner (or noisier) than "
            "a real dataset of the same size would produce. The fixed test user is a cold-start user "
            "with zero interaction history in every experiment (a deliberate choice so 'same user "
            "context' is trivially true across four separate databases -- see "
            "app.experiments.fixed_scenario), so these results say nothing about how a model trained "
            "on skewed data would treat a user with a real, independent history of their own. Absolute "
            "metric values, scores, and rankings describe this specific synthetic run only and must "
            "not be presented as production evidence."
        ),
        "conclusion": conclusion,
    }


def _markdown(comparison: dict[str, Any]) -> str:
    lines = ["# Comparative Experiment Report: Same Algorithm, Different Datasets", ""]
    lines += ["## Same-algorithm verification", ""]
    verification = comparison["sameAlgorithmVerification"]
    lines.append(f"- Algorithm identical across all four experiments: **{verification['algorithmIdenticalAcrossExperiments']}** ({verification['algorithms']})")
    lines.append(f"- Hyperparameters identical: **{verification['hyperparametersIdenticalAcrossExperiments']}**")
    lines.append(f"- Dataset-generation seed identical: **{verification['datasetGenerationSeedIdenticalAcrossExperiments']}** ({verification['datasetGenerationSeeds']})")
    lines.append(f"- Model random_state identical: **{verification['modelRandomStateIdenticalAcrossExperiments']}** ({verification['modelRandomStates']})")
    lines.append(f"- Fixed test user / candidate list / limit identical: **{verification['fixedScenarioIdenticalAcrossExperiments']}** ({verification['fixedCandidateListSource']})")
    lines.append("")

    lines += ["## Comparison table", "",
              "| Experiment | Dataset | Algorithm | Dominant Category | F1 | PR-AUC | ROC-AUC | Top Recommended Categories |",
              "|---|---|---|---|---|---|---|---|"]
    for row in comparison["comparisonTable"]:
        lines.append(
            f"| {row['experiment']} | {row['dataset']} | {row['algorithm']} | {row['dominantCategory']} | "
            f"{row['f1Score']} | {row['prAuc']} | {row['rocAuc']} | {', '.join(row['topRecommendedCategories'])} |"
        )
    lines.append("")

    lines += ["## Recommendation ranking comparison / score differences per candidate", "",
              "| Candidate | Sport rank/score | Entertainment rank/score | Music rank/score | Balanced rank/score | Score spread |",
              "|---|---|---|---|---|---|"]
    for candidate_id, entry in comparison["candidateScoreMatrix"].items():
        cells = []
        for key in _EXPECTED_KEYS:
            per = entry["perExperiment"][key]
            cells.append(f"{per['rank']}/{per['score']}" if per["rank"] is not None else "-")
        lines.append(f"| {candidate_id} | {' | '.join(cells)} | {entry['scoreSpread']} |")
    lines.append("")

    lines += ["## Category distribution of Top 10 recommendations", ""]
    for key in _EXPECTED_KEYS:
        dist = comparison["top10CategoryDistribution"].get(key, {})
        lines.append(f"- **{key}**: {dist}")
    lines.append("")

    lines += ["## Metric comparison", "",
              "| Dataset | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |",
              "|---|---|---|---|---|---|---|"]
    for key in _EXPECTED_KEYS:
        metrics = comparison["metricComparison"].get(key, {})
        lines.append(f"| {key} | {metrics.get('accuracy')} | {metrics.get('precision')} | {metrics.get('recall')} | {metrics.get('f1Score')} | {metrics.get('prAuc')} | {metrics.get('rocAuc')} |")
    lines.append("")

    lines += ["## What the model learned differently", ""]
    for finding in comparison["recommendationLearningFindings"]:
        lines.append(f"- **{finding['experiment']}** (dominant={finding['dominantCategory']}): {finding['observation']}")
    lines.append("")

    lines += ["## Business interpretation", "", comparison["businessInterpretation"], ""]
    lines += ["## Limitations of synthetic data", "", comparison["syntheticDataLimitations"], ""]

    conclusion = comparison["conclusion"]
    lines += ["## Conclusion", "",
              f"**Verdict: {conclusion['verdict']}**", "", conclusion["statement"], ""]
    if conclusion["diagnosticPossibleCauses"]:
        lines.append("Possible causes for the inconsistent/absent effect (not all necessarily apply):")
        for cause in conclusion["diagnosticPossibleCauses"]:
            lines.append(f"- {cause}")
        lines.append("")

    return "\n".join(lines)


# ===================================================================================
# v2 (rigor pass): enforced invariants, ten-section structure, verdict from the
# symmetric primary scenario's raw model probabilities only.
# ===================================================================================

def _raw_score_matrix_v2(reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Category -> per-experiment raw (pre-reranking) model probability. Unlike v1's
    candidate-score matrix, nothing here is ever missing due to Top-K truncation -- the
    primary scenario's limit equals its candidate count, so every category has a raw score in
    every experiment."""
    from app.experiments.fixed_scenario import PRIMARY_CANDIDATE_CATEGORIES

    matrix: dict[str, dict[str, Any]] = {}
    for category in PRIMARY_CANDIDATE_CATEGORIES:
        candidate_id = f"fixed-cand-primary-{category.lower()}"
        per_experiment = {key: (report.get("rawModelScores") or {}).get(candidate_id) for key, report in reports.items()}
        scores = [value for value in per_experiment.values() if value is not None]
        matrix[category] = {
            "perExperiment": per_experiment,
            "scoreSpread": round(max(scores) - min(scores), 6) if len(scores) >= 2 else None,
        }
    return matrix


def _comparison_table_v2(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, report in reports.items():
        metrics = report.get("metrics") or {}
        adjusted = report["primaryScenario"]["response"]["recommendations"]
        top_categories: list[str] = []
        for item in adjusted:
            if item["category"] not in top_categories:
                top_categories.append(item["category"])
        rows.append({
            "experiment": report["experimentId"], "dataset": key, "algorithm": report["algorithm"],
            "dominantCategory": report["dominantCategory"],
            "f1Score": metrics.get("f1Score"), "prAuc": metrics.get("prAuc"), "rocAuc": metrics.get("rocAuc"),
            "topRecommendedCategories": top_categories,
        })
    return rows


def _learning_explanation_v2(reports: dict[str, dict[str, Any]], raw_score_matrix: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Primary-scenario, raw-score-only findings (spec section 5: "only the symmetric primary
    scenario and rawModelScore may determine the experimental verdict"). For each
    dominant-category experiment: (a) its dominant category's raw score vs. the SAME
    category's raw score in the balanced experiment (cross-experiment, now meaningful because
    class balance is controlled/equalized in v2 -- see comparative_dataset_generation), and
    (b) its dominant category's raw score vs. the average of every OTHER category's raw score
    within that SAME experiment (within-experiment, robust to any residual cross-experiment
    difference). `supportsHypothesis` is driven by (a); (b) is reported as corroborating
    evidence, not as a second independent verdict signal."""
    findings = []
    for key, report in reports.items():
        if key == "balanced":
            continue
        dominant_category = report["dominantCategory"]
        entry = raw_score_matrix.get(dominant_category)
        if entry is None:
            findings.append({
                "experiment": report["experimentId"], "dominantCategory": dominant_category,
                "observation": f"No primary candidate exists for category {dominant_category!r}; cannot compare.",
                "evidence": {}, "supportsHypothesis": None,
            })
            continue

        this_score = entry["perExperiment"].get(key)
        balanced_score = entry["perExperiment"].get("balanced")
        peer_scores = [
            raw_score_matrix[category]["perExperiment"].get(key)
            for category in raw_score_matrix if category != dominant_category
        ]
        peer_scores = [value for value in peer_scores if value is not None]
        peer_average = round(sum(peer_scores) / len(peer_scores), 6) if peer_scores else None
        cross_experiment_delta = round(this_score - balanced_score, 6) if (this_score is not None and balanced_score is not None) else None
        within_experiment_delta = round(this_score - peer_average, 6) if (this_score is not None and peer_average is not None) else None
        supports = cross_experiment_delta is not None and cross_experiment_delta > 0

        if cross_experiment_delta is not None:
            observation = (
                f"{dominant_category} raw model score in {key}: {this_score} vs {balanced_score} in balanced "
                f"(cross-experiment delta {cross_experiment_delta:+.6f}); vs. the average of the other "
                f"{len(peer_scores)} categories' raw scores within {key} itself: {peer_average} "
                f"(within-experiment delta {within_experiment_delta:+.6f})."
            )
        else:
            observation = f"{dominant_category} raw model score in {key} or balanced is missing; cannot compute a cross-experiment delta."
        findings.append({
            "experiment": report["experimentId"], "dominantCategory": dominant_category,
            "observation": observation,
            "evidence": {
                "rawScoreInOwnExperiment": this_score, "rawScoreInBalanced": balanced_score,
                "crossExperimentDelta": cross_experiment_delta,
                "peerCategoryAverageRawScoreInOwnExperiment": peer_average,
                "withinExperimentDelta": within_experiment_delta,
            },
            "supportsHypothesis": supports,
        })
    return findings


def _conclusion_v2(findings: list[dict[str, Any]]) -> dict[str, Any]:
    supporting = [f for f in findings if f["supportsHypothesis"] is True]
    contradicting = [f for f in findings if f["supportsHypothesis"] is False]
    inconclusive = [f for f in findings if f["supportsHypothesis"] is None]

    what_raw_model_learned = (
        f"Raw (pre-reranking) model probability, primary symmetric scenario only: "
        f"{len(supporting)} of {len(findings)} dominant-category experiment(s) showed a higher "
        f"raw score for their own dominant category than the balanced-trained model assigned "
        f"the same category."
    )
    what_reranking_changed = (
        "See postRerankingEvidence for the real, adjusted (post-business-reranking) "
        "recommendation order per experiment -- compare against rawModelScoreEvidence to see "
        "whether business reranking (seen-penalty, diversity decay) preserved, amplified, or "
        "reduced the raw model's category ordering. Not itself part of this verdict."
    )
    cannot_conclude = (
        "Whether a model trained on skewed real (non-synthetic) data would show the same "
        "effect; whether the effect generalizes beyond this fixed, cold-start test user and "
        "fixed candidate set; whether the magnitude of any raw-score delta found here is "
        "practically significant for a production ranking (no minimum-effect-size threshold "
        "or statistical-significance test is applied -- deltas are reported as-is)."
    )

    if supporting and not contradicting:
        verdict = "YES"
        statement = (
            f"The same locked LogisticRegression, trained under identical configuration on "
            f"different declared category distributions, produced a measurably different raw "
            f"probability for the dominant category in all {len(supporting)} dominant-category "
            f"experiment(s) tested, relative to the balanced-trained model."
        )
    elif supporting and contradicting:
        verdict = "PARTIALLY"
        statement = (
            f"{len(supporting)} of {len(findings)} dominant-category experiment(s) showed the "
            f"hypothesized raw-score increase; {len(contradicting)} did not -- see "
            f"diagnosticPossibleCauses."
        )
    elif not supporting and not contradicting:
        verdict = "INCONCLUSIVE"
        statement = "Raw scores were not comparable for one or more experiments (see evidence) -- no verdict can be drawn from what's available."
    else:
        verdict = "NO"
        statement = (
            f"The same locked algorithm and configuration did NOT produce the hypothesized "
            f"raw-score increase in any of the {len(contradicting)} dominant-category "
            f"experiment(s) tested against balanced -- see diagnosticPossibleCauses."
        )

    return {
        "verdict": verdict, "statement": statement,
        "whatTheRawModelLearned": what_raw_model_learned,
        "whatBusinessRerankingChanged": what_reranking_changed,
        "whatCannotBeConcludedFromSyntheticData": cannot_conclude,
        "supportingFindings": len(supporting), "contradictingFindings": len(contradicting), "inconclusiveFindings": len(inconclusive),
        "diagnosticPossibleCauses": list(_DIAGNOSTIC_CAUSES) if (contradicting or inconclusive) else [],
    }


def build_comparison_v2(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Raises `app.experiments.comparative_invariants.ComparativeInvariantError` (propagated
    to the caller, uncaught) if any required cross-report invariant does not hold -- no
    section below is ever computed from an unvalidated input, and the caller must not write
    any output file if this raises."""
    from app.experiments.comparative_invariants import (
        validate_cross_report_invariants_v2,
    )

    validate_cross_report_invariants_v2(reports)

    representative = reports["sport"]  # any report works: identity verified above across all four
    experimental_controls = {
        "algorithm": representative["algorithm"],
        "algorithmHyperparameters": representative["algorithmHyperparameters"],
        "datasetGenerationSeed": representative["datasetGenerationSeed"],
        "modelRandomState": representative["modelRandomState"],
        "featureNames": representative["featureNames"],
        "splitStrategy": representative["splitStrategy"],
        "splitLifecycleUsedFallback": representative["splitLifecycleUsedFallback"],
        "fixedScenario": representative["fixedScenario"],
        "identicalAcrossAllFourExperiments": True,
        "note": "Verified programmatically across all four reports before this comparison was built (app.experiments.comparative_invariants.validate_cross_report_invariants_v2) -- not merely asserted.",
    }

    dataset_treatment = {
        key: {
            "experimentId": report["experimentId"], "dominantCategory": report["dominantCategory"],
            "categoryMappingNote": report.get("categoryMappingNote"),
            "categoryWeightsTarget": report["categoryWeightsTarget"],
        }
        for key, report in reports.items()
    }

    realized_dataset_statistics = {
        key: {"dataset": report["dataset"], "categoryDistributionRealized": report["categoryDistributionRealized"]}
        for key, report in reports.items()
    }

    raw_score_matrix = _raw_score_matrix_v2(reports)
    raw_model_score_evidence = {
        "note": (
            "app.experiments.fixed_scenario.PRIMARY_FIXED_CANDIDATES only -- one symmetric "
            "candidate per category, identical popularity/age/creatorFollowed=false/"
            "alreadySeen=false across every candidate. rawModelScore is the trained model's "
            "pre-reranking probability, captured via app.experiments.comparative_scoring "
            "(parity-checked against the real HTTP /recommendations response -- see "
            "postRerankingEvidence)."
        ),
        "rawScoreByCategoryAndExperiment": raw_score_matrix,
    }

    post_reranking_evidence = {
        key: {
            "primaryScenarioAdjustedRecommendations": report["primaryScenario"]["response"]["recommendations"],
            "strategy": report["primaryScenario"]["response"]["strategy"],
            "modelVersion": report["primaryScenario"]["response"]["modelVersion"],
        }
        for key, report in reports.items()
    }

    comparison_table = _comparison_table_v2(reports)
    findings = _learning_explanation_v2(reports, raw_score_matrix)
    conclusion = _conclusion_v2(findings)

    secondary_scenario_evidence = {
        key: {"response": report["secondaryScenario"]["response"], "note": report["secondaryScenario"]["note"]}
        for key, report in reports.items()
    }

    confounding_factors = [
        (
            "The sport/entertainment/music datasets share the identical generation seed and an "
            "isomorphic 80/10/10 category-weight shape -- they differ only in which category "
            "label is assigned the dominant/related/other roles, not in any other statistical "
            "property of the generation process."
        ),
        (
            "The entertainment dataset's dominant category (COMEDY) never appears in the "
            "balanced dataset's training categories at all, whereas sport/music's dominant "
            "categories (SPORT/MUSIC) both appear in balanced at a lower (20%) weight -- the "
            "entertainment-vs-balanced comparison is a 'never seen vs seen' contrast, not a "
            "'seen more vs seen less' contrast like the other two."
        ),
        (
            "Deterministic post-generation class-balance downsampling "
            "(app.experiments.comparative_dataset_generation._stratified_downsample) removes "
            "some point-in-time history context for chronologically-later surviving rows -- "
            "applied identically across all four datasets via the same seeded algorithm, so it "
            "should not differentially bias one dataset over another, but it is a real "
            "methodological effect, not a null one."
        ),
        (
            "The fixed test user is cold-start (zero interaction history) in every experiment, "
            "so every primary-scenario feature except category, popularity, and age is a "
            "constant neutral value -- this isolates the category effect cleanly but says "
            "nothing about personalization for a user with real history."
        ),
    ]

    limitations = [
        (
            "All four datasets are entirely synthetic, generated from a weighted-category "
            "heuristic, not real user behavior -- category preference is injected directly "
            "into generation rather than emerging from organic engagement."
        ),
        (
            "No minimum-effect-size or statistical-significance threshold is applied to any "
            "raw-score delta -- see whatCannotBeConcludedFromSyntheticData in the conclusion."
        ),
        (
            "Absolute metric values, scores, and rankings describe this specific synthetic run "
            "only and must not be presented as production evidence."
        ),
    ]

    return {
        "generatedFrom": {key: report["experimentId"] for key, report in reports.items()},
        "experimentalControls": experimental_controls,
        "datasetTreatment": dataset_treatment,
        "realizedDatasetStatistics": realized_dataset_statistics,
        "rawModelScoreEvidence": raw_model_score_evidence,
        "postRerankingEvidence": post_reranking_evidence,
        "comparisonTable": comparison_table,
        "observations": [finding["observation"] for finding in findings],
        "evidenceSupportingObservations": findings,
        "secondaryScenarioEvidence": secondary_scenario_evidence,
        "confoundingFactors": confounding_factors,
        "limitations": limitations,
        "conclusion": conclusion,
    }


def _markdown_v2(comparison: dict[str, Any]) -> str:
    lines = ["# Comparative Experiment Report v2: Same Algorithm, Different Datasets", ""]

    lines += ["## 1. Experimental controls", ""]
    controls = comparison["experimentalControls"]
    lines.append(f"- Identical across all four experiments (enforced, not just displayed): **{controls['identicalAcrossAllFourExperiments']}**")
    lines.append(f"- Algorithm: `{controls['algorithm']}`")
    lines.append(f"- Hyperparameters: `{controls['algorithmHyperparameters']}`")
    lines.append(f"- Dataset-generation seed: `{controls['datasetGenerationSeed']}`  Model random_state: `{controls['modelRandomState']}`")
    lines.append(f"- Feature names/order: `{controls['featureNames']}`")
    lines.append(f"- Split strategy: {controls['splitStrategy']} (used fallback: {controls['splitLifecycleUsedFallback']})")
    lines.append(f"- Fixed (primary) scenario: `{controls['fixedScenario']}`")
    lines.append("")

    lines += ["## 2. Dataset treatment", ""]
    for key, treatment in comparison["datasetTreatment"].items():
        note = f" -- {treatment['categoryMappingNote']}" if treatment.get("categoryMappingNote") else ""
        lines.append(f"- **{key}** ({treatment['experimentId']}): dominant={treatment['dominantCategory']}, target weights={treatment['categoryWeightsTarget']}{note}")
    lines.append("")

    lines += ["## 3. Realized dataset statistics", ""]
    for key, stats in comparison["realizedDatasetStatistics"].items():
        lines.append(f"- **{key}**: {stats['dataset']}")
        lines.append(f"  - realized category distribution: {stats['categoryDistributionRealized']}")
    lines.append("")

    lines += ["## 4. Raw model-score evidence (primary scenario, pre-reranking)", "", comparison["rawModelScoreEvidence"]["note"], "",
              "| Category | " + " | ".join(_EXPECTED_KEYS) + " | Score spread |",
              "|---|" + "---|" * (len(_EXPECTED_KEYS) + 1)]
    for category, entry in comparison["rawModelScoreEvidence"]["rawScoreByCategoryAndExperiment"].items():
        cells = [str(entry["perExperiment"].get(key)) for key in _EXPECTED_KEYS]
        lines.append(f"| {category} | " + " | ".join(cells) + f" | {entry['scoreSpread']} |")
    lines.append("")

    lines += ["## 5. Post-reranking evidence (primary scenario, real adjusted response)", ""]
    for key, evidence in comparison["postRerankingEvidence"].items():
        ranked = ", ".join(f"{item['category']}:{item['score']}" for item in evidence["primaryScenarioAdjustedRecommendations"])
        lines.append(f"- **{key}** (strategy={evidence['strategy']}): {ranked}")
    lines.append("")

    lines += ["## Comparison table", "",
              "| Experiment | Dataset | Algorithm | Dominant Category | F1 | PR-AUC | ROC-AUC | Top Recommended Categories (post-rerank) |",
              "|---|---|---|---|---|---|---|---|"]
    for row in comparison["comparisonTable"]:
        lines.append(
            f"| {row['experiment']} | {row['dataset']} | {row['algorithm']} | {row['dominantCategory']} | "
            f"{row['f1Score']} | {row['prAuc']} | {row['rocAuc']} | {', '.join(row['topRecommendedCategories'])} |"
        )
    lines.append("")

    lines += ["## 6. Observations", ""]
    for observation in comparison["observations"]:
        lines.append(f"- {observation}")
    lines.append("")

    lines += ["## 7. Evidence supporting each observation", ""]
    for finding in comparison["evidenceSupportingObservations"]:
        lines.append(f"- **{finding['experiment']}** (dominant={finding['dominantCategory']}, supportsHypothesis={finding['supportsHypothesis']}): `{finding['evidence']}`")
    lines.append("")

    lines += ["## Secondary scenario (business-reranking evidence only -- excluded from the verdict)", ""]
    for key, evidence in comparison["secondaryScenarioEvidence"].items():
        top3 = ", ".join(f"{item['category']}:{item['reason']}" for item in evidence["response"]["recommendations"][:3])
        lines.append(f"- **{key}**: top 3 = {top3}")
    lines.append("")

    lines += ["## 8. Confounding factors", ""]
    for factor in comparison["confoundingFactors"]:
        lines.append(f"- {factor}")
    lines.append("")

    lines += ["## 9. Limitations", ""]
    for limitation in comparison["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")

    conclusion = comparison["conclusion"]
    lines += ["## 10. Conclusion", "",
              f"**Verdict: {conclusion['verdict']}**", "", conclusion["statement"], "",
              f"- What the raw model learned: {conclusion['whatTheRawModelLearned']}",
              f"- What business reranking changed: {conclusion['whatBusinessRerankingChanged']}",
              f"- What cannot be concluded from synthetic data: {conclusion['whatCannotBeConcludedFromSyntheticData']}", ""]
    if conclusion["diagnosticPossibleCauses"]:
        lines.append("Possible causes for the inconsistent/absent effect (not all necessarily apply):")
        for cause in conclusion["diagnosticPossibleCauses"]:
            lines.append(f"- {cause}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="comparative_reports")
    parser.add_argument("--output-dir", default=None, help="Defaults to --input-dir.")
    parser.add_argument("--experiment-version", choices=("v1", "v2"), default="v1",
                         help="Defaults to v1 (the original, unchanged documented command). Pass v2 for the rigor-pass pipeline.")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if args.experiment_version == "v1":
        output_dir.mkdir(parents=True, exist_ok=True)
        reports = _load_reports(input_dir)
        comparison = build_comparison(reports)
        (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
        (output_dir / "comparison.md").write_text(_markdown(comparison), encoding="utf-8")
        print(f"Wrote {output_dir / 'comparison.json'}")
        print(f"Wrote {output_dir / 'comparison.md'}")
        print(f"Conclusion: {comparison['conclusion']['verdict']} -- {comparison['conclusion']['statement']}")
        return 0

    # v2: validate everything BEFORE writing anything. On any failure, name exactly which
    # invariant(s) failed, exit non-zero, and never touch the output directory -- an earlier
    # valid comparison.json/comparison.md there is never overwritten by a failed run.
    from app.experiments.comparative_invariants import ComparativeInvariantError
    try:
        reports = _load_reports(input_dir)
        comparison = build_comparison_v2(reports)
    except (FileNotFoundError, ValueError, ComparativeInvariantError) as exc:
        print("v2 comparison could not be built -- no report written:", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    (output_dir / "comparison.md").write_text(_markdown_v2(comparison), encoding="utf-8")
    print(f"Wrote {output_dir / 'comparison.json'}")
    print(f"Wrote {output_dir / 'comparison.md'}")
    print(f"Conclusion: {comparison['conclusion']['verdict']} -- {comparison['conclusion']['statement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
