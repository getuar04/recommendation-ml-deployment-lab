"""Evaluation-system audit: reconciles app.ml.eligibility's production behavioral gates
against the independent app.benchmark ranking benchmark, for all 5 candidate algorithms.

    python -m scripts.run_evaluation_reconciliation

Trains all 5 algorithms on one shared, freshly-generated synthetic training dataset (same
convention as scripts/run_ranking_benchmark.py), then runs:
  - production gate inventory + feature-delta + OOD diagnostics for representative probes
  - NOT_INTERESTED feature-group ablation (same-subtheme vs. diverse-subtheme probe)
  - RandomForest EASY-scenario candidate-level score table
  - score-resolution/tie statistics (raw and final) per algorithm/difficulty
  - calibration before/after ranking comparison
  - the unified ModelBehavior / RerankerPolicy / EndToEnd layer report

This script never trains on benchmark scenario data and never writes to the real active
model artifact (see app.benchmark.runner.use_model). Diagnostic-only: no gate/weight/
hyperparameter is changed anywhere in this script.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_TRAINING_DB_DIR = tempfile.mkdtemp(prefix="eval_reconciliation_training_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TRAINING_DB_DIR}/training.db")

import pandas as pd

from app.benchmark import ablation, diagnostics, gate_inventory, layers
from app.benchmark.runner import (
    run_scenario_isolated,
    train_all_algorithms,
    use_model,
)
from app.benchmark.scenarios import (
    easy_scenario,
    hard_scenario,
)
from app.ml.algorithm_registry import (
    ALL_ALGORITHM_NAMES,
    unavailable_algorithms,
)
from app.ml.dataset_builder import FEATURES, build_dataset
from scripts.generate_synthetic_data import generate

REFERENCE_TIMESTAMP = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)


def _build_training_dataset():
    from sqlalchemy import select

    from app.db.database import SessionLocal
    from app.db.models import Content
    from app.db.repositories import interactions

    generate(reference_timestamp=REFERENCE_TIMESTAMP)
    db = SessionLocal()
    try:
        rows = interactions(db)
        content_by_id = {c.content_id: c for c in db.scalars(select(Content)).all()}
        return build_dataset(rows, content_by_id)
    finally:
        db.close()


def _section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def _print_gate_report() -> None:
    _section("STEP 1/2/3/4: PRODUCTION GATE INVENTORY vs. FEATURE-DELTA SUMMARIES")
    probes = gate_inventory.all_probes()
    for name in ("longTerm", "notInterested", "creator", "alreadySeen"):
        probe = probes[name]
        print(f"\n--- gate={probe.name} critical={probe.critical} tolerant={probe.tolerant} "
              f"tolerance={probe.tolerance} ---")
        print(f"  history: {probe.history_description}")
        print(f"  {probe.left_label} vs {probe.right_label} -> expect {probe.expected_relationship}")
        deltas = diagnostics.feature_delta_summary(probe.left_features, probe.right_features)
        for row in deltas[:6]:
            print(f"    {row['feature']:38} left={row['leftValue']:>10} right={row['rightValue']:>10} "
                  f"|delta|={row['absoluteDifference']:>10}")


def _print_ood_report(training_df, probes) -> None:
    _section("STEP 5: OUT-OF-DISTRIBUTION CHECK FOR PRODUCTION GATE PROBES")
    for name in ("longTerm", "notInterested", "recent", "creator"):
        probe = probes[name]
        for side_label, features in (("left", probe.left_features), ("right", probe.right_features)):
            percentiles = diagnostics.feature_percentiles(training_df, features)
            neighbor = diagnostics.nearest_neighbor_distance(training_df, features)
            key_features = ["category_affinity", "recent_category_affinity", "session_category_affinity"]
            key_percentiles = {f: percentiles.get(f) for f in key_features}
            print(f"  gate={name:32} side={side_label:6} keyPercentiles={key_percentiles} "
                  f"nearestNeighborDist={neighbor['nearestNeighborDistance']}")


def _print_not_interested_diagnosis(algorithms_models: dict[str, dict]) -> None:
    _section("STEP 8/21: NOT_INTERESTED ROOT-CAUSE DIAGNOSIS (same-subtheme vs. diverse-subtheme)")
    probe = gate_inventory.all_probes()["subthemeRejectionLocalization"]
    key_features = ["category_affinity", "recent_category_affinity", "category_negative_count",
                     "category_positive_count", "session_category_affinity", "hashtag_affinity"]
    print(f"  same-subtheme features: {[(k, probe.left_features[k]) for k in key_features]}")
    print(f"  diverse-subtheme features: {[(k, probe.right_features[k]) for k in key_features]}")

    for algorithm, entry in algorithms_models.items():
        model = entry["model"]
        frame = pd.DataFrame([probe.left_features, probe.right_features])[FEATURES]
        same_score, diverse_score = model.predict_proba(frame)[:, 1]
        print(f"\n  --- {algorithm}: same={same_score:.6f} diverse={diverse_score:.6f} "
              f"margin(same-diverse)={same_score - diverse_score:+.6f} ---")
        ablation_result = ablation.run_group_ablation(model, probe.right_features, probe.left_features)
        for group in ablation.FEATURE_GROUPS:
            print(f"    {group:20} scoreDelta={ablation_result[group]['scoreDelta']:+.6f} "
                  f"fractionOfTotalDelta={ablation_result[group]['fractionOfTotalDelta']:+.4f}")
        print(f"    {'TOTAL':20} scoreDelta={ablation_result['_totalDelta']['scoreDelta']:+.6f}")


def _print_random_forest_easy_diagnosis(training_results: dict, tmp_dir: Path) -> None:
    _section("STEP 10/22: RandomForest EASY SCENARIO CANDIDATE-LEVEL DIAGNOSIS")
    if "RandomForestClassifier" not in training_results:
        print("  RandomForestClassifier not available/trained in this run -- skipped.")
        return
    entry = training_results["RandomForestClassifier"]
    scenario = easy_scenario()
    relevance_by_id = {c.content_id: c.relevance for c in scenario.candidates}
    with use_model(entry["model"], "RandomForestClassifier", entry, tmp_dir / "rf_easy"):
        result = run_scenario_isolated(scenario, algorithm="RandomForestClassifier")
    print(f"  {'contentId':38}{'relevance':>10}{'rawScore':>12}{'rawRank':>10}{'finalScore':>12}{'finalRank':>10}")
    for content_id in scenario_candidate_order(scenario):
        raw_score = result.raw.scores_by_content_id[content_id]
        raw_rank = result.raw.ranking.index(content_id) + 1
        final_score = result.reranked.scores_by_content_id[content_id]
        final_rank = result.reranked.ranking.index(content_id) + 1
        print(f"  {content_id:38}{relevance_by_id[content_id]:>10}{raw_score:>12.6f}{raw_rank:>10}"
              f"{final_score:>12.6f}{final_rank:>10}")
    print(f"\n  raw NDCG@10={result.raw.metrics['ndcgAt10']:.6f} final NDCG@10={result.reranked.metrics['ndcgAt10']:.6f}")
    resolution = diagnostics.score_resolution_stats(list(result.raw.scores_by_content_id.values()))
    print(f"  raw score resolution: {resolution}")


def scenario_candidate_order(scenario):
    return [c.content_id for c in scenario.candidates]


def _print_score_resolution(training_results: dict, tmp_dir: Path) -> None:
    _section("STEP 11: SCORE RESOLUTION / TIE STATISTICS (HARD scenario)")
    for algorithm, entry in training_results.items():
        with use_model(entry["model"], algorithm, entry, tmp_dir / "resolution" / algorithm):
            result = run_scenario_isolated(hard_scenario(), algorithm=algorithm)
        raw_stats = diagnostics.score_resolution_stats(list(result.raw.scores_by_content_id.values()))
        final_stats = diagnostics.score_resolution_stats(list(result.reranked.scores_by_content_id.values()))
        print(f"  {algorithm:24} raw: unique={raw_stats['uniqueScores']}/{raw_stats['count']} "
              f"std={raw_stats['std']:.4f} within1e-3={raw_stats['adjacentPairsWithin1e3']}   "
              f"final: unique={final_stats['uniqueScores']}/{final_stats['count']} std={final_stats['std']:.4f}")


def _print_calibration_audit(training_results: dict) -> None:
    _section("STEP 12: CALIBRATION BEFORE/AFTER RANKING COMPARISON (HARD scenario candidates)")
    from app.benchmark.runner import isolated_session, seed_scenario
    from app.ml.dataset_builder import FeatureHistory

    scenario = hard_scenario()
    with isolated_session() as db:
        seed_scenario(db, scenario)
        from sqlalchemy import select

        from app.db.models import Content
        from app.db.repositories import recent_interactions_for_ranking
        rows = recent_interactions_for_ranking(db, scenario.user_id, limit=500)
        content_by_id = {c.content_id: c for c in db.scalars(select(Content)).all()}
        history = FeatureHistory()
        for row in sorted(rows, key=lambda r: r.timestamp):
            history.update(row, content=content_by_id.get(row.content_id))
        now = datetime.now(timezone.utc)
        feature_rows = [
            history.features(
                user_id=scenario.user_id, category=c.category, creator_id=c.creator_id, content_id=c.content_id,
                timestamp=now, content_popularity_score=c.content_popularity_score,
                content_created_at=now, creator_followed=c.creator_followed, already_seen=c.already_seen,
                hashtags=c.hashtags, topics=c.topics, entities=c.entities, subgenres=c.subgenres, title=c.title,
            )
            for c in scenario.candidates
        ]

    for algorithm, entry in training_results.items():
        comparison = diagnostics.calibration_ranking_comparison(entry["uncalibratedModel"], entry["model"], feature_rows)
        print(f"  {algorithm:24} pairwiseInversions={comparison['pairwiseInversions']:>3}/{comparison['totalPairs']} "
              f"inversionRate={comparison['inversionRate']:.4f} rankingIdentical={comparison['rankingIdentical']}")


def _print_unified_layer_report(training_results: dict, tmp_dir: Path) -> None:
    _section("STEP 15/20: UNIFIED ModelBehavior / RerankerPolicy / EndToEnd REPORT")
    print(f"  {'Algorithm':24}{'ModelBehavior':>16}{'RerankerOK':>12}{'EndToEnd':>14}{'E2E Critical':>16}")
    for algorithm, entry in training_results.items():
        report = layers.unified_report(algorithm, entry["model"], entry, tmp_dir / "unified")
        mb = report["modelBehavior"]
        rp = report["rerankerPolicy"]
        e2e = report["endToEnd"]
        print(f"  {algorithm:24}{f'{mb['passed']}/{mb['total']}':>16}"
              f"{rp['passed']!s:>12}{f'{e2e['passed']}/{e2e['total']}':>14}"
              f"{f'{e2e['criticalPassed']}/{e2e['criticalTotal']}':>16}")


def _print_gate_classification() -> None:
    _section("STEP 14: GATE CLASSIFICATION")
    for gate in layers.CLASSIFICATIONS:
        print(f"  {gate.gate:34} layer={gate.layer:16} status={gate.recommended_status}")


def main() -> int:
    print(f"Training {len(ALL_ALGORITHM_NAMES) - len(unavailable_algorithms())} algorithm(s) "
          "on one shared synthetic dataset...")
    training_df = _build_training_dataset()
    training_results = train_all_algorithms(training_df)

    probes = gate_inventory.all_probes()
    _print_gate_report()
    _print_ood_report(training_df, probes)
    _print_not_interested_diagnosis(training_results)

    with tempfile.TemporaryDirectory(prefix="eval_reconciliation_models_") as tmp:
        tmp_dir = Path(tmp)
        _print_random_forest_easy_diagnosis(training_results, tmp_dir)
        _print_score_resolution(training_results, tmp_dir)
        _print_calibration_audit(training_results)
        _print_unified_layer_report(training_results, tmp_dir)

    _print_gate_classification()

    shutil.rmtree(_TRAINING_DB_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
