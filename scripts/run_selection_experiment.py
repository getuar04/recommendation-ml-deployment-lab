"""Task 5: multi-seed layered model-selection experiment.

    python -m scripts.run_selection_experiment

Trains all 5 candidate algorithms through app.ml.trainer.train_and_select (the new layered
PHASE 1 eligibility / PHASE 2 quality-score selection) on the SAME synthetic dataset, across
several deterministic `random_seed` values (estimator-fitting randomness only -- the synthetic
data generator itself is a fixed, unparameterized module constant; see
app.benchmark.runner.train_all_algorithms's own docstring for why), then aggregates the results
with app.ml.selection_stability to answer: is any candidate's eligibility/quality stable enough
across seeds to be considered a new default?

Diagnostic-only: never modifies training data, gates, weights, or model selection itself; never
commits/pushes anything.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

_TRAINING_DB_DIR = tempfile.mkdtemp(prefix="selection_experiment_training_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TRAINING_DB_DIR}/training.db")

from app.ml import selection_stability
from app.ml.trainer import train_and_select
from scripts.generate_synthetic_data import generate

REFERENCE_TIMESTAMP = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
SEEDS = (42, 101, 2026)


def _build_training_dataset():
    from sqlalchemy import select

    from app.db.database import SessionLocal
    from app.db.models import Content
    from app.db.repositories import interactions
    from app.ml.dataset_builder import build_dataset

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


def main() -> int:
    print("Building one shared synthetic training dataset (fixed generator, not seed-parameterized)...")
    training_df = _build_training_dataset()

    per_seed_results: dict[int, dict] = {}
    for seed in SEEDS:
        print(f"\nRunning train_and_select(random_seed={seed})...")
        per_seed_results[seed] = train_and_select(training_df, random_seed=seed)

    algorithms = sorted(per_seed_results[SEEDS[0]]["modelComparison"])

    _section("PER-SEED WINNER")
    for seed in SEEDS:
        result = per_seed_results[seed]
        print(f"  seed={seed:6} winner={result['selectedModel']:24} eligibleSelection={result['eligibleSelection']}")

    _section("MULTI-SEED AGGREGATION (Task 5 required experiment table)")
    header = (f"  {'Algorithm':24}{'ElgRate':>9}{'HardG':>8}{'SoftG':>8}{'MedNDCG':>10}"
              f"{'HardNDCG':>10}{'HardStd':>9}{'AdvNDCG':>10}{'CritRate':>10}{'PRAUC':>8}"
              f"{'QualMean':>10}{'QualStd':>9}{'WinCnt':>8}")
    print(header)
    aggregates: dict[str, dict] = {}
    for algorithm in algorithms:
        records = []
        for seed in SEEDS:
            result = per_seed_results[seed]
            decision = result["eligibilityDecisions"][algorithm]
            severity = result["eligibilitySeverity"][algorithm]
            candidate_eval = result["candidateEvaluations"][algorithm]
            by_difficulty = candidate_eval["endToEnd"]["byDifficulty"]
            records.append({
                "eligible": decision["eligible"],
                "hardPassed": severity["hardPassed"], "hardTotal": severity["hardTotal"],
                "softPassed": severity["softPassed"], "softTotal": severity["softTotal"],
                "mediumNdcg": by_difficulty.get("MEDIUM", {}).get("finalNdcgAt10", 0.0),
                "hardNdcg": by_difficulty.get("HARD", {}).get("finalNdcgAt10", 0.0),
                "adversarialNdcg": by_difficulty.get("ADVERSARIAL", {}).get("finalNdcgAt10", 0.0),
                "criticalPassRate": decision["criticalPassRate"],
                "prAuc": float(result["modelComparison"][algorithm].get("prAuc") or 0.0),
                "qualityScore": result["qualityScores"][algorithm]["score"],
                "isWinner": result["selectedModel"] == algorithm,
            })
        aggregate = selection_stability.aggregate_algorithm(algorithm, records)
        aggregates[algorithm] = aggregate
        hard_gate_col = f"{aggregate['hardGatesMeanPassed']:.1f}/{aggregate['hardGatesTotal']}"
        soft_gate_col = f"{aggregate['softGatesMeanPassed']:.1f}/{aggregate['softGatesTotal']}"
        print(f"  {algorithm:24}{aggregate['eligibilityRate']:>9.2f}"
              f"{hard_gate_col:>8}{soft_gate_col:>8}"
              f"{aggregate['mediumNdcgMean']:>10.4f}{aggregate['hardNdcgMean']:>10.4f}"
              f"{aggregate['hardNdcgStd']:>9.4f}{aggregate['adversarialNdcgMean']:>10.4f}"
              f"{aggregate['criticalPassRateMean']:>10.2f}{aggregate['prAucMean']:>8.4f}"
              f"{aggregate['qualityScoreMean']:>10.4f}{aggregate['qualityScoreStd']:>9.4f}"
              f"{aggregate['winnerCount']:>8}")

    _section("WINNER COUNTS")
    counts = selection_stability.winner_counts([per_seed_results[seed]["selectedModel"] for seed in SEEDS])
    for name, count in counts.items():
        print(f"  {name:24} won {count}/{len(SEEDS)} seeds")

    _section("STABLE-DEFAULT-CANDIDATE POLICY (must be eligible on every evaluated seed)")
    for algorithm in algorithms:
        stable = selection_stability.is_stable_default_candidate(aggregates[algorithm])
        print(f"  {algorithm:24} stableDefaultCandidate={stable}")

    _section("HARD/SOFT GATE MARGINS PER ALGORITHM (seed=42, for reference)")
    result = per_seed_results[SEEDS[0]]
    for algorithm in algorithms:
        severity = result["eligibilitySeverity"][algorithm]
        print(f"  {algorithm:24} hardFailed={severity['hardFailedGates']} softFailed={severity['softFailedGates']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
