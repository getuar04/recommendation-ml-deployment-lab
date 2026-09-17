"""Developer-facing CLI for the independent ranking benchmark (app.benchmark).

    python -m scripts.run_ranking_benchmark
    python -m scripts.run_ranking_benchmark --algorithm LogisticRegression
    python -m scripts.run_ranking_benchmark --difficulty hard
    python -m scripts.run_ranking_benchmark --seed 42 --json

Trains the requested candidate algorithm(s) on one shared, freshly-generated synthetic
training dataset (an isolated database, physically separate from the benchmark's own
database -- see app.benchmark.runner.isolated_session), then scores every benchmark scenario
through the real recommendation_service.recommend() for each algorithm in turn. Prints a
concise plain-text summary; add --json for a machine-readable dump instead.

This script never trains on benchmark scenario data and never writes to the real active
model artifact -- see app.benchmark.runner.use_model.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# A fresh, unique temp file every invocation -- never a stale/reused training database, and
# physically separate from the benchmark scenarios' own in-memory databases (see
# app.benchmark.runner.isolated_session). Must be set before any app module is imported.
_TRAINING_DB_DIR = tempfile.mkdtemp(prefix="ranking_benchmark_training_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TRAINING_DB_DIR}/training.db")

from app.benchmark.runner import run_full_matrix, train_all_algorithms
from app.benchmark.scenario_types import ScenarioResult
from app.ml.algorithm_registry import (
    ALL_ALGORITHM_NAMES,
    unavailable_algorithms,
)
from app.ml.dataset_builder import build_dataset
from app.ml.trainer import RANDOM_SEED
from scripts.generate_synthetic_data import generate

REFERENCE_TIMESTAMP = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)


def _build_training_dataset():
    """Generates one shared synthetic training dataset with the SAME generator training uses
    (scripts/generate_synthetic_data.py) -- deliberately never any benchmark scenario pattern
    (see app.benchmark.scenarios' module docstring). Uses the module-level DATABASE_URL this
    script points at a fresh temp file before any app module is imported (see top of file),
    physically separate from the benchmark scenarios' own in-memory databases
    (app.benchmark.runner.isolated_session)."""
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


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _print_table(matrix: dict[str, dict]) -> None:
    difficulties = ["EASY", "MEDIUM", "HARD", "ADVERSARIAL"]
    print("\n=== Raw vs. final NDCG@10 by difficulty ===")
    header = f"{'Algorithm':22}" + "".join(f"{d + ' raw':>16}{d + ' final':>18}" for d in difficulties)
    print(header)
    for algorithm, result in matrix.items():
        scenario_results: dict[str, ScenarioResult] = result["scenarios"]
        row = f"{algorithm:22}"
        for difficulty in difficulties:
            matching = [r for r in scenario_results.values() if r.difficulty == difficulty]
            raw_ndcg = sum(r.raw.metrics["ndcgAt10"] for r in matching) / len(matching) if matching else float("nan")
            final_ndcg = sum(r.reranked.metrics["ndcgAt10"] for r in matching) / len(matching) if matching else float("nan")
            row += f"{_fmt(raw_ndcg):>16}{_fmt(final_ndcg):>18}"
        print(row)

    print("\n=== Critical constraints (passed/total) and reranker effect ===")
    print(f"{'Algorithm':22}{'Raw crit.':>12}{'Final crit.':>14}{'Improved':>10}{'Degraded':>10}{'Neutral':>10}")
    for algorithm, result in matrix.items():
        all_results: list[ScenarioResult] = [*result["scenarios"].values(), result["dominantCategory"]]
        raw_passed = sum(r.raw.critical_constraints_passed for r in all_results)
        raw_total = sum(r.raw.critical_constraints_total for r in all_results)
        final_passed = sum(r.reranked.critical_constraints_passed for r in all_results)
        final_total = sum(r.reranked.critical_constraints_total for r in all_results)
        improved = sum(len(r.reranker_improved) for r in all_results)
        degraded = sum(len(r.reranker_degraded) for r in all_results)
        neutral = sum(len(r.reranker_neutral) for r in all_results)
        print(f"{algorithm:22}{f'{raw_passed}/{raw_total}':>12}{f'{final_passed}/{final_total}':>14}"
              f"{improved:>10}{degraded:>10}{neutral:>10}")

    print("\n=== Dominant-category diversity (final ranking, Top-10) ===")
    print(f"{'Algorithm':22}{'Top-cat share':>16}{'Unique cats':>14}{'Unique creators':>18}")
    for algorithm, result in matrix.items():
        diversity = result["dominantCategory"].reranked.metrics
        print(f"{algorithm:22}{_fmt(diversity['topCategoryShare']):>16}"
              f"{diversity['uniqueCategories']:>14}{diversity['uniqueCreators']:>18}")

    print("\n=== Temporal personalization (MUSIC candidates) ===")
    print(f"{'Algorithm':22}{'MUSIC rank T0':>16}{'MUSIC rank T1':>16}{'raw scoreDelta':>16}{'final scoreDelta':>18}")
    for algorithm, result in matrix.items():
        t0, t1 = result["temporal"]["t0"], result["temporal"]["t1"]
        for music_id in ("temporal-music-1", "temporal-music-2"):
            rank_t0 = t0.reranked.ranking.index(music_id) + 1 if music_id in t0.reranked.ranking else None
            rank_t1 = t1.reranked.ranking.index(music_id) + 1 if music_id in t1.reranked.ranking else None
            raw_delta = t1.raw.scores_by_content_id.get(music_id, 0) - t0.raw.scores_by_content_id.get(music_id, 0)
            final_delta = t1.reranked.scores_by_content_id.get(music_id, 0) - t0.reranked.scores_by_content_id.get(music_id, 0)
            print(f"{algorithm + ' ' + music_id:22}{rank_t0!s:>16}{rank_t1!s:>16}"
                  f"{_fmt(raw_delta):>16}{_fmt(final_delta):>18}")

    print("\n=== NOT_INTERESTED localization (final scores) ===")
    print(f"{'Algorithm':22}{'Football (same)':>18}{'Football (diverse)':>20}{'Basketball (same)':>20}{'Basketball (diverse)':>22}")
    for algorithm, result in matrix.items():
        same = result["localization"]["sameSubtheme"]
        diverse = result["localization"]["diverseSubthemes"]
        print(f"{algorithm:22}"
              f"{_fmt(same.reranked.scores_by_content_id.get('localization-football', 0)):>18}"
              f"{_fmt(diverse.reranked.scores_by_content_id.get('localization-football', 0)):>20}"
              f"{_fmt(same.reranked.scores_by_content_id.get('localization-basketball-unrejected', 0)):>20}"
              f"{_fmt(diverse.reranked.scores_by_content_id.get('localization-basketball-unrejected', 0)):>22}")


def _to_jsonable(matrix: dict[str, dict]) -> dict:
    def scenario_json(result: ScenarioResult) -> dict:
        return {
            "scenarioId": result.scenario_id, "difficulty": result.difficulty, "algorithm": result.algorithm,
            "raw": {"ranking": result.raw.ranking, "metrics": result.raw.metrics,
                    "criticalConstraintsPassed": result.raw.critical_constraints_passed,
                    "criticalConstraintsTotal": result.raw.critical_constraints_total},
            "reranked": {"ranking": result.reranked.ranking, "metrics": result.reranked.metrics,
                         "criticalConstraintsPassed": result.reranked.critical_constraints_passed,
                         "criticalConstraintsTotal": result.reranked.critical_constraints_total},
            "rerankerImproved": result.reranker_improved, "rerankerDegraded": result.reranker_degraded,
        }

    return {
        algorithm: {
            "scenarios": {sid: scenario_json(result) for sid, result in data["scenarios"].items()},
            "dominantCategory": scenario_json(data["dominantCategory"]),
            "temporal": {"t0": scenario_json(data["temporal"]["t0"]), "t1": scenario_json(data["temporal"]["t1"])},
            "localization": {"sameSubtheme": scenario_json(data["localization"]["sameSubtheme"]),
                              "diverseSubthemes": scenario_json(data["localization"]["diverseSubthemes"])},
        }
        for algorithm, data in matrix.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the independent recommendation-ranking benchmark.")
    parser.add_argument("--algorithm", default="all", help="'all' or one of: " + ", ".join(ALL_ALGORITHM_NAMES))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for training (default: %(default)s)")
    parser.add_argument("--difficulty", default="all", choices=["all", "easy", "medium", "hard", "adversarial"])
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a table")
    args = parser.parse_args(argv)

    available = [name for name in ALL_ALGORITHM_NAMES if name not in unavailable_algorithms()]
    if args.algorithm == "all":
        algorithms = available
    else:
        if args.algorithm not in ALL_ALGORITHM_NAMES:
            parser.error(f"--algorithm must be 'all' or one of {sorted(ALL_ALGORITHM_NAMES)}")
        if args.algorithm not in available:
            parser.error(f"{args.algorithm!r} is unavailable in this environment: {unavailable_algorithms().get(args.algorithm)}")
        algorithms = [args.algorithm]

    # --json must be pure, parseable JSON on stdout: scripts.generate_synthetic_data.generate()
    # prints its own status line, so that (and this script's own status line) is redirected
    # away from stdout in JSON mode rather than left to pollute the machine-readable output.
    status_stream = io.StringIO() if args.json else sys.stdout
    with contextlib.redirect_stdout(status_stream):
        print(f"Training {len(algorithms)} algorithm(s) on one shared synthetic dataset (modelSeed={args.seed})...")
        training_df = _build_training_dataset()
    training_results = train_all_algorithms(training_df, algorithms=algorithms, random_seed=args.seed)

    try:
        with tempfile.TemporaryDirectory(prefix="ranking_benchmark_models_") as tmp_dir:
            matrix = run_full_matrix(training_results, tmp_dir=Path(tmp_dir))

        if args.difficulty != "all":
            wanted = args.difficulty.upper()
            for data in matrix.values():
                data["scenarios"] = {sid: r for sid, r in data["scenarios"].items() if r.difficulty == wanted}

        if args.json:
            print(json.dumps(_to_jsonable(matrix), indent=2, default=str))
        else:
            _print_table(matrix)
        return 0
    finally:
        shutil.rmtree(_TRAINING_DB_DIR, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
