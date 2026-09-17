"""Task 7: ranking-native challenger experiment.

    python -m scripts.run_ranking_challenger_experiment

Compares the 5 existing classifier candidates (trained via the UNCHANGED app.ml.trainer.
train_and_select, on the real flat interaction dataset -- exactly Tasks 4-6's established
baseline) against 3 ranking-native challengers (XGBRanker/LGBMRanker/CatBoostRanker, trained
via app.ml.ranker_trainer on synthetic ranking groups -- see app.ml.ranking_groups for why real
impression groups do not exist in this dataset). Both sides are routed through the IDENTICAL
app.ml.eligibility gates, app.benchmark pipeline, app.ml.reranker, and app.ml.quality_scorer/
app.ml.eligibility_policy layered-evaluation machinery -- no gate, benchmark, or selection logic
is duplicated for rankers.

Never wires a ranker into app.ml.trainer/app.ml.algorithm_registry's production selection --
challenger-only throughout (Task 7 spec section 23). Never tunes classifier hyperparameters/
features/sample weights/gates. Never commits/pushes anything.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

_TRAINING_DB_DIR = tempfile.mkdtemp(prefix="ranking_challenger_training_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TRAINING_DB_DIR}/training.db")

from app.ml.ranker_registry import CLASSIFIER_TO_RANKER
from app.ml.ranker_trainer import train_ranking_challengers
from app.ml.trainer import train_and_select
from scripts.generate_synthetic_data import generate

REFERENCE_TIMESTAMP = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
ESTIMATOR_SEEDS = (42, 101, 2026)
DATASET_SEEDS = (20260721, 42, 101)  # 20260721 == scripts.generate_synthetic_data.SEED (today's default)
GATE_NAMES_OF_INTEREST = ("negative", "notInterested", "recent", "session", "semantic")


def _build_classifier_training_dataset(seed: int = 20260721):
    from sqlalchemy import select

    from app.db.database import SessionLocal
    from app.db.models import Content
    from app.db.repositories import interactions
    from app.ml.dataset_builder import build_dataset

    generate(reference_timestamp=REFERENCE_TIMESTAMP, seed=seed)
    db = SessionLocal()
    try:
        rows = interactions(db)
        content_by_id = {c.content_id: c for c in db.scalars(select(Content)).all()}
        return build_dataset(rows, content_by_id)
    finally:
        db.close()


def _section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def _classifier_row(name: str, result: dict) -> dict:
    report = result["eligibility"][name]
    decision = result["eligibilityDecisions"][name]
    candidate_eval = result["candidateEvaluations"][name]
    by_difficulty = candidate_eval["endToEnd"]["byDifficulty"]
    return {
        "family": "classifier",
        "gates": {g: report["gates"][g]["margin"] for g in GATE_NAMES_OF_INTEREST},
        "hardPassed": result["eligibilitySeverity"][name]["hardPassed"],
        "hardTotal": result["eligibilitySeverity"][name]["hardTotal"],
        "byDifficulty": by_difficulty,
        "criticalPassRate": decision["criticalPassRate"],
        "eligible": decision["eligible"],
        "qualityScore": result["qualityScores"][name]["score"],
    }


def _ranker_row(name: str, result: dict) -> dict:
    report = result["eligibility"][name]
    decision = result["eligibilityDecisions"][name]
    candidate_eval = result["candidateEvaluations"][name]
    by_difficulty = candidate_eval["endToEnd"]["byDifficulty"]
    return {
        "family": "ranker",
        "gates": {g: report["gates"][g]["margin"] for g in GATE_NAMES_OF_INTEREST},
        "hardPassed": result["eligibilitySeverity"][name]["hardPassed"],
        "hardTotal": result["eligibilitySeverity"][name]["hardTotal"],
        "byDifficulty": by_difficulty,
        "criticalPassRate": decision["criticalPassRate"],
        "eligible": decision["eligible"],
        "qualityScore": result["qualityScores"][name]["score"],
    }


def _print_experiment_table(rows: dict[str, dict]) -> None:
    _section("REQUIRED EXPERIMENT TABLE")
    header = (f"  {'Model':24}{'Family':11}{'NegMargin':>11}{'NotIntMargin':>13}{'HardGates':>11}"
              f"{'MedNDCG':>9}{'HardNDCG':>9}{'AdvNDCG':>9}{'CritRate':>9}{'Eligible':>9}")
    print(header)
    for name, row in rows.items():
        by_difficulty = row["byDifficulty"]
        print(f"  {name:24}{row['family']:11}{row['gates']['negative']:>+11.4f}{row['gates']['notInterested']:>+13.4f}"
              f"{f'{row['hardPassed']}/{row['hardTotal']}':>11}"
              f"{by_difficulty.get('MEDIUM', {}).get('finalNdcgAt10', 0.0):>9.4f}"
              f"{by_difficulty.get('HARD', {}).get('finalNdcgAt10', 0.0):>9.4f}"
              f"{by_difficulty.get('ADVERSARIAL', {}).get('finalNdcgAt10', 0.0):>9.4f}"
              f"{row['criticalPassRate']:>9.2f}{row['eligible']!s:>9}")


def _print_pairwise_comparison(rows: dict[str, dict]) -> None:
    _section("REQUIRED PAIRWISE CLASSIFIER -> RANKER COMPARISON")
    for classifier_name, ranker_name in CLASSIFIER_TO_RANKER.items():
        c, r = rows[classifier_name], rows[ranker_name]
        print(f"  {classifier_name} -> {ranker_name}")
        print(f"    negative margin:      {c['gates']['negative']:+.4f} -> {r['gates']['negative']:+.4f}  "
              f"(delta {r['gates']['negative'] - c['gates']['negative']:+.4f})")
        print(f"    notInterested margin: {c['gates']['notInterested']:+.4f} -> {r['gates']['notInterested']:+.4f}  "
              f"(delta {r['gates']['notInterested'] - c['gates']['notInterested']:+.4f})")
        c_hard = c["byDifficulty"].get("HARD", {}).get("finalNdcgAt10", 0.0)
        r_hard = r["byDifficulty"].get("HARD", {}).get("finalNdcgAt10", 0.0)
        c_adv = c["byDifficulty"].get("ADVERSARIAL", {}).get("finalNdcgAt10", 0.0)
        r_adv = r["byDifficulty"].get("ADVERSARIAL", {}).get("finalNdcgAt10", 0.0)
        print(f"    HARD final NDCG@10:   {c_hard:.4f} -> {r_hard:.4f}  (delta {r_hard - c_hard:+.4f})")
        print(f"    ADV final NDCG@10:    {c_adv:.4f} -> {r_adv:.4f}  (delta {r_adv - c_adv:+.4f})")
        print(f"    eligible:             {c['eligible']} -> {r['eligible']}")


def _candidate_level_test(scorer_or_model, feature_columns) -> tuple[float, float]:
    import pandas as pd

    from app.ml.dataset_builder import FeatureHistory

    now = REFERENCE_TIMESTAMP

    class _Row:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Content:
        def __init__(self, **kw):
            self.hashtags = kw.get("hashtags", [])
            self.topics = []
            self.entities = []
            self.subgenres = []
            self.title = None

    def interaction(cid, *, event_type, watch_percentage, when):
        return _Row(event_id=cid, user_id="probe-user", content_id=cid, creator_id="coach", category="SPORT",
                    event_type=event_type, watch_time_seconds=watch_percentage, content_duration_seconds=100.0,
                    watch_percentage=watch_percentage, liked=False, shared=False, favorited=False,
                    commented=False, creator_followed=False, timestamp=when)

    from datetime import timedelta
    history = FeatureHistory()
    for i in range(8):
        history.update(interaction(f"fb{i}", event_type="VIDEO_COMPLETED", watch_percentage=93, when=now - timedelta(days=30 + i)),
                        content=_Content(hashtags=["FOOTBALL"]))
    for i in range(4):
        history.update(interaction(f"tn{i}", event_type="CONTENT_NOT_INTERESTED", watch_percentage=4, when=now - timedelta(days=10 + i)),
                        content=_Content(hashtags=["TENNIS"]))

    football = history.features(user_id="probe-user", category="SPORT", creator_id="coach", content_id="cand-football",
                                 timestamp=now, content_popularity_score=0.6, content_created_at=now, hashtags=["FOOTBALL"])
    tennis = history.features(user_id="probe-user", category="SPORT", creator_id="coach", content_id="cand-tennis",
                               timestamp=now, content_popularity_score=0.6, content_created_at=now, hashtags=["TENNIS"])
    frame = pd.DataFrame([football, tennis])[feature_columns]
    scores = scorer_or_model.predict_proba(frame)[:, 1]
    return float(scores[0]), float(scores[1])


def main() -> int:
    print("=" * 100)
    print("PART 1: CLASSIFIERS (real flat dataset, default seed, app.ml.trainer.train_and_select unmodified)")
    print("=" * 100)
    classifier_df = _build_classifier_training_dataset()
    classifier_result = train_and_select(classifier_df, random_seed=ESTIMATOR_SEEDS[0])

    print("\n" + "=" * 100)
    print("PART 2: RANKING CHALLENGERS (synthetic ranking groups, default seed)")
    print("=" * 100)
    ranker_result = train_ranking_challengers(random_seed=ESTIMATOR_SEEDS[0])

    rows = {}
    for name in classifier_result["modelComparison"]:
        rows[name] = _classifier_row(name, classifier_result)
    for name in ranker_result["eligibility"]:
        rows[name] = _ranker_row(name, ranker_result)

    _print_experiment_table(rows)
    _print_pairwise_comparison(rows)

    _section("CANDIDATE-LEVEL FOOTBALL vs REJECTED-TENNIS TEST")
    from app.ml.dataset_builder import CATEGORICAL, NUMERIC
    feature_columns = CATEGORICAL + list(NUMERIC)
    for name in classifier_result["modelComparison"]:
        if name == classifier_result["selectedModel"]:
            football, tennis = _candidate_level_test(classifier_result["model"], feature_columns)
            print(f"  {name:24} (winner) football={football:.6f} tennis={tennis:.6f} margin={football - tennis:+.6f}")
    for name, scorer in ranker_result["scorers"].items():
        football, tennis = _candidate_level_test(scorer, feature_columns)
        print(f"  {name:24} football={football:.6f} tennis={tennis:.6f} margin={football - tennis:+.6f}")

    _section("ESTIMATOR-SEED ROBUSTNESS (rankers, default dataset, seeds 42/101/2026)")
    for seed in ESTIMATOR_SEEDS:
        result = train_ranking_challengers(random_seed=seed)
        eligible = [name for name, decision in result["eligibilityDecisions"].items() if decision["eligible"]]
        print(f"  seed={seed:6} eligibleRankers={eligible}")

    _section("DATASET-SEED ROBUSTNESS (rankers only -- synthetic ranking groups already support "
              "a `group_seed`; classifier dataset-seed robustness requires a separate isolated-DB "
              "run, see scripts/run_ranking_challenger_dataset_seeds.py)")
    for seed in DATASET_SEEDS:
        result = train_ranking_challengers(random_seed=ESTIMATOR_SEEDS[0], group_seed=seed)
        eligible = [name for name, decision in result["eligibilityDecisions"].items() if decision["eligible"]]
        print(f"  groupSeed={seed:10} eligibleRankers={eligible}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
