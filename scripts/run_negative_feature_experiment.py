"""Negative-feedback feature representation: controlled experiment.

    python -m scripts.run_negative_feature_experiment

Compares BASELINE (today's production NUMERIC) against NEG_A/NEG_B/NEG_C/NEG_V2 (see
app.ml.negative_feedback_features) -- same dataset, split, estimator seed, sample weights,
hyperparameters, calibration, and layered selection architecture (app.ml.trainer.
train_and_select) throughout. ONLY the numeric feature list passed to
app.ml.pipeline_builder.build_classifier_pipeline differs between configurations.

Every new negative-feedback feature is already computed unconditionally by
app.ml.dataset_builder.FeatureHistory.features() (see that module) -- this script never
computes a feature itself, it only selects which already-computed columns feed a given
configuration's `numeric` list, via a temporary, fully-reversible monkeypatch of the exact
module-level FEATURES/NUMERIC names app.ml.eligibility/app.ml.predictor/app.ml.trainer/
app.benchmark.runner/app.services.recommendation_service already read (mirrors this project's
established MODEL_PATH/METADATA_PATH monkeypatch convention, e.g. app.benchmark.runner.
use_model) -- production app.ml.dataset_builder.NUMERIC/FEATURES is never changed by this
script.

Never tunes hyperparameters, sample weights, or gate tolerances. Never trains on benchmark
scenario data. Never commits/pushes/writes to the real active model artifact.
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from datetime import datetime, timezone

_TRAINING_DB_DIR = tempfile.mkdtemp(prefix="negative_feature_experiment_training_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TRAINING_DB_DIR}/training.db")

from app.ml import negative_feedback_features as nff
from app.ml.dataset_builder import CATEGORICAL, NUMERIC
from app.ml.trainer import train_and_select
from scripts.generate_synthetic_data import generate

REFERENCE_TIMESTAMP = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
PRIMARY_SEED = 42
ESTIMATOR_SEEDS = (42, 101, 2026)
GATE_NAMES_OF_INTEREST = ("negative", "notInterested", "recent", "session", "semantic", "subthemeRejectionLocalization")


def _build_training_dataset(seed: int = 20260721):
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


@contextlib.contextmanager
def alternate_feature_set(numeric_features: list[str]):
    """Temporarily repoints the exact module-level FEATURES/NUMERIC names app.ml.eligibility/
    app.ml.predictor/app.ml.trainer/app.benchmark.runner/app.services.recommendation_service
    already read, so a candidate trained on `numeric_features` can be scored through the REAL,
    unmodified gate/benchmark pipeline without ever changing production app.ml.dataset_builder.
    NUMERIC/FEATURES. Restores every one of them on exit, even on error."""
    from app.benchmark import runner as bench_runner
    from app.ml import eligibility, predictor, trainer
    from app.services import recommendation_service

    feature_names = CATEGORICAL + list(numeric_features)
    patches = [
        (eligibility, "FEATURES", feature_names),
        (predictor, "FEATURES", feature_names),
        (predictor, "NUMERIC", list(numeric_features)),
        (trainer, "FEATURES", feature_names),
        (trainer, "NUMERIC", list(numeric_features)),
        (bench_runner, "FEATURES", feature_names),
        (recommendation_service, "FEATURES", feature_names),
    ]
    originals = [(module, attr, getattr(module, attr)) for module, attr, _ in patches]
    for module, attr, value in patches:
        setattr(module, attr, value)
    try:
        yield feature_names
    finally:
        for module, attr, value in originals:
            setattr(module, attr, value)


def _section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def _run_configuration(training_df, config_name: str, extra_features: tuple[str, ...], *, seed: int) -> dict:
    with alternate_feature_set(list(NUMERIC) + list(extra_features)):
        return train_and_select(training_df, random_seed=seed)


def _print_gate_margin_table(results_by_config: dict[str, dict]) -> None:
    _section("HARD-GATE MARGINS BY CONFIGURATION (seed=42): negative / notInterested / recent / session / semantic / subtheme")
    header = f"  {'Config':10}{'Algorithm':24}" + "".join(f"{g[:12]:>14}" for g in GATE_NAMES_OF_INTEREST)
    print(header)
    for config_name, result in results_by_config.items():
        for algorithm, report in result["eligibility"].items():
            gates = report["gates"]
            row = f"  {config_name:10}{algorithm:24}"
            for gate in GATE_NAMES_OF_INTEREST:
                row += f"{gates[gate]['margin']:>+14.4f}"
            print(row)


def _print_ranking_table(results_by_config: dict[str, dict]) -> None:
    _section("RANKING/SELECTION METRICS BY CONFIGURATION (seed=42)")
    header = (f"  {'Config':10}{'Algorithm':24}{'MedNDCG':>10}{'HardNDCG':>10}{'AdvNDCG':>10}"
              f"{'CritRate':>10}{'PRAUC':>8}{'Quality':>9}{'Eligible':>10}")
    print(header)
    for config_name, result in results_by_config.items():
        for algorithm in result["modelComparison"]:
            candidate_eval = result["candidateEvaluations"][algorithm]
            by_difficulty = candidate_eval["endToEnd"]["byDifficulty"]
            decision = result["eligibilityDecisions"][algorithm]
            quality = result["qualityScores"][algorithm]["score"]
            pr_auc = float(result["modelComparison"][algorithm].get("prAuc") or 0.0)
            print(f"  {config_name:10}{algorithm:24}"
                  f"{by_difficulty.get('MEDIUM', {}).get('finalNdcgAt10', 0.0):>10.4f}"
                  f"{by_difficulty.get('HARD', {}).get('finalNdcgAt10', 0.0):>10.4f}"
                  f"{by_difficulty.get('ADVERSARIAL', {}).get('finalNdcgAt10', 0.0):>10.4f}"
                  f"{decision['criticalPassRate']:>10.2f}{pr_auc:>8.4f}{quality:>9.4f}"
                  f"{decision['eligible']!s:>10}")


def _print_winner_summary(results_by_config: dict[str, dict]) -> None:
    _section("WINNER PER CONFIGURATION (seed=42)")
    for config_name, result in results_by_config.items():
        print(f"  {config_name:10} winner={result['selectedModel']:24} eligibleSelection={result['eligibleSelection']}")


def _print_feature_importance(training_df, config_name: str, extra_features: tuple[str, ...], *, seed: int) -> None:
    _section(f"FEATURE IMPORTANCE (diagnostic only, config={config_name}, seed={seed})")
    with alternate_feature_set(list(NUMERIC) + list(extra_features)) as feature_names:
        result = train_and_select(training_df, random_seed=seed)
        for algorithm in result["modelComparison"]:
            uncalibrated = result["uncalibratedModel"] if algorithm == result["selectedModel"] else None
            if uncalibrated is None:
                continue  # only the winner's uncalibrated pipeline is retained by train_and_select
            estimator = uncalibrated.named_steps["model"]
            numeric_names = feature_names[len(CATEGORICAL):]
            print(f"  {algorithm} (winner) -- top negative-feedback-feature importances:")
            if hasattr(estimator, "coef_"):
                # LogisticRegression: coefficients on the SCALED numeric block, in `numeric`
                # column order -- category one-hot columns come first in the ColumnTransformer
                # output, so the numeric block is the trailing len(numeric_names) coefficients.
                coefs = estimator.coef_[0][-len(numeric_names):]
                ranked = sorted(zip(numeric_names, coefs), key=lambda item: abs(item[1]), reverse=True)
            elif hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_[-len(numeric_names):]
                ranked = sorted(zip(numeric_names, importances), key=lambda item: abs(item[1]), reverse=True)
            else:
                print("    (no native coefficients/importances exposed)")
                continue
            for name, value in ranked[:8]:
                marker = " <-- negative-feedback feature" if name in nff.NEG_V2 else ""
                print(f"    {name:42}{value:>+10.4f}{marker}")


def _print_candidate_level_not_interested_test(training_df, config_name: str, extra_features: tuple[str, ...], *, seed: int) -> None:
    """Task 6 spec section 25: same user, strong SPORT/Football, repeated Tennis rejection --
    candidate pool Football/Tennis/generic-SPORT/secondary-MUSIC. Raw-model scores only (no
    reranker/benchmark truth encoded into features)."""
    from datetime import timedelta

    import pandas as pd

    from app.ml.dataset_builder import FeatureHistory

    _section(f"CANDIDATE-LEVEL NOT_INTERESTED TEST (config={config_name}, seed={seed})")
    now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)

    class _Row:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Content:
        def __init__(self, **kw):
            self.hashtags = kw.get("hashtags", [])
            self.topics = kw.get("topics", [])
            self.entities = kw.get("entities", [])
            self.subgenres = kw.get("subgenres", [])
            self.title = None

    def interaction(cid, *, event_type, watch_percentage, when):
        return _Row(event_id=cid, user_id="probe-user", content_id=cid, creator_id="coach", category="SPORT",
                    event_type=event_type, watch_time_seconds=watch_percentage, content_duration_seconds=100.0,
                    watch_percentage=watch_percentage, liked=False, shared=False, favorited=False,
                    commented=False, creator_followed=False, timestamp=when)

    history = FeatureHistory()
    for i in range(8):
        history.update(interaction(f"fb{i}", event_type="VIDEO_COMPLETED", watch_percentage=93, when=now - timedelta(days=30 + i)),
                        content=_Content(hashtags=["FOOTBALL"]))
    for i in range(4):
        history.update(interaction(f"tn{i}", event_type="CONTENT_NOT_INTERESTED", watch_percentage=4, when=now - timedelta(days=10 + i)),
                        content=_Content(hashtags=["TENNIS"]))

    candidates = {
        "football": history.features(user_id="probe-user", category="SPORT", creator_id="coach", content_id="cand-football",
                                      timestamp=now, content_popularity_score=0.6, content_created_at=now, hashtags=["FOOTBALL"]),
        "tennis": history.features(user_id="probe-user", category="SPORT", creator_id="coach", content_id="cand-tennis",
                                    timestamp=now, content_popularity_score=0.6, content_created_at=now, hashtags=["TENNIS"]),
        "genericSport": history.features(user_id="probe-user", category="SPORT", creator_id="coach", content_id="cand-generic",
                                          timestamp=now, content_popularity_score=0.5, content_created_at=now),
        "secondaryMusic": history.features(user_id="probe-user", category="MUSIC", creator_id="dj", content_id="cand-music",
                                            timestamp=now, content_popularity_score=0.5, content_created_at=now),
    }

    with alternate_feature_set(list(NUMERIC) + list(extra_features)) as feature_names:
        result = train_and_select(training_df, random_seed=seed)
        frame = pd.DataFrame(list(candidates.values()))[feature_names]
        for algorithm, model in [(result["selectedModel"], result["model"])]:
            scores = model.predict_proba(frame)[:, 1]
            by_name = dict(zip(candidates, scores))
            print(f"  winner={algorithm}: football={by_name['football']:.6f} tennis={by_name['tennis']:.6f} "
                  f"margin={by_name['football'] - by_name['tennis']:+.6f} "
                  f"(genericSport={by_name['genericSport']:.6f} secondaryMusic={by_name['secondaryMusic']:.6f})")


def main() -> int:
    print("Building one shared synthetic training dataset (default seed)...")
    training_df = _build_training_dataset()

    _section("PRIMARY CONTROLLED COMPARISON (same dataset/split/seed=42/sample weights/hyperparameters/calibration/selection architecture)")
    results_by_config: dict[str, dict] = {}
    for config_name, extra_features in nff.FEATURE_GROUPS.items():
        print(f"\nRunning configuration {config_name} (extra features: {extra_features or 'none'})...")
        results_by_config[config_name] = _run_configuration(training_df, config_name, extra_features, seed=PRIMARY_SEED)

    _print_gate_margin_table(results_by_config)
    _print_ranking_table(results_by_config)
    _print_winner_summary(results_by_config)

    # Feature importance + candidate-level tests for BASELINE and NEG_V2 only (the two
    # endpoints of the comparison) -- avoids re-training every configuration a second time.
    for config_name in ("BASELINE", "NEG_V2"):
        _print_feature_importance(training_df, config_name, nff.FEATURE_GROUPS[config_name], seed=PRIMARY_SEED)
        _print_candidate_level_not_interested_test(training_df, config_name, nff.FEATURE_GROUPS[config_name], seed=PRIMARY_SEED)

    _section("MULTI-SEED VALIDATION OF NEG_V2 (estimator seeds 42/101/2026, same dataset)")
    for seed in ESTIMATOR_SEEDS:
        result = _run_configuration(training_df, "NEG_V2", nff.FEATURE_GROUPS["NEG_V2"], seed=seed)
        eligible = [name for name, decision in result["eligibilityDecisions"].items() if decision["eligible"]]
        print(f"  seed={seed:6} winner={result['selectedModel']:24} eligibleCandidates={eligible}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
