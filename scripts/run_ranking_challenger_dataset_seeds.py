"""Task 7: classifier dataset-seed robustness, using a genuinely isolated engine/session per
seed (never the app's singleton `app.db.database.engine`/`SessionLocal`) -- the gap Task 6
explicitly left open ("did not run multiple dataset distributions"). Mirrors
app.benchmark.runner.isolated_session's own isolation convention (`make_engine("sqlite:///
:memory:")` + a fresh sessionmaker per run), extended here to scripts.generate_synthetic_data.
generate()'s new `db`/`db_engine` injection parameters (Task 7) so several independent
datasets can be generated within ONE process without ever touching global DB state.

    python -m scripts.run_ranking_challenger_dataset_seeds

Fixed estimator seed throughout (spec section 29: "clearly distinguish dataset-seed robustness
from estimator-seed robustness") -- only the underlying synthetic dataset varies.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

# A real DATABASE_URL must exist before any app.* import touches app.db.database at module
# load time, even though this script never uses the resulting global engine for training data.
_PLACEHOLDER_DB_DIR = tempfile.mkdtemp(prefix="ranking_challenger_dataset_seeds_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_PLACEHOLDER_DB_DIR}/placeholder.db")

from sqlalchemy.orm import sessionmaker

from app.db.database import Base, make_engine
from app.ml.trainer import RANDOM_SEED, train_and_select
from scripts.generate_synthetic_data import generate

REFERENCE_TIMESTAMP = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
DATASET_SEEDS = (20260721, 42, 101)  # 20260721 == scripts.generate_synthetic_data.SEED (today's default dataset)


def _isolated_classifier_dataset(seed: int):
    from sqlalchemy import select

    from app.db.models import Content
    from app.db.repositories import interactions
    from app.ml.dataset_builder import build_dataset

    isolated_engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(isolated_engine)
    isolated_session = sessionmaker(bind=isolated_engine, autoflush=False, expire_on_commit=False)()
    try:
        generate(reference_timestamp=REFERENCE_TIMESTAMP, seed=seed, db=isolated_session, db_engine=isolated_engine)
        rows = interactions(isolated_session)
        content_by_id = {c.content_id: c for c in isolated_session.scalars(select(Content)).all()}
        return build_dataset(rows, content_by_id)
    finally:
        isolated_session.close()
        isolated_engine.dispose()


def main() -> int:
    print(f"{'=' * 100}\nCLASSIFIER DATASET-SEED ROBUSTNESS (fixed estimator seed={RANDOM_SEED}, isolated DB per seed)\n{'=' * 100}")
    for seed in DATASET_SEEDS:
        df = _isolated_classifier_dataset(seed)
        result = train_and_select(df, random_seed=RANDOM_SEED)
        eligible = [name for name, decision in result["eligibilityDecisions"].items() if decision["eligible"]]
        print(f"  datasetSeed={seed:10} rows={len(df):6} winner={result['selectedModel']:24} eligibleClassifiers={eligible}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
