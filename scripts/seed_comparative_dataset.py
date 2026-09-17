"""Seed exactly one comparative-experiment dataset (spec: "same algorithm, different
dataset") into whatever database `DATABASE_URL` currently resolves to.

Meant to run inside the app container, against that comparative environment's own isolated
Postgres volume (see README "Comparative Experiments" and `.env.experiment-*.env`):

    docker compose --env-file .env --env-file .env.experiment-sport.env exec app \
        python -m scripts.seed_comparative_dataset --dataset sport

`--experiment-version` defaults to `v1`, so the command above -- the exact command documented
before the v2 rigor pass existed -- is unchanged and produces exactly the v1 dataset it always
has (unmodified generation formulas, `datetime.now()` anchor, no post-generation
stratification). Pass `--experiment-version v2` to seed the v2 dataset instead, against a v2
profile (`.env.experiment-sport-v2.env`): fixed generation timestamp, deterministic
post-generation class-balance stratification (see
`app.experiments.comparative_dataset_generation`), into `comparative-sport-v2`'s own isolated
volume. v1 and v2 are never mixed into the same database -- an `EXPERIMENT_ID` mismatch or the
presence of another comparative experiment's rows fails fast before anything is written.

Deliberately separate from `scripts/generate_synthetic_data.py` (the original fixed-shape
generator) and from `scripts/run_experiment.py` (which trains into an isolated temp SQLite
database, never the real running stack) -- this script writes into the real configured
database, exactly as an operator seeding a real environment would, and is idempotent: rerunning
it against a database that already has this dataset's rows is a safe no-op, not a duplicate
insert.

Usage:
    python -m scripts.seed_comparative_dataset --dataset sport --describe                      # v1, print only
    python -m scripts.seed_comparative_dataset --dataset sport                                  # v1 (default)
    python -m scripts.seed_comparative_dataset --dataset sport --experiment-version v2           # v2
"""
from __future__ import annotations

import argparse
import json
import sys

from app.experiments.comparative_definitions import COMPARATIVE_DEFINITIONS
from app.experiments.comparative_definitions_v2 import COMPARATIVE_DEFINITIONS_V2

_DEFINITIONS_BY_VERSION = {"v1": COMPARATIVE_DEFINITIONS, "v2": COMPARATIVE_DEFINITIONS_V2}


def _describe(dataset: str, *, version: str) -> int:
    # Diagnostic-only output goes to stderr (not stdout): scripts.run_comparative_experiment
    # may invoke this module's _seed() internally (its --seed flag) while reserving stdout
    # exclusively for its own final JSON report, so nothing in this module ever prints to
    # stdout -- keeps `... | some-json-consumer` safe regardless of which entry point ran it.
    definition = _DEFINITIONS_BY_VERSION[version][dataset]
    print(f"experimentVersion={version}", file=sys.stderr)
    print(f"experimentId={definition.experiment_id}", file=sys.stderr)
    print(f"datasetVersion={definition.dataset_version}", file=sys.stderr)
    print(f"synthetic={definition.synthetic}", file=sys.stderr)
    print(f"seed={definition.seed}", file=sys.stderr)
    print(f"algorithm={definition.algorithm}", file=sys.stderr)
    print(f"dominantCategory={definition.dominant_category}", file=sys.stderr)
    print(f"categoryWeights={json.dumps(definition.category_weights)}", file=sys.stderr)
    print(f"users={definition.users} contents={definition.contents} creators={definition.creators} interactions={definition.interactions}", file=sys.stderr)
    print(f"description={definition.description}", file=sys.stderr)
    if definition.category_mapping_note:
        print(f"categoryMappingNote={definition.category_mapping_note}", file=sys.stderr)
    return 0


def _already_seeded(db, experiment_id: str) -> bool:
    from sqlalchemy import select

    from app.db.models import Interaction
    prefix = f"syn-cmp-{experiment_id}-"
    return db.scalar(select(Interaction).where(Interaction.event_id.like(f"{prefix}%")).limit(1)) is not None


def _seed(dataset: str, *, version: str = "v1") -> int:
    from sqlalchemy import select

    from app.db.database import Base, SessionLocal, engine
    from app.db.models import Content, Interaction
    from app.experiments.comparative_dataset_generation import (
        category_distribution,
        generate_comparative_dataset,
    )
    from app.experiments.comparative_invariants import (
        COMPARATIVE_REFERENCE_TIMESTAMP_V2,
        TARGET_LABELLED_SAMPLES_PER_CLASS_V2,
        validate_dataset_complete_v2,
        validate_experiment_id_matches,
        validate_no_foreign_comparative_rows,
    )
    from app.experiments.dataset_summary import video_dataset_summary

    definition = _DEFINITIONS_BY_VERSION[version][dataset]
    Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        # Fail fast, before touching the database, if this environment doesn't identify the
        # requested dataset, or if it already contains another comparative experiment's rows.
        validate_experiment_id_matches(definition)
        validate_no_foreign_comparative_rows(db, definition)

        if _already_seeded(db, definition.experiment_id):
            print(f"Dataset '{definition.experiment_id}' already seeded in this database; nothing inserted.", file=sys.stderr)
        else:
            print(f"Seeding comparative dataset '{definition.experiment_id}' (version={version}, target category_weights={definition.category_weights}) ...", file=sys.stderr)
            if version == "v2":
                generate_comparative_dataset(
                    db, definition,
                    reference_timestamp=COMPARATIVE_REFERENCE_TIMESTAMP_V2,
                    downsample_target_per_class=TARGET_LABELLED_SAMPLES_PER_CLASS_V2,
                )
            else:
                # v1: no new keyword arguments -- byte-for-byte the same call every existing
                # v1 caller has always made.
                generate_comparative_dataset(db, definition)
            print("Seeding complete.", file=sys.stderr)

        rows = db.scalars(
            select(Interaction).where(Interaction.event_id.like(f"syn-cmp-{definition.experiment_id}-%"))
        ).all()
        contents = db.scalars(
            select(Content).where(Content.content_id.like(f"cmp-{definition.experiment_id}-%"))
        ).all()
        summary = video_dataset_summary(rows)
        realized_categories = category_distribution(rows)

        print("--- Dataset verification (measured from the database, not assumed) ---", file=sys.stderr)
        print(f"totalInteractions={summary['totalInteractions']}", file=sys.stderr)
        print(f"labelledSamples={summary['labelledSamples']} positiveSamples={summary['positiveSamples']} negativeSamples={summary['negativeSamples']}", file=sys.stderr)
        print(f"positiveRatio={summary['positiveRatio']} negativeRatio={summary['negativeRatio']}", file=sys.stderr)
        print(f"uniqueUsers={summary['uniqueUsers']} uniqueContents={len(contents)} uniqueCreators={summary['uniqueCreators']}", file=sys.stderr)
        print(f"realizedCategoryDistribution={json.dumps(realized_categories)}", file=sys.stderr)
        print(f"targetCategoryWeights={json.dumps(definition.category_weights)}", file=sys.stderr)

        if version == "v2":
            # Hard, loud gate: raises ComparativeInvariantError (not a soft warning) if the
            # realized class balance isn't *exactly* the configured target -- e.g. because an
            # idempotent skip found stale data from a different generator version.
            validate_dataset_complete_v2(db, definition)
            print(f"v2 exact class balance confirmed: {TARGET_LABELLED_SAMPLES_PER_CLASS_V2}/{TARGET_LABELLED_SAMPLES_PER_CLASS_V2} positive/negative.", file=sys.stderr)
        elif summary["labelledSamples"] < 100 or summary["positiveSamples"] == 0 or summary["negativeSamples"] == 0:
            print("WARNING: fewer than 100 labelled rows or a missing class -- training will fail with INSUFFICIENT_TRAINING_DATA.", file=sys.stderr)
            return 1
        return 0
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(COMPARATIVE_DEFINITIONS), help="Comparative dataset id to seed.")
    parser.add_argument("--experiment-version", choices=("v1", "v2"), default="v1",
                         help="Defaults to v1 (the original, unchanged documented command). Pass v2 for the rigor-pass pipeline.")
    parser.add_argument("--describe", action="store_true", help="Print the definition and exit. Does not touch the database.")
    args = parser.parse_args(argv)

    if args.describe:
        return _describe(args.dataset, version=args.experiment_version)
    return _seed(args.dataset, version=args.experiment_version)


if __name__ == "__main__":
    sys.exit(main())
