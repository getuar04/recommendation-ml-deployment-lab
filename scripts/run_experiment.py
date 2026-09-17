"""Run exactly one named experiment (spec §44-47), or describe it without mutating
anything. Never touches the real .env, the real database, or the real models/ directory --
uses an isolated SQLite database and isolated model artifact paths under --work-dir, and
writes its JSON report under --experiment-dir (defaults to the EXPERIMENT_DIR setting).

Usage:
    python -m scripts.run_experiment --experiment small_balanced --describe
    python -m scripts.run_experiment --experiment small_balanced --domain video

Requires an explicit --experiment selection; never runs all three scenarios automatically.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from app.experiments.definitions import DEFINITIONS


def _describe(experiment_id: str) -> int:
    definition = DEFINITIONS[experiment_id]
    print(f"experimentId={definition.experiment_id}")
    print(f"datasetVersion={definition.dataset_version}")
    print(f"synthetic={definition.synthetic}")
    print(f"seed={definition.seed}")
    print(f"users={definition.users} contents={definition.contents} creators={definition.creators} interactions={definition.interactions}")
    print(f"categories={','.join(definition.categories)}")
    print(f"description={definition.description}")
    print(f"imbalanceNote={definition.imbalance_note}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, choices=sorted(DEFINITIONS), help="Experiment definition id to run.")
    parser.add_argument("--domain", choices=("video", "live"), default="video")
    parser.add_argument("--describe", action="store_true", help="Print the definition and exit. Does not mutate anything.")
    parser.add_argument("--experiment-dir", default=None, help="Where the JSON report is written (defaults to EXPERIMENT_DIR).")
    parser.add_argument("--work-dir", default=None, help="Scratch directory for the isolated DB/model files (defaults to a fresh temp directory, deleted on success).")
    parser.add_argument("--keep-work-dir", action="store_true", help="Do not delete --work-dir after a successful run (useful for inspection).")
    args = parser.parse_args(argv)

    if args.describe:
        return _describe(args.experiment)

    from app.experiments.runner import run_experiment
    definition = DEFINITIONS[args.experiment]

    if args.experiment_dir is not None:
        experiment_dir = Path(args.experiment_dir)
    else:
        from app.core.config import EXPERIMENT_DIR
        experiment_dir = EXPERIMENT_DIR

    owns_work_dir = args.work_dir is None
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="experiment-run-"))

    print(f"Running experiment '{definition.experiment_id}' (domain={args.domain}, synthetic=True) ...")
    outcome = run_experiment(definition, domain=args.domain, experiment_dir=experiment_dir, work_dir=work_dir)

    if owns_work_dir and not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"status={outcome.status} runId={outcome.run_id} reportPath={outcome.report_path}")
    if outcome.status != "SUCCEEDED":
        print(f"errorCode={outcome.error_code} message={outcome.error_message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
