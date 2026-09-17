"""Pre-flight validation for a v2 comparative-experiment Docker Compose profile -- must be
run, and must pass, BEFORE every `docker compose up` for a v2 environment (spec: "Before
every v2 Docker start, validate that..."). Inspects only the *resolved* configuration (via
`docker compose ... config --format json`); never touches a live container or volume, so it
is always safe to run.

Usage (mirrors the real startup command -- pass the exact same --env-file flags):

    python -m scripts.validate_v2_compose_profile --dataset sport \
        --env-file .env --env-file .env.experiment-sport-v2.env

Checks (app.experiments.comparative_invariants.validate_v2_compose_profile):
- the Compose project name is not the default stack's project name;
- the project name identifies this dataset and v2;
- all three volume names (postgres/model/experiment) identify this dataset and v2;
- EXPERIMENT_ID/DATASET_VERSION resolve to this dataset's v2 experiment id;
- the resolved host ports do not collide with the default stack's or any v1 profile's ports.

Exits non-zero with a message naming exactly which check(s) failed if any invariant does not
hold; never starts, stops, or modifies any container or volume.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from app.experiments.comparative_invariants import (
    DATASET_KEYS,
    ComparativeInvariantError,
    validate_v2_compose_profile,
)


def _resolved_config(env_files: list[str]) -> dict:
    command = ["docker", "compose"]
    for env_file in env_files:
        command += ["--env-file", env_file]
    command += ["config", "--format", "json"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"`{' '.join(command)}` failed (exit {result.returncode}):\n{result.stderr}")
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASET_KEYS)
    parser.add_argument("--env-file", action="append", required=True,
                         help="Pass once per --env-file you intend to give `docker compose up` (in the same order), e.g. --env-file .env --env-file .env.experiment-sport-v2.env")
    args = parser.parse_args(argv)

    try:
        config = _resolved_config(args.env_file)
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"Could not resolve Compose configuration: {exc}", file=sys.stderr)
        return 2

    try:
        validate_v2_compose_profile(config, dataset_key=args.dataset)
    except ComparativeInvariantError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"v2 Compose profile OK for dataset={args.dataset!r} (project={config.get('name')!r}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
