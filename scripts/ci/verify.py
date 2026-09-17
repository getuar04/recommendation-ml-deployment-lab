"""One command to run locally, before opening a Merge Request, that mirrors what
`.gitlab-ci.yml`'s `validate` + `static-analysis` + `test` stages check (PostgreSQL and
Docker are intentionally NOT included here -- see README "CI pipeline" for why, and run
`scripts/ci/check_postgres_integration.py` / `docker build` separately if you need those).

Usage (from the repository root):
    python -m scripts.ci.verify            # fast: forbidden files, migrations, ruff, mypy,
                                            # dependency consistency, a small fast test subset
    python -m scripts.ci.verify --full      # same, but the full pytest suite instead of the
                                            # fast subset (this is what CI's `test:full` job runs)

Exits non-zero if any step fails; runs every step regardless of earlier failures so a single
`verify` invocation reports everything that's wrong, not just the first failure.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASETEMP_FAST = REPO_ROOT / "data" / "pytest-verify-fast-basetemp"
BASETEMP_FULL = REPO_ROOT / "data" / "pytest-verify-full-basetemp"

# A small, fast, representative subset -- unit/contract-shaped tests that don't train a real
# model. Mirrors .gitlab-ci.yml's `test:fast` job selection; kept in this one place so the
# local command and the CI job can never silently diverge (verify.py is the actual
# implementation `test:fast` invokes -- see .gitlab-ci.yml).
FAST_TEST_PATHS = [
    "tests/test_health.py",
    "tests/test_migrations.py",
    "tests/test_auth_me_route.py",
    "tests/test_candidate_service.py",
    "tests/test_artifact_lifecycle.py",
    "tests/test_training.py",
    "tests/test_user_profile_contract.py",
]


def _run(cmd: list[str], *, label: str) -> bool:
    print(f"\n=== {label} ===")
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    ok = result.returncode == 0
    print(f"--- {label}: {'PASSED' if ok else 'FAILED'} (exit {result.returncode}) ---")
    return ok


def build_steps(*, full: bool) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [
        ("forbidden files", [sys.executable, "-m", "scripts.ci.check_forbidden_files"]),
        ("migration sanity (single head, imports cleanly)", [sys.executable, "-m", "scripts.ci.check_migrations"]),
        ("dependency consistency", [sys.executable, "-m", "scripts.ci.check_dependency_consistency"]),
        ("ruff", [sys.executable, "-m", "ruff", "check", "app", "tests", "scripts"]),
        ("mypy", [sys.executable, "-m", "mypy", "app", "scripts"]),
    ]
    if full:
        BASETEMP_FULL.mkdir(parents=True, exist_ok=True)
        steps.append((
            "pytest (full suite)",
            [sys.executable, "-m", "pytest", "-q", f"--basetemp={BASETEMP_FULL}", "-p", "no:cacheprovider"],
        ))
    else:
        BASETEMP_FAST.mkdir(parents=True, exist_ok=True)
        steps.append((
            "pytest (fast subset)",
            [sys.executable, "-m", "pytest", "-q", f"--basetemp={BASETEMP_FAST}", "-p", "no:cacheprovider", *FAST_TEST_PATHS],
        ))
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="run the full pytest suite instead of the fast subset")
    args = parser.parse_args(argv)

    results = [(label, _run(cmd, label=label)) for label, cmd in build_steps(full=args.full)]

    print("\n=== summary ===")
    for label, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {label}")

    failed = [label for label, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
