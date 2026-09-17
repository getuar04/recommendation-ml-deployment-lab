"""Fails (non-zero exit) if any forbidden file is tracked by Git.

Run from the repository root: `python -m scripts.ci.check_forbidden_files`. Used both as a
GitLab CI job (`.gitlab-ci.yml`, `validate` stage) and locally via `scripts/ci/verify.py`.

Inspects `git ls-files` output only -- this script never opens or reads the *contents* of
any tracked file, so a real secret that slips past every other safeguard is still never
echoed by this one. Only safe filenames are ever printed.

Cross-platform by construction: matching is done against the POSIX-style relative paths
`git ls-files` always reports (forward slashes on every OS, Windows included), and the only
external process invoked is `git` itself via `subprocess` (no shell, no OS-specific tool).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Exact tracked paths that are legitimate, reviewed fixtures -- never an accidental commit.
# This is the ONLY way to suppress a match; every entry must be justified by a comment.
ALLOWLIST: frozenset[str] = frozenset({
    ".env.example",  # placeholder-only template, no real values (README "Setup").
    # Layer-on-top-of-.env override files for the four comparative-experiment datasets
    # (v1 and v2) -- container names/ports/volume names/TRAINING_ALGORITHM_LOCK only, never
    # DB_PASSWORD/INTERNAL_API_KEY (those still come from the real, gitignored .env). See
    # README "Comparative Experiments" / "Docker volume structure".
    ".env.experiment-sport.env", ".env.experiment-entertainment.env",
    ".env.experiment-music.env", ".env.experiment-balanced.env",
    ".env.experiment-sport-v2.env", ".env.experiment-entertainment-v2.env",
    ".env.experiment-music-v2.env", ".env.experiment-balanced-v2.env",
    # README "Artifact schema and compatibility": a deliberate, immutable point-in-time
    # snapshot predating the current metadata schema, intentionally left in place as a
    # fixture -- not a live/generated artifact.
    "models/presentation-verification/model_metadata.json",
})

# 5 MiB is generous for any real source file in this repository; a tracked file above this
# is itself suspicious even if no pattern below happens to name it explicitly.
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024

# (pattern, human description). Patterns are matched with `search()` against the POSIX-style
# path git reports, so `(^|/)` anchors a path *segment* start without requiring a full-path match.
FORBIDDEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(^|/)\.env(\..+)?$"), "dotenv file (may contain secrets/credentials)"),
    (re.compile(r"(^|/)id_rsa(\.pub)?$"), "SSH private/public key"),
    (re.compile(r"\.(pem|key|p12|pfx|crt)$"), "certificate or private key material"),
    (re.compile(r"(^|/)data/.*\.db$"), "local SQLite database under data/"),
    (re.compile(r"^data/pytest-"), "pytest temporary/basetemp directory"),
    (re.compile(r"\.log$"), "log file"),
    (re.compile(r"(^|/)\.coverage(\..*)?$"), "coverage data file"),
    (re.compile(r"(^|/)coverage\.xml$"), "coverage report"),
    (re.compile(r"(^|/)htmlcov/"), "HTML coverage report directory"),
    (re.compile(r"(^|/)reports/junit.*\.xml$"), "generated JUnit report"),
    (re.compile(r"\.joblib$"), "trained model artifact"),
    (re.compile(r"(^|/)model_metadata\.json$"), "generated VIDEO model metadata"),
    (re.compile(r"(^|/)live_model_metadata\.json$"), "generated LIVE model metadata"),
    (re.compile(r"(^|/)\.idea/"), "IDE (JetBrains) project metadata"),
    (re.compile(r"(^|/)\.vscode/"), "IDE (VS Code) project metadata"),
    (re.compile(r"(^|/)__pycache__/"), "Python bytecode cache directory"),
    (re.compile(r"\.py[co]$"), "compiled Python bytecode"),
    (re.compile(r"(^|/)\.pytest_cache/"), "pytest cache directory"),
    (re.compile(r"(^|/)\.mypy_cache/"), "mypy cache directory"),
    (re.compile(r"(^|/)\.ruff_cache/"), "ruff cache directory"),
)


def tracked_files(repo_root: Path = REPO_ROOT) -> list[str]:
    """Every path Git currently tracks, relative to `repo_root`, POSIX-separated."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        capture_output=True, check=True,
    )
    return [p for p in result.stdout.decode("utf-8", errors="replace").split("\0") if p]


def tracked_file_size(path: str, repo_root: Path = REPO_ROOT) -> int | None:
    """The committed blob size for `path` at HEAD (never reads working-tree content
    directly). Returns None if the size can't be determined (e.g. no commits yet)."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-s", f"HEAD:{path}"],
        capture_output=True, check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.decode().strip())
    except ValueError:
        return None


def find_violations(
    tracked: list[str], *, repo_root: Path = REPO_ROOT, allowlist: frozenset[str] = ALLOWLIST,
) -> list[str]:
    """Pure function over an already-collected file list -- no Git calls of its own except
    the optional oversized-file size lookup, so this is directly unit-testable with a fake
    `tracked` list and without a real repository."""
    violations = []
    for path in tracked:
        if path in allowlist:
            continue
        matched = False
        for pattern, description in FORBIDDEN_PATTERNS:
            if pattern.search(path):
                violations.append(f"{path}  [{description}]")
                matched = True
                break
        if not matched:
            size = tracked_file_size(path, repo_root)
            if size is not None and size > MAX_TRACKED_FILE_BYTES:
                mib = MAX_TRACKED_FILE_BYTES // (1024 * 1024)
                violations.append(f"{path}  [tracked file exceeds {mib} MiB -- unexpected generated artifact?]")
    return violations


def main() -> int:
    tracked = tracked_files()
    violations = find_violations(tracked)
    if violations:
        print("Forbidden tracked files detected:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        print(
            "\nIf one of these is a deliberate, reviewed fixture, add its exact path to "
            "ALLOWLIST in scripts/ci/check_forbidden_files.py with a comment explaining why.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {len(tracked)} tracked files, no forbidden files detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
