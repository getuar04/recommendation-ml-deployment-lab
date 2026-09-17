"""Focused checks for Docker/production-readiness properties that aren't exercised by any
other test: graceful shutdown (the Dockerfile CMD must not swallow SIGTERM), the non-root
runtime user, the health endpoint's live reachability, and that `.dockerignore` actually
excludes local secrets. These read the real `Dockerfile`/`.dockerignore` text for the two
structural checks -- not a brittle full-file parse, just presence of the specific properties
this project's Docker checklist requires.

The X-Internal-API-Key inbound gate has been removed entirely -- the consuming backend team
confirmed it is no longer part of this service's security architecture.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dockerfile_text() -> str:
    return (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_cmd_execs_uvicorn_so_docker_stop_reaches_it_directly():
    """Without `exec`, the shell (`sh -c "..."`) stays PID 1 and uvicorn runs as its child --
    `docker stop`'s SIGTERM goes to the shell, which does not reliably forward it, so shutdown
    stalls until the SIGKILL timeout. `exec` replaces the shell with uvicorn so it receives
    the signal directly."""
    text = _dockerfile_text()
    cmd_line = next(line for line in text.splitlines() if line.startswith("CMD"))
    assert "exec uvicorn" in cmd_line, cmd_line


def test_dockerfile_runs_as_a_non_root_user():
    text = _dockerfile_text()
    assert "USER app" in text
    # No later instruction switches back to root after that.
    after = text.split("USER app", 1)[1]
    assert "USER root" not in after


def test_dockerfile_healthcheck_targets_the_real_unauthenticated_health_path():
    text = _dockerfile_text()
    assert "/api/v1/recommendation-ml-service/health" in text


def test_dockerignore_excludes_local_secrets_and_caches():
    text = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    patterns = set(text.splitlines())
    for required in (".env", ".git", "**/__pycache__"):
        assert required in patterns, f"{required!r} missing from .dockerignore"


def test_health_endpoint_is_reachable(client):
    """Live-request complement to test_api_safety.py's OpenAPI-spec-only check: a real
    request to GET /health always succeeds (never 401/403) -- container health/readiness
    probes never send any auth header."""
    response = client.get("/api/v1/recommendation-ml-service/health")
    assert response.status_code in (200, 503)  # never 401/403
