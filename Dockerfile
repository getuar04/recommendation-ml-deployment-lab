# Pinned by digest, not just the floating `3.12-slim` tag, so the base image is immutable and
# reproducible. Verified 2026-08-07 via `docker pull python:3.12-slim && docker inspect
# --format '{{index .RepoDigests 0}}' python:3.12-slim`: resolves to Python 3.12.13 on Debian 13
# (trixie). Re-pin deliberately with the same command when a newer patch/security update is
# needed -- do not replace this digest without re-verifying it the same way.
FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36

# GET /health metadata (app.core.config / app.api.health_routes). SERVICE_COMMIT/
# SERVICE_BUILD_TIME have no meaningful value known to the container at runtime -- they only
# exist at build time (the invoking git checkout / CI job clock), so they must be threaded
# through as build args, not left as plain ENV defaults. Never hardcode a real-looking commit
# hash or timestamp here: an unset build arg deliberately stays "unknown" (see
# app.core.config.SERVICE_COMMIT/SERVICE_BUILD_TIME), which is an honest "not supplied" value
# rather than a fabricated one. A real build supplies both, e.g.:
#   docker build \
#     --build-arg SERVICE_COMMIT="$(git rev-parse HEAD)" \
#     --build-arg SERVICE_BUILD_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)" .
# docker-compose.yml passes these through the same way (see its `build.args`); both can still
# be overridden without a rebuild via `docker compose run/up -e SERVICE_COMMIT=...` or the
# container's `environment:` block, since a plain ENV set here is only the default.
ARG SERVICE_NAME=recommendation-ml-service
ARG SERVICE_VERSION=1.0.0
ARG SERVICE_COMMIT=unknown
ARG SERVICE_BUILD_TIME=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3500 \
    SERVICE_NAME=${SERVICE_NAME} \
    SERVICE_VERSION=${SERVICE_VERSION} \
    SERVICE_COMMIT=${SERVICE_COMMIT} \
    SERVICE_BUILD_TIME=${SERVICE_BUILD_TIME}

WORKDIR /app

# Runtime system dependency required by LightGBM.
# libgomp1 provides the GNU OpenMP runtime (libgomp.so.1).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt ./

# requirements.txt (the shared, pip-compile-generated lock file) still lists `pytest` as a
# direct dependency -- it is not imported anywhere under app/, scripts/, or migrations/
# (verified), only by tests/, which is never copied into this image. Uninstalling it here
# (rather than editing requirements.in/requirements.txt, which are a separately-owned lock
# file requiring a deliberate pip-compile regeneration and full test re-run per their own
# header comment) keeps the resolved/pinned dependency graph identical while dropping a
# test-only package from the runtime image. `httpx` is deliberately kept: unlike pytest, it
# has a real documented operational use inside a running container (see README "Docker").
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y --no-input pytest

COPY app ./app
COPY scripts ./scripts
COPY alembic.ini ./
COPY migrations ./migrations

# Belt-and-braces: strip any bytecode cache that made it into the build context from the host
# checkout despite .dockerignore (observed to happen on some Windows/Docker Desktop setups,
# where `.dockerignore`'s recursive `**/__pycache__` pattern does not reliably exclude nested
# directories from the context) -- deterministic regardless of host platform or dockerignore
# matching quirks.
RUN find . -depth -type d -name '__pycache__' -exec rm -rf {} \; \
    && find . -type f -name '*.py[cod]' -delete \
    && mkdir -p /app/data /app/models /app/experiments \
    && chown -R app:app /app

USER app

EXPOSE 3500

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3500/api/v1/recommendation-ml-service/health', timeout=3)"

# scripts/run_migrations.py brings PostgreSQL to the current Alembic head (skips entirely
# for a SQLite DATABASE_URL) BEFORE uvicorn starts -- the app must never serve requests
# against a schema behind the code. `&&` means a migration failure (non-zero exit) stops
# here: uvicorn never starts, and the container exits non-zero instead of silently serving
# against an incompatible schema. `exec` on the uvicorn stage (unchanged) still replaces
# the shell as PID 1 for correct signal handling once migrations succeed.
CMD ["sh", "-c", "python -m scripts.run_migrations && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-3500}"]