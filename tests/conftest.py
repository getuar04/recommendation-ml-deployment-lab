import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["DATABASE_URL"]="sqlite://"
import pytest
from fastapi.testclient import TestClient

from app.db.database import Base, engine
from app.main import app

# Shared deterministic anchor for scripts.generate_synthetic_data.generate() in tests that
# train a real model from its output. generate()'s reference_timestamp defaults to None, which
# falls back to real wall-clock time (datetime.now(timezone.utc)) -- harmless for the relative
# structure of the generated data, but it shifts absolute timestamp-derived features (e.g.
# hour_of_day, and which point-in-time split each row lands in) on every run. Proven directly
# (repeated fresh-process runs): this is enough to move LogisticRegression's end-to-end critical
# constraint pass rate across app.ml.eligibility_policy.MINIMUM_CRITICAL_PASS_RATE's boundary,
# intermittently producing NO_ELIGIBLE_MODEL in tests whose generate() call omitted it. Several
# other test modules already define an equivalent local `REFERENCE_TIMESTAMP` constant with this
# same value; this is the shared, importable home for tests that don't already have one.
SYNTHETIC_DATA_REFERENCE_TIMESTAMP = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.drop_all(engine); Base.metadata.create_all(engine); yield
@pytest.fixture
def client():
    return TestClient(app)

# Production-safety net: MODEL_PATH/METADATA_PATH/MODEL_DIR (app.core.config) are each
# imported by value (`from app.core.config import MODEL_PATH`) into app.api.training_routes,
# app.ml.model_store, app.services.recommendation_service, and app.services.training_service
# independently -- monkeypatching one does NOT patch the others, so a test/diagnostic that
# forgets even one of the four (or the LIVE equivalents) silently trains into the developer's
# real local models/ directory instead of an isolated tmp_path. This already happened once
# during this project's review; this fixture turns a future recurrence into an immediate, named
# test failure instead of a silent, discovered-much-later corruption of local state.
_REAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
_REAL_ARTIFACT_PATHS = [
    _REAL_MODEL_DIR / "recommendation_model.joblib",
    _REAL_MODEL_DIR / "model_metadata.json",
    _REAL_MODEL_DIR / "live_recommendation_model.joblib",
    _REAL_MODEL_DIR / "live_model_metadata.json",
]


def _snapshot_real_artifacts() -> dict[Path, tuple[int, int] | None]:
    snapshot: dict[Path, tuple[int, int] | None] = {}
    for path in _REAL_ARTIFACT_PATHS:
        try:
            stat = path.stat()
            snapshot[path] = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            snapshot[path] = None
    return snapshot


@pytest.fixture(autouse=True)
def _guard_real_model_artifacts():
    before = _snapshot_real_artifacts()
    yield
    after = _snapshot_real_artifacts()
    changed = [str(path) for path in _REAL_ARTIFACT_PATHS if before[path] != after[path]]
    if changed:
        pytest.fail(
            "This test wrote to the real local models/ directory instead of an isolated "
            f"tmp_path: {changed}. Every test/diagnostic that trains, promotes, or rolls back "
            "a model MUST monkeypatch MODEL_PATH/METADATA_PATH (and MODEL_DIR) in EVERY module "
            "that imports them -- app.api.training_routes, app.ml.model_store, "
            "app.services.recommendation_service, app.services.training_service, and the LIVE "
            "equivalents (app.api.training_routes' /live routes, app.ml.live_trainer, "
            "app.services.live_recommendation_service). See tests/test_end_to_end.py for a "
            "correct example.",
            pytrace=False,
        )

def ensure_content(client, content_id, *, content_type="VIDEO", creator_id="cr-1", category="FOOD"):
    """Group B (Amendment 1): interaction integrity now requires the referenced content to
    exist and be active before an event is accepted. This is the single shared place tests
    create that prerequisite content -- best-effort (a second call for the same content_id
    hits the existing 409 CONTENT_ALREADY_EXISTS and is ignored), so call sites don't need
    to track whether a given content_id was already created."""
    client.post("/api/v1/recommendation-ml-service/contents", json={"contentId": content_id, "creatorId": creator_id,
                                          "contentType": content_type, "category": category})
