import os
from pathlib import Path

from dotenv import load_dotenv

from app.experiments.validation import validate_identifier


def _getenv_int(name: str, default: str) -> int:
    """`int(os.getenv(name, default))`, except a malformed/empty EXPLICITLY-PROVIDED value
    raises a ValueError naming the variable and the rejected raw value instead of Python's
    unlabeled `invalid literal for int()` message. An absent variable still falls back to
    `default` exactly as before -- this never hides a broken Secret/ConfigMap value behind a
    silent default, it only makes the resulting fail-fast error legible."""
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Invalid {name}={raw!r}: expected integer") from None


def _getenv_float(name: str, default: str) -> float:
    """Same contract as `_getenv_int` above, for float-valued settings."""
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Invalid {name}={raw!r}: expected float") from None


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
(ROOT / "data").mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "experiments").mkdir(parents=True, exist_ok=True)
# Phase A: deployment-mode gate. "local" is the default so an unconfigured checkout behaves
# exactly as before; "test" is also inferred automatically whenever DATABASE_URL points at
# SQLite (the existing, established signal the test suite already relies on -- see
# tests/conftest.py and app/main.py's own create_all() gate), so no test or local workflow
# needs a new environment variable to keep working unchanged. Any other APP_ENV value
# (dev/staging/production/anything else) is treated as a real deployment: it must fail
# startup loudly when a required deployment setting is missing (DATABASE_URL below)
# instead of silently falling back to a local default.
_APP_ENV_RAW = os.getenv("APP_ENV", "local").strip().lower() or "local"
_DATABASE_URL_RAW = os.getenv("DATABASE_URL") or None
APP_ENV = "test" if (_DATABASE_URL_RAW or "").startswith("sqlite") else _APP_ENV_RAW
_KEY_OPTIONAL_ENVS = {"local", "test"}

# The localhost default below is a LOCAL-DEVELOPMENT convenience only (a developer's own
# Postgres, see .env.example / docker-compose.yml). It is deliberately never a deployment
# fallback: in Kubernetes/EKS the database lives behind its own Service DNS name and the URL
# is injected from a Kubernetes Secret (see deploy/values-*.yaml), so a deployment that
# somehow starts without DATABASE_URL must fail here, at import time, rather than come up
# "healthy-looking" while every query dies against a localhost that has no database.
_DATABASE_URL_LOCAL_DEFAULT = "postgresql+psycopg://recommendation_user:change-me@localhost:5432/recommendation_ml"
if APP_ENV not in _KEY_OPTIONAL_ENVS and not _DATABASE_URL_RAW:
    raise ValueError(
        f"DATABASE_URL must be set when APP_ENV={APP_ENV!r} (only {sorted(_KEY_OPTIONAL_ENVS)} "
        "may fall back to the local-development default). Point it at the deployment's "
        "PostgreSQL service, e.g. postgresql+psycopg://<user>:<password>@<postgres-service>:5432/<database>"
    )
DATABASE_URL = _DATABASE_URL_RAW or _DATABASE_URL_LOCAL_DEFAULT
MODEL_ARTIFACT_ROOT = Path(os.getenv("MODEL_ARTIFACT_ROOT", os.getenv("MODEL_DIR", str(ROOT / "models"))))
MODEL_DIR = MODEL_ARTIFACT_ROOT
# Active artifact paths: these two names and their meaning are unchanged from before this
# phase (many existing tests monkeypatch them directly) -- candidate/previous are additive
# sibling paths computed at call time by app.ml.artifact_lifecycle.sibling_paths(), so a
# monkeypatched MODEL_PATH still gets its candidate/previous files alongside it.
MODEL_PATH = Path(os.getenv("VIDEO_MODEL_ACTIVE_PATH", str(MODEL_DIR / "recommendation_model.joblib")))
METADATA_PATH = MODEL_DIR / "model_metadata.json"
LIVE_MODEL_PATH = Path(os.getenv("LIVE_MODEL_ACTIVE_PATH", str(MODEL_DIR / "live_recommendation_model.joblib")))
LIVE_METADATA_PATH = MODEL_DIR / "live_model_metadata.json"
# Documented default candidate/previous locations (see app.ml.artifact_lifecycle for the
# actual runtime derivation, which follows MODEL_PATH/LIVE_MODEL_PATH even when overridden).
VIDEO_MODEL_CANDIDATE_PATH = Path(os.getenv("VIDEO_MODEL_CANDIDATE_PATH", str(MODEL_PATH.with_name(f"candidate_{MODEL_PATH.name}"))))
VIDEO_MODEL_PREVIOUS_PATH = Path(os.getenv("VIDEO_MODEL_PREVIOUS_PATH", str(MODEL_PATH.with_name(f"previous_{MODEL_PATH.name}"))))
LIVE_MODEL_CANDIDATE_PATH = Path(os.getenv("LIVE_MODEL_CANDIDATE_PATH", str(LIVE_MODEL_PATH.with_name(f"candidate_{LIVE_MODEL_PATH.name}"))))
LIVE_MODEL_PREVIOUS_PATH = Path(os.getenv("LIVE_MODEL_PREVIOUS_PATH", str(LIVE_MODEL_PATH.with_name(f"previous_{LIVE_MODEL_PATH.name}"))))

# Content-understanding category classifier (app.ml.content_classifier): a wholly separate
# artifact from the VIDEO/LIVE recommendation models above -- no promotion/rollback lifecycle,
# no candidate/previous siblings, just load-and-cache (app.services.content_enrichment_service).
# Never touched by app.ml.artifact_lifecycle/model_store/model_cache.
CONTENT_CLASSIFIER_MODEL_PATH = Path(os.getenv("CONTENT_CLASSIFIER_MODEL_PATH", str(MODEL_DIR / "content_classifier.joblib")))
CONTENT_CLASSIFIER_METADATA_PATH = Path(os.getenv("CONTENT_CLASSIFIER_METADATA_PATH", str(MODEL_DIR / "content_classifier_metadata.json")))
# Content Understanding dataset foundation (data/content_understanding/) -- real,
# human-reviewed/pseudo-labeled/weak-supervision training data, read by
# app.ml.content_classifier_dataset_source.build_training_dataframe. Currently 0 records
# (see data/content_understanding/real/manifest.json) -- this path existing/being readable
# does not mean real data exists yet.
CONTENT_UNDERSTANDING_REAL_DATASET_PATH = Path(os.getenv(
    "CONTENT_UNDERSTANDING_REAL_DATASET_PATH", str(ROOT / "data" / "content_understanding" / "real" / "dataset.jsonl"),
))
# Confidence thresholds and creator-prior blending (app.ml.content_classifier.
# combine_with_creator_prior): >=HIGH means current-content evidence is trusted alone;
# between LOW and HIGH the creator's historical category distribution is blended in as
# supporting evidence only; below LOW (even after blending) the content is UNKNOWN rather
# than a fabricated guess. Bootstrap starting points, not measured production thresholds --
# see README/docs for the same caveat every other repo-wide threshold constant already
# carries.
CONTENT_CATEGORY_LOW_CONFIDENCE_THRESHOLD = _getenv_float("CONTENT_CATEGORY_LOW_CONFIDENCE_THRESHOLD", "0.35")
CONTENT_CATEGORY_HIGH_CONFIDENCE_THRESHOLD = _getenv_float("CONTENT_CATEGORY_HIGH_CONFIDENCE_THRESHOLD", "0.70")
CONTENT_CATEGORY_CREATOR_PRIOR_WEIGHT = _getenv_float("CONTENT_CATEGORY_CREATOR_PRIOR_WEIGHT", "0.3")
CONTENT_CATEGORY_CREATOR_PROFILE_MIN_SAMPLES = _getenv_int("CONTENT_CATEGORY_CREATOR_PROFILE_MIN_SAMPLES", "3")
CONTENT_CATEGORY_CREATOR_PROFILE_MAX_ROWS = _getenv_int("CONTENT_CATEGORY_CREATOR_PROFILE_MAX_ROWS", "50")
VERSION = "1.0.0"

# The one authoritative place defining this service's public API base path, per the
# company-wide `/api/v1/<service-name>/...` convention used by every other microservice
# (faq-service, follow-service, interaction-count-service, ranking-service, ...).
# app.main is the only place that reads this to mount every router -- nowhere else should
# hardcode "/api/v1/recommendation-ml-service" directly. Deliberately a plain constant, not
# derived from SERVICE_NAME above: SERVICE_NAME is operator-overridable cosmetic /health
# metadata, whereas the URL routing contract must not silently change with it.
API_V1_PREFIX = "/api/v1/recommendation-ml-service"

# GET /health metadata. SERVICE_NAME already exists as a Docker Compose env var (see
# docker-compose.yml's `app.environment`) even though nothing previously read it -- reused
# here rather than inventing a second name for the same concept. SERVICE_VERSION defaults to
# the existing VERSION constant so an unconfigured checkout reports the same version it
# always has. SERVICE_COMMIT/SERVICE_BUILD_TIME have no sensible default (there is no real
# commit/build time to report outside a built image) -- "unknown" is an honest placeholder,
# never a fabricated hash or timestamp; a real deployment supplies both via Docker build
# args (see Dockerfile) or a CI-injected environment variable. "environment" in the /health
# response body is intentionally sourced from the existing APP_ENV below, not a second
# ENVIRONMENT variable -- APP_ENV is already this repository's established deployment-mode
# concept and giving it a second name here would let the two drift apart.
SERVICE_NAME = os.getenv("SERVICE_NAME", "recommendation-ml-service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", VERSION)
SERVICE_COMMIT = os.getenv("SERVICE_COMMIT", "unknown")
SERVICE_BUILD_TIME = os.getenv("SERVICE_BUILD_TIME", "unknown")

# Group C: a training lock older than this is treated as abandoned (the process that held
# it crashed or was killed without releasing it) and is reclaimed rather than blocking
# training forever. See app.services.training_lock.
TRAINING_STALE_LOCK_SECONDS = _getenv_int("TRAINING_STALE_LOCK_SECONDS", "1800")

# Group B: a small allowance for legitimate distributed-system clock skew between the
# client/producer and this server -- an event whose OWN timestamp is further in the future
# than this is rejected rather than silently accepted (see app.schemas.event_schemas), since
# it would otherwise be able to leak into the future relative to training data built as of
# the time it is queried, undermining the point-in-time guarantees the chronological split
# lifecycle depends on (see app.ml.splitting). Was a hardcoded, non-overridable constant
# inside event_schemas.py -- centralized here to match every other comparable threshold's
# existing env-var-overridable convention (e.g. TRAINING_STALE_LOCK_SECONDS above,
# PERSONALISED_RECOMMENDATION_MIN_INTERACTIONS below).
MAX_FUTURE_EVENT_SKEW_SECONDS = _getenv_int("MAX_FUTURE_EVENT_SKEW_SECONDS", "300")
if MAX_FUTURE_EVENT_SKEW_SECONDS < 0:
    raise ValueError(f"MAX_FUTURE_EVENT_SKEW_SECONDS ({MAX_FUTURE_EVENT_SKEW_SECONDS}) must be >= 0.")

# LIVE session reconstruction (app.ml.live_session_builder): a LIVE_JOINED that reopens
# within this many seconds of the same (user, content) pair's last LIVE_LEFT is treated as a
# reconnect of the SAME logical viewing session (network blip/app backgrounding) rather than
# a deliberate new visit -- env-overridable, matching every other comparable threshold's
# existing convention (e.g. MAX_FUTURE_EVENT_SKEW_SECONDS above).
LIVE_RECONNECT_GRACE_SECONDS = _getenv_int("LIVE_RECONNECT_GRACE_SECONDS", "30")
if LIVE_RECONNECT_GRACE_SECONDS < 0:
    raise ValueError(f"LIVE_RECONNECT_GRACE_SECONDS ({LIVE_RECONNECT_GRACE_SECONDS}) must be >= 0.")

# LIVE dynamic viewer state (app.services.providers.live_dynamic_state_provider): a real,
# GENUINELY DIFFERENT question from LIVE_RECONNECT_GRACE_SECONDS above -- that constant governs
# whether a re-JOIN after an explicit LEAVE is the same logical viewing session (an offline/
# session-reconstruction concern); this one governs whether a viewer who has not sent ANY event
# (LIVE_JOINED or a LIVE_WATCHED "still watching" ping -- see app.ml.live_session_builder's own
# module docstring, which already documents LIVE_WATCHED as a periodic ping) recently enough is
# still counted as CURRENTLY present, for real-time popularity ranking. No explicit client
# heartbeat-interval contract exists anywhere in this repository or its sibling-service search
# (see app.services.kafka_behavior_consumer's own real-contract search) -- this default is a
# documented, deliberately conservative ASSUMPTION (a client pinging roughly every 15-30s should
# survive 2-3 missed pings before being considered stale), not a verified platform contract.
# Env-overridable, matching every comparable threshold's existing convention.
LIVE_VIEWER_STALE_SECONDS = _getenv_int("LIVE_VIEWER_STALE_SECONDS", "90")
if LIVE_VIEWER_STALE_SECONDS < 0:
    raise ValueError(f"LIVE_VIEWER_STALE_SECONDS ({LIVE_VIEWER_STALE_SECONDS}) must be >= 0.")

# Bounded lookback window for viewer-momentum (join-rate in the recent half vs. the earlier
# half of this window) and recent-engagement (likes/gifts within this window) computation --
# see app.services.providers.live_dynamic_state_provider. Deliberately its own constant, not
# reused from LIVE_RECENT_WINDOW (app.ml.live_dataset_builder, 7 DAYS -- a training-data
# recency window, three orders of magnitude coarser than a real-time momentum signal needs).
LIVE_DYNAMIC_WINDOW_SECONDS = _getenv_int("LIVE_DYNAMIC_WINDOW_SECONDS", "600")
if LIVE_DYNAMIC_WINDOW_SECONDS <= 0:
    raise ValueError(f"LIVE_DYNAMIC_WINDOW_SECONDS ({LIVE_DYNAMIC_WINDOW_SECONDS}) must be > 0.")

# INTERNAL_API_KEY no longer gates any of this service's own routes (the inbound
# require_internal_api_key dependency was removed -- the consuming backend team confirmed
# X-Internal-API-Key protection is no longer part of this service's security architecture).
# The value is kept, optional in every APP_ENV, purely because app.services.service_clients
# still uses it for the OUTBOUND direction: an unrelated, unchanged convention that
# follow-service's own internalAuthMiddleware still enforces on ITS side (see README
# "Service-to-service auth"). No production startup validation requires it any more.
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
REQUEST_ID_HEADER = os.getenv("REQUEST_ID_HEADER", "X-Request-ID")

# Phase A §4/§5: shared bounds for the VIDEO ranking hot path's bounded interaction-history
# query, and for inbound candidate/stream array sizes. Centralized here (not hardcoded per
# call site/schema) so both the bound and its rationale live in one place. Defaults are
# starting points, not measured production-scale numbers -- see README/docs for the caveat.
RECOMMENDATION_HISTORY_MAX_INTERACTIONS = _getenv_int("RECOMMENDATION_HISTORY_MAX_INTERACTIONS", "500")
RECOMMENDATION_MAX_CANDIDATES = _getenv_int("RECOMMENDATION_MAX_CANDIDATES", "200")

# Phase 3.4: guarded Two-Tower retrieval integration. OFF by default -- when false, POST
# /recommendations runs the exact pre-Phase-3.4 flow (request.candidates, unchanged) with zero
# behavioral or performance difference. When true, app.services.recommendation_service.recommend()
# tries Two-Tower retrieval first and falls back to request.candidates automatically on ANY
# failure (missing/corrupt artifact, retrieval error, or torch/two_tower unavailable) -- see
# that module's _try_two_tower_candidates(). TWO_TOWER_RETRIEVAL_K bounds how many candidates
# Two-Tower hands to the existing RandomForest/reranker, named here (not hardcoded inline)
# per Phase 3.4 Step 3.
TWO_TOWER_RETRIEVAL_ENABLED = os.getenv("TWO_TOWER_RETRIEVAL_ENABLED", "false").strip().lower() == "true"
TWO_TOWER_RETRIEVAL_K = _getenv_int("TWO_TOWER_RETRIEVAL_K", "30")

# Spec §26/§27/§51: recommendation-strategy thresholds are configuration, not scattered
# hard-coded numbers. Strategy by prior interaction count:
#   0                          -> COLD_START
#   1 .. PERSONALISED_MIN-1    -> HYBRID
#   >= PERSONALISED_MIN        -> PERSONALISED_ML
PERSONALISED_RECOMMENDATION_MIN_INTERACTIONS = _getenv_int(
    "PERSONALISED_RECOMMENDATION_MIN_INTERACTIONS", "10"
)
USER_PROFILE_MIN_INTERACTIONS = _getenv_int("USER_PROFILE_MIN_INTERACTIONS", "1")

# Session 4 (spec §44-47): experiment configuration. No source-code change is required to
# switch experiments -- everything here is environment-driven. EXPERIMENT_ID/DATASET_VERSION
# are optional (only meaningful when actually running an experiment) and validated only when
# set, so normal API/test operation (which never sets them) is unaffected. The three volume
# name defaults below are exactly the names Docker Compose already resolves for the current
# real stack (confirmed via `docker compose config` before choosing these defaults) -- adding
# configurability here must not silently orphan the existing postgres_data/model_data
# volumes.
EXPERIMENT_DIR = Path(os.getenv("EXPERIMENT_DIR", str(ROOT / "experiments")))

EXPERIMENT_ID = os.getenv("EXPERIMENT_ID") or None
if EXPERIMENT_ID:
    validate_identifier(EXPERIMENT_ID, field="EXPERIMENT_ID")

DATASET_VERSION = os.getenv("DATASET_VERSION") or None
if DATASET_VERSION:
    validate_identifier(DATASET_VERSION, field="DATASET_VERSION")

# Comparative-experiment algorithm lock (see app/experiments/comparative_definitions.py and
# README "Comparative Experiments"). Unset by default -- app.ml.trainer.train_models() keeps
# comparing every candidate model exactly as before for normal API/test operation. Only a
# comparative-experiment Docker environment (see .env.experiment-*.env) sets this, so the
# real /model/train endpoint's default behavior for everyone else is completely unaffected.
# Also the general-purpose "same algorithm, different dataset" lock any caller of
# app.ml.trainer.train_models(restrict_algorithm=...) can use -- see app.ml.algorithm_registry
# for the full candidate set this name is validated against.
_ALLOWED_TRAINING_ALGORITHMS = {
    "LogisticRegression", "RandomForestClassifier",
    "XGBoostClassifier", "LightGBMClassifier", "CatBoostClassifier",
}
TRAINING_ALGORITHM_LOCK = os.getenv("TRAINING_ALGORITHM_LOCK") or None
if TRAINING_ALGORITHM_LOCK and TRAINING_ALGORITHM_LOCK not in _ALLOWED_TRAINING_ALGORITHMS:
    raise ValueError(
        f"TRAINING_ALGORITHM_LOCK must be one of {sorted(_ALLOWED_TRAINING_ALGORITHMS)} "
        f"(got {TRAINING_ALGORITHM_LOCK!r})"
    )

# Per-algorithm operator kill switches for the three optional gradient-boosting candidates
# (app.ml.algorithm_registry). Independent of whether the library is actually installed --
# this lets an operator exclude, say, CatBoost's heavy transitive deps (matplotlib/plotly/
# graphviz) from the candidate pool at runtime without uninstalling anything or restricting
# to a single algorithm via TRAINING_ALGORITHM_LOCK. Default true (opt-out, not opt-in): once
# a library is actually installed, it participates in comparison like any other candidate.
TRAINING_ENABLE_XGBOOST = os.getenv("TRAINING_ENABLE_XGBOOST", "true").strip().lower() == "true"
TRAINING_ENABLE_LIGHTGBM = os.getenv("TRAINING_ENABLE_LIGHTGBM", "true").strip().lower() == "true"
TRAINING_ENABLE_CATBOOST = os.getenv("TRAINING_ENABLE_CATBOOST", "true").strip().lower() == "true"

# Cross-family production selection (XGBRanker promotion task): whether ranking-native
# challengers (app.ml.ranker_registry.PRODUCTION_RANKER_NAMES -- currently XGBRanker only)
# are trained and compared alongside classifiers in app.ml.trainer.train_and_select_cross_family.
# Independent of TRAINING_ENABLE_XGBOOST (which already separately gates XGBRanker's own
# availability/kill-switch in app.ml.ranker_registry) -- this flag is the higher-level "does
# production selection even attempt cross-family comparison at all" switch, so an operator can
# fall back to classifier-only selection without touching XGBoost availability itself. Default
# true: once a production ranker is available, it participates like any other candidate.
TRAINING_ENABLE_PRODUCTION_RANKERS = os.getenv("TRAINING_ENABLE_PRODUCTION_RANKERS", "true").strip().lower() == "true"

# app.ml.sample_weight_policy: per-row training-importance weighting (CONTENT_NOT_INTERESTED/
# strong-positive rows get more influence on the fitted decision boundary; the binary target
# itself never changes). Default on -- it changes only *how much* each already-labeled row
# influences fitting, never the split structure, the target, or determinism for a fixed seed.
# Kept togglable (mirrors TRAINING_ALGORITHM_LOCK's own rationale) so a controlled "same
# dataset/seed/algorithm, weights on vs. off" comparison stays possible without a code change.
TRAINING_USE_SAMPLE_WEIGHTS = os.getenv("TRAINING_USE_SAMPLE_WEIGHTS", "true").strip().lower() == "true"

POSTGRES_VOLUME_NAME = validate_identifier(
    os.getenv("POSTGRES_VOLUME_NAME", "machine-learning_postgres_data"), field="POSTGRES_VOLUME_NAME"
)
MODEL_VOLUME_NAME = validate_identifier(
    os.getenv("MODEL_VOLUME_NAME", "machine-learning_model_data"), field="MODEL_VOLUME_NAME"
)
EXPERIMENT_VOLUME_NAME = validate_identifier(
    os.getenv("EXPERIMENT_VOLUME_NAME", "machine-learning_experiment_data"), field="EXPERIMENT_VOLUME_NAME"
)

# Drift monitoring (app/ml/drift_baseline.py, app/ml/drift_detector.py, app/api/drift_routes.py).
# All thresholds below are starting points, not calibrated production values -- this project
# has no real production traffic history to calibrate against yet. The commonly-cited PSI
# convention (< 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift) is used only as a
# documented starting point; operators must recalibrate every threshold here against their
# own real traffic before trusting WARNING/CRITICAL as actionable signals.
DRIFT_PSI_WARNING_THRESHOLD = _getenv_float("DRIFT_PSI_WARNING_THRESHOLD", "0.1")
DRIFT_PSI_CRITICAL_THRESHOLD = _getenv_float("DRIFT_PSI_CRITICAL_THRESHOLD", "0.25")
DRIFT_MISSING_RATE_WARNING_DELTA = _getenv_float("DRIFT_MISSING_RATE_WARNING_DELTA", "0.1")
DRIFT_MISSING_RATE_CRITICAL_DELTA = _getenv_float("DRIFT_MISSING_RATE_CRITICAL_DELTA", "0.25")
# Corrective pass (Finding 1): renamed from DRIFT_UNSEEN_CATEGORY_*_RATE. The old name
# implied a genuine "never seen in training" signal; the actual metric it classified was
# only ever the *absolute* observed OTHER-bucket rate, which baseline high-cardinality
# training data can legitimately put above these thresholds on its own (a false positive on
# an observation batch identical to training). The metric this now classifies is the
# *change* in OTHER-bucket share (see app.ml.drift_detector.evaluate_categorical_feature),
# which is what "categorical tail drift" actually is. No backward-compatible alias is kept
# -- the old name was misleading, not merely renamed.
DRIFT_OTHER_RATE_WARNING_DELTA = _getenv_float("DRIFT_OTHER_RATE_WARNING_DELTA", "0.05")
DRIFT_OTHER_RATE_CRITICAL_DELTA = _getenv_float("DRIFT_OTHER_RATE_CRITICAL_DELTA", "0.20")
DRIFT_OUT_OF_RANGE_WARNING_RATE = _getenv_float("DRIFT_OUT_OF_RANGE_WARNING_RATE", "0.05")
DRIFT_OUT_OF_RANGE_CRITICAL_RATE = _getenv_float("DRIFT_OUT_OF_RANGE_CRITICAL_RATE", "0.20")

for _warning, _critical, _name in (
    (DRIFT_PSI_WARNING_THRESHOLD, DRIFT_PSI_CRITICAL_THRESHOLD, "DRIFT_PSI"),
    (DRIFT_MISSING_RATE_WARNING_DELTA, DRIFT_MISSING_RATE_CRITICAL_DELTA, "DRIFT_MISSING_RATE"),
    (DRIFT_OTHER_RATE_WARNING_DELTA, DRIFT_OTHER_RATE_CRITICAL_DELTA, "DRIFT_OTHER_RATE"),
    (DRIFT_OUT_OF_RANGE_WARNING_RATE, DRIFT_OUT_OF_RANGE_CRITICAL_RATE, "DRIFT_OUT_OF_RANGE"),
):
    if not (0 < _warning < _critical):
        raise ValueError(
            f"{_name}_WARNING_* must be positive and strictly less than {_name}_CRITICAL_* "
            f"(got warning={_warning!r}, critical={_critical!r})"
        )
del _warning, _critical, _name

# Sample-count/array-size bounds. DRIFT_MIN_OBSERVATIONS below this, a report is
# INSUFFICIENT_DATA rather than a misleadingly precise-looking OK/WARNING/CRITICAL.
# DRIFT_MAX_OBSERVATIONS bounds both POST .../evaluate request bodies (Pydantic max_length)
# and how many recent local VIDEO interactions GET /model/drift reconstructs -- this is the
# hot path's only "unbounded until proven otherwise" input, so it is bounded the same way
# RECOMMENDATION_HISTORY_MAX_INTERACTIONS bounds the ranking hot path above.
DRIFT_MIN_OBSERVATIONS = _getenv_int("DRIFT_MIN_OBSERVATIONS", "30")
DRIFT_MAX_OBSERVATIONS = _getenv_int("DRIFT_MAX_OBSERVATIONS", "5000")
# GET /model/drift's own query-param default -- deliberately smaller than the hard ceiling
# above, so an un-parameterized call stays cheap; a caller can still opt into up to
# DRIFT_MAX_OBSERVATIONS explicitly via ?limit=.
DRIFT_OBSERVATION_DEFAULT_LIMIT = _getenv_int("DRIFT_OBSERVATION_DEFAULT_LIMIT", "1000")
if DRIFT_MIN_OBSERVATIONS < 1 or DRIFT_MAX_OBSERVATIONS < DRIFT_MIN_OBSERVATIONS:
    raise ValueError(
        f"DRIFT_MIN_OBSERVATIONS ({DRIFT_MIN_OBSERVATIONS}) must be >= 1 and "
        f"DRIFT_MAX_OBSERVATIONS ({DRIFT_MAX_OBSERVATIONS}) must be >= DRIFT_MIN_OBSERVATIONS."
    )
if not (DRIFT_MIN_OBSERVATIONS <= DRIFT_OBSERVATION_DEFAULT_LIMIT <= DRIFT_MAX_OBSERVATIONS):
    raise ValueError(
        f"DRIFT_OBSERVATION_DEFAULT_LIMIT ({DRIFT_OBSERVATION_DEFAULT_LIMIT}) must be between "
        f"DRIFT_MIN_OBSERVATIONS ({DRIFT_MIN_OBSERVATIONS}) and DRIFT_MAX_OBSERVATIONS ({DRIFT_MAX_OBSERVATIONS})."
    )

# Corrective pass (Finding 2): most VIDEO features are history-dependent (all-time or
# 30-day-window accumulators -- see app.ml.dataset_builder.build_feature_rows's docstring
# for the full per-feature audit), so reconstructing GET /model/drift's observation window
# from a bare, freshly-initialized FeatureHistory silently turns every established user
# into an artificial cold start for those features. This bounds a single additional warm-up
# query (app.db.repositories.warmup_interactions_for_drift) that primes FeatureHistory with
# prior interactions for exactly the users represented in the observation window, without
# ever emitting a feature row for a warm-up row itself. Bounded in total (not per-user), so
# it stays a single query regardless of how many distinct users appear in the window --
# never one query per user (no N+1).
DRIFT_WARMUP_MAX_ROWS = _getenv_int("DRIFT_WARMUP_MAX_ROWS", "2000")
if DRIFT_WARMUP_MAX_ROWS < 0:
    raise ValueError(f"DRIFT_WARMUP_MAX_ROWS must be >= 0 (got {DRIFT_WARMUP_MAX_ROWS}).")

# Baseline construction bounds -- both directly bound driftBaseline's serialized size (see
# app/ml/drift_baseline.py), which is why they are deliberately small and fixed by default.
DRIFT_HISTOGRAM_BINS = _getenv_int("DRIFT_HISTOGRAM_BINS", "10")
DRIFT_MAX_CATEGORIES = _getenv_int("DRIFT_MAX_CATEGORIES", "20")
if DRIFT_HISTOGRAM_BINS < 2:
    raise ValueError(f"DRIFT_HISTOGRAM_BINS must be >= 2 (got {DRIFT_HISTOGRAM_BINS}).")
if DRIFT_MAX_CATEGORIES < 1:
    raise ValueError(f"DRIFT_MAX_CATEGORIES must be >= 1 (got {DRIFT_MAX_CATEGORIES}).")

# Floor applied to every bin/category proportion before it is used as a PSI/JS-divergence
# denominator or inside a logarithm, so an empty (zero-probability) bin never produces NaN/Inf.
DRIFT_PROBABILITY_EPSILON = _getenv_float("DRIFT_PROBABILITY_EPSILON", "0.0001")
if not (0 < DRIFT_PROBABILITY_EPSILON < 0.01):
    raise ValueError(f"DRIFT_PROBABILITY_EPSILON must be in (0, 0.01) (got {DRIFT_PROBABILITY_EPSILON}).")

# Real-service integration mode (see app/services/providers/ and README "Boundary with the
# real platform"). LOCAL (default) is the existing, fully-local demo/test behavior --
# request.candidates and/or the local interactions table, completely unchanged. REAL
# additionally lets `app.services.recommendation_service.recommend()` source user behavior
# from a real User Behavior Service and candidates from a real Candidate Service when the
# caller doesn't already supply them (request.userProfile / request.candidates) -- but only
# where UBS_BASE_URL/CANDIDATE_SERVICE_BASE_URL below are also configured; setting
# RECOMMENDATION_DATA_MODE=REAL alone, with no base URLs set, behaves like every REAL
# dependency being unavailable (see each provider's own fallback policy), never a crash and
# never a hardcoded localhost guess.
_ALLOWED_RECOMMENDATION_DATA_MODES = {"LOCAL", "REAL"}
RECOMMENDATION_DATA_MODE = os.getenv("RECOMMENDATION_DATA_MODE", "LOCAL").strip().upper()
if RECOMMENDATION_DATA_MODE not in _ALLOWED_RECOMMENDATION_DATA_MODES:
    raise ValueError(
        f"RECOMMENDATION_DATA_MODE must be one of {sorted(_ALLOWED_RECOMMENDATION_DATA_MODES)} "
        f"(got {RECOMMENDATION_DATA_MODE!r})"
    )

# User Behavior Service (UBS) client (app.services.providers.user_behavior_provider). Used
# only in REAL mode, and only when the request did not already supply userProfile.
#
# Real-contract search (see README "Real-service contract verification"): the local
# workspace's sibling service repositories (follow-service, ranking-service,
# interaction-count-service, viewer-count-service, faq-service) were searched for a "User
# Behavior Service" -- no such repository, OpenAPI/Swagger doc, or Postman collection exists
# anywhere locally. UBS_BASE_URL therefore has NO confirmed real value to default to. Unset
# means "not configured" -- no production URL is invented here. On any UBS failure
# (unreachable/timeout/malformed/schema-invalid), UBS_FALLBACK_TO_LOCAL decides whether
# recommend() falls back to the local interactions table (default, same behavior a genuinely
# absent userProfile already has today) or raises a clear dependency failure
# (UserBehaviorSourceUnavailable, mapped to 503 by app.api.recommendation_routes).
UBS_BASE_URL = os.getenv("UBS_BASE_URL") or None
UBS_TIMEOUT_MS = _getenv_int("UBS_TIMEOUT_MS", "1000")
UBS_FALLBACK_TO_LOCAL = os.getenv("UBS_FALLBACK_TO_LOCAL", "true").strip().lower() == "true"

# Candidate Service client (app.services.providers.candidate_provider). Used only in REAL
# mode. Same real-contract search and same "unset == unconfigured, never invented" result as
# UBS_BASE_URL above -- no "Candidate Service" repository/contract exists locally either. On
# failure, recommend() falls back to request.candidates when the caller supplied any,
# otherwise raises a clear dependency failure (CandidateSourceUnavailable, mapped to 503).
CANDIDATE_SERVICE_BASE_URL = os.getenv("CANDIDATE_SERVICE_BASE_URL") or None
CANDIDATE_SERVICE_TIMEOUT_MS = _getenv_int("CANDIDATE_SERVICE_TIMEOUT_MS", "1000")

# Optional Kafka behavioral-event consumer (Event Tracking Service -> Kafka -> this service;
# see app/services/kafka_behavior_consumer.py and README "Boundary with the real platform").
# OFF by default: local/demo/test operation never needs Kafka, and a missing/unreachable
# broker must never block startup or break POST /events. No Kafka client library is a hard
# dependency of this project (see requirements.txt) -- the consumer module imports one
# lazily, exactly like the Two-Tower/torch integration already does for its own optional
# dependency (see app.services.recommendation_service._try_two_tower_candidates).
#
# KAFKA_BROKERS (not KAFKA_BOOTSTRAP_SERVERS): this name -- along with KAFKA_CLIENT_ID and
# the KAFKA_<DOMAIN>_GROUP_ID / KAFKA_TOPIC_<NAME> shape below -- is the ACTUAL, confirmed
# naming convention every real Kafka-consuming sibling service in this workspace uses
# (follow-service/.env.example, ranking-service/README.md, interaction-count-service),
# reused here deliberately instead of the generic "KAFKA_BOOTSTRAP_SERVERS" name this module
# used before that search. No repository for the platform's actual event-tracking-service
# exists locally, and no topic/schema was found anywhere carrying VIDEO watch-time/completion
# signals (the two real event envelopes found -- follow-service's Avro `BaseEvent`
# {eventId,eventType,userId,timestamp,schemaVersion,region,device,sessionId,requestId,
# appVersion,deviceId} and ranking-service's JSON {eventId,type,target,data,occurredAt} --
# are each domain-specific and neither is a VIDEO-engagement topic) -- so KAFKA_BEHAVIOR_TOPIC
# has NO confirmed default either; it must be set explicitly before the consumer will start
# (see kafka_behavior_consumer.start_consumer_in_background), never a guessed topic name.
KAFKA_BEHAVIOR_ENABLED = os.getenv("KAFKA_BEHAVIOR_ENABLED", "false").strip().lower() == "true"
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS") or None
KAFKA_BEHAVIOR_TOPIC = os.getenv("KAFKA_BEHAVIOR_TOPIC") or None
KAFKA_BEHAVIOR_GROUP_ID = os.getenv("KAFKA_BEHAVIOR_GROUP_ID", "recommendation-ml-service")

# Optional Kafka user.registered consumer (User Service -> Kafka -> this service; see
# app/services/kafka_user_consumer.py, Task: continuous user-projection ingestion). Same
# OFF-by-default/no-guessed-topic posture as KAFKA_BEHAVIOR_* above, and shares KAFKA_BROKERS
# (one cluster, many topics/consumer groups) -- a separate group id keeps this consumer's
# offsets independent of the behavioral-event consumer's. NO confirmed real topic name for
# User Service's user.registered stream was found anywhere in this workspace (unlike the
# envelope SHAPE, which matches ranking-service's real, confirmed JSON convention -- see that
# module's own docstring) -- KAFKA_USER_REGISTERED_TOPIC therefore has no default either and
# must be set explicitly before this consumer will start. Still needs confirmation with the
# platform/User Service team.
KAFKA_USER_REGISTERED_ENABLED = os.getenv("KAFKA_USER_REGISTERED_ENABLED", "false").strip().lower() == "true"
KAFKA_USER_REGISTERED_TOPIC = os.getenv("KAFKA_USER_REGISTERED_TOPIC") or None
KAFKA_USER_REGISTERED_GROUP_ID = os.getenv("KAFKA_USER_REGISTERED_GROUP_ID", "recommendation-ml-service-user-registered")

# JWT/JWKS authentication (Task: first isolated JWT/JWKS verification vertical slice --
# GET /auth/me only, see app.core.jwt_auth). Platform contract confirmed via Follow Service's
# own real, independently-audited Keycloak integration: RS256 only, JWT header carries `kid`,
# the signing key is resolved through the issuer's JWKS, a small configurable clock-skew
# allowance on `exp` (default 10s), audience validation conditional/configurable, and
# authenticated identity is EXACTLY the verified `sub` claim.
#
# JWT_ALLOWED_ISSUERS: a strict, comma-separated allow-list. A token's own (still-
# unverified-at-that-point) `iss` claim is used ONLY to select among these operator-
# configured entries before any cryptographic work happens -- it is NEVER concatenated into
# a JWKS URL or otherwise trusted to construct one. Empty/unset (the default) means no
# issuer is allowed at all, so GET /auth/me fails closed (401) until this is explicitly
# configured -- never a guessed/hardcoded development Keycloak issuer.
JWT_ALLOWED_ISSUERS = [v.strip() for v in os.getenv("JWT_ALLOWED_ISSUERS", "").split(",") if v.strip()]


def _parse_issuer_jwks_map(raw: str) -> dict[str, str]:
    """Parses "issuer1=https://.../certs,issuer2=https://.../certs" into a dict. One
    explicit, operator-supplied JWKS URL per allowed issuer -- never derived from the
    token's own claims or from OIDC discovery at request time."""
    mapping: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        issuer, _, jwks_uri = entry.partition("=")
        issuer, jwks_uri = issuer.strip(), jwks_uri.strip()
        if issuer and jwks_uri:
            mapping[issuer] = jwks_uri
    return mapping


# An issuer present in JWT_ALLOWED_ISSUERS with no corresponding entry here is a
# configuration gap, not a security hole: app.core.jwt_auth treats it identically to
# "issuer not allowed" (fail closed), never silently skipping JWKS verification.
JWT_ISSUER_JWKS_URIS = _parse_issuer_jwks_map(os.getenv("JWT_ISSUER_JWKS_URIS", ""))
# Optional: unset (the default) means audience validation is skipped entirely, matching the
# platform contract's own "audience validation is conditional/configurable" note -- never
# invented/assumed for this service without confirming current platform policy.
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE") or None
JWT_CLOCK_SKEW_SECONDS = _getenv_int("JWT_CLOCK_SKEW_SECONDS", "10")
if JWT_CLOCK_SKEW_SECONDS < 0:
    raise ValueError(f"JWT_CLOCK_SKEW_SECONDS ({JWT_CLOCK_SKEW_SECONDS}) must be >= 0.")
# How long a fetched JWKS key set is reused before an unseen kid triggers a real refetch
# (kid-rotation support) -- see app.core.jwt_auth's PyJWKClient usage.
JWT_JWKS_CACHE_TTL_SECONDS = _getenv_int("JWT_JWKS_CACHE_TTL_SECONDS", "300")
if JWT_JWKS_CACHE_TTL_SECONDS <= 0:
    raise ValueError(f"JWT_JWKS_CACHE_TTL_SECONDS ({JWT_JWKS_CACHE_TTL_SECONDS}) must be > 0.")
# Bounded network timeout for a JWKS fetch -- never an unbounded/blocking call, and never a
# retry loop (a timed-out fetch is one failed request, surfaced as a safe 401).
JWT_JWKS_HTTP_TIMEOUT_SECONDS = _getenv_float("JWT_JWKS_HTTP_TIMEOUT_SECONDS", "5")
if JWT_JWKS_HTTP_TIMEOUT_SECONDS <= 0:
    raise ValueError(f"JWT_JWKS_HTTP_TIMEOUT_SECONDS ({JWT_JWKS_HTTP_TIMEOUT_SECONDS}) must be > 0.")

# VIDEO cohort cold-start preference system (app.services.cohort_preference_provider /
# app.services.cohort_aggregation_service). Replaces the removed
# LOCAL_POC_REGIONAL_CATEGORY_COHORT_PRIOR hand-authored table (app.ml.reranker) -- there is
# NO hardcoded demographic-preference fallback anywhere in this system; disabled, unconfigured,
# or genuinely empty cohort data all degrade to zero cohort influence, never a guess.
# COHORT_PREFERENCES_ENABLED: master switch. Default on, but harmless when the
# `recommendation_cohort_preferences` table has never been populated (rebuild_cohort_
# preferences has never run) -- resolve() then always returns NULL_COHORT_PROFILE and
# cold-start recommendation continues exactly as it would with the flag off.
COHORT_PREFERENCES_ENABLED = os.getenv("COHORT_PREFERENCES_ENABLED", "true").strip().lower() == "true"
# Minimum evidence a cohort level (REGION_AGE/REGION/AGE/GLOBAL) must clear before it is
# trusted at all -- below either bound, resolution falls through to the next broader level
# (spec §F/§G). Defaults are starting points, not measured production-scale numbers, same
# caveat as this module's existing DRIFT_MIN_OBSERVATIONS/RECOMMENDATION_HISTORY_MAX_
# INTERACTIONS defaults above.
COHORT_MIN_USERS = _getenv_int("COHORT_MIN_USERS", "30")
COHORT_MIN_INTERACTIONS = _getenv_int("COHORT_MIN_INTERACTIONS", "200")
# Shrinkage strength (app.services.cohort_preference_provider's own docstring has the exact
# formula): larger = more aggressive pull toward the broader/global prior for a given sample
# size. A cohort whose sample_interactions equals this constant is blended 50/50 with the
# broader prior.
COHORT_SHRINKAGE_K = _getenv_float("COHORT_SHRINKAGE_K", "200")
# Upper bound on the cohort signal's multiplicative reranking effect -- same role/shape as this
# module's own TWO_TOWER_RETRIEVAL_K-style "boost cap" constants in app.ml.reranker (e.g.
# SEARCH_RELEVANCE_BOOST_MAX), just made configurable here because operators may want to tune
# cohort influence without a code change once real data exists.
COHORT_MAX_COLD_START_BOOST = _getenv_float("COHORT_MAX_COLD_START_BOOST", "0.20")
# Strategy-dependent cohort weight (spec §J, "personal history must override cohort
# assumptions"): COLD_START always gets full weight (1.0) and PERSONALISED_ML always gets zero
# -- both structural, not tunable -- HYBRID sits between the two and IS tunable, since it is
# the one genuinely subjective choice ("how much should a partially-known user still lean on
# demographic evidence").
COHORT_HYBRID_STRATEGY_WEIGHT = _getenv_float("COHORT_HYBRID_STRATEGY_WEIGHT", "0.4")
if COHORT_MIN_USERS < 1 or COHORT_MIN_INTERACTIONS < 1:
    raise ValueError(
        f"COHORT_MIN_USERS ({COHORT_MIN_USERS}) and COHORT_MIN_INTERACTIONS ({COHORT_MIN_INTERACTIONS}) "
        "must both be >= 1."
    )
if COHORT_SHRINKAGE_K <= 0:
    raise ValueError(f"COHORT_SHRINKAGE_K must be > 0 (got {COHORT_SHRINKAGE_K}).")
if not (0 <= COHORT_MAX_COLD_START_BOOST <= 1):
    raise ValueError(f"COHORT_MAX_COLD_START_BOOST must be in [0, 1] (got {COHORT_MAX_COLD_START_BOOST}).")
if not (0 <= COHORT_HYBRID_STRATEGY_WEIGHT <= 1):
    raise ValueError(f"COHORT_HYBRID_STRATEGY_WEIGHT must be in [0, 1] (got {COHORT_HYBRID_STRATEGY_WEIGHT}).")

# Session/search intent persistence (app.services.session_intent_provider). "DATABASE" (default)
# uses this project's own existing PostgreSQL infrastructure (session_search_intent table) so a
# recorded search survives a restart and is visible to every worker/process -- "MEMORY" keeps
# the previous single-process/PoC-only behavior (still available for tests/local debugging that
# want to avoid DB round-trips). Never a new external dependency (no Redis/cache service is
# introduced here).
_ALLOWED_SESSION_INTENT_STORES = {"DATABASE", "MEMORY"}
SESSION_INTENT_STORE = os.getenv("SESSION_INTENT_STORE", "DATABASE").strip().upper()
if SESSION_INTENT_STORE not in _ALLOWED_SESSION_INTENT_STORES:
    raise ValueError(
        f"SESSION_INTENT_STORE must be one of {sorted(_ALLOWED_SESSION_INTENT_STORES)} "
        f"(got {SESSION_INTENT_STORE!r})"
    )

# Adapter-boundary configuration for the two REAL-mode placeholder integrations (spec §Q/§R):
# moves each endpoint path out of the adapter's business logic and into configuration, so a
# real contract replacement is a config change, not a code change. Defaults are byte-identical
# to the paths each adapter already had hardcoded -- see app.services.providers.user_behavior_
# provider/candidate_provider's own docstrings for why these are this project's own documented
# placeholders, not verified external contracts. `{userId}` in UBS_BEHAVIOR_PROFILE_PATH is a
# literal format placeholder substituted by the adapter, not a real path template syntax claim.
UBS_BEHAVIOR_PROFILE_PATH = os.getenv("UBS_BEHAVIOR_PROFILE_PATH", "/api/v1/users/{userId}/behavior-profile")
CANDIDATE_SERVICE_GENERATE_PATH = os.getenv("CANDIDATE_SERVICE_GENERATE_PATH", "/api/v1/candidates/generate")
