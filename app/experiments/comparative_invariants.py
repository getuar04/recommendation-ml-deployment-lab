"""Fail-fast invariant validation for the comparative-experiment pipeline (v2 rigor pass).

Every function here either returns quietly (all checked invariants held) or raises
`ComparativeInvariantError` naming exactly which invariant(s) failed -- never returns a
boolean for a caller to optionally act on. Callers (scripts/seed_comparative_dataset.py,
scripts/run_comparative_experiment.py, scripts/build_comparative_report.py,
scripts/validate_v2_compose_profile.py) are expected to let the exception propagate: a failed
invariant must stop the run, not be silently downgraded to a warning or a displayed-but-
ignored boolean.

Policy constants (`TARGET_LABELLED_SAMPLES_PER_CLASS_V2`, `COMPARATIVE_REFERENCE_TIMESTAMP_V2`,
the forbidden-port sets) live here because they are as much a part of "what must be true"
as the checks that enforce them -- a single source of truth for both dataset generation
call sites and the validators that later confirm generation actually produced it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------------------
# v2 policy constants
# ---------------------------------------------------------------------------------------

# Fixed generation anchor for the v2 comparative pipeline (spec: full reproducibility --
# never datetime.now()). Chosen once, never derived from wall-clock time.
COMPARATIVE_REFERENCE_TIMESTAMP_V2 = datetime(2026, 8, 1, tzinfo=timezone.utc)

# Deterministic post-generation stratified-downsample target (spec: class-balance control
# without touching engagement/label-generation parameters). Chosen below the minority-class
# count every v2 dataset is expected to produce, with margin -- verified, never assumed, by
# validate_dataset_complete_v2 after generation.
TARGET_LABELLED_SAMPLES_PER_CLASS_V2 = 450

# Model-level constants the v2 pipeline locks and verifies (mirrors app.ml.trainer.RANDOM_SEED
# and app.experiments.comparative_definitions.LOCKED_ALGORITHM -- restated here as explicit,
# named expectations for the invariant checks below to compare reports against).
EXPECTED_MODEL_RANDOM_STATE = 42
EXPECTED_ALGORITHM = "LogisticRegression"

DEFAULT_STACK_PROJECT_NAME = "machine-learning"
DEFAULT_STACK_PORTS = {3500, 6003}
# The four v1 comparative profiles' ports (app/postgres), documented once here so both the
# v2 Docker pre-flight check and any future port assignment can avoid colliding with them.
V1_PROFILE_PORTS = {3510, 3511, 3512, 3513, 6010, 6011, 6012, 6013}

DATASET_KEYS = ("sport", "entertainment", "music", "balanced")


class ComparativeInvariantError(Exception):
    """A required comparative-experiment invariant did not hold. The message always names
    exactly which check(s) failed -- never a generic/opaque failure."""


def _raise_if_any(errors: list[str], *, context: str) -> None:
    if errors:
        raise ComparativeInvariantError(f"{context}:\n- " + "\n- ".join(errors))


# ---------------------------------------------------------------------------------------
# Docker Compose v2-profile pre-flight validation (host-side, no container/DB access --
# operates on already-parsed `docker compose ... config --format json` output)
# ---------------------------------------------------------------------------------------

def validate_v2_compose_profile(config: dict[str, Any], *, dataset_key: str) -> None:
    """Must be called, and must pass, before every v2 `docker compose up`. Validates the
    *resolved* configuration only -- never inspects live containers or volumes, so it is safe
    to run before anything starts."""
    if dataset_key not in DATASET_KEYS:
        raise ComparativeInvariantError(f"Unknown dataset key {dataset_key!r}; expected one of {DATASET_KEYS}.")
    errors: list[str] = []

    project_name = config.get("name")
    if project_name == DEFAULT_STACK_PROJECT_NAME:
        errors.append(f"Compose project name is the default stack's project name ({project_name!r}) -- refusing to start a v2 profile under it.")
    expected_project = f"machine-learning-comparative-{dataset_key}-v2"
    if project_name != expected_project:
        errors.append(f"Compose project name is {project_name!r}, expected {expected_project!r} (must identify v2 and this dataset).")

    volumes = config.get("volumes") or {}
    expected_volume_names = {
        "postgres_data": f"recommendation_postgres_{dataset_key}_v2",
        "model_data": f"recommendation_models_{dataset_key}_v2",
        "experiment_data": f"recommendation_reports_{dataset_key}_v2",
    }
    for volume_key, expected_name in expected_volume_names.items():
        actual = (volumes.get(volume_key) or {}).get("name")
        if actual != expected_name:
            errors.append(f"Volume {volume_key!r} resolved to {actual!r}, expected {expected_name!r} (must identify {dataset_key} and v2).")

    app_environment = ((config.get("services") or {}).get("app") or {}).get("environment") or {}
    expected_experiment_id = f"comparative-{dataset_key}-v2"
    if app_environment.get("EXPERIMENT_ID") != expected_experiment_id:
        errors.append(f"App EXPERIMENT_ID resolved to {app_environment.get('EXPERIMENT_ID')!r}, expected {expected_experiment_id!r}.")
    if app_environment.get("DATASET_VERSION") != expected_experiment_id:
        errors.append(f"App DATASET_VERSION resolved to {app_environment.get('DATASET_VERSION')!r}, expected {expected_experiment_id!r}.")

    forbidden_ports = DEFAULT_STACK_PORTS | V1_PROFILE_PORTS
    resolved_ports: set[int] = set()
    services = config.get("services") or {}
    for service_name in ("app", "postgres"):
        for port_entry in (services.get(service_name) or {}).get("ports") or []:
            published = port_entry.get("published")
            if published:
                resolved_ports.add(int(published))
    collisions = resolved_ports & forbidden_ports
    if collisions:
        errors.append(f"Resolved host ports {sorted(collisions)} collide with the default stack's or a v1 profile's known ports ({sorted(forbidden_ports)}).")

    _raise_if_any(errors, context=f"v2 Compose profile validation failed for dataset {dataset_key!r}")


# ---------------------------------------------------------------------------------------
# Pre-training database invariants (version-aware: v1 keeps the existing loose threshold
# production training already enforces; v2 requires exact, deterministic class balance)
# ---------------------------------------------------------------------------------------

def validate_experiment_id_matches(definition) -> None:
    """An unset EXPERIMENT_ID counts as a mismatch (fail-closed), not a bypass."""
    from app.core.config import EXPERIMENT_ID
    if EXPERIMENT_ID != definition.experiment_id:
        raise ComparativeInvariantError(
            f"EXPERIMENT_ID is {EXPERIMENT_ID!r}, expected {definition.experiment_id!r}. "
            "Refusing to proceed: this environment's EXPERIMENT_ID does not identify the "
            "requested dataset (an unset EXPERIMENT_ID is treated as a mismatch)."
        )


def _own_and_foreign_comparative_rows(db, definition) -> tuple[list, list]:
    from sqlalchemy import select

    from app.db.models import Interaction
    own_prefix = f"syn-cmp-{definition.experiment_id}-"
    all_comparative = db.scalars(select(Interaction).where(Interaction.event_id.like("syn-cmp-%"))).all()
    own = [row for row in all_comparative if row.event_id.startswith(own_prefix)]
    foreign = [row for row in all_comparative if not row.event_id.startswith(own_prefix)]
    return own, foreign


def validate_no_foreign_comparative_rows(db, definition) -> None:
    _, foreign = _own_and_foreign_comparative_rows(db, definition)
    if foreign:
        sample = sorted({row.event_id for row in foreign})[:5]
        raise ComparativeInvariantError(
            f"Database contains {len(foreign)} interaction row(s) belonging to a different "
            f"comparative experiment than {definition.experiment_id!r} (e.g. {sample}). Each "
            "comparative dataset must live in its own isolated database."
        )


def validate_dataset_complete_v1(db, definition) -> None:
    """Mirrors the existing production threshold (>=100 labelled rows, both classes present --
    app.services.training_service.InsufficientData) as an earlier, clearer pre-flight check;
    does not change what training itself requires."""
    from app.experiments.dataset_summary import video_dataset_summary
    own, _ = _own_and_foreign_comparative_rows(db, definition)
    summary = video_dataset_summary(own)
    errors = []
    if summary["labelledSamples"] < 100:
        errors.append(f"Only {summary['labelledSamples']} labelled rows (need >= 100) -- dataset missing or partially seeded.")
    if summary["positiveSamples"] == 0 or summary["negativeSamples"] == 0:
        errors.append(f"Missing a class: {summary['positiveSamples']} positive / {summary['negativeSamples']} negative.")
    _raise_if_any(errors, context=f"v1 dataset readiness check failed for {definition.experiment_id!r}")


def validate_dataset_complete_v2(db, definition, *, target_per_class: int = TARGET_LABELLED_SAMPLES_PER_CLASS_V2) -> None:
    """v2's stronger guarantee: the realized labelled sample counts must *exactly* match the
    deterministic downsample target (not merely clear a loose minimum) -- this is what proves
    the dataset was fully and correctly seeded by the v2 generator, not partially seeded, not
    seeded by a different generator version, and not missing rows."""
    from app.experiments.dataset_summary import video_dataset_summary
    own, _ = _own_and_foreign_comparative_rows(db, definition)
    summary = video_dataset_summary(own)
    errors = []
    if summary["positiveSamples"] != target_per_class or summary["negativeSamples"] != target_per_class:
        errors.append(
            f"Realized labelled samples are {summary['positiveSamples']} positive / "
            f"{summary['negativeSamples']} negative; expected exactly "
            f"{target_per_class}/{target_per_class}."
        )
    _raise_if_any(errors, context=f"v2 dataset readiness check failed for {definition.experiment_id!r}")


def validate_pre_training(db, definition, *, version: str) -> None:
    """The complete pre-training gate (spec section 1 / section 6): call this immediately
    before triggering `/model/train`. Raises on the first category of failure encountered;
    all checks still run so the error message can name every problem found, not just one."""
    if version not in ("v1", "v2"):
        raise ComparativeInvariantError(f"Unknown experiment version {version!r}; expected 'v1' or 'v2'.")
    validate_experiment_id_matches(definition)
    validate_no_foreign_comparative_rows(db, definition)
    if version == "v2":
        validate_dataset_complete_v2(db, definition)
    else:
        validate_dataset_complete_v1(db, definition)


# ---------------------------------------------------------------------------------------
# Post-training, pre-report-write invariants (v2 only -- spec section 5)
# ---------------------------------------------------------------------------------------

def validate_report_invariants_v2(
    report: dict[str, Any], definition, *, expected_hyperparameters: dict[str, Any], expected_fixed_scenario: dict[str, Any],
) -> None:
    """Every check spec section 5 requires, checked against one experiment's real, executed
    training + recommendation result -- called immediately before `write_report()`. On any
    failure, the caller must not write the report."""
    from app.ml.dataset_builder import FEATURES

    errors: list[str] = []

    if report.get("algorithm") != definition.algorithm:
        errors.append(f"algorithm is {report.get('algorithm')!r}, expected {definition.algorithm!r}.")
    if report.get("selectedModel") != EXPECTED_ALGORITHM:
        errors.append(f"selectedModel is {report.get('selectedModel')!r}, expected {EXPECTED_ALGORITHM!r}.")
    model_comparison = report.get("modelComparison") or {}
    if set(model_comparison) != {EXPECTED_ALGORITHM}:
        errors.append(f"modelComparison keys are {sorted(model_comparison)}, expected exactly [{EXPECTED_ALGORITHM!r}] (automatic model selection must not have run).")
    if report.get("algorithmHyperparameters") != expected_hyperparameters:
        errors.append("algorithmHyperparameters do not match the expected locked configuration.")
    if report.get("modelRandomState") != EXPECTED_MODEL_RANDOM_STATE:
        errors.append(f"modelRandomState is {report.get('modelRandomState')!r}, expected {EXPECTED_MODEL_RANDOM_STATE!r}.")
    if report.get("datasetGenerationSeed") != definition.seed:
        errors.append(f"datasetGenerationSeed is {report.get('datasetGenerationSeed')!r}, expected {definition.seed!r}.")
    if report.get("featureNames") != FEATURES:
        errors.append("featureNames/order do not match app.ml.dataset_builder.FEATURES.")
    if report.get("splitLifecycleUsedFallback") is not False:
        errors.append(f"splitLifecycleUsedFallback is {report.get('splitLifecycleUsedFallback')!r}, expected False (a dataset that fell back to a coarser split tier is not comparable to the others).")
    if not report.get("artifactChecksum"):
        errors.append("artifactChecksum is missing or empty.")
    model_status = report.get("modelStatus") or {}
    if model_status.get("status") != "READY":
        errors.append(f"modelStatus.status is {model_status.get('status')!r}, expected 'READY'.")
    if not report.get("modelVersion") or model_status.get("modelVersion") != report.get("modelVersion"):
        errors.append(f"modelStatus.modelVersion ({model_status.get('modelVersion')!r}) does not match the report's own modelVersion ({report.get('modelVersion')!r}).")
    if report.get("fixedScenario") != expected_fixed_scenario:
        errors.append("fixedScenario does not match the expected primary scenario (user id / limit / candidate categories / candidate count).")
    dataset = report.get("dataset") or {}
    if dataset.get("positiveSamples") != TARGET_LABELLED_SAMPLES_PER_CLASS_V2 or dataset.get("negativeSamples") != TARGET_LABELLED_SAMPLES_PER_CLASS_V2:
        errors.append(f"Realized class balance ({dataset.get('positiveSamples')} pos / {dataset.get('negativeSamples')} neg) does not match the configured v2 target ({TARGET_LABELLED_SAMPLES_PER_CLASS_V2}/{TARGET_LABELLED_SAMPLES_PER_CLASS_V2}).")
    if not report.get("rawModelScores"):
        errors.append("rawModelScores is missing or empty -- the primary scenario's raw model probabilities were not captured.")

    _raise_if_any(errors, context=f"Post-training invariant validation failed for {definition.experiment_id!r} -- report not written")


def validate_parity(in_process_response: dict[str, Any], http_response: dict[str, Any]) -> None:
    """Proves the experiment-only raw-score capture path (which calls the real
    recommendation_service.recommend() in-process) and the real HTTP /recommendations
    endpoint agree exactly for the same model version and candidates: same modelVersion, same
    strategy, same adjusted scores, same final order, same reasons. Must be called, and must
    pass, before any report is written."""
    errors: list[str] = []
    if in_process_response.get("modelVersion") != http_response.get("modelVersion"):
        errors.append(f"modelVersion differs: in-process={in_process_response.get('modelVersion')!r} vs HTTP={http_response.get('modelVersion')!r}.")
    if in_process_response.get("strategy") != http_response.get("strategy"):
        errors.append(f"strategy differs: in-process={in_process_response.get('strategy')!r} vs HTTP={http_response.get('strategy')!r}.")

    in_recs = in_process_response.get("recommendations") or []
    http_recs = http_response.get("recommendations") or []
    if len(in_recs) != len(http_recs):
        errors.append(f"recommendations length differs: in-process={len(in_recs)} vs HTTP={len(http_recs)}.")
    else:
        for index, (in_item, http_item) in enumerate(zip(in_recs, http_recs)):
            if in_item.get("contentId") != http_item.get("contentId"):
                errors.append(f"recommendations[{index}].contentId differs: {in_item.get('contentId')!r} vs {http_item.get('contentId')!r} (final order differs).")
            if in_item.get("category") != http_item.get("category"):
                errors.append(f"recommendations[{index}].category differs.")
            if in_item.get("rank") != http_item.get("rank"):
                errors.append(f"recommendations[{index}].rank differs: {in_item.get('rank')!r} vs {http_item.get('rank')!r}.")
            if in_item.get("reason") != http_item.get("reason"):
                errors.append(f"recommendations[{index}].reason differs: {in_item.get('reason')!r} vs {http_item.get('reason')!r}.")
            in_score, http_score = in_item.get("score"), http_item.get("score")
            if in_score is None or http_score is None or abs(in_score - http_score) > 1e-9:
                errors.append(f"recommendations[{index}].score differs: {in_score!r} vs {http_score!r}.")

    _raise_if_any(errors, context="Raw/adjusted parity check failed between the in-process experiment path and the real HTTP /recommendations response")


# ---------------------------------------------------------------------------------------
# Cross-report invariants (build_comparative_report.py, before writing any output file)
# ---------------------------------------------------------------------------------------

def validate_cross_report_invariants_v2(reports: dict[str, Any]) -> None:
    """All four v2 reports must agree on everything the Core Experiment Rule requires to be
    identical. Called before comparison.json/comparison.md are written; on failure, neither
    file is written."""
    errors: list[str] = []
    missing = [key for key in DATASET_KEYS if key not in reports]
    if missing:
        raise ComparativeInvariantError(f"Missing report(s) for: {missing}. Cannot build a v2 comparison without all four.")

    for key, report in reports.items():
        if report.get("status") != "SUCCEEDED":
            errors.append(f"Report {key!r} has status={report.get('status')!r}, not SUCCEEDED.")
        if not (report.get("rawModelScores")):
            errors.append(f"Report {key!r} has no rawModelScores -- cannot compute a raw-score-based verdict.")
    _raise_if_any(errors, context="Cross-report validation failed (per-report issues)")

    def _all_equal(field: str, transform=lambda v: v) -> list[str]:
        values = {key: transform(report.get(field)) for key, report in reports.items()}
        distinct = list(values.values())
        if not all(value == distinct[0] for value in distinct):
            return [f"{field} differs across experiments: {values}"]
        return []

    for field in (
        "algorithm", "algorithmHyperparameters", "datasetGenerationSeed", "modelRandomState",
        "featureNames", "splitLifecycleUsedFallback", "fixedScenario",
    ):
        errors += _all_equal(field)

    _raise_if_any(errors, context="Cross-report validation failed (v2 reports are not comparable)")
