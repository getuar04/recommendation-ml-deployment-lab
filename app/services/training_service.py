"""Orchestrates VIDEO training: build the dataset, train/select (classifier-only or
cross-family, see app.ml.trainer), assemble the versioned artifact-metadata contract
(app.ml.artifact_metadata, validated against app.ml.model_store's family-aware required-field
sets), and persist atomically.
"""
from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from sqlalchemy import select

from app.core.config import TRAINING_ALGORITHM_LOCK
from app.core.logging import logger
from app.db.models import Content
from app.db.repositories import interactions
from app.ml import artifact_lifecycle, artifact_metadata, model_cache, model_store
from app.ml.dataset_builder import CATEGORICAL, FEATURES, build_dataset
from app.ml.dataset_diagnostics import dataset_diagnostics
from app.ml.split_lifecycle import InsufficientLifecycleDataError
from app.ml.trainer import train_and_select_cross_family, train_models


class InsufficientData(Exception):
    pass


class NoEligibleModel(Exception):
    """Finalization spec Step 3: every candidate algorithm failed at least one mandatory
    behavioral gate (app.ml.eligibility.evaluate_eligibility) -- a model must never be
    promoted just because it happened to have the best aggregate metric among a set of
    behaviorally-ineligible candidates. Raised before any save/promote call, so the active
    model is left completely untouched. Applies identically to a classifier or ranker
    candidate -- see app.ml.eligibility_policy, shared unmodified by both families."""


def train(db) -> dict[str, Any]:
    started = perf_counter()
    rows = interactions(db)
    content_by_id = {item.content_id: item for item in db.scalars(select(Content)).all()}
    # VIDEO-only training scope: `interactions` (app.db.repositories.interactions, a shared
    # helper also used elsewhere) carries no VIDEO/LIVE discriminator of its own -- event
    # ingestion (app.services.event_service.store_event) accepts LIVE_*/domain-ambiguous event
    # types into this same table with no cross-check against the referenced content's own type.
    # A row whose Content is PROVABLY "LIVE" is excluded here (this module only, not the shared
    # repository, so any other caller of `interactions()` is unaffected); a row whose Content is
    # missing/unavailable is left in unchanged -- build_dataset() already tolerates a missing
    # Content row via neutral defaults, and "unavailable" is not evidence of being LIVE.
    video_rows = [row for row in rows if getattr(content_by_id.get(row.content_id), "content_type", "VIDEO") != "LIVE"]
    df = build_dataset(video_rows, content_by_id)
    logger.info("Built point-in-time training dataset with %d labeled rows", len(df))
    if len(df) < 100 or df.target.nunique() < 2:
        raise InsufficientData("At least 100 labeled interactions containing positive and negative samples are required.")

    diagnostics = dataset_diagnostics(df, total_interaction_count=len(video_rows))
    logger.info(
        "Dataset diagnostics: labeled=%d positive=%.1f%% negative=%.1f%% notInterested=%d (%.1f%% of negatives) "
        "users=%d categories=%d",
        diagnostics["labeledRows"], diagnostics["positiveRatio"] * 100, diagnostics["negativeRatio"] * 100,
        diagnostics["negativeBreakdown"]["contentNotInterestedCount"],
        (diagnostics["negativeBreakdown"]["contentNotInterestedPercentOfNegatives"] or 0) * 100,
        diagnostics["uniqueUsers"], diagnostics["uniqueCategories"],
    )

    # Task 5 / XGBRanker promotion task: TRAINING_ALGORITHM_LOCK still means "only train/
    # evaluate that one classifier algorithm" -- with exactly one candidate, layered selection
    # has nothing to select among, so the locked path keeps using the simpler, benchmark-free
    # `train_models` (raw ModelBehavior gates only) exactly as before; it never considers
    # rankers. Unlocked (the default) training now compares every enabled classifier candidate
    # PLUS app.ml.ranker_registry.PRODUCTION_RANKER_NAMES through
    # `train_and_select_cross_family`'s layered, cross-family eligibility/quality policy
    # (app.ml.eligibility_policy/app.ml.quality_scorer/app.ml.candidate_pool).
    try:
        if TRAINING_ALGORITHM_LOCK:
            result = train_models(df)
        else:
            result = train_and_select_cross_family(df)
    except InsufficientLifecycleDataError as exc:
        raise InsufficientData(str(exc)) from exc

    if not result["eligibleSelection"]:
        if "eligibilityDecisions" in result:
            failing = {
                name: decision["rejectionReasons"] for name, decision in result["eligibilityDecisions"].items()
            }
            raise NoEligibleModel(
                "No candidate algorithm was eligible under the layered selection policy (every "
                "HARD model-behavior gate, the HARD reranker-policy check, and a minimum "
                f"end-to-end critical-constraint pass rate must all pass); rejection reasons per "
                f"candidate: {failing}. The active model was left unchanged."
            )
        failing = {
            name: [gate for gate, report in eg["gates"].items() if not report["pass"]]
            for name, eg in result["eligibility"].items()
        }
        raise NoEligibleModel(
            "No candidate algorithm passed every mandatory behavioral gate (long-term/recent/"
            "session/negative/notInterested/semantic/creator/coldStart/"
            f"subthemeRejectionLocalization); failing gates per candidate: {failing}. "
            "The active model was left unchanged."
        )

    now = datetime.now(timezone.utc).isoformat()
    version = f"recommendation-prod-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    model = result.pop("model")

    metadata = artifact_metadata.build_metadata(
        result, model=model, version=version, now=now, started=started, rows=video_rows, diagnostics=diagnostics,
    )
    # Group D: never write straight to the active path. Save the candidate to its sibling
    # path first, validate it (schema/feature/checksum + inference smoke test), and only
    # then atomically promote it onto the active path -- retiring whatever was active
    # before into `previous` so a rollback target is always available. Applies identically
    # regardless of winner family: app.ml.model_store.load_validated (used inside `promote`)
    # already validates classifier- and ranker-shaped metadata against the correct required
    # field set for whichever `modelFamily` the candidate declares.
    #
    # Active paths are read as `model_store.MODEL_PATH`/`model_store.METADATA_PATH` (not
    # imported into this module's own namespace) -- exactly as the pre-Phase-2 code did by
    # calling `model_store.save(model, metadata)` with no explicit path args. That is the
    # one monkeypatch point existing tests already target; reading it dynamically here
    # keeps `monkeypatch.setattr(model_store, "MODEL_PATH", ...)` sufficient without also
    # needing to patch this module.
    active_model_path, active_metadata_path = model_store.MODEL_PATH, model_store.METADATA_PATH
    paths = artifact_lifecycle.sibling_paths(active_model_path, active_metadata_path)
    model_store.save(model, metadata, model_path=paths["candidate_model"],
                      metadata_path=paths["candidate_metadata"], model_dir=model_store.MODEL_DIR)
    promoted_metadata = artifact_lifecycle.promote(
        candidate_model_path=paths["candidate_model"], candidate_metadata_path=paths["candidate_metadata"],
        active_model_path=active_model_path, active_metadata_path=active_metadata_path,
        previous_model_path=paths["previous_model"], previous_metadata_path=paths["previous_metadata"],
        feature_names=FEATURES, categorical_features=CATEGORICAL,
    )
    model_cache.video_cache.invalidate()
    logger.info(
        "Selected %s (modelFamily=%s) by %s, promoted candidate to active model %s",
        result["selectedModel"], metadata["modelFamily"], result["selectionCriterion"], version,
    )
    return promoted_metadata
