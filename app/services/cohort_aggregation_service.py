"""Batch rebuild of VIDEO cohort cold-start preference statistics.

Never called on the recommendation hot path (see app.services.cohort_preference_provider,
which only ever reads the persisted output of this job). Invoked explicitly -- today via
scripts/rebuild_cohort_preferences.py, spec §O's "a callable service/script/internal
operation is sufficient for now" -- whenever an operator wants cohort preferences to reflect
newer interaction history; no code change or restart is required for a rebuild to take effect
(serving always reads the single latest `version` row, see cohort_preference_provider.
_latest_version).

Flow (spec §C): `interactions` (this project's own table, the same source
app.services.training_service.train() already reads via app.db.repositories.interactions),
excluding rows whose content is PROVABLY `Content.content_type == "LIVE"` (VIDEO/LIVE isolation
fix -- see `rebuild_cohort_preferences`'s own comment for the exact compatibility rule and why
it reuses an already-established column rather than inventing a new one), joined in-process
against `user_demographic_context` (the region/age-bucket a user most recently supplied on a
VIDEO recommendation request -- see cohort_preference_provider.record_demographic_context) ->
aggregate positive/negative evidence per (cohort level, category) -> persist one new
`version` of `recommendation_cohort_preferences` -> done.

Label semantics are NOT reimplemented here: `app.ml.feature_builder.target_for` (already used
by app.ml.dataset_builder for the exact same interaction rows, for VIDEO model training) is
reused verbatim -- 1 (positive: watch_percentage>=70, or liked/shared/favorited/creator_
followed), 0 (negative: explicit CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED, or fast/short
watch with no positive flag), or None (ambiguous/no signal -- excluded from cohort evidence,
same as it is excluded from training rows).
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.db.models import CohortPreference, Content, UserDemographicContext
from app.db.repositories import interactions as all_interactions
from app.ml.feature_builder import target_for
from app.ml.replay_saturation_policy import (
    is_saturation_controlled_interaction,
    repeat_influence_weight,
)
from app.services import training_lock

__all__ = ["rebuild_cohort_preferences"]

# Serializes concurrent rebuilds using this project's own existing DB-backed, PK-based mutual
# exclusion (app.services.training_lock -- one row per `model_type`, restart-/multi-worker-safe,
# with stale-lock reclaim). Without this, two overlapping rebuilds each read
# MAX(version) before either commits, both allocate the SAME next version number (no unique
# constraint on `version` or on (version, cohort_type, region, age_bucket, category) prevents
# this at the database level either), and the resulting version silently mixes rows from both
# runs -- exactly the "partially written/corrupted version" this table's own docstring says can
# never happen. Reusing `training_lock` here (keyed by this dedicated pseudo-model-type) avoids
# inventing a second, parallel locking mechanism for what is the same class of problem
# (app.services.training_lock's own docstring: "restart-safe, per-model-type lock").
_REBUILD_LOCK_KEY = "COHORT_REBUILD"


@dataclass
class _Accumulator:
    # Raw, unweighted counts -- UNCHANGED meaning, still what CohortPreference.positive_count/
    # negative_count/sample_interactions persist (observability/provenance: "how many labeled
    # interactions actually fed this cohort", an integer-shaped fact the DB schema already
    # commits to). NOT used to compute preference_score anymore -- see weighted_positive/
    # weighted_negative below.
    positive: int = 0
    negative: int = 0
    # Replay/exposure saturation (app.ml.replay_saturation_policy): the SAME per-(user,
    # content) occurrence-weighted evidence used for serving/training, applied here so one
    # user's repeated passive replay of ONE content cannot disproportionately swing this
    # cohort's preference_score for every OTHER user sharing it. Only this weighted pair feeds
    # preference_score -- deliberately kept SEPARATE from the raw positive/negative counts
    # above (no schema change: CohortPreference.preference_score is already a float column;
    # positive_count/negative_count/sample_interactions stay raw integers exactly as
    # documented).
    weighted_positive: float = 0.0
    weighted_negative: float = 0.0
    users: set[str] = field(default_factory=set)


def _cohort_keys(region: str | None, age_bucket: str | None) -> list[tuple[str, str | None, str | None]]:
    """Every aggregation-level key a single user's demographic context contributes evidence
    to, in one pass over their interactions -- mirrors cohort_preference_provider._fallback_
    levels' set of levels (minus the "which one wins" fallback logic, irrelevant here: every
    level this user has data for is simply built, so THAT resolution can choose at read time)."""
    keys: list[tuple[str, str | None, str | None]] = [("GLOBAL", None, None)]
    if region:
        keys.append(("REGION", region, None))
    if age_bucket:
        keys.append(("AGE", None, age_bucket))
    if region and age_bucket:
        keys.append(("REGION_AGE", region, age_bucket))
    return keys


def _next_version(db: Session) -> int:
    current_max = db.scalar(select(func.max(CohortPreference.version)))
    return (current_max or 0) + 1


def rebuild_cohort_preferences(db: Session, *, now: datetime | None = None) -> dict[str, Any]:
    """Rebuilds cohort preferences from the current contents of `interactions` +
    `user_demographic_context` and persists them as a brand-new `version` (additive: prior
    versions are left in place, not deleted, so a concurrent reader mid-rebuild always sees a
    complete, consistent version). Returns a small summary dict (never a per-user or per-
    interaction payload -- spec §M, aggregates only).

    Serialized via `_REBUILD_LOCK_KEY` (see its own comment above): a second, concurrent call
    while a rebuild is already in flight raises `training_lock.TrainingAlreadyRunningError`
    instead of silently allocating the same `version` as the in-flight rebuild."""
    job_id = uuid.uuid4().hex
    training_lock.acquire(db, _REBUILD_LOCK_KEY, job_id)
    try:
        return _rebuild_cohort_preferences_locked(db, now=now)
    finally:
        training_lock.release(db, _REBUILD_LOCK_KEY, job_id)


def _rebuild_cohort_preferences_locked(db: Session, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    demographics = {row.user_id: row for row in db.scalars(select(UserDemographicContext)).all()}
    if not demographics:
        logger.info("cohort rebuild skipped: no user_demographic_context rows yet")
        return {"version": None, "cohortsWritten": 0, "usersConsidered": 0}

    # VIDEO-only scope: `interactions` (app.db.repositories.interactions, reused unfiltered by
    # app.services.training_service.train() too) has no VIDEO/LIVE discriminator of its own --
    # event ingestion (app.services.event_service.store_event) accepts the full EventType enum,
    # LIVE_*/domain-ambiguous CONTENT_*/CREATOR_FOLLOWED types included, into this SAME table,
    # with no cross-check against the referenced content's own type. `Content.content_type`
    # (existing column, default "VIDEO") is the one reliable, already-established discriminator.
    #
    # Compatibility rule (matches training_service/user_behavior_provider/drift_service/
    # candidate_routes exactly -- a row PROVABLY LIVE is excluded; a row whose Content is
    # missing/unavailable is KEPT, since "unavailable" is not evidence of being LIVE): a naive
    # VIDEO allowlist (only keep rows whose content_id IS a known VIDEO content_id) was tried
    # and reverted -- it silently dropped every interaction whose Content row happens to be
    # missing/deleted, which is a real, expected state (e.g. this exact test suite's own
    # fixtures build Interaction rows with no matching Content row at all). One bounded query
    # for every content_id actually referenced by `rows` below -- never N+1.
    rows = all_interactions(db)
    referenced_content_ids = {row.content_id for row in rows}
    content_by_id = (
        {item.content_id: item for item in db.scalars(select(Content).where(Content.content_id.in_(referenced_content_ids))).all()}
        if referenced_content_ids else {}
    )

    # key: (cohort_type, region, age_bucket, category) -> Accumulator
    aggregates: dict[tuple[str, str | None, str | None, str], _Accumulator] = defaultdict(_Accumulator)
    # Replay/exposure saturation (app.ml.replay_saturation_policy): `rows` (all_interactions)
    # is already in chronological (ascending timestamp) order -- a single pass, one
    # occurrence tally per (user_id, content_id), computed ONCE per row and reused across
    # every cohort level that row contributes to below (a row's replay weight does not depend
    # on which cohort level is being asked about).
    content_occurrences: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        if getattr(content_by_id.get(row.content_id), "content_type", "VIDEO") == "LIVE":
            continue
        demo = demographics.get(row.user_id)
        if demo is None or (demo.region is None and demo.age_bucket is None):
            continue
        if getattr(row, "is_training_context_only", False):
            continue
        label = target_for(row)
        if label is None:
            continue
        weight = 1.0
        if is_saturation_controlled_interaction(row):
            content_key = (row.user_id, row.content_id)
            content_occurrences[content_key] += 1
            weight = repeat_influence_weight(content_occurrences[content_key])
        category = row.category.upper()
        for cohort_type, region, age_bucket in _cohort_keys(demo.region, demo.age_bucket):
            acc = aggregates[(cohort_type, region, age_bucket, category)]
            acc.positive += label
            acc.negative += 1 - label
            acc.weighted_positive += weight * label
            acc.weighted_negative += weight * (1 - label)
            acc.users.add(row.user_id)

    if not aggregates:
        logger.info("cohort rebuild skipped: no labeled interactions for any demographic-tagged user")
        return {"version": None, "cohortsWritten": 0, "usersConsidered": len(demographics)}

    version = _next_version(db)
    persisted = []
    for (cohort_type, region, age_bucket, category), acc in aggregates.items():
        # `total`/positive_count/negative_count remain the RAW, unweighted counts (unchanged
        # meaning -- see _Accumulator). preference_score alone uses the replay-weighted pair,
        # so a repeated-content-spamming user's excess passive evidence cannot disproportionately
        # move this cohort's preference for every other user sharing it.
        total = acc.positive + acc.negative
        weighted_total = acc.weighted_positive + acc.weighted_negative
        preference_score = acc.weighted_positive / weighted_total if weighted_total else 0.5
        persisted.append(CohortPreference(
            version=version, cohort_type=cohort_type, region=region, age_bucket=age_bucket,
            category=category, preference_score=preference_score, sample_users=len(acc.users),
            sample_interactions=total, positive_count=acc.positive, negative_count=acc.negative,
            generated_at=now,
        ))
    db.add_all(persisted)
    db.commit()
    logger.info("cohort rebuild complete version=%d cohortsWritten=%d usersConsidered=%d",
                version, len(persisted), len(demographics))
    return {"version": version, "cohortsWritten": len(persisted), "usersConsidered": len(demographics)}
