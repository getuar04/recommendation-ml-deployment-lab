"""Serving-side read boundary for the VIDEO cohort cold-start preference system.

Replaces the removed `LOCAL_POC_REGIONAL_CATEGORY_COHORT_PRIOR` hand-authored table
(app.ml.reranker): every category preference score returned here is derived from persisted,
versioned aggregates built by `app.services.cohort_aggregation_service.rebuild_cohort_
preferences` from real historical `interactions` data -- never a hardcoded mapping, and never
recomputed from raw interaction rows on this (per-request) path. `resolve()` issues a small,
indexed read against `recommendation_cohort_preferences` (bounded to the single latest
`version`) -- it never scans the `interactions` table.

Fallback hierarchy (spec §G): REGION_AGE -> REGION -> AGE -> GLOBAL, stopping at the first
level that clears the configured minimum-evidence bar (COHORT_MIN_USERS/COHORT_MIN_
INTERACTIONS). A level that exists but is too small to trust is skipped entirely, exactly like
a level that doesn't exist -- both fall through to the next broader level. If nothing in the
chain (including GLOBAL) clears the bar, `NULL_COHORT_PROFILE` is returned: zero cohort
influence, never a guess, never a crash.

Shrinkage (spec §H): once a non-GLOBAL level is selected, its raw per-category scores are
blended toward the GLOBAL row's own score (when GLOBAL data exists) using the sample-size-
weighted formula documented on `_shrink_toward_global` below -- a small cohort that JUST clears
the minimum-evidence bar still can't produce an extreme, low-confidence score.

app.ml.reranker never imports this module (kept out of its "no I/O" boundary) -- callers
(app.services.recommendation_service) resolve a profile here and pass the plain
`app.ml.cohort_profile.CohortPreferenceProfile` value into `rerank()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import (
    COHORT_MIN_INTERACTIONS,
    COHORT_MIN_USERS,
    COHORT_SHRINKAGE_K,
)
from app.core.logging import logger
from app.db.models import CohortPreference, UserDemographicContext
from app.ml.cohort_profile import NULL_COHORT_PROFILE, CohortPreferenceProfile

__all__ = [
    "CohortPreferenceProvider", "DatabaseCohortPreferenceProvider",
    "provider", "record_demographic_context",
]


@dataclass(frozen=True)
class _CategoryEvidence:
    """One category's own evidence within a cohort level -- sample_users/sample_interactions
    are genuinely PER-CATEGORY (app.services.cohort_aggregation_service accumulates a separate
    user-set/interaction-count per (level, category), never one shared count for the whole
    level), so a level with strong SPORT evidence and almost none for MUSIC must not let SPORT's
    numbers stand in for MUSIC's."""

    sample_users: int
    sample_interactions: int
    score: float


@dataclass(frozen=True)
class _CohortRows:
    cohort_type: str
    region: str | None
    age_bucket: str | None
    by_category: dict[str, _CategoryEvidence]

    @property
    def sample_users(self) -> int:
        """Order-independent, deterministic LEVEL-level proxy: the largest per-category
        sample_users observed anywhere in this level -- never an arbitrary "whichever category
        row the database happened to return first" pick (the previous `rows[0]` approach), and
        never dependent on row order. This only gates whether the level is worth considering AT
        ALL; `_shrink_toward_global` below is what protects an individual weakly-evidenced
        category once the level is accepted."""
        return max((evidence.sample_users for evidence in self.by_category.values()), default=0)

    @property
    def sample_interactions(self) -> int:
        """Total LEVEL-level interaction evidence: the SUM of every category's own
        sample_interactions, not max(). Each interaction belongs to exactly one category row
        (app.services.cohort_aggregation_service accumulates positive_count+negative_count
        per (level, category), never a shared total), so SPORT=70 + MUSIC=60 + GAMING=70 is
        genuinely 200 interactions of evidence for this level -- reporting max()=70 here would
        undercount real evidence and could incorrectly fail COHORT_MIN_INTERACTIONS for a level
        that actually has enough data, forcing an unnecessary fallback. Unlike `sample_users`
        (above, still max()-based -- the same user can appear in multiple categories, so
        summing users WOULD double-count), summing is safe here: interactions, unlike users,
        cannot double-count across categories -- a single interaction event has exactly one
        category."""
        return sum(evidence.sample_interactions for evidence in self.by_category.values())


def _meets_minimum_evidence(rows: _CohortRows) -> bool:
    return rows.sample_users >= COHORT_MIN_USERS and rows.sample_interactions >= COHORT_MIN_INTERACTIONS


def _fallback_levels(region: str | None, age_bucket: str | None) -> list[tuple[str, str | None, str | None]]:
    """Deterministic REGION_AGE -> REGION -> AGE -> GLOBAL order (spec §G), skipping any level
    that isn't even applicable given what was supplied (e.g. no REGION-only step when region
    is None) -- GLOBAL is always attempted last regardless of what was supplied, since it is
    the final safety net before falling back to non-cohort cold-start signals entirely."""
    levels: list[tuple[str, str | None, str | None]] = []
    if region and age_bucket:
        levels.append(("REGION_AGE", region, age_bucket))
    if region:
        levels.append(("REGION", region, None))
    if age_bucket:
        levels.append(("AGE", None, age_bucket))
    levels.append(("GLOBAL", None, None))
    return levels


def _latest_version(db: Session) -> int | None:
    return db.scalar(select(func.max(CohortPreference.version)))


def _rows_for(db: Session, version: int, cohort_type: str, region: str | None, age_bucket: str | None) -> _CohortRows | None:
    query = select(CohortPreference).where(
        CohortPreference.version == version,
        CohortPreference.cohort_type == cohort_type,
        CohortPreference.region == region,
        CohortPreference.age_bucket == age_bucket,
    )
    rows = db.scalars(query).all()
    if not rows:
        return None
    return _CohortRows(
        cohort_type=cohort_type, region=region, age_bucket=age_bucket,
        by_category={
            row.category.upper(): _CategoryEvidence(
                sample_users=row.sample_users, sample_interactions=row.sample_interactions,
                score=row.preference_score,
            )
            for row in rows
        },
    )


def _shrink_toward_global(rows: _CohortRows, global_rows: _CohortRows | None) -> dict[str, float]:
    """James-Stein-style linear shrinkage toward the broader/global prior, computed
    INDEPENDENTLY per category:

        shrunk[category] = w * cohort_score + (1 - w) * global_score,  where  w = n / (n + k)

    `n` = THAT category's own sample_interactions (never a level-wide/borrowed count from a
    different category), `k` = COHORT_SHRINKAGE_K (the sample size at which the cohort-specific
    evidence and the global prior are weighted equally). n -> 0 collapses to the global score
    outright; n >> k trusts the cohort-specific score almost entirely. A category present in the
    cohort but absent from the global aggregate (should not happen once GLOBAL has been built
    from the same categories, but handled defensively) blends toward a neutral 0.5 instead of
    guessing. When no GLOBAL data exists at all (only possible for a GLOBAL-level result itself,
    or a not-yet-rebuilt table), the raw cohort scores are returned unshrunk -- there is nothing
    broader to blend toward."""
    if global_rows is None:
        return {category: evidence.score for category, evidence in rows.by_category.items()}
    shrunk: dict[str, float] = {}
    for category, evidence in rows.by_category.items():
        global_evidence = global_rows.by_category.get(category)
        global_score = global_evidence.score if global_evidence is not None else 0.5
        weight = evidence.sample_interactions / (evidence.sample_interactions + COHORT_SHRINKAGE_K)
        shrunk[category] = weight * evidence.score + (1 - weight) * global_score
    return shrunk


class CohortPreferenceProvider(Protocol):
    """Narrow read boundary app.services.recommendation_service depends on. A future
    provider (e.g. one backed by a real analytics warehouse instead of this project's own
    `recommendation_cohort_preferences` table) implements this identical Protocol -- callers
    never change."""

    def resolve(self, db: Session, *, region: str | None, age_bucket: str | None) -> CohortPreferenceProfile: ...


class DatabaseCohortPreferenceProvider:
    """Reads `recommendation_cohort_preferences` (this project's own PostgreSQL/SQLite-backed
    table, populated by app.services.cohort_aggregation_service). See module docstring for the
    full fallback/shrinkage contract."""

    def resolve(self, db: Session, *, region: str | None, age_bucket: str | None) -> CohortPreferenceProfile:
        if region is None and age_bucket is None:
            return NULL_COHORT_PROFILE
        version = _latest_version(db)
        if version is None:
            return NULL_COHORT_PROFILE

        global_rows = _rows_for(db, version, "GLOBAL", None, None)
        for cohort_type, cohort_region, cohort_age_bucket in _fallback_levels(region, age_bucket):
            if cohort_type == "GLOBAL":
                rows = global_rows
            else:
                rows = _rows_for(db, version, cohort_type, cohort_region, cohort_age_bucket)
            if rows is None or not _meets_minimum_evidence(rows):
                continue
            shrunk = (
                {category: evidence.score for category, evidence in rows.by_category.items()}
                if cohort_type == "GLOBAL" else _shrink_toward_global(rows, global_rows)
            )
            return CohortPreferenceProfile(
                source=cohort_type, region=cohort_region, age_bucket=cohort_age_bucket,
                sample_users=rows.sample_users, sample_interactions=rows.sample_interactions,
                preferences=shrunk, generated_at=None, version=version, reliable=True,
            )
        return NULL_COHORT_PROFILE


provider: CohortPreferenceProvider = DatabaseCohortPreferenceProvider()


def record_demographic_context(db: Session, user_id: str, *, region: str | None, age_bucket: str | None) -> None:
    """Opportunistic, best-effort partial upsert of the calling user's most-recently-observed
    region/age bucket -- the only write path that ever populates `user_demographic_context`,
    the join key app.services.cohort_aggregation_service uses to attribute historical
    interactions to a cohort. A no-op when neither field is supplied (nothing to record).

    Partial-update semantics: `region`/`age_bucket` here mean "supplied on THIS request", not
    "the user's complete demographic state" -- a field the caller did not supply is preserved
    as whatever was already stored (a new row simply leaves an unsupplied field NULL), never
    overwritten with None. This is what stops a request that only carries a region (age
    omitted, e.g. the client didn't ask, or the user is below the cohort age floor) from
    silently erasing a previously recorded age bucket, and vice versa -- "missing" is never
    "delete the previous known value".

    Write-skip: if the merged result is byte-identical to the already-stored row (both fields
    unchanged), this returns without an UPDATE or commit at all -- a recommendation request
    that repeats the same region/age it already recorded must not write to the database every
    single time.

    Failure here must NEVER break a recommendation response (this is a side-effect write on
    the serving hot path, not something the caller is waiting on) -- any exception is caught,
    rolled back, and logged, matching this codebase's existing defensive posture for other
    best-effort writes (see app.services.recommendation_service's ranking-snapshot capture)."""
    if region is None and age_bucket is None:
        return
    try:
        row = db.get(UserDemographicContext, user_id)
        if row is None:
            # Brand-new row: an unsupplied field simply has nothing to preserve yet, so it
            # stays NULL (never fabricated).
            row = UserDemographicContext(
                user_id=user_id, region=region, age_bucket=age_bucket,
                updated_at=datetime.now(timezone.utc),
            )
            db.add(row)
            db.commit()
            return

        merged_region = region if region is not None else row.region
        merged_age_bucket = age_bucket if age_bucket is not None else row.age_bucket
        if merged_region == row.region and merged_age_bucket == row.age_bucket:
            return  # Unchanged -- nothing to persist.

        row.region = merged_region
        row.age_bucket = merged_age_bucket
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:  # noqa: BLE001 -- best-effort side write, must never break serving
        # `db` can legitimately be None (offline `recommend(None, request)` callers reaching
        # this path with a userContext.region/age set) -- nothing to roll back in that case.
        if db is not None:
            db.rollback()
        logger.info("cohort demographic-context write failed userId=%s failureType=%s", user_id, type(exc).__name__)
