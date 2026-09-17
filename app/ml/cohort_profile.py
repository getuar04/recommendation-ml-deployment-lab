"""Pure data shape for an already-resolved cohort cold-start preference profile.

This module exists ONLY so `app.ml.reranker` (explicitly "no I/O, no database access" -- see
its own module docstring) can type/document what it receives from
`app.services.cohort_preference_provider` without importing anything from the DB/service
layer. The actual resolution (querying `recommendation_cohort_preferences`, applying the
REGION+AGE/REGION/AGE/GLOBAL fallback hierarchy, minimum-evidence checks, and shrinkage) lives
entirely in that service module; `app.ml.reranker.regional_cohort_relevance` only ever reads
the already-computed fields below, exactly as it already treats `candidate`/`user_context` as
opaque duck-typed inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class CohortPreferenceProfile:
    """`source` is the resolved cohort level: "REGION_AGE" | "REGION" | "AGE" | "GLOBAL" |
    "NONE" (no evidence anywhere in the fallback chain, including GLOBAL). `preferences` maps
    an UPPER-CASED category name to a bounded [0, 1] preference score, already shrunk toward
    the broader/global prior (see cohort_preference_provider.DatabaseCohortPreferenceProvider's
    own docstring for the exact formula) -- reranker never re-derives or re-blends anything,
    it only reads a category key back out. `reliable` is the single gate reranker checks before
    applying ANY influence: False means "insufficient evidence was found anywhere in the
    fallback chain", and every consumer must treat that identically to "no profile at all".
    """

    source: str
    region: str | None
    age_bucket: str | None
    sample_users: int
    sample_interactions: int
    preferences: dict[str, float] = field(default_factory=dict)
    generated_at: datetime | None = None
    version: int | None = None
    reliable: bool = False


# Canonical "nothing resolved" value -- returned by the provider whenever the fallback chain
# (including GLOBAL) has no cohort data at all, or cohort preferences are disabled/unconfigured.
# Reranker's own `regional_cohort_relevance` treats this identically to `cohort_profile=None`.
NULL_COHORT_PROFILE = CohortPreferenceProfile(
    source="NONE", region=None, age_bucket=None, sample_users=0, sample_interactions=0,
    preferences={}, generated_at=None, version=None, reliable=False,
)
