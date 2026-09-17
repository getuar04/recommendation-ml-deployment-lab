"""Single source of truth for cold-start cohort CONTEXT derivation: region normalization and
age->age-bucket mapping. VIDEO cold-start cohort system only (see
app.services.cohort_preference_provider) -- never used for the 34-feature ML model contract.

Centralizing this here (instead of duplicating age-bucket ranges across
app.ml.reranker/app.services.cohort_preference_provider/app.services.cohort_aggregation_service/
tests) is what "age bucket logic must exist in ONE maintainable location" means in practice: a
bucket boundary changes in exactly one place, and every caller (reranker resolution, cohort
aggregation, tests) reads back the same mapping.

Privacy/data-minimization: `age_bucket_for` is the ONLY sanctioned way raw age is allowed to
flow anywhere near cohort storage -- callers should call this as early as practical (request
validation time) and pass only the resulting bucket string onward, never the raw integer. No
function here persists or logs raw age.
"""
from __future__ import annotations

# Ordered, non-overlapping, inclusive [low, high] ranges; high=None means "and above". Reviewed
# against app.schemas.recommendation_schemas.UserContext.age's existing validation bound
# (0 <= age <= 120) -- MIN_COHORT_AGE below is intentionally stricter than that schema bound
# (a real product constraint: under-13 users are excluded from demographic cohort inference
# entirely, not bucketed into "13-17"), so a schema-valid age can still legitimately resolve to
# "no bucket" here.
AGE_BUCKETS: tuple[tuple[int, int | None, str], ...] = (
    (13, 17, "13-17"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
    (65, None, "65+"),
)

MIN_COHORT_AGE = AGE_BUCKETS[0][0]

# Every bucket label, in order -- useful for tests/aggregation that need to enumerate buckets
# without re-deriving them from AGE_BUCKETS' tuple shape.
AGE_BUCKET_LABELS: tuple[str, ...] = tuple(label for _, _, label in AGE_BUCKETS)


def age_bucket_for(age: int | None) -> str | None:
    """Maps a raw age to its cohort bucket label, or None when no bucket applies.

    None is returned (never a fabricated bucket, never a raised exception) for: a missing age
    (`age is None`), an age below MIN_COHORT_AGE, or a negative age -- the schema layer already
    rejects an age outside [0, 120] before this is ever called, but this function stays total
    and defensive on its own regardless of what validation happened upstream, so a caller can
    never crash the recommendation path merely by asking for a bucket. "No bucket" is exactly
    the same as "no age supplied" downstream: both mean the AGE-cohort dimension is simply
    unavailable for this request, never an error condition.
    """
    if age is None or age < MIN_COHORT_AGE:
        return None
    for low, high, label in AGE_BUCKETS:
        if age >= low and (high is None or age <= high):
            return label
    return None


def normalize_region(value: str | None) -> str | None:
    """Canonical region identifier: stripped, upper-cased, empty-to-None. Byte-identical to
    app.schemas.recommendation_schemas.UserContext's own pre-existing `_normalize_region`
    validator (that validator now delegates here) -- one normalization rule for the whole
    codebase, not a second copy that could quietly drift (e.g. accepting "xk" in one place and
    only ever comparing against "XK" in another). Never attempts geographic intelligence
    (country-name resolution, aliasing "Kosovo" to "XK", etc.) -- an unrecognized or free-text
    region simply normalizes to its upper-cased form and, if no cohort evidence exists for it,
    falls through the cohort fallback hierarchy safely (see cohort_preference_provider)."""
    if not value:
        return None
    normalized = value.strip().upper()
    return normalized or None
