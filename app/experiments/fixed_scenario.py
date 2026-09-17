"""The fixed prediction scenario(s) shared, byte-for-byte, across every comparative
experiment (spec: "Fixed Prediction Scenario" -- same userId, user context, candidate IDs,
categories, candidate metadata, candidate order, and recommendation limit). Defined exactly
once and imported everywhere it's used (comparative seeding/run scripts, tests) so identity
across experiments is provable by construction, not by independently-typed copies drifting
apart.

`FIXED_TEST_USER_ID` deliberately has zero interactions seeded in *any* experiment's
database. That is what makes "same user context" trivially and provably true across
otherwise-completely-separate isolated databases: a cold-start user's point-in-time feature
history (`app.ml.dataset_builder.FeatureHistory`) is empty by construction in every one of
them, so the fixed candidates are scored using only what each dataset's trained model learned
about `category` and `content_popularity_score` -- exactly the thing this experiment is
trying to isolate and compare, uncontaminated by any per-database synthetic user history.

Two scenarios, kept strictly separate (v2 -- see README "Comparative Experiments"):

- `PRIMARY_FIXED_CANDIDATES`: one candidate per supported category, with *identical*
  popularity, age, `creatorFollowed=False`, and `alreadySeen=False` across every candidate --
  the only thing that varies is `category`. This is the *only* scenario allowed to feed the
  ML-learning verdict (raw model probability comparisons), because it removes every business-
  rule/metadata confound (a followed creator, a higher-popularity candidate, etc.) that would
  otherwise entangle "the algorithm learned about this category" with "this candidate simply
  had a metadata advantage."
- `FIXED_CANDIDATES` (module-level; also exposed as `SECONDARY_FIXED_CANDIDATES`): the
  original, deliberately asymmetric list -- varied popularity/age, one `creatorFollowed=True`
  candidate, one `alreadySeen=True` candidate -- kept unchanged for v1 compatibility and
  reused, under its new "secondary" name, to exercise the business-reranking logic (seen
  penalty, diversity decay, follow boost) in isolation. Its results must never be mixed into
  the primary ML-learning conclusion.
"""
from __future__ import annotations

from app.experiments.definitions import DEFAULT_CATEGORIES

FIXED_TEST_USER_ID = "comparative-fixed-test-user"
FIXED_RECOMMENDATION_LIMIT = 10

# Order matters -- this is "the same candidate order before ranking" the spec requires, and
# must never be reordered/regenerated between runs.
FIXED_CANDIDATES: list[dict] = [
    {"contentId": "fixed-cand-sport-1", "creatorId": "fixed-creator-sport-a", "category": "SPORT",
     "contentPopularityScore": 0.82, "contentAgeHours": 4, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-sport-2", "creatorId": "fixed-creator-sport-b", "category": "SPORT",
     "contentPopularityScore": 0.55, "contentAgeHours": 96, "creatorFollowed": False, "alreadySeen": True},
    {"contentId": "fixed-cand-comedy-1", "creatorId": "fixed-creator-comedy-a", "category": "COMEDY",
     "contentPopularityScore": 0.78, "contentAgeHours": 6, "creatorFollowed": True, "alreadySeen": False},
    {"contentId": "fixed-cand-comedy-2", "creatorId": "fixed-creator-comedy-b", "category": "COMEDY",
     "contentPopularityScore": 0.48, "contentAgeHours": 120, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-music-1", "creatorId": "fixed-creator-music-a", "category": "MUSIC",
     "contentPopularityScore": 0.80, "contentAgeHours": 3, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-music-2", "creatorId": "fixed-creator-music-b", "category": "MUSIC",
     "contentPopularityScore": 0.51, "contentAgeHours": 72, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-tech-1", "creatorId": "fixed-creator-tech-a", "category": "TECH",
     "contentPopularityScore": 0.70, "contentAgeHours": 12, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-food-1", "creatorId": "fixed-creator-food-a", "category": "FOOD",
     "contentPopularityScore": 0.65, "contentAgeHours": 20, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-gaming-1", "creatorId": "fixed-creator-gaming-a", "category": "GAMING",
     "contentPopularityScore": 0.73, "contentAgeHours": 8, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-travel-1", "creatorId": "fixed-creator-travel-a", "category": "TRAVEL",
     "contentPopularityScore": 0.60, "contentAgeHours": 30, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-fitness-1", "creatorId": "fixed-creator-fitness-a", "category": "FITNESS",
     "contentPopularityScore": 0.58, "contentAgeHours": 40, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-news-1", "creatorId": "fixed-creator-news-a", "category": "NEWS",
     "contentPopularityScore": 0.45, "contentAgeHours": 2, "creatorFollowed": False, "alreadySeen": False},
    {"contentId": "fixed-cand-fashion-1", "creatorId": "fixed-creator-fashion-a", "category": "FASHION",
     "contentPopularityScore": 0.52, "contentAgeHours": 50, "creatorFollowed": False, "alreadySeen": False},
]

FIXED_CANDIDATE_CATEGORIES = sorted({candidate["category"] for candidate in FIXED_CANDIDATES})


def fixed_recommendation_request() -> dict:
    """The exact JSON body POSTed to `/api/v1/recommendation-ml-service/recommendations` for the *secondary* (business-
    reranking) scenario -- a fresh dict each call so no caller can mutate the shared
    module-level list/constants by accident. Kept under its original v1 name for backward
    compatibility; see `secondary_fixed_recommendation_request` for the version-explicit alias."""
    return {
        "userId": FIXED_TEST_USER_ID,
        "limit": FIXED_RECOMMENDATION_LIMIT,
        "candidates": [dict(candidate) for candidate in FIXED_CANDIDATES],
    }


# --- Secondary scenario: version-explicit names (same data/function as above -- v1's
# FIXED_CANDIDATES/FIXED_RECOMMENDATION_LIMIT/fixed_recommendation_request are the single
# source of truth; these are aliases, not copies, so there is nothing to drift out of sync). ---
SECONDARY_FIXED_CANDIDATES = FIXED_CANDIDATES
SECONDARY_RECOMMENDATION_LIMIT = FIXED_RECOMMENDATION_LIMIT
SECONDARY_CANDIDATE_CATEGORIES = FIXED_CANDIDATE_CATEGORIES
secondary_fixed_recommendation_request = fixed_recommendation_request


# --- Primary scenario: symmetric across categories, the only scenario allowed to feed the
# ML-learning verdict. Every field except `category`/`contentId`/`creatorId` is identical. ---
PRIMARY_CANDIDATE_POPULARITY_SCORE = 0.65
PRIMARY_CANDIDATE_AGE_HOURS = 24

PRIMARY_FIXED_CANDIDATES: list[dict] = [
    {
        "contentId": f"fixed-cand-primary-{category.lower()}",
        "creatorId": f"fixed-creator-primary-{category.lower()}",
        "category": category,
        "contentPopularityScore": PRIMARY_CANDIDATE_POPULARITY_SCORE,
        "contentAgeHours": PRIMARY_CANDIDATE_AGE_HOURS,
        "creatorFollowed": False,
        "alreadySeen": False,
    }
    for category in DEFAULT_CATEGORIES
]

# The limit equals the candidate count, so the "Top 10" *is* every primary candidate ranked --
# no Top-K truncation to worry about when comparing categories that don't make a cut.
PRIMARY_RECOMMENDATION_LIMIT = len(PRIMARY_FIXED_CANDIDATES)
PRIMARY_CANDIDATE_CATEGORIES = sorted({candidate["category"] for candidate in PRIMARY_FIXED_CANDIDATES})


def primary_fixed_recommendation_request() -> dict:
    """The exact JSON body POSTed to `/api/v1/recommendation-ml-service/recommendations` for the *primary* (symmetric,
    category-learning) scenario -- a fresh dict each call so no caller can mutate the shared
    module-level list/constants by accident."""
    return {
        "userId": FIXED_TEST_USER_ID,
        "limit": PRIMARY_RECOMMENDATION_LIMIT,
        "candidates": [dict(candidate) for candidate in PRIMARY_FIXED_CANDIDATES],
    }
