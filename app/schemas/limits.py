"""Shared default/min/max bounds for request limits, centralized so each domain's actual
numeric bound is defined once (not duplicated per schema) while still letting different
domains keep different, independently-appropriate maximums -- e.g. the experiment listing
endpoint's 200 has nothing to do with a reasonable VIDEO recommendation page size, and
forcing them to share one constant would be wrong, not DRY.
"""

RECOMMENDATION_LIMIT_DEFAULT = 10
RECOMMENDATION_LIMIT_MIN = 1
RECOMMENDATION_LIMIT_MAX = 100

CANDIDATE_LIMIT_DEFAULT = 50
CANDIDATE_LIMIT_MIN = 1
CANDIDATE_LIMIT_MAX = 100

LIVE_CANDIDATE_LIMIT_DEFAULT = 30
LIVE_CANDIDATE_LIMIT_MIN = 1
LIVE_CANDIDATE_LIMIT_MAX = 100

LIVE_RECOMMENDATION_LIMIT_DEFAULT = 10
LIVE_RECOMMENDATION_LIMIT_MIN = 1
LIVE_RECOMMENDATION_LIMIT_MAX = 100

EXPERIMENT_LIMIT_DEFAULT = 50
EXPERIMENT_LIMIT_MIN = 1
EXPERIMENT_LIMIT_MAX = 200
EXPERIMENT_OFFSET_MIN = 0
EXPERIMENT_OFFSET_MAX = 10_000  # bounds pathological scans of a very large report history
