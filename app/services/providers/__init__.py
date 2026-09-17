"""Data-adapter boundary for dual-mode (LOCAL/REAL) recommendation input -- see
app.services.providers.user_behavior_provider and app.services.providers.candidate_provider.
Both providers normalize their respective real-service or local input into the SAME schemas
app.services.recommendation_service.recommend() already scores; the difference between
LOCAL and REAL lives entirely here, never inside the ranking core itself.
"""
