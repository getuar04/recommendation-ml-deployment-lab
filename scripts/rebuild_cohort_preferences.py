"""Callable operation (spec §O -- no scheduler required "for now") to rebuild VIDEO cohort
cold-start preferences from the current `interactions` + `user_demographic_context` tables.

Usage:
    python -m scripts.rebuild_cohort_preferences

Safe to rerun at any time: each run persists a brand-new `version` of
`recommendation_cohort_preferences` (app.services.cohort_aggregation_service.
rebuild_cohort_preferences) without touching or deleting any prior version -- serving
(app.services.cohort_preference_provider) always reads the single latest version, so a rebuild
takes effect immediately for new requests with no code change and no restart.
"""
from app.db.database import SessionLocal
from app.services.cohort_aggregation_service import rebuild_cohort_preferences


def main() -> None:
    db = SessionLocal()
    try:
        result = rebuild_cohort_preferences(db)
        print(result)
    finally:
        db.close()


if __name__ == "__main__":
    main()
