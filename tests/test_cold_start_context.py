"""Finalization spec Decision 11: cold-start language/region compatibility ONLY, never a
demographic stereotype. app.ml.reranker.cold_start_context_relevance/rerank tested directly
(no HTTP, no training), mirroring tests/test_search_intent.py's pattern."""
from types import SimpleNamespace

from app.ml.reranker import (
    COLD_START_CONTEXT_BOOST_MAX,
    cold_start_context_relevance,
    rerank,
)
from app.schemas.recommendation_schemas import UserContext


def candidate(content_id, category, creator="creator-1", *, popularity=0.5, language=None, regions=None):
    return SimpleNamespace(
        content_id=content_id, category=category, creator_id=creator, content_popularity_score=popularity,
        title=None, hashtags=[], topics=[], entities=[], subgenres=[],
        candidate_source=None, social_context=None, language=language, regions=regions or [],
    )


def test_zero_effect_when_not_cold_start():
    c = candidate("c1", "MUSIC", language="en", regions=["US"])
    ctx = UserContext.model_validate({"language": "en", "region": "US"})
    assert cold_start_context_relevance(c, ctx, cold_start=False) == 0.0


def test_zero_effect_without_user_context():
    c = candidate("c1", "MUSIC", language="en", regions=["US"])
    assert cold_start_context_relevance(c, None, cold_start=True) == 0.0


def test_zero_effect_when_candidate_has_no_language_or_region_metadata():
    """Honesty requirement: a candidate that never mentions language/region must be a pure
    no-op even for a cold-start user with a fully populated userContext -- never a fabricated
    effect."""
    c = candidate("c1", "MUSIC")  # no language, no regions
    ctx = UserContext.model_validate({"age": 21, "language": "en", "region": "US"})
    assert cold_start_context_relevance(c, ctx, cold_start=True) == 0.0


def test_full_match_is_maximal():
    c = candidate("c1", "MUSIC", language="en", regions=["US", "CA"])
    ctx = UserContext.model_validate({"language": "en", "region": "US"})
    assert cold_start_context_relevance(c, ctx, cold_start=True) == 1.0


def test_partial_mismatch_is_between_zero_and_one():
    c = candidate("c1", "MUSIC", language="fr", regions=["US"])
    ctx = UserContext.model_validate({"language": "en", "region": "US"})
    relevance = cold_start_context_relevance(c, ctx, cold_start=True)
    assert 0.0 < relevance < 1.0


def test_age_never_influences_relevance_regardless_of_value():
    """Mandatory: no demographic stereotype. Two userContexts differing ONLY in age must
    produce identical relevance for the same candidate."""
    c = candidate("c1", "GAMING", language="en", regions=["US"])
    young = UserContext.model_validate({"age": 16, "language": "en", "region": "US"})
    old = UserContext.model_validate({"age": 70, "language": "en", "region": "US"})
    assert cold_start_context_relevance(c, young, cold_start=True) == cold_start_context_relevance(c, old, cold_start=True)


def test_age_alone_with_no_language_or_region_never_produces_a_category_bias():
    """No age->category mapping exists anywhere: a candidate in ANY category, matched against
    a userContext carrying only age (no language/region), must score zero relevance."""
    for category in ("GAMING", "NEWS", "SPORT", "MUSIC", "FASHION"):
        c = candidate("c1", category)
        ctx = UserContext.model_validate({"age": 21})
        assert cold_start_context_relevance(c, ctx, cold_start=True) == 0.0, f"age-only context leaked into {category} relevance"


def test_cold_start_boost_is_bounded_and_does_not_override_personalization():
    """A strongly personalized candidate must still win over a merely language/region-matched
    one -- cold-start context is a tiny nudge among popularity/freshness/exploration, not a
    dominant signal (this scenario uses cold_start=True structurally to isolate the boost;
    the honesty point is that COLD_START_CONTEXT_BOOST_MAX itself is small). No cohort_profile
    is supplied here (defaults to None), so only cold_start_context_relevance's own bound is
    in play -- cohort stacking is covered separately in tests/test_cold_start_personalization.py."""
    strong = candidate("strong", "SPORT", popularity=0.9)
    matched = candidate("matched", "MUSIC", popularity=0.2, language="en", regions=["US"])
    scored = [
        {"candidate": strong, "features": {"already_seen": 0}, "model_score": 0.9, "reason": "POPULAR_CONTENT"},
        {"candidate": matched, "features": {"already_seen": 0}, "model_score": 0.2, "reason": "EXPLORATION"},
    ]
    ctx = UserContext.model_validate({"language": "en", "region": "US"})
    chosen = rerank([dict(i) for i in scored], 2, user_context=ctx, cold_start=True)
    assert chosen[0]["candidate"].content_id == "strong"
    boosted = next(item["adjusted_score"] for item in chosen if item["candidate"].content_id == "matched")
    max_multiplier = 1.0 + COLD_START_CONTEXT_BOOST_MAX
    assert boosted <= 0.2 * max_multiplier * 1.1  # small logit-space slack


def test_rerank_without_cold_start_or_user_context_args_is_unaffected():
    """Regression safety: existing callers that don't pass user_context/cold_start at all
    (the default False/None) get byte-identical behavior to before this parameter existed."""
    c1 = candidate("c1", "MUSIC", language="en", regions=["US"])
    scored = [{"candidate": c1, "features": {"already_seen": 0}, "model_score": 0.5, "reason": "EXPLORATION"}]
    default_call = rerank([dict(i) for i in scored], 1)
    explicit_false = rerank([dict(i) for i in scored], 1, user_context=None, cold_start=False)
    assert default_call[0]["adjusted_score"] == explicit_false[0]["adjusted_score"]
