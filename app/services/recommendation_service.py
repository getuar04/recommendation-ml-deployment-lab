"""ML scoring plus deliberately light, documented business reranking.

Reranking itself (and the heuristic explanation reason codes) live in the pure,
I/O-free `app.ml.reranker` module so the exact same logic can be reused for
offline pre-/post-rerank evaluation (`app.ml.reranking_eval`) without drifting
from what production actually serves.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any

from app.core.cohort_context import age_bucket_for
from app.core.config import (
    COHORT_PREFERENCES_ENABLED,
    MODEL_PATH,
    PERSONALISED_RECOMMENDATION_MIN_INTERACTIONS,
    RECOMMENDATION_DATA_MODE,
    TWO_TOWER_RETRIEVAL_ENABLED,
    TWO_TOWER_RETRIEVAL_K,
)
from app.core.logging import logger
from app.ml import model_cache, model_store
from app.ml.dataset_builder import FEATURES
from app.ml.predictor import probabilities
from app.ml.ranking_snapshot_capture import build_ranking_decision_snapshots
from app.ml.ranking_snapshot_repository import (
    NULL_RANKING_SNAPSHOT_REPOSITORY,
    RankingSnapshotRepository,
)
from app.ml.reranker import explanation, rerank
from app.services.cohort_preference_provider import (
    provider as cohort_preference_provider,
)
from app.services.cohort_preference_provider import (
    record_demographic_context,
)
from app.services.providers import candidate_provider, user_behavior_provider
from app.services.providers.candidate_provider import CandidateSourceUnavailable
from app.services.providers.user_behavior_provider import UserBehaviorSourceUnavailable
from app.services.session_intent_provider import provider as session_intent_provider
from app.services.user_context_provider import load_user_context

__all__ = [
    "CandidateSourceUnavailable", "ModelArtifactInvalid", "ModelNotTrained",
    "UserBehaviorSourceUnavailable", "explanation", "recommend", "rerank",
]


class ModelNotTrained(Exception):
    pass


def strategy_for(interaction_count: float) -> str:
    """Spec §27: configurable, deterministic strategy tiers by prior interaction count.
    The model still scores every request (the artifact contract is always enforced);
    the strategy labels how much personalisation the history can actually support.

    `interaction_count` accepts a float (widened for
    app.services.providers.user_behavior_provider.UserBehaviorState.evidence_count(), a
    replay-saturation-discounted "effective" count -- see recommend()'s call site) as well as
    the plain raw int every pre-existing caller still passes; the three-tier comparison logic
    itself is completely unchanged."""
    if interaction_count == 0:
        return "COLD_START"
    if interaction_count < PERSONALISED_RECOMMENDATION_MIN_INTERACTIONS:
        return "HYBRID"
    return "PERSONALISED_ML"


class ModelArtifactInvalid(Exception):
    """The model artifact exists but is corrupted, incompatible, or transiently unstable
    (retraining in progress); `reason` is one of "INCOMPATIBLE" / "CORRUPTED" / "BUSY"."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def _try_two_tower_candidates(db, history, user_id: str, seen_content_ids: set, now: datetime):
    """Phase 3.4 guarded integration: only called when TWO_TOWER_RETRIEVAL_ENABLED is true.
    Returns a candidate list on a genuinely usable Two-Tower result, or None if Two-Tower
    should be skipped for this request (missing/corrupt artifact, any unexpected runtime
    error -- including `app.services.two_tower_shadow_service`/`torch` being unimportable --
    or zero candidates produced). The caller (`recommend()`) always falls back to
    `request.candidates` when this returns None -- Two-Tower must never break this endpoint,
    never fabricate candidates, and never train anything at request time. The import below is
    intentionally LAZY (function-local): `app.services.recommendation_service` itself never
    gains a hard `torch`/two-tower dependency, so this path costs nothing at all when the flag
    is off (the default). Logs a concise, structured outcome only -- never feature vectors or
    embeddings."""
    started = perf_counter()
    try:
        from app.services.two_tower_shadow_service import two_tower_retrieval_candidates
        candidates = two_tower_retrieval_candidates(
            db, history, user_id, seen_content_ids, retrieval_k=TWO_TOWER_RETRIEVAL_K, now=now,
        )
    except Exception as exc:  # noqa: BLE001 -- Two-Tower must never break this endpoint (see docstring); always falls back
        # Best-effort: a failed DB-bound statement inside two_tower_retrieval_candidates could
        # otherwise leave this request-scoped `db` session needing an explicit rollback before
        # any later use in this same request (candidate/model scoring queries, cohort
        # resolution, etc.) -- see the identical rationale on the session-intent/cohort-resolve
        # exception handlers below. `db` can legitimately be None (e.g. direct calls in
        # tests/test_recommendation_service_gaps.py, or an offline `recommend(None, request)`
        # caller reaching this path) -- nothing to roll back in that case.
        if db is not None:
            db.rollback()
        logger.info(
            "two_tower retrieval failed userId=%s failureType=%s error=%s fallbackUsed=true",
            user_id, type(exc).__name__, str(exc)[:200],
        )
        return None

    latency_ms = (perf_counter() - started) * 1000
    if not candidates:
        logger.info(
            "two_tower retrieval returned no candidates userId=%s retrievalK=%d retrievalLatencyMs=%.1f fallbackUsed=true",
            user_id, TWO_TOWER_RETRIEVAL_K, latency_ms,
        )
        return None

    logger.info(
        "two_tower retrieval succeeded userId=%s retrievalK=%d retrievedCount=%d retrievalLatencyMs=%.1f",
        user_id, TWO_TOWER_RETRIEVAL_K, len(candidates), latency_ms,
    )
    return candidates


def recommend(
    db, request, *, exclude_already_seen: bool = False,
    internal_correlation_id: str | None = None,
    snapshot_repository: RankingSnapshotRepository = NULL_RANKING_SNAPSHOT_REPOSITORY,
):
    """Score and rerank a VIDEO candidate request.

    The production HTTP boundary sets ``exclude_already_seen=True`` to enforce the feed
    eligibility contract.  Offline benchmark/evaluation callers intentionally retain the
    default so they can still score paired seen/unseen probes and measure the model feature
    plus defensive reranker penalty without changing the 34-feature artifact contract.

    ``internal_correlation_id``/``snapshot_repository`` (RMS real-data integration prep,
    pre-contract): opt-in, additive only -- every existing caller omits both and gets
    byte-identical behavior (no snapshot is built, `NullRankingSnapshotRepository` is never
    even asked to record anything). When a caller DOES supply a correlation id (today: never,
    from the production HTTP route -- see `app.api.recommendation_routes`'s own docstring for
    why that wiring is deliberately not enabled yet), this function becomes technically
    capable of producing an immutable `app.ml.ranking_snapshot.RankedCandidateSnapshot` for
    every ranked candidate, built from the SAME `scored`/`chosen` values already computed
    below -- never a second, parallel scoring pass. See `app.ml.ranking_snapshot_capture` for
    why `internal_correlation_id` is never treated as an authoritative training request id.
    Failures here can never break a recommendation response -- same defensive posture as
    `_try_two_tower_candidates` above.
    """
    # Validated consistently regardless of whether there turn out to be any candidates to
    # score: an incompatible/corrupted artifact must be reported the same way for an
    # empty request as for a normal one, not masked by a cheaper, permissive read.
    try:
        model, metadata = model_cache.video_cache.get(MODEL_PATH, model_store.METADATA_PATH, expected_features=FEATURES)
    except model_store.ArtifactNotFoundError as exc:
        raise ModelNotTrained() from exc
    except model_store.ArtifactIncompatibleError as exc:
        raise ModelArtifactInvalid(str(exc), reason="INCOMPATIBLE") from exc
    except model_store.ArtifactCorruptedError as exc:
        raise ModelArtifactInvalid(str(exc), reason="CORRUPTED") from exc
    except model_store.ArtifactBusyError as exc:
        raise ModelArtifactInvalid(str(exc), reason="BUSY") from exc

    # Dual-mode data-adapter boundary (app.services.providers.user_behavior_provider): an
    # explicit request.userProfile always wins outright (unchanged Phase A behavior -- the
    # production path a real Feed Service, or a REAL-mode caller that already resolved UBS
    # itself, already uses). Its absence falls back to RECOMMENDATION_DATA_MODE: REAL tries a
    # real User Behavior Service (with a configurable fallback to the local path below on
    # failure); LOCAL (and REAL's own fallback) uses this project's own `interactions` table,
    # bounded to RECOMMENDATION_HISTORY_MAX_INTERACTIONS rows -- byte-identical to the
    # pre-dual-mode legacy/demo path. Every line below this point is the SAME shared ranking
    # core regardless of which of the three sources supplied `state` -- see
    # user_behavior_provider.UserBehaviorState.
    state = user_behavior_provider.resolve(db, request)
    history = state.history
    interaction_count = state.interaction_count
    cold_start = state.cold_start
    seen_content_ids = state.seen_content_ids
    now = datetime.now(timezone.utc)

    # Phase 3.4 guarded Two-Tower integration: OFF by default (TWO_TOWER_RETRIEVAL_ENABLED).
    # When off, `candidate_source` is exactly `request.candidates` and every line below this
    # point runs byte-identical to the pre-Phase-3.4 flow -- no behavioral or performance
    # difference. When on, a successful Two-Tower retrieval REPLACES the candidate source for
    # this request (the caller-supplied `request.candidates` are simply not scored that
    # request); any Two-Tower failure falls back to `request.candidates` unchanged. Either way,
    # everything from here on (feature building, RandomForest scoring, reranking, response
    # shape) is the SAME existing code, applied to whichever candidate list was chosen --
    # never duplicated per source.
    # Dual-mode data-adapter boundary (app.services.providers.candidate_provider): LOCAL mode
    # (default) is unchanged -- request.candidates only. REAL mode additionally sources
    # candidates from a real Candidate Service when the caller supplied none, with a
    # controlled fallback to request.candidates (see candidate_provider.resolve's own
    # docstring for the exact priority/fallback policy). Two-Tower below may still replace
    # whichever candidate_source was chosen here -- unchanged from before this module existed.
    retrieval_mode = "CURRENT"
    # Local candidate generation must not consume `request.limit` capacity on content this
    # request is about to hard-exclude anyway (a real, runtime-proven gap: generation used to
    # fill its quota from the full active-VIDEO catalog, then already-seen exclusion below
    # could drop several of them with no backfill, silently under-filling the response even
    # when plenty of other eligible unseen content existed). Passing this request's own
    # already-resolved authoritative `seen_content_ids` through only when this caller actually
    # enforces hard exclusion (`exclude_already_seen`) keeps every offline/benchmark caller
    # that intentionally scores paired seen/unseen candidates (default False) byte-identical.
    candidate_source, candidate_source_label = candidate_provider.resolve(
        db, request, seen_content_ids=frozenset(seen_content_ids) if exclude_already_seen else None,
    )
    if TWO_TOWER_RETRIEVAL_ENABLED:
        two_tower_candidates = _try_two_tower_candidates(db, history, request.user_id, seen_content_ids, now)
        if two_tower_candidates is not None:
            candidate_source = two_tower_candidates
            retrieval_mode = "TWO_TOWER"
        else:
            retrieval_mode = "TWO_TOWER_FALLBACK"

    feature_rows: list[dict[str, Any]] = []
    valid = []
    content_ids: set[str] = set()
    # Feed eligibility contract: an explicitly seen candidate is not merely a weak scoring
    # signal -- it is ineligible for serving.  Collect IDs first so conflicting duplicate
    # representations cannot re-introduce an item through an `alreadySeen=false` copy.  The
    # model feature remains in the offline/artifact contract, and the reranker's seen penalty
    # remains as defense in depth for other/offline callers.
    #
    # Two independent sources feed this set, and EITHER one marking a contentId seen excludes
    # it -- a caller-supplied `alreadySeen=false` must never override RMS's own resolved
    # history (`state.seen_content_ids`, already computed above regardless of source:
    # REQUEST_PROFILE/UBS profile.seen_content_ids or LOCAL_DB's queried interaction rows). No
    # second DB query here -- `seen_content_ids` is already the authoritative resolved view.
    already_seen_content_ids = (
        {candidate.content_id for candidate in candidate_source if candidate.already_seen} | seen_content_ids
        if exclude_already_seen
        else set()
    )
    for candidate in candidate_source:
        # A candidate with a blank identifier is silently dropped from this batch, not a
        # reason to fail the whole request with 422 -- consistent with content_id/category
        # below and with the documented Candidate Service boundary (README "Candidate Service
        # / Recommendation Service boundary"): a single malformed entry from a large,
        # trusted-caller-supplied batch shouldn't block every other valid candidate.
        # creator_id previously had no check here at all (schema allows a blank string, and
        # this loop didn't skip it either), so a blank creatorId candidate was scored with ""
        # as a literal creator identifier -- fixed to match content_id/category's treatment.
        if (
            not candidate.content_id
            or not candidate.creator_id.strip()
            or not candidate.category.strip()
            or candidate.content_id in already_seen_content_ids
            or candidate.content_id in content_ids
        ):
            continue
        content_ids.add(candidate.content_id)
        valid.append(candidate)
        feature_rows.append(history.features(
            user_id=request.user_id,
            category=candidate.category,
            creator_id=candidate.creator_id,
            content_id=candidate.content_id,
            timestamp=now,
            content_popularity_score=candidate.content_popularity_score,
            content_created_at=now - timedelta(hours=candidate.content_age_hours),
            creator_followed=candidate.creator_followed,
            already_seen=candidate.already_seen,
            hashtags=candidate.hashtags,
            topics=candidate.topics,
            entities=candidate.entities,
            subgenres=candidate.subgenres,
            title=candidate.title,
        ))
    # Search -> session intent (finalization spec, "search -> persistent session intent"):
    # an explicit request.search_intent always wins outright (never merged/duplicated with
    # the stored one -- see app.services.session_intent_provider module docstring); its
    # absence falls back to whatever active, TTL-decayed intent the provider has for this
    # user (None if the user never searched or the search has fully decayed). Either way
    # this is the ONLY search_intent handed to rerank() below, so a search is applied once.
    # `db` is passed through so a persistent (DATABASE-mode) provider can read its shared,
    # cross-worker store -- see app.services.session_intent_provider.SessionIntentProvider.
    effective_search_intent = request.search_intent
    if effective_search_intent is None:
        effective_search_intent = session_intent_provider.get_active_intent(db, request.user_id, now=now)

    # Persisted onboarding context (app.services.user_context_provider, app.api.user_routes.
    # create_user): an explicit request.user_context always wins outright (unchanged,
    # backward-compatible -- an existing caller that still resends userContext on every
    # request keeps working byte-identically); its absence falls back to whatever context
    # this user was created with (None if they were created without one, predate this
    # feature, or don't exist at all). Every use of "user context" below this point --
    # cohort resolution and the rerank() call -- reads this single resolved value, so
    # region/age/interests/language personalization behave identically regardless of
    # whether the context came from the request body or from storage. Mirrors
    # effective_search_intent's own fallback pattern immediately above.
    effective_user_context = request.user_context
    if effective_user_context is None:
        effective_user_context = load_user_context(db, request.user_id)

    # Replay/exposure saturation (app.ml.replay_saturation_policy): strategy maturity is
    # decided from `state.evidence_count()` -- the raw `interaction_count` discounted for
    # repeated passive replay of the SAME content -- NOT the raw `interaction_count` itself,
    # so N passive repeats of ONE content cannot alone advance COLD_START -> HYBRID ->
    # PERSONALISED_ML the way N distinct/legitimate interactions do. `interactionCount` in the
    # response below is deliberately untouched (still the raw row count) -- a public,
    # pre-existing contract field, not silently redefined.
    strategy = strategy_for(state.evidence_count())

    # VIDEO cohort cold-start preference system (replaces the removed
    # LOCAL_POC_REGIONAL_CATEGORY_COHORT_PRIOR hardcoded table -- see app.ml.reranker). Two
    # independent, both best-effort/non-fatal steps, gated by COHORT_PREFERENCES_ENABLED:
    #   1. Opportunistically record this request's region/derived age bucket (never raw age --
    #      app.core.cohort_context.age_bucket_for) so future cohort rebuilds
    #      (app.services.cohort_aggregation_service) have evidence to aggregate.
    #   2. Resolve an already-computed cohort profile for THIS request's region/age bucket.
    #      Skipped entirely for PERSONALISED_ML (cohort weight is always 0 there -- see
    #      app.ml.reranker.COHORT_STRATEGY_WEIGHT) and whenever neither region nor age was
    #      supplied, since a cohort lookup would have nothing user-specific to key off.
    cohort_profile = None
    if COHORT_PREFERENCES_ENABLED and effective_user_context is not None:
        cohort_region = effective_user_context.region
        cohort_age_bucket = age_bucket_for(effective_user_context.age)
        if cohort_region or cohort_age_bucket:
            record_demographic_context(db, request.user_id, region=cohort_region, age_bucket=cohort_age_bucket)
            if strategy != "PERSONALISED_ML":
                try:
                    cohort_profile = cohort_preference_provider.resolve(
                        db, region=cohort_region, age_bucket=cohort_age_bucket,
                    )
                except Exception as exc:  # noqa: BLE001 -- cohort resolution must never break a recommendation response
                    # See app.services.session_intent_provider.DatabaseSessionIntentProvider.peek's
                    # own comment: a failed statement leaves this request-scoped `db` session
                    # needing an explicit rollback before any further use (e.g. a snapshot-
                    # capture write later in this same request). `db` can legitimately be None
                    # (offline `recommend(None, request)` callers) -- nothing to roll back then.
                    if db is not None:
                        db.rollback()
                    logger.info("cohort preference resolution failed userId=%s failureType=%s",
                                request.user_id, type(exc).__name__)
                    cohort_profile = None

    if not feature_rows:
        logger.info(
            "recommendation completed userId=%s strategy=%s modelVersion=%s count=0 retrievalMode=%s "
            "dataMode=%s userBehaviorSource=%s candidateSource=%s",
            request.user_id, strategy, metadata["modelVersion"], retrieval_mode,
            RECOMMENDATION_DATA_MODE, state.source, candidate_source_label,
        )
        return {"userId": request.user_id, "modelVersion": metadata["modelVersion"],
                "strategy": strategy, "interactionCount": interaction_count, "recommendations": []}

    model_scores = probabilities(model, feature_rows)
    scored = [
        {"candidate": candidate, "features": features, "model_score": float(score),
         "reason": explanation(features, candidate, cold_start)}
        for candidate, features, score in zip(valid, feature_rows, model_scores)
    ]
    scored.sort(key=lambda item: item["model_score"], reverse=True)
    chosen = rerank(
        scored, request.limit,
        search_intent=effective_search_intent, user_context=effective_user_context, cold_start=cold_start,
        user_id=request.user_id, cohort_profile=cohort_profile, strategy=strategy,
    )
    if internal_correlation_id:
        try:
            snapshots = build_ranking_decision_snapshots(
                internal_correlation_id=internal_correlation_id,
                ranking_timestamp=now,
                model_version=metadata["modelVersion"],
                feature_names=FEATURES,
                user_snapshot={"userId": request.user_id, "interactionCount": interaction_count, "strategy": strategy},
                pre_rerank_order=scored,
                final_order=chosen,
            )
            snapshot_repository.record(snapshots)
        except Exception as exc:  # noqa: BLE001 -- internal capture must never break a recommendation response
            # Best-effort: a failed snapshot_repository.record() write could leave this
            # request-scoped `db` session needing an explicit rollback before any later use --
            # same rationale as the Two-Tower/session-intent/cohort-resolve handlers above.
            # `db` can legitimately be None (offline `recommend(None, request)` callers) --
            # nothing to roll back then.
            if db is not None:
                db.rollback()
            logger.info(
                "ranking snapshot capture failed userId=%s failureType=%s error=%s",
                request.user_id, type(exc).__name__, str(exc)[:200],
            )
    logger.info(
        "recommendation completed userId=%s strategy=%s modelVersion=%s count=%d retrievalMode=%s "
        "dataMode=%s userBehaviorSource=%s candidateSource=%s",
        request.user_id, strategy, metadata["modelVersion"], len(chosen), retrieval_mode,
        RECOMMENDATION_DATA_MODE, state.source, candidate_source_label,
    )
    return {
        "userId": request.user_id,
        "modelVersion": metadata["modelVersion"],
        "strategy": strategy,
        "interactionCount": interaction_count,
        "recommendations": [
            {"contentId": item["candidate"].content_id, "category": item["candidate"].category,
             "score": round(item["adjusted_score"], 6), "rank": rank, "reason": item["reason"],
             # Pass-through response enrichment (see app.schemas.response_models.
             # RecommendationItem) -- echoes the candidate's own request-supplied fields,
             # never re-derived or re-scored here.
             "creatorId": item["candidate"].creator_id,
             "contentPopularityScore": item["candidate"].content_popularity_score,
             "contentAgeHours": item["candidate"].content_age_hours,
             "creatorFollowed": item["candidate"].creator_followed,
             "alreadySeen": item["candidate"].already_seen,
             "title": item["candidate"].title,
             "hashtags": item["candidate"].hashtags,
             "topics": item["candidate"].topics,
             "entities": item["candidate"].entities,
             "subgenres": item["candidate"].subgenres,
             "language": item["candidate"].language,
             "regions": item["candidate"].regions,
             "candidateSource": item["candidate"].candidate_source,
             "localBucketSource": item["candidate"].local_bucket_source}
            for rank, item in enumerate(chosen, 1)
        ],
    }
