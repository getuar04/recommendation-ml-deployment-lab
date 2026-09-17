"""Deterministic trace of the REAL production VIDEO recommendation pipeline reacting to one
dummy user's ("User X") interaction history, scored by ONE real global trained model.

This is NOT a research script and NOT a new algorithm: every scoring/reranking step below
calls the actual production functions (`app.services.recommendation_service.recommend`,
`app.ml.dataset_builder.FeatureHistory`, `app.ml.predictor.probabilities`,
`app.ml.reranker.rerank`/`explanation`) unmodified. The only "local" logic here is building
deterministic dummy input (User X's history, a candidate set) and printing what the real
pipeline did with it.

Architecture (see README "Comparative Experiments" / Phase A userProfile section for the
production paths this reuses):

1. ONE isolated global model is trained once, via the real production trainer
   (`app.services.training_service.train`), on the existing generic multi-user synthetic
   dataset (`scripts.generate_synthetic_data.generate`) -- never on User X. Saved to an
   isolated temp directory, never the repository's real `models/` directory (see
   `_isolate_model_paths` -- same pattern `tests/test_final_acceptance.py` already uses).
   This step is skipped entirely if a schema-compatible model already exists at the real
   `models/` path (see `resolve_active_model`).
2. User X's 6 conceptual interactions are turned into real `app.db.models.Interaction` rows
   AND into an equivalent `UserProfile` snapshot (Phase A schema), so both production input
   paths can be exercised and compared (`build_feature_history_from_rows` /
   `build_user_profile_from_history`).
3. Candidates use the real `Candidate` schema.
4. Scoring/reranking go through the real `recommendation_service.recommend()` (UserProfile
   path -- zero DB reads, matching the future Feed Service -> this service boundary), plus a
   side-by-side direct call to the same `FeatureHistory.features()` /
   `app.ml.predictor.probabilities()` the service uses internally, purely so this script can
   print the RAW (pre-rerank) ranking that `recommend()` computes but does not return.
5. One new interaction is added, User X's state is rebuilt, and the SAME model scores again
   -- the model is never retrained on this new interaction.

Usage (from the repository root):
    python -m scripts.demo_production_recommendation_trace
    python -m scripts.demo_production_recommendation_trace --verbose
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite://")

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.api import training_routes
from app.db.database import Base, SessionLocal, engine
from app.db.models import Interaction
from app.ml import model_store
from app.ml.dataset_builder import (
    FEATURES,
    RECENT_WINDOW,
    FeatureHistory,
    history_from_rows,
)
from app.ml.predictor import probabilities
from app.ml.reranker import explanation as explain_reason
from app.schemas.recommendation_schemas import (
    Candidate,
    CategoryProfile,
    CreatorProfile,
    RecentWatchEvent,
    RecommendationRequest,
    UserProfile,
)
from app.services import (
    recommendation_service,
    training_service,
)
from app.services.recommendation_service import (
    ModelArtifactInvalid,
    ModelNotTrained,
    recommend,
)
from scripts.generate_synthetic_data import generate

USER_ID = "demo-user-x"
# Bulk training data (many synthetic users, never User X) is anchored to a fixed past date so
# the trained model is reproducible run to run regardless of when this script executes.
BULK_TRAINING_REFERENCE_TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class HistoricalEvent:
    content_id: str
    creator_id: str
    category: str
    event_type: str
    watch_percentage: float
    timestamp: datetime
    liked: bool = False
    shared: bool = False

    @property
    def completed(self) -> bool:
        return self.event_type == "VIDEO_COMPLETED"


# --- User X's story (fixed by the lead): historically SPORT, recently shifting to MUSIC, ---
# --- then one very recent fast-skip that rejects a SPORT recommendation. ---
#
# Timestamps are offsets from `now` (the real wall-clock moment the trace runs), not a fixed
# calendar date: `recommendation_service.recommend()` always scores against
# `datetime.now(timezone.utc)` internally (there is no as_of/reference-timestamp parameter on
# the real production function, and this script must not modify it -- see module docstring).
# Anchoring User X's history to a fixed past date would silently desynchronize this script's
# own directly-computed "raw" scoring pass (which does accept an explicit `as_of`) from what
# `recommend()` actually scores against, so the two would stop being comparable.
def historical_events(now: datetime) -> list[HistoricalEvent]:
    return [
        HistoricalEvent("hist-football-1", "creator-alpha-sports", "SPORT", "VIDEO_COMPLETED", 95.0,
                         now - timedelta(days=20), liked=True),
        HistoricalEvent("hist-ucl-1", "creator-uefa", "SPORT", "VIDEO_COMPLETED", 90.0,
                         now - timedelta(days=15), shared=True),
        HistoricalEvent("hist-football-2", "creator-alpha-sports", "SPORT", "VIDEO_WATCHED", 85.0,
                         now - timedelta(days=10)),
        HistoricalEvent("hist-concert-1", "creator-dj-nova", "MUSIC", "VIDEO_COMPLETED", 92.0,
                         now - timedelta(days=3), liked=True),
        HistoricalEvent("hist-live-1", "creator-dj-nova", "MUSIC", "VIDEO_COMPLETED", 96.0,
                         now - timedelta(days=1), liked=True),
        HistoricalEvent("hist-football-3", "creator-alpha-sports", "SPORT", "VIDEO_SKIPPED", 8.0,
                         now - timedelta(minutes=10)),
    ]


def feedback_event_for(candidate: Candidate, at: datetime) -> HistoricalEvent:
    """The lead's feedback-loop interaction (STEP 14): User X watches the top-ranked
    candidate from the FIRST recommendation call, shortly after it was served."""
    return HistoricalEvent(candidate.content_id, candidate.creator_id, candidate.category,
                            "VIDEO_COMPLETED", 98.0, at, liked=True, shared=True)


def build_candidates() -> list[Candidate]:
    """8 deterministic candidates using the real production Candidate schema (Phase A)."""
    return [
        Candidate(contentId="video-v1", creatorId="creator-alpha-sports", category="SPORT",
                  contentPopularityScore=0.55, contentAgeHours=6, creatorFollowed=False, alreadySeen=False,
                  title="Football Highlights", language=None, candidateSource=None, socialContext=None,
                  localBucketSource=None),
        Candidate(contentId="video-v2", creatorId="creator-hoops", category="SPORT",
                  contentPopularityScore=0.40, contentAgeHours=10, creatorFollowed=False, alreadySeen=False,
                  title="Basketball Highlights", language=None, candidateSource=None, socialContext=None, localBucketSource=None),
        Candidate(contentId="video-v3", creatorId="creator-dj-nova", category="MUSIC",
                  contentPopularityScore=0.60, contentAgeHours=4, creatorFollowed=False, alreadySeen=False,
                  title="Live Concert Highlights", language=None, candidateSource=None, socialContext=None, localBucketSource=None),
        Candidate(contentId="video-v4", creatorId="creator-dj-nova", category="MUSIC",
                  contentPopularityScore=0.65, contentAgeHours=8, creatorFollowed=False, alreadySeen=False,
                  title="Live Performance Session", language=None, candidateSource=None, socialContext=None, localBucketSource=None),
        Candidate(contentId="video-v5", creatorId="creator-gamer1", category="GAMING",
                  contentPopularityScore=0.35, contentAgeHours=12, creatorFollowed=False, alreadySeen=False,
                  title="Gaming Marathon", language=None, candidateSource=None, socialContext=None, localBucketSource=None),
        Candidate(contentId="video-v6", creatorId="creator-funny1", category="COMEDY",
                  contentPopularityScore=0.45, contentAgeHours=20, creatorFollowed=False, alreadySeen=False,
                  title="Stand-up Comedy Special", language=None, candidateSource=None, socialContext=None, localBucketSource=None),
        Candidate(contentId="video-v7", creatorId="creator-alpha-sports", category="SPORT",
                  contentPopularityScore=0.50, contentAgeHours=14, creatorFollowed=False, alreadySeen=False,
                  title="Football Weekly Recap", language=None, candidateSource=None, socialContext=None, localBucketSource=None),
        Candidate(contentId="video-v8", creatorId="creator-explore1", category="TRAVEL",
                  contentPopularityScore=0.20, contentAgeHours=180, creatorFollowed=False, alreadySeen=False,
                  title="Travel Vlog Around The World", language=None, candidateSource=None, socialContext=None, localBucketSource=None),
    ]


# --------------------------------------------------------------------------------------
# Step 1: ONE isolated global model, trained via the real production pipeline, never on
# User X. Isolated to a temp directory -- the repository's real models/ dir is untouched.
# --------------------------------------------------------------------------------------

def _isolate_model_paths(model_dir: Path) -> tuple[Path, Path]:
    """Same isolation pattern as tests/test_final_acceptance.py's `_isolate_model_paths`:
    MODEL_PATH/METADATA_PATH/MODEL_DIR are imported by value into multiple modules, so each
    must be patched independently or a mistake here would silently train into the real
    repository models/ directory."""
    # setattr (not `module.X = ...`) throughout: iterating a tuple of differently-typed module
    # objects widens each to plain `types.ModuleType` under mypy, so direct attribute
    # assignment spuriously fails static "Module has no attribute" checks even where the
    # attribute genuinely exists at runtime. tests/test_final_acceptance.py's own
    # `_isolate_model_paths` sidesteps the same issue via `monkeypatch.setattr(...,
    # raising=False)`; this mirrors that, without the pytest-only `monkeypatch` fixture.
    model_path = model_dir / "recommendation_model.joblib"
    metadata_path = model_dir / "model_metadata.json"
    for module in (training_routes, model_store, recommendation_service):
        setattr(module, "MODEL_PATH", model_path)  # noqa: B010
    for module in (training_routes, model_store):
        setattr(module, "METADATA_PATH", metadata_path)  # noqa: B010
    setattr(model_store, "MODEL_DIR", model_dir)  # noqa: B010
    # training_service reads model_store.MODEL_PATH/METADATA_PATH dynamically (see its own
    # module docstring) rather than importing them by value, so these two are defensive
    # redundancy only -- mirroring the same `raising=False` monkeypatch of a currently
    # nonexistent module attribute in tests/test_final_acceptance.py.
    setattr(training_service, "MODEL_PATH", model_path)  # noqa: B010
    setattr(training_service, "METADATA_PATH", metadata_path)  # noqa: B010
    return model_path, metadata_path


def train_isolated_global_model(model_dir: Path) -> dict[str, Any]:
    """Trains ONE real global model on the existing generic, multi-user synthetic dataset
    (many synthetic users, not User X) via the real production trainer
    (`app.services.training_service.train`). User X never appears in this training data."""
    _isolate_model_paths(model_dir)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    generate(reference_timestamp=BULK_TRAINING_REFERENCE_TIMESTAMP)
    db = SessionLocal()
    try:
        return training_service.train(db)
    finally:
        db.close()


def resolve_active_model(scratch_dir: Path) -> tuple[Path, Path, dict[str, Any], bool]:
    """Prefers the real repository `models/` artifact if it is present AND schema/feature
    compatible with the current code -- only trains a fresh isolated model (see
    `train_isolated_global_model`) when no compatible artifact exists locally. Returns
    (model_path, metadata_path, metadata, trained_fresh)."""
    from app.core.config import METADATA_PATH as REAL_METADATA_PATH
    from app.core.config import MODEL_PATH as REAL_MODEL_PATH
    try:
        metadata = model_store.check_artifact_compatibility(
            FEATURES, model_path=REAL_MODEL_PATH, metadata_path=REAL_METADATA_PATH,
        )
        return REAL_MODEL_PATH, REAL_METADATA_PATH, metadata, False
    except model_store.ArtifactError as exc:
        print(f"[blocker] Real production model artifact at {REAL_MODEL_PATH} is not usable: "
              f"{type(exc).__name__}: {exc}")
        print("[action] Training ONE isolated global model via the real production pipeline "
              "(app.services.training_service.train) on the existing generic multi-user "
              "synthetic dataset -- saved outside the repository, never on User X, and the "
              "real models/ directory is left untouched.")
        metadata = train_isolated_global_model(scratch_dir)
        model_path, metadata_path = scratch_dir / "recommendation_model.joblib", scratch_dir / "model_metadata.json"
        return model_path, metadata_path, metadata, True


# --------------------------------------------------------------------------------------
# Step 2: User X's history -> real FeatureHistory (DB-row path) AND an equivalent UserProfile
# (Phase A snapshot path).
# --------------------------------------------------------------------------------------

def _interaction_row(event: HistoricalEvent) -> Interaction:
    return Interaction(
        event_id=f"demo-{event.content_id}-{int(event.timestamp.timestamp())}", user_id=USER_ID,
        content_id=event.content_id, creator_id=event.creator_id, category=event.category,
        event_type=event.event_type, watch_time_seconds=event.watch_percentage,
        content_duration_seconds=100, watch_percentage=event.watch_percentage, timestamp=event.timestamp,
        liked=event.liked, shared=event.shared, favorited=False, commented=False, creator_followed=False,
    )


def build_feature_history_from_rows(events: list[HistoricalEvent]) -> FeatureHistory:
    """Path A: the legacy/demo DB-row path (`app.ml.dataset_builder.history_from_rows`),
    fed real `Interaction` ORM rows built from User X's events. No `content_by_id` is
    supplied, so semantic (hashtag/topic/entity/subgenre/title) history stays neutral --
    User X's events carry no content metadata, which is realistic and not a shortcut."""
    return history_from_rows([_interaction_row(event) for event in events])


def build_user_profile_from_history(
    history: FeatureHistory, events: list[HistoricalEvent], *, as_of: datetime,
) -> UserProfile:
    """Path B: an equivalent Phase A `UserProfile` snapshot, the production path a real Feed
    Service request will use. All-time aggregate counters (interactionCount/positiveCount/.../
    rawAffinityScore) are read directly off the REAL `FeatureHistory` accumulator state built
    by Path A above -- not recomputed here -- so the two paths are equivalent by construction,
    not by a second, parallel reimplementation of the affinity formula.
    `recentRawAffinityScore` uses the exact same RECENT_WINDOW filter
    `FeatureHistory.features()` itself applies to `recent_events`. Only `recentWatchEvents`
    (raw watch_percentage/completed/liked/shared per event) comes from `events` directly --
    that per-event detail is not retained in `FeatureHistory`'s aggregated state, so this is
    the one place the schema's own raw event facts (not a formula) are used verbatim.
    """
    by_category: dict[str, list[HistoricalEvent]] = defaultdict(list)
    for event in events:
        by_category[event.category].append(event)

    categories: list[CategoryProfile] = []
    for (user_id, category), cat in history.categories.items():
        if user_id != USER_ID:
            continue
        recent_raw = sum(delta for t, delta in cat["recent_events"] if as_of - t <= RECENT_WINDOW)
        recent_watch_events = [
            RecentWatchEvent(timestamp=event.timestamp, watchPercentage=event.watch_percentage,
                              completed=event.completed, liked=event.liked, shared=event.shared,
                              notInterested=False)
            for event in by_category[category] if as_of - event.timestamp <= RECENT_WINDOW
        ]
        categories.append(CategoryProfile(
            category=category, interactionCount=cat["interactions"], positiveCount=cat["positive"],
            negativeCount=cat["negative"], completedCount=cat["completed"], watchCount=cat["watch_count"],
            watchPercentageSum=cat["watch_percentage_sum"], rawAffinityScore=cat["raw"],
            lastInteractionAt=cat["last"], recentRawAffinityScore=recent_raw,
            recentWatchEvents=recent_watch_events,
            explicitNegativeCount=0, lastExplicitNegativeAt=None,
        ))

    creators = [
        # round(), not int(): app.ml.replay_saturation_policy weights these counters by a
        # fractional replay-influence multiplier -- this demo/trace script only needs a
        # human-readable summary, never further computation, so rounding to the nearest
        # whole interaction for display is a safe, honest approximation here.
        CreatorProfile(creatorId=creator_id, interactionCount=round(c["interactions"]), completedCount=round(c["completed"]))
        for (user_id, creator_id), c in history.creators.items() if user_id == USER_ID
    ]
    followed = [creator_id for (user_id, creator_id) in history.followed_creators if user_id == USER_ID]
    seen = [content_id for (user_id, content_id) in history.seen if user_id == USER_ID]
    return UserProfile(
        totalInteractionCount=round(history.users.get(USER_ID, 0.0)), followedCreatorIds=followed,
        seenContentIds=seen, categories=categories, creators=creators,
    )


# --------------------------------------------------------------------------------------
# Step 3: real scoring -- both the direct FeatureHistory/predictor calls (for RAW ranking
# visibility, which `recommend()` computes internally but does not return) and the full
# `recommendation_service.recommend()` call (for the authoritative FINAL response).
# --------------------------------------------------------------------------------------

def raw_score_candidates(
    history: FeatureHistory, candidates: list[Candidate], model: Any, *, as_of: datetime, cold_start: bool,
) -> list[dict[str, Any]]:
    """The exact sequence `recommendation_service.recommend()` runs internally
    (FeatureHistory.features() -> app.ml.predictor.probabilities() ->
    app.ml.reranker.explanation()), called directly so this script can print the RAW,
    pre-rerank model ranking that `recommend()`'s return value does not expose."""
    feature_rows = [
        history.features(
            user_id=USER_ID, category=c.category, creator_id=c.creator_id, content_id=c.content_id,
            timestamp=as_of, content_popularity_score=c.content_popularity_score,
            content_created_at=as_of - timedelta(hours=c.content_age_hours),
            creator_followed=c.creator_followed, already_seen=c.already_seen,
            hashtags=c.hashtags, topics=c.topics, entities=c.entities, subgenres=c.subgenres, title=c.title,
        )
        for c in candidates
    ]
    scores = probabilities(model, feature_rows)
    scored: list[dict[str, Any]] = [
        {"candidate": c, "features": f, "model_score": float(s), "reason": explain_reason(f, c, cold_start)}
        for c, f, s in zip(candidates, feature_rows, scores)
    ]
    scored.sort(key=lambda item: item["model_score"], reverse=True)
    return scored


def run_recommend(user_profile: UserProfile, candidates: list[Candidate], *, limit: int = 10) -> dict[str, Any]:
    """The real, unmodified production service function -- the same one
    app/api/recommendation_routes.py's POST /recommendations calls."""
    request = RecommendationRequest(userId=USER_ID, limit=limit, candidates=candidates, userProfile=user_profile,
                                     searchIntent=None, userContext=None)
    return recommend(None, request)


# --------------------------------------------------------------------------------------
# Explainability (Step 13): safe wording, LR-coefficient contribution only in verbose mode
# and only when the active model really is a (calibrated) LogisticRegression.
# --------------------------------------------------------------------------------------

def try_linear_contributions(model: Any, feature_row: dict[str, Any]) -> list[tuple[str, float]] | None:
    """Best-effort, read directly off the real fitted pipeline -- returns None (never a
    fabricated number) if the active model is not a calibrated linear model or introspection
    fails for any reason."""
    try:
        import pandas as pd

        calibrated_classifier = model.calibrated_classifiers_[0]
        base_estimator = calibrated_classifier.estimator
        pipeline = getattr(base_estimator, "estimator", base_estimator)  # unwrap FrozenEstimator
        preprocessor = pipeline.named_steps["features"]
        classifier = pipeline.named_steps["model"]
        coef = getattr(classifier, "coef_", None)
        if coef is None:
            return None
        frame = pd.DataFrame([feature_row])[FEATURES]
        transformed = preprocessor.transform(frame)
        transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
        names = preprocessor.get_feature_names_out()
        contributions = list(zip(names, (transformed[0] * coef[0]).tolist()))
        contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return contributions[:8]
    except Exception:  # noqa: BLE001 -- deliberately tolerant: any introspection failure (wrong
        # model type, unexpected pipeline shape, sklearn version differences) degrades to "not
        # available" rather than crashing the demo or fabricating a number.
        return None


# --------------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------------

KEY_FEATURES = [
    "category_affinity", "recent_category_affinity", "session_category_affinity",
    "session_intent_confidence", "has_session_activity", "creator_followed",
    "creator_interaction_count", "average_semantic_affinity", "recent_category_watch_percentage",
    "category_negative_count", "content_popularity_score", "already_seen",
]


def _hr(title: str = "") -> None:
    print()
    print("=" * 78)
    if title:
        print(title)
        print("=" * 78)


def print_active_model(metadata: dict[str, Any], *, trained_fresh: bool) -> None:
    _hr("ACTIVE MODEL")
    print(f"source: {'isolated demo training run (real prod trainer, generic synthetic data)' if trained_fresh else 'existing repository models/ artifact'}")
    print(f"modelVersion: {metadata['modelVersion']}")
    print(f"selectedModel: {metadata['selectedModel']}")
    print(f"modelType: {metadata['modelType']}")
    print(f"schemaVersion: {metadata.get('schemaVersion')}  featureCount: {len(metadata['featureNames'])}")
    print(f"trainingSamples: {metadata['trainingSamples']}  testSamples: {metadata['testSamples']}")
    print(f"decisionThreshold: {metadata['decisionThreshold']:.4f}")
    m = metadata.get("metrics", {}) or {}
    print(f"metrics: prAuc={m.get('prAuc')} rocAuc={m.get('rocAuc')} f1Score={m.get('f1Score')}")
    ds = metadata.get("datasetSource", {}) or {}
    print(f"trained on: {ds.get('totalRowCount')} rows, synthetic={ds.get('synthetic')} (User X is NOT in this data)")


def print_interaction_history(events: list[HistoricalEvent], *, now: datetime) -> None:
    _hr("USER X INTERACTION HISTORY (6 events)")
    for i, e in enumerate(events, 1):
        age = now - e.timestamp
        flags = ",".join(f for f, v in (("liked", e.liked), ("shared", e.shared)) if v) or "-"
        print(f"{i}. {e.category:6s} {e.content_id:16s} creator={e.creator_id:20s} "
              f"watch={e.watch_percentage:5.1f}% event={e.event_type:16s} flags={flags:12s} "
              f"{age.days}d{age.seconds // 3600}h ago")


def print_user_state(profile: UserProfile, *, label: str) -> None:
    _hr(f"USER X STATE -- {label}")
    print(f"totalInteractionCount: {profile.total_interaction_count}")
    for cat in sorted(profile.categories, key=lambda c: c.category):
        print(f"  [{cat.category}] interactions={cat.interaction_count} positive={cat.positive_count} "
              f"negative={cat.negative_count} rawAffinity={cat.raw_affinity_score:.1f} "
              f"recentRawAffinity={cat.recent_raw_affinity_score:.1f} "
              f"avgWatch%={cat.watch_percentage_sum / cat.watch_count if cat.watch_count else 0:.1f}")
    for cr in sorted(profile.creators, key=lambda c: c.creator_id):
        print(f"  creator[{cr.creator_id}] interactions={cr.interaction_count} completed={cr.completed_count}")


def print_candidates(candidates: list[Candidate]) -> None:
    _hr("CANDIDATES")
    for c in candidates:
        print(f"{c.content_id:10s} {c.category:7s} creator={c.creator_id:20s} "
              f"popularity={c.content_popularity_score:.2f} ageHours={c.content_age_hours:.0f} "
              f"title={c.title!r}")


def print_key_features(scored: list[dict[str, Any]]) -> None:
    _hr("KEY FEATURES PER CANDIDATE (subset -- see --verbose for the full vector)")
    for item in scored:
        c, f = item["candidate"], item["features"]
        print(f"{c.content_id:10s} {c.category:7s} " + " ".join(
            f"{name}={f[name]:.3f}" if isinstance(f[name], float) else f"{name}={f[name]}"
            for name in KEY_FEATURES
        ))


def print_full_feature_vector(scored: list[dict[str, Any]]) -> None:
    _hr("FULL FEATURE VECTOR (verbose)")
    for item in scored:
        c = item["candidate"]
        print(f"--- {c.content_id} ({c.category}) ---")
        for name in FEATURES:
            print(f"  {name}: {item['features'][name]}")


def print_raw_ranking(scored: list[dict[str, Any]]) -> None:
    _hr("RAW MODEL RANKING (before reranking)")
    for rank, item in enumerate(scored, 1):
        c = item["candidate"]
        print(f"{rank}. {c.content_id:10s} | {c.category:7s} | score={item['model_score']:.4f} | {item['reason']}")


def print_final(response: dict[str, Any]) -> None:
    _hr(f"FINAL RERANKED TOP-N (strategy={response['strategy']}, interactionCount={response['interactionCount']})")
    for r in response["recommendations"]:
        print(f"{r['rank']}. {r['contentId']:10s} | {r['category']:7s} | score={r['score']:.4f} | {r['reason']}")


def explain_rank_changes(raw: list[dict[str, Any]], final: dict[str, Any]) -> None:
    _hr("RAW ORDER vs FINAL RERANKED ORDER")
    raw_rank = {item["candidate"].content_id: i + 1 for i, item in enumerate(raw)}
    final_rank = {r["contentId"]: r["rank"] for r in final["recommendations"]}
    dropped = set(raw_rank) - set(final_rank)
    for content_id in sorted(final_rank, key=lambda cid: final_rank[cid]):
        before, after = raw_rank[content_id], final_rank[content_id]
        marker = "same" if before == after else f"moved {before} -> {after}"
        print(f"{content_id:10s} raw#{before:<3d} final#{after:<3d} ({marker})")
    if dropped:
        print(f"dropped by reranker (diversity/creator-repetition limits): {sorted(dropped)}")


def print_lr_contributions(model: Any, scored: list[dict[str, Any]]) -> None:
    _hr("LOGISTIC REGRESSION COEFFICIENT CONTRIBUTIONS (verbose, top candidate only)")
    top = scored[0]
    contributions = try_linear_contributions(model, top["features"])
    if contributions is None:
        print("Not available for this active model/run (either not a linear model, or "
              "introspection failed) -- omitted rather than fabricated.")
        return
    print(f"For {top['candidate'].content_id} ({top['candidate'].category}), largest |contribution| terms:")
    for name, value in contributions:
        print(f"  {name}: {value:+.4f}")


def print_before_after(before: dict[str, Any], after: dict[str, Any], *, watched_content_id: str) -> None:
    _hr("BEFORE vs AFTER (same model, same candidate set, User X state updated)")
    before_by_id = {r["contentId"]: r for r in before["recommendations"]}
    after_by_id = {r["contentId"]: r for r in after["recommendations"]}
    all_ids = sorted(set(before_by_id) | set(after_by_id))
    changed_any = False
    for content_id in all_ids:
        b, a = before_by_id.get(content_id), after_by_id.get(content_id)
        b_desc = f"rank={b['rank']} score={b['score']:.4f}" if b else "not in top-N"
        a_desc = f"rank={a['rank']} score={a['score']:.4f}" if a else "not in top-N"
        star = " <-- just watched" if content_id == watched_content_id else ""
        if b_desc != a_desc:
            changed_any = True
        print(f"{content_id:10s} BEFORE[{b_desc:24s}] AFTER[{a_desc:24s}]{star}")
    if not changed_any:
        print("\nNo rank/score changes were observed in the top-N between BEFORE and AFTER.")


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_trace(*, verbose: bool = False) -> dict[str, Any]:
    """Runs the full trace and returns the key artifacts (for reuse by tests). Prints the
    presentation-friendly narrative as it goes."""
    with TemporaryDirectory(prefix="rms-demo-model-") as scratch:
        model_path, metadata_path, metadata, trained_fresh = resolve_active_model(Path(scratch))
        print_active_model(metadata, trained_fresh=trained_fresh)
        model, _ = model_store.load_validated(FEATURES, model_path=model_path, metadata_path=metadata_path)

        # Captured once and reused for every "raw" computation below. `recommend()` itself
        # always uses its own internal `datetime.now(timezone.utc)` (no as_of parameter --
        # see module docstring), so this value and recommend()'s internal "now" can only ever
        # agree to within the few milliseconds of Python execution between the two calls
        # below -- close enough that no feature (day/hour/window-membership granularity)
        # differs, without modifying the real production function to accept an injected clock.
        now_before = datetime.now(timezone.utc)
        events_before = historical_events(now_before)
        print_interaction_history(events_before, now=now_before)

        history_a = build_feature_history_from_rows(events_before)
        history_b_source = build_feature_history_from_rows(events_before)
        profile_before = build_user_profile_from_history(history_b_source, events_before, as_of=now_before)
        history_b = FeatureHistory.from_profile(USER_ID, profile_before)

        print_user_state(profile_before, label="BEFORE new interaction")

        candidates = build_candidates()
        print_candidates(candidates)

        cold_start = profile_before.total_interaction_count == 0
        scored_raw = raw_score_candidates(history_b, candidates, model, as_of=now_before, cold_start=cold_start)
        scored_raw_dbpath = raw_score_candidates(history_a, candidates, model, as_of=now_before, cold_start=cold_start)
        _hr("PATH EQUIVALENCE CHECK (DB-row FeatureHistory vs UserProfile-derived FeatureHistory)")
        max_diff = max(
            abs(a["model_score"] - b["model_score"]) for a, b in zip(scored_raw_dbpath, scored_raw)
        )
        print(f"Max |model_score| difference between the two paths across all candidates: {max_diff:.6f}")
        print("(Expected to be ~0 for category/creator/session-driven features; semantic "
              "features are always neutral on the UserProfile path -- see module docstring "
              "'Known limitation' in app.ml.dataset_builder.)")

        print_key_features(scored_raw)
        if verbose:
            print_full_feature_vector(scored_raw)
        print_raw_ranking(scored_raw)

        response_before = run_recommend(profile_before, candidates, limit=10)
        print_final(response_before)
        explain_rank_changes(scored_raw, response_before)
        if verbose:
            print_lr_contributions(model, scored_raw)

        top_music = next((r for r in response_before["recommendations"] if r["category"] == "MUSIC"), None)
        if top_music is None:
            print("\n[observation] No MUSIC candidate reached the top-N -- the feedback-loop "
                  "interaction below will target the highest-ranked candidate overall instead.")
            watched = response_before["recommendations"][0]
        else:
            watched = top_music
        watched_candidate = next(c for c in candidates if c.content_id == watched["contentId"])

        # feedback_at: just after `now_before` -- a reaction to the FIRST recommendation call.
        # now_after: a second, independently-captured real "now" for the AFTER call (same
        # reasoning as now_before -- recommend() always uses its own internal wall-clock time,
        # see above), captured naturally later than feedback_at by however long the script
        # itself took to run the first call and print its trace. Both the feedback event and
        # the earlier fast-skip event stay well inside the 30-minute session window relative
        # to now_after, and inside the 30-day recent window -- nothing here depends on the
        # exact elapsed duration.
        feedback_at = now_before + timedelta(seconds=1)
        feedback_event = feedback_event_for(watched_candidate, feedback_at)
        _hr("NEW USER INTERACTION")
        print(f"User X watches {feedback_event.content_id} ({feedback_event.category}) "
              f"watch={feedback_event.watch_percentage}% liked={feedback_event.liked} shared={feedback_event.shared}")
        print("The GLOBAL model is NOT retrained. Only User X's state is updated.")

        now_after = datetime.now(timezone.utc)
        events_after = [*events_before, feedback_event]
        history_after_source = build_feature_history_from_rows(events_after)
        profile_after = build_user_profile_from_history(history_after_source, events_after, as_of=now_after)
        print_user_state(profile_after, label="AFTER new interaction")

        response_after = run_recommend(profile_after, candidates, limit=10)
        print_final(response_after)

        print_before_after(response_before, response_after, watched_content_id=watched_candidate.content_id)

        return {
            "metadata": metadata, "trained_fresh": trained_fresh,
            "profile_before": profile_before, "profile_after": profile_after,
            "scored_raw": scored_raw, "response_before": response_before, "response_after": response_after,
            "watched_candidate": watched_candidate,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="also print full feature vectors and LR coefficient debug info")
    args = parser.parse_args(argv)
    try:
        run_trace(verbose=args.verbose)
    except (ModelNotTrained, ModelArtifactInvalid) as exc:
        print(f"\n[BLOCKER] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _hr("KEY OBSERVATION")
    print("See the BEFORE vs AFTER section above for what the real production model actually did.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
