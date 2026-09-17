import json
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _decode_token_list(raw: str | None) -> list[str]:
    """Decodes a `*_json` column back into a token list. Corrupt or absent JSON degrades to
    `[]` (neutral semantic features), never a crash -- matches how a NULL column (content
    created before semantic metadata existed, or by a caller that never sent any) already
    reads back as no semantic history in app.ml.dataset_builder."""
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return decoded if isinstance(decoded, list) else []

class User(Base):
    """Spec §4: a user exists independently of any behaviour profile. Creating a user must
    not assign interests, train anything, or fabricate interactions."""
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    status: Mapped[str] = mapped_column(String, default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Interaction(Base):
    __tablename__ = "interactions"
    # Phase A: backs app.db.repositories.recent_interactions_for_ranking()'s
    # WHERE user_id = ? ORDER BY timestamp DESC LIMIT N query on the VIDEO ranking hot path;
    # kept in sync with migrations/versions/0b6779c4f8fb_*.py's op.create_index() so
    # Base.metadata (SQLite tests) and Alembic (Postgres) agree on the schema.
    __table_args__ = (Index("ix_interactions_user_id_timestamp", "user_id", "timestamp"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    content_id: Mapped[str] = mapped_column(String, index=True)
    creator_id: Mapped[str] = mapped_column(String, index=True)
    category: Mapped[str] = mapped_column(String, index=True)
    event_type: Mapped[str] = mapped_column(String)
    watch_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    content_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    watch_percentage: Mapped[float | None] = mapped_column(Float, nullable=True)
    liked: Mapped[bool] = mapped_column(Boolean, default=False)
    shared: Mapped[bool] = mapped_column(Boolean, default=False)
    favorited: Mapped[bool] = mapped_column(Boolean, default=False)
    commented: Mapped[bool] = mapped_column(Boolean, default=False)
    creator_followed: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    # Training-dataset context/label separation (internal, training-only; see
    # app.ml.dataset_builder.build_dataset's own docstring for the full mechanism). Real event
    # ingestion (app.routes/app.services.interaction_service -- whatever accepts external event
    # payloads) never sets this; it has no Pydantic/API schema field and defaults False for
    # every existing and every real production row, so build_dataset's behavior is unchanged
    # for every caller that doesn't explicitly opt in. Only scripts/generate_synthetic_data.py's
    # dedicated context-only cohort ever sets it True, to update FeatureHistory (the same as any
    # real historical event) without that same event becoming a labeled training row -- avoiding
    # the point-in-time checkpoint-collision problem a literal N-length skip streak causes (see
    # the repair report). Never read by the live recommendation/inference path, which never
    # calls build_dataset at all.
    is_training_context_only: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Eventual-consistency audit (Task: local-existence blocking validation, later corrected by
    # Task: content_pending evidence-leak audit): RMS is not the authoritative source of truth
    # for Content -- a Content projection may simply not have arrived locally yet when an
    # otherwise-valid interaction referencing it does. True means
    # app.services.event_service.store_event could not find a local Content row for this
    # interaction's content_id at ingestion time, so the content-active/content-type-mismatch
    # checks (which require that row) were skipped rather than treated as failures, and no
    # Content-derived data was fabricated onto this row -- every other column is populated
    # from the event payload itself, exactly like a resolved interaction. False (the default,
    # and every pre-existing row) means Content was present and locally known at ingestion
    # time, so those checks ran normally, OR a previously-pending row has since been resolved
    # (see app.api.content_routes._resolve_pending_interactions).
    #
    # Corrective pass finding: this column previously claimed to be "never read... observability/
    # future-reconciliation metadata only" -- that was false. Every evidence-generating read
    # path (app.db.repositories.interactions/recent_interactions_for_ranking/
    # recent_interactions_for_drift/warmup_interactions_for_drift/interactions_for_content, plus
    # the LIVE providers that query Interaction directly -- app.services.providers.
    # live_dynamic_state_provider/live_personalization_provider) now filters on
    # content_pending.is_(False); a True row is excluded from the long-term/recent/session
    # profile, VIDEO/LIVE training and drift datasets, candidate generation, cohort
    # aggregation, and LIVE dynamic-state/personalization signals until
    # `app.api.content_routes.create_content`'s reconciliation clears it.
    content_pending: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

class Content(Base):
    """Spec §5: contentType is part of the persisted contract, not just accepted and
    discarded -- callers must be able to read back what they created."""
    __tablename__ = "contents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    creator_id: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(String, default="VIDEO", server_default="VIDEO")
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    popularity_score: Mapped[float] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    # Semantic metadata (VIDEO recommendation only): all nullable, additive columns -- content
    # created before this existed (or by a caller that only sends category) reads back as
    # title=None and empty token lists, which app.ml.dataset_builder treats as neutral
    # semantic features, never a crash. Stored as JSON-encoded lists of already-normalized
    # tokens (app.ml.semantic_tokens) rather than a separate association table: bounded
    # per-row size (see semantic_tokens.MAX_TOKENS_PER_FIELD) makes a relational fan-out table
    # pure overhead here -- tokens are only ever read whole, alongside their own content row,
    # never queried by token.
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    hashtags_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    topics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    entities_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    subgenres_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Content-understanding (app.ml.content_classifier / app.services.content_enrichment_
    # service): additive, nullable -- every pre-existing row (and every caller that still
    # supplies category explicitly) reads back as category_source=None/category_confidence=
    # None, never a fabricated value. Populated only when `category` was LOCALLY INFERRED
    # (no caller-supplied category at creation time): category_source is "INFERRED" and
    # category_confidence is the classifier's (possibly creator-prior-blended) confidence in
    # [0, 1]. An explicitly caller-supplied category leaves both None -- category_source
    # "EXPLICIT" is not persisted, since None already unambiguously means "not inferred" for
    # every existing and future caller that never sends this field.
    category_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    category_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Future canonical semantic taxonomy foundation (additive, Task: taxonomy/version
    # compatibility infrastructure) -- deliberately NOT populated with any final taxonomy by
    # this change (no proposed category list is product-approved yet; see the taxonomy
    # stress-test/decision-package reports). `category` above remains, unchanged, the ONLY
    # field the currently-active VIDEO model's categorical feature and every existing
    # retrieval/reranking/affinity consumer reads -- this task does not switch any of them
    # over. These three columns exist purely so a FUTURE canonical value has somewhere to
    # land without ever touching `category`:
    #   primary_category -- future canonical broad classification (nullable; NULL means "no
    #     canonical classification exists for this row yet", true for every row today).
    #   subcategory -- future governed second-level classification under primary_category
    #     (nullable, independent of primary_category being set).
    #   taxonomy_version -- which versioned vocabulary a NON-NULL primary_category/subcategory
    #     value belongs to (never invented/silently reinterpreted later). Also stamped for the
    #     EXISTING legacy classifier's own inferred `category` (using its own already-existing
    #     CATEGORY_TAXONOMY_VERSION, "v1-bootstrap") -- an accurate description of data that
    #     already exists, not a new taxonomy claim -- so a caller-supplied `category` (no
    #     provenance claim possible) still leaves this NULL, exactly like category_source does.
    primary_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subcategory: Mapped[str | None] = mapped_column(String(64), nullable=True)
    taxonomy_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    @property
    def hashtags(self) -> list[str]:
        return _decode_token_list(self.hashtags_json)

    @property
    def topics(self) -> list[str]:
        return _decode_token_list(self.topics_json)

    @property
    def entities(self) -> list[str]:
        return _decode_token_list(self.entities_json)

    @property
    def subgenres(self) -> list[str]:
        return _decode_token_list(self.subgenres_json)

    __table_args__ = (
        Index("ix_contents_content_type_is_active", "content_type", "is_active"),
    )


class TrainingJob(Base):
    """Spec §6/Group C: training job state persisted in the database so it survives a
    process restart -- the previous implementation kept this in an in-process dict."""
    __tablename__ = "training_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    model_type: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="PENDING")
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String, nullable=True)
    # Full training-result metadata as JSON text, so API callers keep seeing the same rich
    # `result` object the old in-memory job store returned (spec's own required-columns list
    # is "at least" the others above; this one is additive).
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class UserDemographicContext(Base):
    """Cold-start cohort CONTEXT only (VIDEO cohort preference system) -- never a behaviour
    profile, never an ML feature source. Populated opportunistically by
    app.services.cohort_preference_provider.record_demographic_context() whenever a VIDEO
    recommendation request supplies `userContext.region`/`age` (see
    app.schemas.recommendation_schemas.UserContext): the caller's most-recently-observed
    normalized region/derived age bucket, nothing else. Deliberately stores `age_bucket`, never
    raw age (privacy/data-minimization boundary, spec §M) -- app.core.cohort_context.
    age_bucket_for() is the only place raw age is read at all. One row per user (last-write-
    wins on the latest request), consumed only by app.services.cohort_aggregation_service's
    batch rebuild job, never read on the per-request hot path."""
    __tablename__ = "user_demographic_context"
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    region: Mapped[str | None] = mapped_column(String(8), nullable=True)
    age_bucket: Mapped[str | None] = mapped_column(String(16), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class UserOnboardingContext(Base):
    """Persisted onboarding context (age/region/language/interests), supplied once via
    `userContext` on `POST /users` and automatically reloaded by
    app.services.recommendation_service.recommend() on every later recommendation request
    for that user that doesn't supply its own `userContext` (see
    app.services.user_context_provider.load_user_context and that function's caller for the
    exact precedence rule).

    Deliberately a SEPARATE table from UserDemographicContext above, not an extension of it:
    that table is an anonymized (age_bucket, never raw age), no-interests/no-language,
    cohort-REBUILD-only snapshot with its own single offline-aggregation purpose -- mixing
    this table's richer, individually-attributable onboarding profile into it would blur two
    different responsibilities (population-level cohort statistics vs. one user's own
    declared profile). Raw age IS stored here (unlike UserDemographicContext) because this
    row belongs to, and was declared by, that one user -- not pooled cohort-aggregate
    storage; it still only ever reaches cohort lookups through
    app.core.cohort_context.age_bucket_for(), exactly like a request-supplied `userContext.
    age` already does, so the population-level data-minimization boundary is unaffected.

    One row per user, written once at creation (see app.api.user_routes.create_user) --
    there is currently no update endpoint, matching this task's scope. `interests` is stored
    JSON-encoded via the same convention app.db.models.Content already uses for its own
    hashtags/topics/entities/subgenres columns."""
    __tablename__ = "user_onboarding_context"
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    region: Mapped[str | None] = mapped_column(String(8), nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    interests_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    @property
    def interests(self) -> list[str]:
        return _decode_token_list(self.interests_json)


class CohortPreference(Base):
    """Persisted, versioned cohort cold-start preference statistics (VIDEO only). One row per
    (version, cohort_type, region, age_bucket, category) -- see
    app.services.cohort_aggregation_service.rebuild_cohort_preferences for how this is
    (re)computed from historical `interactions` + `user_demographic_context`, and
    app.services.cohort_preference_provider for how serving reads it back (always the single
    latest `version`, never a mix of versions, never a live recomputation). `region`/
    `age_bucket` are both nullable so one table represents all four fallback levels:
    REGION_AGE (both set), REGION (age_bucket NULL), AGE (region NULL), GLOBAL (both NULL) --
    `cohort_type` is the authoritative discriminator, not "which columns happen to be NULL",
    so a lookup never has to guess."""
    __tablename__ = "recommendation_cohort_preferences"
    __table_args__ = (
        Index("ix_cohort_pref_lookup", "version", "cohort_type", "region", "age_bucket"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, index=True)
    cohort_type: Mapped[str] = mapped_column(String(16))
    region: Mapped[str | None] = mapped_column(String(8), nullable=True)
    age_bucket: Mapped[str | None] = mapped_column(String(16), nullable=True)
    category: Mapped[str] = mapped_column(String(64))
    preference_score: Mapped[float] = mapped_column(Float)
    sample_users: Mapped[int] = mapped_column(Integer)
    sample_interactions: Mapped[int] = mapped_column(Integer)
    positive_count: Mapped[int] = mapped_column(Integer)
    negative_count: Mapped[int] = mapped_column(Integer)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SessionSearchIntent(Base):
    """Shared/persistent backing store for app.services.session_intent_provider's
    `DatabaseSessionIntentProvider` -- replaces the old single-process in-memory dict so a
    recorded search survives a restart and is visible to every RMS worker/process, not just
    the one that received the POST /users/{userId}/search-intent call. One row per user (a new
    search REPLACES the previous row outright, matching the pre-existing in-memory semantics);
    `expires_at` is computed at write time from the same SESSION_WINDOW-derived TTL the
    in-memory provider already used, so lookup is a single indexed comparison, never a scan."""
    __tablename__ = "session_search_intent"
    __table_args__ = (Index("ix_session_search_intent_expires_at", "expires_at"),)
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    intent_json: Mapped[str] = mapped_column(Text)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TrainingLock(Base):
    """Spec §7/Group C: one row per model type. Presence of a (non-stale) row means a
    training job is in flight; the primary key on `model_type` is what makes concurrent
    acquisition attempts race safely at the database level instead of relying on a
    Python-process-local lock, which would not be restart- or multi-worker-safe."""
    __tablename__ = "training_locks"
    model_type: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(String)
    locked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

