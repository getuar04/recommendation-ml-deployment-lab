from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.cohort_context import normalize_region
from app.core.config import RECOMMENDATION_MAX_CANDIDATES
from app.ml.dataset_builder import MAX_CONTENT_AGE_HOURS, SEMANTIC_FIELDS
from app.ml.semantic_tokens import normalize_title, normalize_token, normalize_tokens
from app.schemas.limits import (
    RECOMMENDATION_LIMIT_DEFAULT,
    RECOMMENDATION_LIMIT_MAX,
    RECOMMENDATION_LIMIT_MIN,
)

# Finalization spec Decision 10, extended for related-user candidate generation: the only two
# candidateSource values with any defined reranker behavior today. Kept as an explicit
# allow-list (not a free string) so an unrecognized value fails fast at request validation
# rather than silently having zero effect further downstream.
#
# SOCIAL vs COLLABORATIVE is a provenance distinction owned entirely by the caller (Candidate
# Service, in production) -- RMS scores both identically (app.ml.reranker.social_relevance),
# on purpose: the boost is a pure function of the same three bounded SocialContext signals
# (interestSimilarity/relationshipStrength/sourceUserEngagement) regardless of which produced
# the candidate, so there is nothing for RMS to compute differently between them. The
# distinction exists for the CALLER's own bookkeeping/explainability (a follow-graph traversal
# vs. a behavior-similarity match are different retrieval strategies upstream), not because RMS
# ranks them differently:
#   SOCIAL:        candidate surfaced via a direct relationship (follow/mutual-follow).
#   COLLABORATIVE: candidate surfaced via behavior/interest similarity between two users with
#                  no required direct relationship (e.g. two users who don't follow each other
#                  but have highly similar long-term category/semantic interests).
ALLOWED_CANDIDATE_SOURCES = frozenset({"SOCIAL", "COLLABORATIVE"})


def _reject_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


# Finalization spec Decision 10: bounded, generic social provenance for a candidate sourced
# from a followed/related user rather than the general catalog. Every field is a plain 0..1
# credibility signal (or a boolean) -- no user IDs, no relationship graph, nothing that would
# require RMS to own social state; the caller (Follow/Event Tracking Service, in production)
# supplies these as pre-computed values, exactly like contentPopularityScore already is.
class SocialContext(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    interest_similarity: float=Field(0.0, alias="interestSimilarity", ge=0, le=1)
    relationship_strength: float=Field(0.0, alias="relationshipStrength", ge=0, le=1)
    source_user_engagement: float=Field(0.0, alias="sourceUserEngagement", ge=0, le=1)
    mutual_follow: bool=Field(False, alias="mutualFollow")


# Phase A §7: max_length values below match the bound app/api/content_routes.py's
# ContentCreate already applies to the same logical fields (contentId/creatorId=128,
# category=64) -- reusing an existing convention, not inventing a new one.
class Candidate(BaseModel):
    """title/hashtags/topics/entities/subgenres are optional, additive semantic metadata
    (VIDEO only): a caller sending only category/creatorId/etc. keeps working unchanged,
    scored with neutral semantic features. Tokens are normalized the exact same way
    app.api.content_routes normalizes them at ingestion time (app.ml.semantic_tokens), which
    is what the training/inference normalization-parity requirement depends on.

    language/regions/candidateSource/socialContext (finalization spec Decisions 10/11) are
    reranker-only context: none of them feed FeatureHistory.features() or the model, they are
    read directly off the candidate by app.ml.reranker. A caller that omits them keeps
    scoring/ranking byte-identical to today.

    `localBucketSource` (candidateSource/reason observability audit) is a DIFFERENT concept
    from `candidateSource` above -- that field is caller-declared SOCIAL/COLLABORATIVE
    provenance for an explicit candidate from an external system (validated against
    `ALLOWED_CANDIDATE_SOURCES`); this one is RMS's OWN local bucket label
    (TRENDING/NEW_CONTENT/PREFERRED_CATEGORY/etc., see
    `app.services.providers.video_candidate_provider.SOURCE_ORDER`) for a candidate this
    service generated itself. Set only by local generation; never read by
    `FeatureHistory.features()`, the model, or the reranker -- informational/observability
    only, exactly like `candidateSource`. Not caller-authoritative: a caller may set it on an
    explicit candidate (harmless -- it is passed through unread by scoring/ranking/eligibility,
    unlike `popularityScore`, which is why this field has no ownership-enforcing validator)."""
    model_config=ConfigDict(populate_by_name=True)
    content_id: str=Field(alias="contentId", max_length=128); creator_id: str=Field(alias="creatorId", max_length=128); category: str=Field(max_length=64)
    content_popularity_score: float=Field(alias="contentPopularityScore", ge=0, le=1)
    content_age_hours: float=Field(alias="contentAgeHours", ge=0, le=MAX_CONTENT_AGE_HOURS)
    creator_followed: bool=Field(False, alias="creatorFollowed"); already_seen: bool=Field(False, alias="alreadySeen")
    title: str | None = Field(None)
    hashtags: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    subgenres: list[str] = Field(default_factory=list)
    language: str | None = Field(None, max_length=16)
    regions: list[str] = Field(default_factory=list, max_length=50)
    candidate_source: str | None = Field(None, alias="candidateSource")
    social_context: SocialContext | None = Field(None, alias="socialContext")
    local_bucket_source: str | None = Field(None, alias="localBucketSource")

    @field_validator("title")
    @classmethod
    def _normalize_title(cls, value: str | None) -> str | None:
        return normalize_title(value)

    @field_validator("hashtags", "topics", "entities", "subgenres", mode="before")
    @classmethod
    def _normalize_field_tokens(cls, value: object, info: Any) -> list[str]:
        return normalize_tokens(value, field=info.field_name)

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str | None) -> str | None:
        return value.strip().lower() if value and value.strip() else None

    @field_validator("regions", mode="before")
    @classmethod
    def _normalize_regions(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)) or isinstance(value, str):
            # ValueError, not TypeError: pydantic v2 field_validators only convert ValueError/
            # AssertionError into a clean 422 ValidationError -- a raised TypeError propagates
            # raw and becomes an unhandled 500 instead (verified against pydantic==2.13.4).
            raise ValueError("regions must be a list of strings")  # noqa: TRY004
        return list(dict.fromkeys(str(v).strip().upper() for v in value if str(v).strip()))

    @field_validator("candidate_source")
    @classmethod
    def _validate_candidate_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if normalized not in ALLOWED_CANDIDATE_SOURCES:
            raise ValueError(f"candidateSource must be one of {sorted(ALLOWED_CANDIDATE_SOURCES)}")
        return normalized


# Phase A §1/§3: optional VIDEO behavior-profile snapshot. Every field here is derived 1:1
# from what app.ml.dataset_builder.FeatureHistory.features() actually consumes (see
# FeatureHistory.from_profile()) -- nothing here is invented beyond that pipeline's existing
# inputs. When supplied, the recommendation service builds history entirely from this object
# and never queries the local interactions table (app/services/recommendation_service.py).
class RecentWatchEvent(BaseModel):
    """One raw watch event within the existing 30-day RECENT_WINDOW
    (app.ml.dataset_builder.RECENT_WINDOW) -- naturally small, not full lifetime history."""
    model_config=ConfigDict(populate_by_name=True)
    timestamp: datetime
    # Bound matches EventCreate's own existing "watchTimeSeconds unreasonably exceeds
    # duration" rule (app/schemas/event_schemas.py: rejects > 3x contentDurationSeconds) --
    # a rewatch/replay can legitimately push watch_percentage above 100. Confirmed against
    # real data during Phase A verification: values up to ~122% occur in practice, so a
    # le=100 bound (this schema's first draft) was genuinely too strict, not just cautious.
    watch_percentage: float=Field(alias="watchPercentage", ge=0, le=300)
    completed: bool
    # Session-behavior addition (current-session productionization): current-session evidence
    # strength distinguishes a bare watch from a watch+like/share (see
    # app.ml.dataset_builder._session_event_delta). Optional, defaulting to False, so an
    # upstream caller that hasn't been updated to populate these two fields yet degrades
    # safely to "no explicit signal on this event" rather than crashing or being rejected --
    # the same additive-field convention recentRawAffinityScore already established on
    # CategoryProfile below.
    liked: bool=False
    shared: bool=False
    # Negative-feedback session bug fix: whether this specific event was an explicit rejection
    # (CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED, or a skip the caller wants scored the same
    # way) -- see app.ml.dataset_builder._session_event_delta/_session_label's docstrings for
    # the bug this closes (a high watch_percentage on a since-rejected video used to read as
    # positive session evidence). Same additive/safe-default convention as liked/shared above:
    # a caller that predates this field keeps sending valid requests, just without the fix.
    not_interested: bool=Field(False, alias="notInterested")


class CategoryProfile(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    category: str=Field(min_length=1, max_length=64)
    interaction_count: int=Field(alias="interactionCount", ge=0)
    positive_count: int=Field(alias="positiveCount", ge=0)
    negative_count: int=Field(alias="negativeCount", ge=0)
    completed_count: int=Field(alias="completedCount", ge=0)
    watch_count: int=Field(alias="watchCount", ge=0)
    watch_percentage_sum: float=Field(alias="watchPercentageSum", ge=0, allow_inf_nan=False)
    raw_affinity_score: float=Field(alias="rawAffinityScore", allow_inf_nan=False)
    last_interaction_at: datetime|None=Field(None, alias="lastInteractionAt")
    recent_watch_events: list[RecentWatchEvent]=Field(default_factory=list, alias="recentWatchEvents", max_length=200)
    # Recent/session behavior (minimum addition for recent_category_affinity, see
    # app.ml.dataset_builder.FeatureHistory.features()): the same pre-aggregated,
    # "trusted, not re-derived" convention as rawAffinityScore above, restricted to only
    # the events within the existing RECENT_WINDOW -- computed by the supplying service
    # (User Behavior Service, in production) using the exact same weighting formula
    # FeatureHistory.update() already applies for rawAffinityScore, just windowed.
    # Defaults to 0.0 (neutral: affinity_score(0.0) == 0.5) so a caller that doesn't yet
    # populate this field degrades safely instead of crashing or fabricating a signal.
    recent_raw_affinity_score: float=Field(0.0, alias="recentRawAffinityScore", allow_inf_nan=False)
    # Negative-feedback feature representation task: additive, optional, same convention as
    # recentRawAffinityScore above -- a caller that doesn't yet populate these (every pre-this-
    # task caller) degrades safely to "never explicitly rejected" (see
    # app.ml.dataset_builder.FeatureHistory.from_profile), not a crash or a fabricated signal.
    # `explicit_negative_count` is the EXPLICIT_NEGATIVE_EVENT_TYPES-only subset of
    # `negative_count` -- pre-aggregated by the caller exactly like every other counter here.
    explicit_negative_count: int=Field(0, alias="explicitNegativeCount", ge=0)
    last_explicit_negative_at: datetime|None=Field(None, alias="lastExplicitNegativeAt")

    @field_validator("category")
    @classmethod
    def _category_not_blank(cls, value: str) -> str:
        return _reject_blank(value, "category")

    # Counts below are pre-aggregated by the caller (User Behavior Service in production),
    # not recomputed here -- these checks only reject internally-inconsistent snapshots that
    # would otherwise silently corrupt FeatureHistory.from_profile()'s reconstructed state
    # (e.g. category_completion_rate > 1). positiveCount+negativeCount<=interactionCount
    # mirrors TARGET_DEFINITION (app/ml/feature_builder.py): a single interaction is never
    # both a positive and a negative label.
    @model_validator(mode="after")
    def _counts_consistent_with_interaction_count(self) -> "CategoryProfile":
        for name, value in (
            ("positiveCount", self.positive_count),
            ("negativeCount", self.negative_count),
            ("completedCount", self.completed_count),
            ("watchCount", self.watch_count),
        ):
            if value > self.interaction_count:
                raise ValueError(
                    f"category {self.category!r}: {name} ({value}) must not exceed interactionCount ({self.interaction_count})"
                )
        if self.positive_count + self.negative_count > self.interaction_count:
            raise ValueError(
                f"category {self.category!r}: positiveCount + negativeCount "
                f"({self.positive_count + self.negative_count}) must not exceed interactionCount ({self.interaction_count})"
            )
        if self.explicit_negative_count > self.negative_count:
            raise ValueError(
                f"category {self.category!r}: explicitNegativeCount ({self.explicit_negative_count}) "
                f"must not exceed negativeCount ({self.negative_count})"
            )
        return self


# Finalization spec Decision 5: the smallest clean contract that lets FeatureHistory.from_profile
# populate the SAME per-(user, token) semantic accumulators app.ml.dataset_builder.FeatureHistory.
# update() already builds from real interaction rows (see FeatureHistory.tokens) -- one entry per
# (field, token) pair the supplying service has pre-aggregated, mirroring how CategoryProfile/
# CreatorProfile already carry pre-aggregated counters instead of raw events. Does not add a new
# model feature: hashtag_affinity/topic_affinity/.../average_semantic_affinity already exist in
# FEATURES (see app.ml.dataset_builder.NUMERIC) and are already computed from FeatureHistory.tokens
# -- this only lets that existing computation see non-neutral values on the userProfile path.
class SemanticAffinityEntry(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    field: str
    token: str=Field(min_length=1, max_length=64)
    interaction_count: int=Field(alias="interactionCount", ge=0)
    positive_count: int=Field(alias="positiveCount", ge=0)
    negative_count: int=Field(alias="negativeCount", ge=0)
    raw_affinity_score: float=Field(alias="rawAffinityScore", allow_inf_nan=False)
    # Negative-feedback feature representation task: additive/optional, same convention as
    # CategoryProfile.explicit_negative_count above.
    explicit_negative_count: int=Field(0, alias="explicitNegativeCount", ge=0)

    @field_validator("field")
    @classmethod
    def _field_must_be_known(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SEMANTIC_FIELDS:
            raise ValueError(f"semanticAffinities.field must be one of {SEMANTIC_FIELDS}, got {value!r}")
        return normalized

    @field_validator("token")
    @classmethod
    def _normalize_and_reject_blank_token(cls, value: str) -> str:
        normalized = normalize_token(value)
        if not normalized:
            raise ValueError(f"semanticAffinities.token {value!r} normalizes to an empty token")
        return normalized

    @model_validator(mode="after")
    def _counts_consistent(self) -> "SemanticAffinityEntry":
        if self.positive_count + self.negative_count > self.interaction_count:
            raise ValueError(
                f"semantic entry ({self.field!r}, {self.token!r}): positiveCount + negativeCount "
                f"({self.positive_count + self.negative_count}) must not exceed interactionCount ({self.interaction_count})"
            )
        if self.explicit_negative_count > self.negative_count:
            raise ValueError(
                f"semantic entry ({self.field!r}, {self.token!r}): explicitNegativeCount "
                f"({self.explicit_negative_count}) must not exceed negativeCount ({self.negative_count})"
            )
        return self


class CreatorProfile(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    creator_id: str=Field(alias="creatorId", min_length=1, max_length=128) 
    interaction_count: int=Field(alias="interactionCount", ge=0)
    completed_count: int=Field(alias="completedCount", ge=0)

    @field_validator("creator_id")
    @classmethod
    def _creator_id_not_blank(cls, value: str) -> str:
        return _reject_blank(value, "creatorId")

    @model_validator(mode="after")
    def _completed_count_consistent(self) -> "CreatorProfile":
        if self.completed_count > self.interaction_count:
            raise ValueError(
                f"creator {self.creator_id!r}: completedCount ({self.completed_count}) "
                f"must not exceed interactionCount ({self.interaction_count})"
            )
        return self


class UserProfile(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    version: int=1
    total_interaction_count: int=Field(alias="totalInteractionCount", ge=0)
    followed_creator_ids: list[str]=Field(default_factory=list, alias="followedCreatorIds", max_length=500)
    seen_content_ids: list[str]=Field(default_factory=list, alias="seenContentIds", max_length=2000)
    categories: list[CategoryProfile]=Field(default_factory=list, max_length=200)
    creators: list[CreatorProfile]=Field(default_factory=list, max_length=1000)
    semantic_affinities: list[SemanticAffinityEntry]=Field(default_factory=list, alias="semanticAffinities", max_length=1000)

    @field_validator("version")
    @classmethod
    def _version_must_be_supported(cls, value: int) -> int:
        if value != 1:
            raise ValueError("userProfile.version must be 1")
        return value

    # Duplicate entries are rejected rather than silently deduplicated: FeatureHistory.from_profile
    # keys categories by their upper-cased name and creators by their literal id (see
    # app/ml/dataset_builder.py), so a duplicate would either silently overwrite an earlier
    # entry (categories differing only by case) or signal an inconsistent upstream snapshot
    # (exact duplicates) -- neither should be scored as if it were valid, deterministic input.
    @model_validator(mode="after")
    def _no_duplicate_entries(self) -> "UserProfile":
        seen_categories: set[str] = set()
        for category_profile in self.categories:
            key = category_profile.category.strip().upper()
            if key in seen_categories:
                raise ValueError(f"duplicate category in userProfile.categories: {category_profile.category!r}")
            seen_categories.add(key)

        seen_creators: set[str] = set()
        for creator_profile in self.creators:
            if creator_profile.creator_id in seen_creators:
                raise ValueError(f"duplicate creatorId in userProfile.creators: {creator_profile.creator_id!r}")
            seen_creators.add(creator_profile.creator_id)

        if len(set(self.followed_creator_ids)) != len(self.followed_creator_ids):
            raise ValueError("userProfile.followedCreatorIds must not contain duplicates")
        if len(set(self.seen_content_ids)) != len(self.seen_content_ids):
            raise ValueError("userProfile.seenContentIds must not contain duplicates")

        seen_semantic: set[tuple[str, str]] = set()
        for entry in self.semantic_affinities:
            semantic_key = (entry.field, entry.token)
            if semantic_key in seen_semantic:
                raise ValueError(f"duplicate semantic entry in userProfile.semanticAffinities: {semantic_key!r}")
            seen_semantic.add(semantic_key)
        return self


# Finalization spec Decision 2/3: explicit, CURRENT-REQUEST/session-scoped search intent -- not
# part of UserProfile, never persisted by RMS. Matched generically (category/title/hashtags/
# topics/entities/subgenres token overlap, see app.ml.reranker.search_relevance) against every
# candidate; no per-entity logic anywhere in that matching.
class SearchIntent(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    query: str | None = Field(None)
    category: str | None = Field(None, max_length=64)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    subgenres: list[str] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0, le=1)

    @field_validator("query")
    @classmethod
    def _normalize_query(cls, value: str | None) -> str | None:
        return normalize_title(value)

    @field_validator("topics", "entities", "subgenres", mode="before")
    @classmethod
    def _normalize_field_tokens(cls, value: object, info: Any) -> list[str]:
        return normalize_tokens(value, field=info.field_name)


# Finalization spec Decision 11 (updated by the cohort cold-start finalization pass): `age`'s
# ONLY ranking influence anywhere in this codebase is indirect and bounded -- app.ml.reranker
# never reads raw age at all; app.services.recommendation_service derives a coarse age BUCKET
# from it (app.core.cohort_context.age_bucket_for) and uses that bucket, together with
# `region` below, to look up an already-aggregated, data-driven cohort preference profile
# (app.services.cohort_preference_provider) built from real historical interactions -- never a
# hardcoded demographic mapping. That signal is cold-start-only, bounded, shrunk toward the
# global prior for small samples, and weakens to zero as the user's own interaction history
# grows (see app.ml.reranker.COHORT_STRATEGY_WEIGHT). region/language separately also
# influence ranking directly via explicit candidate.language/candidate.regions compatibility
# (see app.ml.reranker.cold_start_context_relevance), also only while the user is a genuine
# cold start.
#
# `interests` (cold-start personalization upgrade, onboarding-interest task): explicit,
# user-declared topics/categories collected at onboarding -- e.g. a Feed/onboarding flow's
# "pick a few things you like" step. Request/session-scoped only: RMS does not own or persist
# `interests` itself (no DB column, no migration -- the caller supplies it fresh on whichever
# request wants it personalized). Unlike `interests`, `region`/the derived age bucket ARE
# opportunistically persisted (app.services.cohort_preference_provider.record_demographic_
# context, see the comment above this class) -- but only as cohort-aggregation CONTEXT, never
# merged into a user's long-term behavior profile and never read back as an "interest". Reuses
# `normalize_tokens` -- the EXACT validator app.schemas.recommendation_
# schemas.Candidate already applies to hashtags/topics/entities/subgenres -- so an onboarding
# interest and a candidate's own semantic token normalize identically ("Football" and
# "#FOOTBALL" both become "FOOTBALL"), and app.ml.reranker.explicit_interest_relevance can
# match on exact-token equality without a second, differently-behaved normalizer. Bounded by
# normalize_tokens' own existing MAX_TOKENS_PER_FIELD (15) -- the same cap already applied to
# Candidate.hashtags/topics/entities/subgenres -- not a free-form tag cloud.
class UserContext(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    age: int | None = Field(None, ge=0, le=120)
    region: str | None = Field(None, max_length=8)
    language: str | None = Field(None, max_length=16)
    interests: list[str] = Field(default_factory=list)

    @field_validator("region")
    @classmethod
    def _normalize_region(cls, value: str | None) -> str | None:
        return normalize_region(value)

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str | None) -> str | None:
        return value.strip().lower() if value and value.strip() else None

    @field_validator("interests", mode="before")
    @classmethod
    def _normalize_interests(cls, value: object) -> list[str]:
        return normalize_tokens(value, field="interests")


class RecommendationRequest(BaseModel):
    model_config=ConfigDict(populate_by_name=True)
    user_id: str=Field(alias="userId", min_length=1, max_length=128)
    limit: int=Field(RECOMMENDATION_LIMIT_DEFAULT, ge=RECOMMENDATION_LIMIT_MIN, le=RECOMMENDATION_LIMIT_MAX)
    # Optional (userId -> internal candidates integration): omitting/emptying candidates no
    # longer means "score nothing" -- app.services.providers.candidate_provider.resolve falls
    # back to local VIDEO candidate generation when this is empty. Explicit non-empty
    # candidates, when supplied, are still used unchanged and take precedence -- purely
    # additive for every existing caller.
    candidates: list[Candidate]=Field(default_factory=list, max_length=RECOMMENDATION_MAX_CANDIDATES)
    # Optional and additive: a request with only userId/candidates/limit (no userProfile at
    # all) behaves exactly as before -- the local-database legacy/demo path, unchanged. When
    # present, the recommendation service uses this snapshot instead and performs zero
    # interaction-table reads.
    user_profile: UserProfile|None=Field(None, alias="userProfile")
    # Request/session-scoped only (Decision 2/3/11): never merged into userProfile, never
    # persisted by RMS. Omitting either keeps behavior byte-identical to today.
    search_intent: SearchIntent|None=Field(None, alias="searchIntent")
    user_context: UserContext|None=Field(None, alias="userContext")

    @field_validator("user_id")
    @classmethod
    def _user_id_not_blank(cls, value: str) -> str:
        return _reject_blank(value, "userId")

