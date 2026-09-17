"""Record contract for the Content Understanding DATASET/LABELING foundation.

This is infrastructure for accumulating and validating training/evaluation examples for a
future content classifier -- it is NOT the classifier itself, does not train anything, and
does not touch `app.ml.content_classifier` / `app.ml.content_classifier_data` (the existing
synthetic bootstrap dataset -- see that module's own docstring for its provenance).

No final taxonomy is encoded here. `primary_category`/`subcategory` are free strings with no
enum, exactly like the storage-layer contract in `app.db.models.Content` (see that module's
comment) -- validation in this file is structural/provenance-based only (Part 13 of the
dataset-foundation task), never against a proposed category list.
"""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ml.semantic_tokens import (
    MAX_TITLE_LENGTH,
    MAX_TOKEN_LENGTH,
    MAX_TOKENS_PER_FIELD,
    normalize_title,
)


class DatasetRecordError(ValueError):
    """A dataset record violates the structural/provenance contract below. Raised as
    ValueError so it surfaces as a normal pydantic validation error, never an uncaught
    exception."""


class LabelSource(str, Enum):
    """Provenance of a record's semantic label(s). Never mixed into one undifferentiated
    pool -- every consumer (training, evaluation, reporting) must branch on this, not assume
    uniform quality. See `app.ml.content_dataset_eligibility` for what each source is actually
    allowed to be used for."""

    HUMAN_REVIEWED = "HUMAN_REVIEWED"
    SYNTHETIC = "SYNTHETIC"
    PSEUDO_LABELED = "PSEUDO_LABELED"
    WEAK_SUPERVISION = "WEAK_SUPERVISION"


class ReviewStatus(str, Enum):
    """Small controlled review lifecycle -- not an annotation platform, just enough state to
    stop unreviewed/pseudo data from silently entering a human-reviewed evaluation set."""

    UNREVIEWED = "UNREVIEWED"
    REVIEWED = "REVIEWED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    DISPUTED = "DISPUTED"


# Language contract (Part 7): a lowercase BCP-47-style primary tag ("sq", "en", "sq-al", ...)
# or one of two sentinel values below. Deliberately NOT a closed enum of just
# {sq, en, mixed, unknown} -- the classifier must support at least Albanian/English/mixed, but
# the schema should not hardcode a fixed language universe. No language detector is
# implemented or implied here; callers supply this value, it is only normalized/validated.
LANGUAGE_MIXED = "mixed"
LANGUAGE_UNKNOWN = "unknown"
_LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$")


def normalize_language_tag(raw: str | None) -> str | None:
    """None means "not recorded" (real data may legitimately lack a language label -- Part 3).
    "" / whitespace-only also normalizes to None. Any other value must be `mixed`, `unknown`,
    or a BCP-47-shaped primary tag (lowercased); anything else raises."""
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value in (LANGUAGE_MIXED, LANGUAGE_UNKNOWN):
        return value
    if not _LANGUAGE_TAG_RE.match(value):
        raise DatasetRecordError(
            f"language {raw!r} is not a recognized language tag, {LANGUAGE_MIXED!r}, or {LANGUAGE_UNKNOWN!r}."
        )
    return value


def _validate_raw_tag_list(values: object, *, field: str) -> list[str]:
    """Bounds/shape validation shared with `app.ml.semantic_tokens.normalize_tokens`
    (same MAX_TOKENS_PER_FIELD/MAX_TOKEN_LENGTH bounds, same strict-shape/blank/type checks),
    but deliberately does NOT uppercase or collapse punctuation the way that classifier-
    feature-space normalizer does: a dataset record preserves hashtags/topics as authored
    (case, diacritics, punctuation) so it is not tied to one specific future feature
    representation. Only exact-string whitespace-trimmed duplicates are dropped."""
    if values is None:
        return []
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise DatasetRecordError(f"{field} must be a list of strings (got {type(values).__name__}).")
    if len(values) > MAX_TOKENS_PER_FIELD:
        raise DatasetRecordError(f"{field} must not exceed {MAX_TOKENS_PER_FIELD} entries (got {len(values)}).")
    cleaned: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            raise DatasetRecordError(f"{field} entries must be strings (got {type(raw).__name__}: {raw!r}).")
        stripped = raw.strip()
        if not stripped:
            raise DatasetRecordError(f"{field} entries must not be blank.")
        if len(stripped) > MAX_TOKEN_LENGTH:
            raise DatasetRecordError(f"{field} entry {stripped!r} exceeds the maximum length of {MAX_TOKEN_LENGTH}.")
        if stripped not in cleaned:
            cleaned.append(stripped)
    return cleaned


class DatasetRecord(BaseModel):
    """One Content Understanding training/evaluation example.

    Field groups:
      identity        -- example_id (required, dataset-row primary key), content_id,
                          creator_id (Part 10: preserved for future known/unseen-creator
                          split, never used as a text-classifier feature by this module).
      content          -- content_type, title, hashtags, language, content_created_at
                          (Part 11: preserved for future chronological holdout; this module
                          never assigns a split itself).
      label            -- primary_category, subcategory, topics, taxonomy_version,
                          label_source, label_confidence.
      review provenance -- review_status, reviewer_id, reviewed_at, notes (Part 4/5).
      dataset metadata -- dataset_split (structural only -- never populated by this module,
                          see Part 11), source_metadata (free-form, e.g. ingestion batch id).

    Every field except example_id/label_source is optional: real data may legitimately lack
    any of them (Part 3), and requiring them would make this contract unusable for genuine
    partial records.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    example_id: str = Field(..., alias="exampleId", min_length=1, max_length=128)
    content_id: str | None = Field(None, alias="contentId", max_length=128)
    creator_id: str | None = Field(None, alias="creatorId", max_length=128)
    content_type: str | None = Field(None, alias="contentType", max_length=32)

    title: str | None = Field(None, max_length=MAX_TITLE_LENGTH)
    hashtags: list[str] = Field(default_factory=list)
    language: str | None = Field(None, max_length=32)
    content_created_at: datetime | None = Field(None, alias="contentCreatedAt")

    primary_category: str | None = Field(None, alias="primaryCategory", max_length=64)
    subcategory: str | None = Field(None, max_length=64)
    topics: list[str] = Field(default_factory=list)
    taxonomy_version: str | None = Field(None, alias="taxonomyVersion", max_length=32)
    label_source: LabelSource = Field(..., alias="labelSource")
    label_confidence: float | None = Field(None, alias="labelConfidence", ge=0.0, le=1.0)

    review_status: ReviewStatus = Field(ReviewStatus.UNREVIEWED, alias="reviewStatus")
    reviewer_id: str | None = Field(None, alias="reviewerId", max_length=64)
    reviewed_at: datetime | None = Field(None, alias="reviewedAt")
    notes: str | None = Field(None, max_length=2000)

    dataset_split: str | None = Field(None, alias="datasetSplit", max_length=32)
    source_metadata: dict[str, Any] | None = Field(None, alias="sourceMetadata")

    @property
    def normalized_title(self) -> str | None:
        return normalize_title(self.title)

    @property
    def normalized_language(self) -> str | None:
        return normalize_language_tag(self.language)

    @model_validator(mode="after")
    def _validate(self) -> DatasetRecord:
        normalize_title(self.title)
        normalize_language_tag(self.language)

        if not (self.title and self.title.strip()) and not self.hashtags:
            raise DatasetRecordError(
                f"record {self.example_id!r} has no usable text signal (title or hashtags required)."
            )

        has_label = bool(self.primary_category or self.subcategory)
        if has_label and not self.taxonomy_version:
            raise DatasetRecordError(
                f"record {self.example_id!r} has a primary_category/subcategory but no taxonomyVersion -- "
                "every semantic label must be tied to a taxonomy version."
            )

        if self.review_status is ReviewStatus.REVIEWED and (not self.reviewer_id or not self.reviewed_at):
            raise DatasetRecordError(
                f"record {self.example_id!r} has reviewStatus=REVIEWED but is missing reviewerId/reviewedAt -- "
                "REVIEWED must mean genuinely reviewed, not merely a string in the label column."
            )

        if (
            self.label_source is LabelSource.HUMAN_REVIEWED
            and self.review_status is ReviewStatus.REVIEWED
            and not self.primary_category
        ):
            raise DatasetRecordError(
                f"record {self.example_id!r} is HUMAN_REVIEWED and REVIEWED but has no primaryCategory -- "
                "ambiguous content must be flagged NEEDS_REVIEW/DISPUTED, not marked REVIEWED with a blank label."
            )

        return self

    @model_validator(mode="before")
    @classmethod
    def _validate_tag_lists(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field in ("hashtags", "topics"):
                if field in data:
                    data[field] = _validate_raw_tag_list(data[field], field=field)
        return data
