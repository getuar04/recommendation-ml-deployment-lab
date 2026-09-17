"""Legacy VIDEO ranking-model category-compatibility projection (Task: additive taxonomy/
version compatibility foundation).

ONE centralized place a future canonical `primaryCategory` value would be translated into the
category token the CURRENTLY-ACTIVE VIDEO model's `OneHotEncoder` (app.ml.pipeline_builder,
`handle_unknown="ignore"`) was actually trained on -- so this translation logic is never
independently re-derived/duplicated across app.ml.dataset_builder, app.ml.feature_builder,
app.services.providers.video_candidate_provider, or app.ml.reranker.

NOT WIRED INTO ANY OF THOSE CONSUMERS YET, deliberately: this task adds infrastructure only.
Every existing consumer continues reading `Content.category` directly, unchanged (see this
task's own report, section F). This module exists so that a FUTURE consumer switch has exactly
one place to call, and so that a FUTURE approved canonical->legacy mapping has exactly one
place to be registered, instead of being hand-copied into each of those modules independently.

Contains NO unapproved category mappings (e.g. a future FASHION_BEAUTY -> FASHION merge) -- the
mapping registry below starts empty. Populate a taxonomy version's mapping only once product
has actually approved it (see the taxonomy stress-test / decision-package reports); never guess.
"""
from __future__ import annotations

__all__ = [
    "LEGACY_MODEL_UNKNOWN_CATEGORY",
    "LEGACY_VIDEO_MODEL_CATEGORY_VOCABULARY",
    "project_to_legacy_model_category",
    "register_legacy_mapping",
]

# The exact 10-category vocabulary the currently-active VIDEO model's OneHotEncoder was
# trained on (app.ml.content_classifier.CATEGORIES / app.experiments.definitions.
# DEFAULT_CATEGORIES -- reused verbatim, not re-derived, so this can never silently drift from
# the one real vocabulary confirmed against the live model artifact in a prior audit).
LEGACY_VIDEO_MODEL_CATEGORY_VOCABULARY: frozenset[str] = frozenset({
    "FOOD", "SPORT", "MUSIC", "TECH", "GAMING", "TRAVEL", "COMEDY", "NEWS", "FASHION", "FITNESS",
})

# Mirrors app.ml.content_classifier.UNKNOWN_CATEGORY -- the same "no usable classification"
# value, reused here for "no usable legacy-model projection" for the identical reason: never a
# fabricated guess, and OneHotEncoder(handle_unknown="ignore") already treats an unrecognized
# string as a safe all-zero categorical feature regardless of which unrecognized string it is.
LEGACY_MODEL_UNKNOWN_CATEGORY = "UNKNOWN"

# {taxonomy_version: {canonical_primary_category: legacy_model_category}}. Deliberately empty:
# no taxonomy version's canonical->legacy mapping has been product-approved yet.
_VERSIONED_MAPPINGS: dict[str, dict[str, str]] = {}


def register_legacy_mapping(taxonomy_version: str, mapping: dict[str, str]) -> None:
    """Registers (or replaces) the approved canonical->legacy mapping for one taxonomy
    version. Not called anywhere in this codebase yet -- exists so a future, product-approved
    mapping has exactly one place to be registered, instead of being hand-copied into
    dataset_builder/feature_builder/video_candidate_provider/reranker independently. Stores a
    defensive copy so the caller's own dict can be mutated afterward without affecting the
    registry."""
    _VERSIONED_MAPPINGS[taxonomy_version] = dict(mapping)


def project_to_legacy_model_category(
    primary_category: str | None, *, taxonomy_version: str | None = None,
) -> str:
    """Returns the category token the ACTIVE VIDEO model's `OneHotEncoder` should receive for
    a piece of content described by `primary_category` (a future canonical value) and, when
    known, the `taxonomy_version` it belongs to.

    Precedence:
    1. `primary_category` is already a legacy-vocabulary token (identity projection) -- the
       only case that can occur today, since nothing yet writes a genuinely new canonical
       value anywhere in this codebase.
    2. A mapping was registered for `taxonomy_version` and contains `primary_category` --
       version-aware translation. Currently always a miss: the registry starts empty.
    3. Otherwise: `LEGACY_MODEL_UNKNOWN_CATEGORY` -- exactly the same safe fallback
       `OneHotEncoder(handle_unknown="ignore")` already provides today for any unrecognized
       category string (see app.ml.pipeline_builder's own docstring) -- never a crash, never a
       fabricated guess.

    `primary_category=None` also returns `LEGACY_MODEL_UNKNOWN_CATEGORY`: "no canonical
    category exists yet for this content" is exactly as uninformative to the legacy model as
    an unrecognized one.
    """
    if primary_category is None:
        return LEGACY_MODEL_UNKNOWN_CATEGORY
    if primary_category in LEGACY_VIDEO_MODEL_CATEGORY_VOCABULARY:
        return primary_category
    if taxonomy_version is not None:
        mapping = _VERSIONED_MAPPINGS.get(taxonomy_version)
        if mapping and primary_category in mapping:
            return mapping[primary_category]
    return LEGACY_MODEL_UNKNOWN_CATEGORY
