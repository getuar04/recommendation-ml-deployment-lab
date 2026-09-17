"""Single normalization contract for candidate semantic metadata (title/hashtags/topics/
entities/subgenres), shared byte-for-byte by request-time validation
(app.schemas.recommendation_schemas, app.api.content_routes) and feature construction
(app.ml.dataset_builder). This is what the training/inference-parity requirement depends on:
if any caller normalized tokens differently, the same real-world tag ("#Messi" vs "Messi" vs
"messi") would silently become two different features in training and in serving.

Normalization is intentionally simple and deterministic: uppercase, non-alphanumeric runs
collapsed to a single underscore, leading/trailing underscores stripped. No stemming, no
synonym resolution -- a bounded, explicit design tradeoff, not an NLP pipeline.
"""
from __future__ import annotations

import re

MAX_TITLE_LENGTH = 200
MAX_TOKENS_PER_FIELD = 15
MAX_TOKEN_LENGTH = 40
# Title-token extraction bounds (app.ml.dataset_builder folds these into the same
# hashtag/topic/entity/subgenre semantic-affinity aggregation -- see extract_title_tokens()).
MAX_TITLE_TOKENS = 8

_NON_ALNUM_RUN = re.compile(r"[^A-Za-z0-9]+")
_WORD_SPLIT = re.compile(r"[^A-Za-z0-9]+")

# Low-information words dropped from title tokenization. Deliberately small and generic --
# never a content-specific word (a team/artist/genre name), which would make title-token
# extraction a disguised per-title rule instead of a generic linguistic filter.
#
# Two categories: grammatical connectives (AND/THE/OF/...), and generic superlative/
# descriptor words that show up in titles across every domain (sports "Best/Road/Final",
# music "Live/Highlights", any recap-style title's "New/Top/Moments/Greatest") and would
# otherwise become a semantic match purely from being a common title word, independent of
# what the title is actually about.
TITLE_STOPWORDS = frozenset({
    "AND", "THE", "OF", "A", "AN", "TO", "IN", "ON", "WITH", "FOR", "AT", "BY", "IS", "IT",
    "AS", "OR", "FROM", "VS",
    "BEST", "ROAD", "FINAL", "NEW", "TOP", "LIVE", "HIGHLIGHTS", "MOMENTS", "GREATEST",
})


class SemanticMetadataError(ValueError):
    """A semantic metadata field (title/hashtags/topics/entities/subgenres) violates the
    bounded-payload contract: too many tokens, a blank token, or a token that is too long.
    Raised as ValueError so pydantic field validators surface it as a normal 422 without any
    extra wiring."""


def normalize_token(raw: str) -> str:
    """"heavy-metal" -> "HEAVY_METAL", "#Messi" -> "MESSI", "Lionel Messi" ->
    "LIONEL_MESSI", "FC Barcelona" -> "FC_BARCELONA". Returns "" for input that normalizes
    to nothing (e.g. only punctuation) -- callers that must reject blanks check for that."""
    if raw is None:
        return ""
    cleaned = raw.strip().lstrip("#")
    cleaned = _NON_ALNUM_RUN.sub("_", cleaned).strip("_")
    return cleaned.upper()


def normalize_tokens(values: object, *, field: str = "tokens") -> list[str]:
    """Normalize, deduplicate (order-preserving), and bounds-check a raw token list.

    Strict about shape, not just content: `values` must be a real list/tuple (a JSON array),
    and every entry must be a `str` -- a bare string is never treated as an iterable-of-
    characters, a non-list payload never reaches Python's `len()`/iteration and crashes with
    an uncaught TypeError, and a non-string entry (int, bool, dict, nested list, None) is
    rejected instead of being silently `str()`-coerced into a token. Every rejection raises
    `SemanticMetadataError` (a `ValueError`), which pydantic turns into a normal 422, never a
    500.

    Also enforces: max MAX_TOKENS_PER_FIELD entries, no blank entries (before or after
    normalization), max MAX_TOKEN_LENGTH per normalized token. Deduplication happens after
    normalization so "messi" and "#Messi" in the same list count once, so duplicate
    hashtags/topics do not artificially inflate affinity.
    """
    if values is None:
        return []
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise SemanticMetadataError(
            f"{field} must be a list of strings (got {type(values).__name__})."
        )
    if not values:
        return []
    if len(values) > MAX_TOKENS_PER_FIELD:
        raise SemanticMetadataError(f"{field} must not exceed {MAX_TOKENS_PER_FIELD} entries (got {len(values)}).")
    seen: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            raise SemanticMetadataError(f"{field} entries must be strings (got {type(raw).__name__}: {raw!r}).")
        if not raw.strip():
            raise SemanticMetadataError(f"{field} entries must not be blank.")
        token = normalize_token(raw)
        if not token:
            raise SemanticMetadataError(f"{field} entry {raw!r} normalizes to an empty token.")
        if len(token) > MAX_TOKEN_LENGTH:
            raise SemanticMetadataError(f"{field} token {token!r} exceeds the maximum length of {MAX_TOKEN_LENGTH}.")
        if token not in seen:
            seen.append(token)
    return seen


def normalize_title(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    if len(trimmed) > MAX_TITLE_LENGTH:
        raise SemanticMetadataError(f"title must not exceed {MAX_TITLE_LENGTH} characters (got {len(trimmed)}).")
    return trimmed


def extract_title_tokens(title: str | None) -> list[str]:
    """Deterministic word-level tokenization of a free-text title into the same normalized
    token alphabet `normalize_token` produces for hashtags/topics/entities/subgenres, so a
    title word and an identically-spelled hashtag ("Messi" the title word vs. "MESSI" the
    hashtag) land on the exact same feature-space token.

    "Messi Master Class Live" -> ["MESSI", "MASTER", "CLASS"] ("live" is a stopword).
    "Messi and Argentina Win the World Cup" -> ["MESSI", "ARGENTINA", "WIN", "WORLD", "CUP"]
    ("and"/"the" are stopwords).

    Used identically by content ingestion (app.api.content_routes, for bounds validation),
    training dataset construction and historical profile construction
    (app.ml.dataset_builder._token_fields), and online candidate scoring
    (app.services.recommendation_service) -- one implementation, so train/serve title
    tokenization can never drift.

    Bounded and deterministic: left-to-right first-occurrence order, deduplicated, capped at
    MAX_TITLE_TOKENS tokens, each token capped at MAX_TOKEN_LENGTH characters (truncated, not
    rejected -- this is an internal feature-engineering bound on free text, not a payload
    validation rule; MAX_TITLE_LENGTH already bounds the raw payload in normalize_title).
    Returns [] for a blank/missing title, exactly like a candidate with no title at all.
    """
    if not title:
        return []
    tokens: list[str] = []
    for word in _WORD_SPLIT.split(title.strip()):
        if not word:
            continue
        token = word.upper()
        if token in TITLE_STOPWORDS:
            continue
        if len(token) > MAX_TOKEN_LENGTH:
            token = token[:MAX_TOKEN_LENGTH]
        if token not in tokens:
            tokens.append(token)
        if len(tokens) >= MAX_TITLE_TOKENS:
            break
    return tokens
