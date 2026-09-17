"""Exact normalized-duplicate detection for the Content Understanding dataset foundation
(Part 9 of the dataset-foundation task) -- prevents the same or trivially-reformatted caption
from silently appearing in both a future train split and a future evaluation split.

Only EXACT (post-normalization) duplicates are handled here. Near-duplicate detection
(paraphrases, reordered hashtags with different wording, embedding-similarity clustering) is
explicitly OUT OF SCOPE for this task -- see `NEAR_DUPLICATE_STRATEGY_NOTE` below for the
documented future direction; introducing an embedding system now would be exactly the
complexity this task's non-goals rule out.
"""
from __future__ import annotations

import hashlib
import unicodedata

from app.ml.content_dataset_text import normalize_dataset_text

NEAR_DUPLICATE_STRATEGY_NOTE = (
    "Not implemented in this task. Future direction: once a real embedding/sentence-similarity "
    "dependency is justified (it is not today -- see app.ml.content_classifier's own module "
    "docstring on why this repo has zero such dependencies), compute a similarity score between "
    "normalized title+hashtag text and cluster pairs above a chosen threshold for manual review "
    "before any train/eval split is finalized. Until then, exact-fingerprint matching "
    "(compute_content_fingerprint) is the only automated leakage defense."
)


def _fold_diacritics(text: str) -> str:
    """Fingerprint-only aggressive folding (NFKD decomposition, drop combining marks, e.g.
    Albanian "e" -> "e"): deliberately more aggressive than
    `app.ml.content_dataset_text.normalize_dataset_text`, which preserves diacritics for
    display/storage. Two captions that a person would call "the same text" should collide on
    a fingerprint even if one used precomposed and the other decomposed/plain-ASCII spelling;
    that same folding would be lossy and wrong for the stored record itself."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def compute_content_fingerprint(title: str | None, hashtags: list[str] | None) -> str:
    """Deterministic fingerprint over normalized title + hashtag set. Hashtag order does not
    affect the fingerprint (a set, not a sequence -- the same tags in a different order are
    still the same content); title text does (title text order matters for meaning).

    Returns a `sha256:<hex>` string. Two records with the same fingerprint are exact
    normalized duplicates; the caller decides what to do about it (see
    `app.ml.content_dataset_validator`, which flags conflicting labels on a shared
    fingerprint as an error)."""
    normalized_title = normalize_dataset_text(title) or ""
    folded_title = _fold_diacritics(normalized_title).casefold()

    folded_hashtags = sorted(
        {_fold_diacritics(normalize_dataset_text(tag) or "").casefold() for tag in (hashtags or [])} - {""}
    )

    payload = f"title:{folded_title}|hashtags:{','.join(folded_hashtags)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
