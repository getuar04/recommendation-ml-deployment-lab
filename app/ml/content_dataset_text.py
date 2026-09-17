"""Safe, deterministic text preprocessing for the Content Understanding dataset foundation.

Deliberately minimal and non-destructive: this is NOT `app.ml.content_classifier.feature_text`
(which uppercases and reduces title/hashtags to a bag of normalized tokens -- a classifier-
specific feature-space encoding). A dataset record's raw text is a durable artifact that may
outlive any one classifier's feature representation, so this module only does Unicode/
whitespace hygiene and explicitly preserves every signal a short-video title/caption carries:
hashtags, mentions, URLs, emoji, punctuation, case, and Albanian diacritics (e.g. e/E, c/C).

No language detection, no stemming, no synonym resolution, no heavyweight NLP dependency --
same bounded-scope philosophy as `app.ml.semantic_tokens`.
"""
from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_dataset_text(text: str | None) -> str | None:
    """Unicode NFC normalization (canonical composition -- e.g. a combining-mark-decomposed
    "e" + COMBINING DIAERESIS above becomes a single precomposed "e" codepoint, so visually
    identical Albanian/other text compares equal regardless of how it was typed/encoded) plus
    whitespace-run collapsing and edge trimming. Returns None for None/blank input, exactly
    like `app.ml.semantic_tokens.normalize_title`.

    Everything else is preserved verbatim: case, diacritics, hashtags, @mentions, URLs, emoji,
    punctuation. This is a hygiene pass, not a feature-engineering pass.
    """
    if text is None:
        return None
    composed = unicodedata.normalize("NFC", text)
    collapsed = _WHITESPACE_RUN.sub(" ", composed).strip()
    return collapsed or None
