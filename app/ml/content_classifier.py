"""Content-understanding: local category inference for content whose real upstream contract
(Content Service) supplies only contentId/creatorId/title(caption)/hashtags -- never a
category, topics, entities, or subgenres (see the event-driven architecture audit; no
Content Service repository/schema exists locally that proves otherwise). The active VIDEO
recommendation model's only categorical feature is `category`
(app.ml.dataset_builder.CATEGORICAL); this module exists to derive it locally instead of
requiring an upstream field that will not be sent, WITHOUT retraining or touching that model.

Approach chosen (see the accompanying audit report for the full evaluation of alternatives):
a small, deterministic TF-IDF + LogisticRegression multi-class text classifier over
normalized title tokens + hashtags (reusing `app.ml.semantic_tokens`'s existing, already-
tested normalization -- never a second tokenizer). Rejected: (a) a pretrained sentence-
embedding/zero-shot model -- this project has zero NLP/embedding dependencies today
(requirements.txt: no transformers/sentence-transformers/torch as a hard dependency) and
adding one is not justified without first proving a much simpler, already-available approach
(plain scikit-learn, already a hard dependency) is insufficient; (b) a hardcoded keyword-to-
category lookup -- explicitly disallowed by the task, and would not degrade gracefully or
produce a calibrated confidence the way a trained probabilistic classifier does.

Category taxonomy (`CATEGORIES` below, versioned via `CATEGORY_TAXONOMY_VERSION`): reused
verbatim from the only concrete category vocabulary that exists anywhere in this repository
today (`scripts/generate_synthetic_data.py` / `app.experiments.definitions.
DEFAULT_CATEGORIES`) -- itself a synthetic-data artifact used to seed this project's own demo/
experiment database, not a declared product taxonomy. THIS TAXONOMY STILL REQUIRES REAL
PRODUCT/DOMAIN CONFIRMATION -- it is reused here (not invented from nothing) specifically so
the architecture has an explicit, versioned, swappable taxonomy from day one rather than a
silently-arbitrary one; replacing it later (a new `CATEGORY_TAXONOMY_VERSION`) requires
retraining this classifier, never the active VIDEO recommendation model, whose OneHotEncoder
(app.ml.pipeline_builder, `handle_unknown="ignore"`) already tolerates any category value it
has never seen -- including a full taxonomy swap -- safely (see that module's own docstring).

Creator history is supporting evidence only, never primary: see
`combine_with_creator_prior`'s own docstring for the exact precedence rule.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml.semantic_tokens import extract_title_tokens, normalize_tokens

CATEGORY_TAXONOMY_VERSION = "v1-bootstrap"
# Reused verbatim from scripts/generate_synthetic_data.py / app.experiments.definitions --
# see module docstring: NOT a confirmed product taxonomy yet.
CATEGORIES: tuple[str, ...] = ("FOOD", "SPORT", "MUSIC", "TECH", "GAMING", "TRAVEL", "COMEDY", "NEWS", "FASHION", "FITNESS")
UNKNOWN_CATEGORY = "UNKNOWN"

RANDOM_SEED = 42


@dataclass(frozen=True)
class ContentClassification:
    """`source` is one of "MODEL" (current-content evidence alone, either confident enough to
    skip the creator prior entirely, or with no usable creator prior available),
    "MODEL_WITH_CREATOR_PRIOR" (current-content evidence blended with the creator's
    historical category distribution), or "UNKNOWN" (confidence too low either way --
    never a fabricated guess).

    `taxonomy_version` (Task: taxonomy/version compatibility infrastructure): always
    `CATEGORY_TAXONOMY_VERSION` -- every classification this module produces, `UNKNOWN`
    included, comes from running THIS classifier's vocabulary/pipeline, so it is accurately
    described by the same version regardless of which branch produced it. Additive-only field;
    does not change which category/confidence/source is chosen."""
    category: str
    confidence: float
    source: str
    taxonomy_version: str = CATEGORY_TAXONOMY_VERSION


def feature_text(title: str | None, hashtags: list[str] | None) -> str:
    """Single deterministic text representation fed to the classifier. Reuses the EXACT
    same title/hashtag normalization content ingestion and the VIDEO semantic-affinity
    pipeline already use (app.ml.semantic_tokens) -- never a second, parallel tokenizer, so
    a real title/hashtag and this classifier's training data are normalized identically."""
    title_tokens = extract_title_tokens(title)
    hashtag_tokens = normalize_tokens(hashtags or [])
    return " ".join([*title_tokens, *hashtag_tokens])


def build_pipeline(random_seed: int = RANDOM_SEED) -> Pipeline:
    """Unigrams, not bigrams: measured directly against this bootstrap dataset's actual size
    (~150 rows) -- bigram features are so sparse across ~11-18 examples per category that
    they fragment the signal and systematically under-confidence every prediction (measured:
    average held-out max-probability ~0.90 with unigrams vs. ~0.92 with a much higher C
    needed just to reach the SAME confidence with bigrams added, and still lower per-example
    confidence on both a verbatim training example and novel-but-on-topic wording). C=200
    (vs. scikit-learn's default 1.0) was likewise chosen by directly measuring predicted-
    probability calibration on this dataset: at C=1 a verbatim training example scores
    ~0.14 confidence (systematically under-confident, since the classes are in fact cleanly
    separable -- held-out accuracy is 1.0 at every C tried); C=200 produces well-separated
    confidence between clearly-on-topic text (measured ~0.96) and genuinely ambiguous/
    nonsense text (measured ~0.15-0.19), which is what `combine_with_creator_prior`'s
    LOW/HIGH confidence thresholds actually depend on to distinguish the two. Not a generic
    default -- re-measure both choices if the training dataset's size/shape changes
    materially (e.g. once real labeled data replaces/supplements the synthetic bootstrap
    set)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, C=200.0, random_state=random_seed)),
    ])


_DEFAULT_SYNTHETIC_ONLY_DATASET_SOURCE: dict[str, Any] = {
    "type": "synthetic-bootstrap",
    "synthetic": True,
    "note": (
        "Deterministic, hand-authored bootstrap dataset (app.ml.content_classifier_data) "
        "-- NOT real labeled production data, and has no creator dimension so creator/"
        "near-duplicate leakage-resistant splitting does not apply to it. Sufficient to "
        "prove the classification pipeline works end-to-end; must be replaced or "
        "supplemented with real labeled content before this metric is presented as "
        "production evidence."
    ),
}


def train_classifier(
    dataset: pd.DataFrame, *, random_seed: int = RANDOM_SEED, dataset_source: dict[str, Any] | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """`dataset` columns: `text` (already run through `feature_text`), `category`. Stratified
    holdout split for an honest accuracy estimate. Creator/near-duplicate leakage-resistant
    splitting (required for any FUTURE real dataset -- see the audit report) does not apply
    to `app.ml.content_classifier_data`'s bootstrap set, which has no creator dimension at
    all; this is called out explicitly in `datasetSource` below rather than silently
    presenting a stratified-random-split accuracy as more rigorous than it is.

    `dataset_source` (classifier release-boundary/dataset-integration audit): describes what
    `dataset` actually contains, for the saved metadata's `datasetSource` field. Defaults to
    the historical "always synthetic-bootstrap" claim ONLY when omitted -- every existing
    caller that still passes a bare synthetic-only DataFrame (e.g.
    `scripts/train_content_classifier.py` before this integration, and any test that builds
    `dataset` from `app.ml.content_classifier_data.build_dataframe()` directly) keeps
    producing byte-identical metadata. A caller that trains on a real/mixed dataset (see
    `app.ml.content_classifier_dataset_source.build_training_dataframe`) MUST pass its own
    honestly-computed `dataset_source` -- this function never inspects `dataset` itself to
    guess composition, so it can never silently mislabel a real/mixed dataset as
    synthetic-only (or vice versa)."""
    if dataset["category"].nunique() < 2:
        raise ValueError("training dataset must contain at least 2 distinct categories")
    train_df, test_df = train_test_split(
        dataset, test_size=0.25, random_state=random_seed, stratify=dataset["category"],
    )
    pipeline = build_pipeline(random_seed)
    pipeline.fit(train_df["text"], train_df["category"])
    test_predictions = pipeline.predict(test_df["text"])
    metadata = {
        "categoryTaxonomyVersion": CATEGORY_TAXONOMY_VERSION,
        "categories": list(pipeline.named_steps["clf"].classes_),
        "trainingSamples": len(train_df),
        "testSamples": len(test_df),
        "holdoutAccuracy": float(accuracy_score(test_df["category"], test_predictions)),
        "datasetSource": dataset_source if dataset_source is not None else dict(_DEFAULT_SYNTHETIC_ONLY_DATASET_SOURCE),
        "randomSeed": random_seed,
    }
    return pipeline, metadata


def save_classifier(pipeline: Pipeline, metadata: dict[str, Any], *, model_path: Path, metadata_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_classifier(*, model_path: Path, metadata_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return pipeline, metadata


def classify_text(pipeline: Pipeline, title: str | None, hashtags: list[str] | None) -> dict[str, float]:
    """{category: probability} for every class the pipeline was trained on. Returns {} for
    blank/unusable text (no title, no hashtags) -- there is nothing for the classifier to
    read, and `combine_with_creator_prior` treats an empty dict as UNKNOWN, never a guess."""
    text = feature_text(title, hashtags)
    if not text:
        return {}
    probabilities = pipeline.predict_proba([text])[0]
    classes = pipeline.named_steps["clf"].classes_
    return {category: float(probability) for category, probability in zip(classes, probabilities)}


def combine_with_creator_prior(
    current_probs: dict[str, float], creator_category_counts: dict[str, int], *,
    low_confidence_threshold: float, high_confidence_threshold: float,
    creator_prior_weight: float, creator_profile_min_samples: int,
) -> ContentClassification:
    """Current-content evidence (title + hashtags of THIS piece of content) is primary and
    is never overridden by creator history: a creator who usually posts one category can
    still publish something different, and this function must reflect that when the current
    content clearly says so.

    Precedence, in order:
    1. No usable current-content evidence at all (`current_probs` empty) -> UNKNOWN,
       regardless of creator history: a creator prior alone is never enough to assert what a
       SPECIFIC piece of content is about.
    2. Current-content confidence >= `high_confidence_threshold` -> use it alone, source
       "MODEL". Strong evidence from the current content is never diluted by a creator prior.
    3. Otherwise, if the creator has at least `creator_profile_min_samples` historical rows,
       blend the current-content distribution with the creator's historical category
       distribution (weighted `creator_prior_weight` toward the creator prior, so current
       content still dominates the blend) and re-score from the blended distribution --
       source "MODEL_WITH_CREATOR_PRIOR" if the blended confidence still clears
       `low_confidence_threshold`, else UNKNOWN.
    4. Otherwise (no usable creator prior): use current-content evidence alone if it clears
       `low_confidence_threshold` (source "MODEL"), else UNKNOWN -- never fabricate certainty
       for genuinely ambiguous content just because there is no creator history to fall back on.
    """
    if not current_probs:
        return ContentClassification(UNKNOWN_CATEGORY, 0.0, "UNKNOWN")

    top_category = max(current_probs, key=lambda category: current_probs[category])
    top_confidence = current_probs[top_category]

    if top_confidence >= high_confidence_threshold:
        return ContentClassification(top_category, top_confidence, "MODEL")

    total_creator_rows = sum(creator_category_counts.values())
    if total_creator_rows >= creator_profile_min_samples:
        creator_probs = {category: count / total_creator_rows for category, count in creator_category_counts.items()}
        blended = {
            category: (1 - creator_prior_weight) * current_probs.get(category, 0.0)
            + creator_prior_weight * creator_probs.get(category, 0.0)
            for category in set(current_probs) | set(creator_probs)
        }
        blended_category = max(blended, key=lambda category: blended[category])
        blended_confidence = blended[blended_category]
        if blended_confidence >= low_confidence_threshold:
            return ContentClassification(blended_category, blended_confidence, "MODEL_WITH_CREATOR_PRIOR")
        return ContentClassification(UNKNOWN_CATEGORY, blended_confidence, "UNKNOWN")

    if top_confidence >= low_confidence_threshold:
        return ContentClassification(top_category, top_confidence, "MODEL")
    return ContentClassification(UNKNOWN_CATEGORY, top_confidence, "UNKNOWN")
