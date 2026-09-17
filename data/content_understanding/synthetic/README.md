# Synthetic bootstrap dataset -- pointer, not a copy

The repository's only existing content-classification dataset (~150 hand-authored rows,
10-category `v1-bootstrap` taxonomy) lives in code at
`app/ml/content_classifier_data.py`, NOT in this directory. It is intentionally not
duplicated here so there is exactly one source of truth.

Provenance (already explicit in that module and reused here for consistency with this
dataset foundation's vocabulary):

- **Label source:** `SYNTHETIC` (see `app.ml.content_dataset_schema.LabelSource`).
- **Taxonomy version:** `v1-bootstrap` (`app.ml.content_classifier.CATEGORY_TAXONOMY_VERSION`).
- **Purpose:** proves the classification pipeline (text normalization -> TF-IDF ->
  LogisticRegression -> confidence threshold -> creator-prior blending) works end-to-end.
  **Not real labeled production data**, and must never be reported as such -- see
  `train_classifier`'s own `datasetSource` metadata, which carries this caveat into every
  trained artifact.

This dataset foundation does not change, retrain, or re-label this dataset.
