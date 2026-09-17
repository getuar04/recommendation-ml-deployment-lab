# Content Understanding dataset foundation

Infrastructure for accumulating and validating real, human-reviewed training/evaluation
examples for a future Content Understanding classifier. This directory is the DATASET/
LABELING foundation only -- it does not contain a classifier, does not train anything, and
does not encode a final category taxonomy.

## Status

- **Real, human-reviewed data: NOT AVAILABLE.** `real/dataset.jsonl` is currently empty (0
  records, see `real/manifest.json`). No real Content Service integration and no human
  annotation process exist yet. This is expected and honest, not a gap to paper over.
- **Final taxonomy: NOT APPROVED.** No category enum is encoded anywhere in this directory
  or in the code under `app/ml/content_dataset_*.py`.
- **Existing 150-row synthetic classifier dataset:** lives in code, at
  `app/ml/content_classifier_data.py` -- not duplicated here. It is explicitly labeled
  synthetic/bootstrap in its own module docstring and in the `datasetSource` metadata written
  alongside every trained classifier artifact (`app.ml.content_classifier.train_classifier`).
  See `synthetic/README.md` in this directory for a pointer, not a copy.

## Layout

```
data/content_understanding/
  README.md                 -- this file
  SCHEMA.md                 -- the record contract, with an illustrative (non-training) example
  LABELING_GUIDELINES.md    -- guidelines for future human annotators
  real/
    dataset.jsonl            -- the real, human-reviewed dataset. Currently 0 records.
    manifest.json            -- honest manifest for the file above (record_count: 0 today)
  synthetic/
    README.md                -- pointer to the existing bootstrap dataset (app/ml/content_classifier_data.py)
```

## Code

The record contract, validator, eligibility policy, and manifest builder live under
`app/ml/`:

- `app.ml.content_dataset_schema` -- `DatasetRecord`, `LabelSource`, `ReviewStatus`, language
  normalization.
- `app.ml.content_dataset_text` -- safe, non-destructive text normalization.
- `app.ml.content_dataset_fingerprint` -- exact normalized-duplicate fingerprinting.
- `app.ml.content_dataset_io` -- JSONL read/write.
- `app.ml.content_dataset_validator` -- structural/provenance dataset validation.
- `app.ml.content_dataset_eligibility` -- training-set and gold-evaluation-set selection.
- `app.ml.content_dataset_manifest` -- dataset release manifest.

Validate any dataset file with:

```
python -m scripts.validate_content_dataset data/content_understanding/real/dataset.jsonl
```
