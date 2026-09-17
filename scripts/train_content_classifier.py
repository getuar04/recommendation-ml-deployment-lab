"""Trains and saves the content-category classifier artifact
(app.ml.content_classifier, app.services.content_enrichment_service).

Training data (classifier release-boundary/dataset-integration audit):
`app.ml.content_classifier_dataset_source.build_training_dataframe` is the single source of
truth for what gets trained on -- it combines any real, training-eligible, taxonomy-
compatible records from the Content Understanding dataset foundation
(data/content_understanding/real/dataset.jsonl; currently 0 records) with the deterministic
SYNTHETIC bootstrap dataset (app.ml.content_classifier_data), and always records an honest
`datasetSource` composition in the saved metadata (never presenting synthetic-augmented
holdout accuracy as production-quality evidence -- see that module's own docstring). Today,
with 0 real records, this trains on the exact same synthetic-only data as before this
integration existed.

This is a wholly separate artifact from the VIDEO/LIVE recommendation models: it has no
promotion/rollback lifecycle and never touches MODEL_PATH/LIVE_MODEL_PATH or
app.ml.artifact_lifecycle/model_store.

Usage:
    python -m scripts.train_content_classifier
    python -m scripts.train_content_classifier --help   # prints this, does NOT train
"""
from __future__ import annotations

import argparse

from app.core.config import (
    CONTENT_CLASSIFIER_METADATA_PATH,
    CONTENT_CLASSIFIER_MODEL_PATH,
)
from app.ml.content_classifier import save_classifier, train_classifier
from app.ml.content_classifier_dataset_source import build_training_dataframe


def main() -> None:
    # Standard CLI behavior: `-h`/`--help` prints usage and exits (argparse's own
    # `parse_args()` calls `sys.exit(0)` before returning), and any unrecognized argument
    # is rejected with exit code 2 -- neither case falls through to training. Previously
    # this function took no arguments at all, so `--help` was silently ignored by Python
    # itself and execution fell straight through into a real training run.
    argparse.ArgumentParser(
        prog="python -m scripts.train_content_classifier",
        description="Train and save the content-category classifier artifact "
                     "(app.ml.content_classifier) from the Content Understanding dataset "
                     "foundation's real training-eligible records (currently none) combined "
                     "with the deterministic SYNTHETIC bootstrap dataset "
                     "(app.ml.content_classifier_data). Takes no arguments.",
    ).parse_args()
    dataset, dataset_source = build_training_dataframe()
    pipeline, metadata = train_classifier(dataset, dataset_source=dataset_source)
    save_classifier(pipeline, metadata, model_path=CONTENT_CLASSIFIER_MODEL_PATH, metadata_path=CONTENT_CLASSIFIER_METADATA_PATH)
    print(f"Saved content classifier to {CONTENT_CLASSIFIER_MODEL_PATH}")
    print(f"categoryTaxonomyVersion={metadata['categoryTaxonomyVersion']} "
          f"holdoutAccuracy={metadata['holdoutAccuracy']:.3f} "
          f"trainingSamples={metadata['trainingSamples']} testSamples={metadata['testSamples']}")
    print(f"datasetSource: realRecordsTrainingEligible={dataset_source['realRecordsTrainingEligible']} "
          f"syntheticBootstrapIncluded={dataset_source['syntheticBootstrapIncluded']} "
          f"containsAnyHumanReviewedData={dataset_source['containsAnyHumanReviewedData']}")


if __name__ == "__main__":
    main()
