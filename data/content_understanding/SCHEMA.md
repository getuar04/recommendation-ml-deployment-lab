# Dataset record schema

Canonical definition: `app.ml.content_dataset_schema.DatasetRecord` (pydantic model). This
document is a human-readable companion, not a second source of truth -- if it ever disagrees
with the code, the code wins.

One JSONL line = one `DatasetRecord`, alias-serialized (camelCase on disk, matching this
repository's existing API convention).

| Field | Required | Notes |
|---|---|---|
| `exampleId` | yes | Dataset-row primary key. Unique within a file. |
| `contentId` | no | Real upstream Content identifier, if this example corresponds to ingested content. |
| `creatorId` | no | Preserved for future known/unseen-creator evaluation splitting -- never used as a text-classifier feature. |
| `contentType` | no | Free string (typically `VIDEO`/`LIVE`, not enum-enforced here). |
| `title` | no* | Caption/title text, as authored (case and diacritics preserved). |
| `hashtags` | no* | List of raw hashtag strings, as authored. |
| `language` | no | Lowercase BCP-47-style tag (`sq`, `en`, `sq-al`, ...), or the sentinel `mixed`/`unknown`. See `app.ml.content_dataset_schema.normalize_language_tag`. |
| `contentCreatedAt` | no | Preserved for future chronological train/eval holdout splitting. |
| `primaryCategory` | no | Free string. **No enum -- no taxonomy is product-approved.** |
| `subcategory` | no | Free string, finer classification under `primaryCategory`. |
| `topics` | no | Secondary/cross-domain topic tags, as authored. |
| `taxonomyVersion` | conditional | **Required if `primaryCategory` or `subcategory` is set.** |
| `labelSource` | yes | One of `HUMAN_REVIEWED`, `SYNTHETIC`, `PSEUDO_LABELED`, `WEAK_SUPERVISION`. |
| `labelConfidence` | no | `[0, 1]`. Meaningful mainly for `PSEUDO_LABELED`/`WEAK_SUPERVISION`. |
| `reviewStatus` | no (default `UNREVIEWED`) | One of `UNREVIEWED`, `REVIEWED`, `NEEDS_REVIEW`, `DISPUTED`. |
| `reviewerId` | conditional | **Required if `reviewStatus` is `REVIEWED`.** A stable anonymous ID -- no reviewer accounts/PII. |
| `reviewedAt` | conditional | **Required if `reviewStatus` is `REVIEWED`.** |
| `notes` | no | Free text, e.g. disagreement notes. |
| `datasetSplit` | no | Structural only. Never auto-populated by this foundation -- see `app.ml.content_dataset_eligibility`; splitting is a later task. |
| `sourceMetadata` | no | Free-form dict, e.g. an ingestion batch id. |

\* At least one of `title`/`hashtags` must be present -- a record needs some text signal.

A `HUMAN_REVIEWED` record with `reviewStatus=REVIEWED` must also carry a `primaryCategory`
(ambiguous content should be `NEEDS_REVIEW`/`DISPUTED` instead of forced into `REVIEWED` with
no label).

## Illustrative example (NON-TRAINING -- do not load this into any dataset file)

```json
{
  "exampleId": "example-schema-illustration-1",
  "contentId": "content-000123",
  "creatorId": "creator-000045",
  "contentType": "VIDEO",
  "title": "Receta e sushi-t shqiptar me lakror",
  "hashtags": ["#ushqim", "#recete", "#shqiperi"],
  "language": "sq",
  "primaryCategory": "FOOD",
  "subcategory": "RECIPES",
  "topics": ["ALBANIAN_CUISINE"],
  "taxonomyVersion": "REPLACE_WITH_AN_ACTUALLY_APPROVED_TAXONOMY_VERSION",
  "labelSource": "HUMAN_REVIEWED",
  "labelConfidence": null,
  "reviewStatus": "REVIEWED",
  "reviewerId": "reviewer-anon-07",
  "reviewedAt": "2026-09-16T00:00:00Z",
  "notes": "Clear primary category; mixed Albanian/English hashtags in original caption."
}
```

This example is for schema illustration only. Its `taxonomyVersion` is a placeholder, not a
real value -- no taxonomy version is approved yet (see `../README.md`). It is not present in
`real/dataset.jsonl` and must not be copied into it.
