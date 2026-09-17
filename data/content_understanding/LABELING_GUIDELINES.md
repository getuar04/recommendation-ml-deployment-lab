# Content Understanding labeling guidelines

For future human annotators labeling real content for `real/dataset.jsonl`. This document
does not define a final category taxonomy -- none is approved yet (see `../README.md`). When
a taxonomy is approved, its category list is published separately; these guidelines describe
*how to label*, not *what the categories are*.

## Principles

1. **Label the content's meaning, not the creator.** A sports creator who occasionally posts
   a recipe video should have that video labeled `FOOD`-equivalent, not `SPORT`, just because
   of who posted it. Creator identity is context, never ground truth for a specific item.

2. **Choose exactly one primary category.** `primaryCategory` is singular by design. If
   content genuinely spans multiple domains, put the dominant one in `primaryCategory` and
   the rest in `topics`.

3. **Use `subcategory` for a finer, stable distinction within the primary category** (e.g. a
   specific cuisine within FOOD), not as a second opinion on the primary category itself.

4. **Use `topics` for secondary/cross-domain meaning** that doesn't fit `primaryCategory`/
   `subcategory` -- a travel vlog that happens to feature a lot of food can be primary
   `TRAVEL` with `FOOD` in `topics`.

5. **When content is genuinely ambiguous, flag it -- do not force a label.** Set
   `reviewStatus` to `NEEDS_REVIEW` or `DISPUTED` rather than picking an arbitrary primary
   category just to fill the field. A record marked `reviewStatus=REVIEWED` must carry a real
   decided `primaryCategory`; if you can't decide, it isn't reviewed yet.

6. **Every labeled record needs a known `taxonomyVersion`.** Do not label against a taxonomy
   that hasn't actually been approved and versioned. If no approved version exists yet, you
   cannot produce a canonical `HUMAN_REVIEWED` record with a `primaryCategory` -- wait for one
   rather than inventing a version string.

7. **Language labeling.** Set `language` to the caption/title's primary language as a
   lowercase tag (`sq`, `en`, ...). If the caption genuinely mixes Albanian and English (not
   just one borrowed word), use `mixed`. If you cannot tell, use `unknown` rather than
   guessing. See `app.ml.content_dataset_schema.normalize_language_tag` for the exact accepted
   values.

8. **Duplicate handling.** If you recognize a caption/hashtag combination you've already
   labeled (a repost, a near-identical reupload), do not create a second record with a
   different label "just in case." Either skip it, or if you believe the earlier label was
   wrong, mark the *existing* record `DISPUTED` with a note rather than adding a conflicting
   duplicate -- the dataset validator (`app.ml.content_dataset_validator`) will flag exact
   duplicate fingerprints that carry conflicting labels as an error.

9. **Do not infer labels from recommendation/engagement behavior.** A video getting lots of
   engagement from sports fans does not make it `SPORT` -- label what the content actually is
   about, independent of how it performed or who watched it.

10. **Do not treat the current classifier's output as unquestioned truth.** The existing
    classifier (`app.ml.content_classifier`) was trained on a small synthetic bootstrap
    dataset (`app.ml.content_classifier_data`) against an unconfirmed 10-category taxonomy.
    Its predictions are a starting hint at most, never a substitute for your own judgment.

11. **Creator history is context, not ground truth.** You may use a creator's typical content
    as a tiebreaker for genuinely ambiguous cases, but it must never override what the
    specific piece of content in front of you actually is.

## Provenance you must fill in

- `reviewerId` -- a stable anonymous identifier for yourself (not a name or account you'd
  need to protect). No reviewer-account system exists; any consistent short ID is enough.
- `reviewedAt` -- when you made the decision.
- `notes` -- record any real disagreement or uncertainty, even on a record you ultimately
  marked `REVIEWED`. This is what lets a future reviewer understand a borderline call.

## What NOT to do

- Do not use the proposed 18-category taxonomy discussed elsewhere as if it were approved --
  it is not, and using it here would encode unapproved product truth into real data.
- Do not backfill `taxonomyVersion` with a version string you made up.
- Do not label content you have not actually looked at.
