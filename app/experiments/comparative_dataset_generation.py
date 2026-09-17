"""Parameterized synthetic VIDEO dataset generation for the "same algorithm, different
dataset" comparative experiment (see app.experiments.comparative_definitions).

Deliberately a separate module from `app.experiments.dataset_generation` (the pre-existing
small/medium/large *scale* experiments), which this session must not change: that generator
assigns content categories round-robin and user preferences via a uniform, unweighted sample,
which cannot express "80% of interactions concentrate on one dominant category". This module
reuses the exact same realistic-variation formulas (label thresholds, watch-time/skip/like
distributions, chronological ordering, ID namespacing) -- the *only* thing that differs is
that category selection (content catalog, user preference, and interaction targeting) is
drawn from `definition.category_weights` instead of uniformly/round-robin, so the requested
dominant-category skew is real, not merely labeled.

Every count here is a generation *target*; the realized distribution is always measured after
generation (`category_distribution`, `class_distribution_from_rows`), never assumed -- the
same philosophy as `app.experiments.dataset_summary`.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.db.models import Content, Interaction
from app.experiments.comparative_definitions import ComparativeExperimentDefinition


def _proportional_category_labels(rng: random.Random, category_weights: dict[str, float], total_count: int) -> list[str]:
    """`total_count` category labels whose *exact* counts are the largest-remainder
    proportional allocation of `category_weights` (not a random weighted draw, which has
    sampling noise that can visibly distort a small catalog -- e.g. an 80/10/10 target
    landing at 71/17/12 by chance). The returned list is then shuffled (deterministically,
    from `rng`) so which catalog position gets which category is still randomized; only the
    *counts* are exact."""
    categories = list(category_weights.keys())
    raw = [category_weights[c] * total_count for c in categories]
    counts = [int(value) for value in raw]  # floor
    remainder = total_count - sum(counts)
    # Largest-remainder method: give the leftover units to the categories with the biggest
    # fractional part first, so the realized allocation is as close to the target weights as
    # an integer count can be.
    order = sorted(range(len(categories)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    labels = [category for category, count in zip(categories, counts) for _ in range(count)]
    rng.shuffle(labels)
    return labels


def _weighted_pick(rng: random.Random, categories: list[str], weights: list[float]) -> str:
    return rng.choices(categories, weights=weights, k=1)[0]


def _weighted_pair_without_replacement(rng: random.Random, categories: list[str], weights: list[float]) -> list[str]:
    """Two distinct categories drawn without replacement, still respecting the relative
    weights (renormalized after the first draw) -- used for a user's home-category
    preference so most users' primary preference concentrates on the heavily-weighted
    (dominant) category, matching "users with strong interest in X"."""
    if len(categories) == 1:
        return list(categories)
    first = _weighted_pick(rng, categories, weights)
    rest_categories = [c for c in categories if c != first]
    rest_weights = [w for c, w in zip(categories, weights) if c != first]
    second = _weighted_pick(rng, rest_categories, rest_weights)
    return [first, second]


class InsufficientForDownsampleError(Exception):
    """`downsample_target_per_class` could not be met because generation produced fewer than
    that many rows in the positive or negative class. This pipeline never upsamples or
    fabricates data to hit a target -- only downsampling (dropping already-generated rows) is
    in scope, so a shortfall here means the target itself needs lowering, or generation scaled
    up, not that this function should invent rows."""


def _stratified_downsample(rng: random.Random, rows: list[Interaction], *, target_per_class: int) -> list[Interaction]:
    """Deterministic, minority-preserving stratified undersampling, applied via the exact same
    algorithm regardless of which dataset is calling it. Classifies every row via the real,
    unmodified `app.ml.feature_builder.target_for` (imported read-only -- the same function
    production training already uses; label semantics are never redefined here), then keeps
    exactly `target_per_class` positive and `target_per_class` negative rows (uniform random
    selection within each class, via the same seeded `rng` the rest of generation already
    used) and every neutral row untouched. No engagement/watch-time/noise/label-generation
    parameter is read or altered by this function -- it only decides which already-generated
    rows get persisted."""
    from app.ml.feature_builder import target_for

    positive = [row for row in rows if target_for(row) == 1]
    negative = [row for row in rows if target_for(row) == 0]
    neutral = [row for row in rows if target_for(row) is None]
    if len(positive) < target_per_class or len(negative) < target_per_class:
        raise InsufficientForDownsampleError(
            f"Cannot downsample to {target_per_class} rows per class: generation produced "
            f"{len(positive)} positive and {len(negative)} negative labelled rows. This "
            "pipeline never upsamples/fabricates data -- lower the downsample target or "
            "increase the generation target instead."
        )
    kept_positive = rng.sample(positive, target_per_class)
    kept_negative = rng.sample(negative, target_per_class)
    kept = neutral + kept_positive + kept_negative
    kept.sort(key=lambda row: row.timestamp)  # random.sample() does not preserve source order
    return kept


def generate_comparative_dataset(
    db: Session,
    definition: ComparativeExperimentDefinition,
    *,
    reference_timestamp: datetime | None = None,
    downsample_target_per_class: int | None = None,
) -> None:
    """Deterministic for a given definition (fixed `random.Random(definition.seed)`).
    Content/Interaction IDs are namespaced with `definition.experiment_id` so they can never
    collide with rows from a different comparative experiment (or a different scale
    experiment from `app.experiments.dataset_generation`) even in a shared database.

    `reference_timestamp` and `downsample_target_per_class` are additive, opt-in parameters
    used only by the v2 comparative pipeline (`app.experiments.comparative_definitions_v2`).
    The v1 pipeline never passes either, so `generate_comparative_dataset(db, v1_definition)`
    -- the exact call every existing v1 caller already makes -- is byte-for-byte unchanged:
    `reference_timestamp=None` anchors to real wall-clock time as before, and
    `downsample_target_per_class=None` skips the post-generation class-balance step entirely,
    so no row is ever dropped and the periodic every-1000-rows commit behavior is unchanged.

    `reference_timestamp`, when given, replaces `datetime.now(timezone.utc)` as the generation
    anchor so the entire dataset (IDs, timestamps, category allocation, interaction values,
    labels) is reproducible bit-for-bit for a given definition, not merely structurally
    similar.

    `downsample_target_per_class`, when given, applies `_stratified_downsample` once, after
    the full candidate row set has been generated (batched commits are skipped in this mode so
    the full set can be seen at once -- trivial at this dataset's scale).
    """
    rng = random.Random(definition.seed)
    now = reference_timestamp if reference_timestamp is not None else datetime.now(timezone.utc)
    start = now - timedelta(days=90)
    window_seconds = 90 * 24 * 3600

    categories = list(definition.category_weights.keys())
    weights = [definition.category_weights[c] for c in categories]

    creator_ids = [f"cmp-{definition.experiment_id}-creator-{i}" for i in range(1, definition.creators + 1)]
    creator_quality = {c: rng.betavariate(2.5, 2) for c in creator_ids}
    content_categories = _proportional_category_labels(rng, definition.category_weights, definition.contents)

    contents: list[Content] = []
    for i in range(definition.contents):
        creator = creator_ids[i % definition.creators]
        created = start + timedelta(seconds=rng.randint(0, window_seconds // 2))
        contents.append(Content(
            content_id=f"cmp-{definition.experiment_id}-content-{i + 1}",
            creator_id=creator,
            category=content_categories[i],
            content_type="VIDEO",
            popularity_score=round(max(0.0, min(1.0, .6 * creator_quality[creator] + .4 * rng.random())), 4),
            is_active=True,
            created_at=created,
            updated_at=created,
        ))
    db.add_all(contents)
    db.flush()

    user_ids = [f"cmp-{definition.experiment_id}-user-{i}" for i in range(1, definition.users + 1)]
    preferred_categories = {user: _weighted_pair_without_replacement(rng, categories, weights) for user in user_ids}

    rows: list[Interaction] = []
    for i in range(definition.interactions):
        timestamp = start + timedelta(seconds=int(i * window_seconds / max(1, definition.interactions)) + rng.randint(0, 300))
        # SQLite has no native timezone-aware datetime type: after db.flush(), re-reading
        # content.created_at can come back offset-naive even though it was written
        # offset-aware -- compare on naive values to sidestep the mismatch (same pattern as
        # app.experiments.dataset_generation and app/api/candidate_routes.py).
        available = [
            content for content in contents
            if (content.created_at.replace(tzinfo=None) if content.created_at.tzinfo else content.created_at) <= timestamp.replace(tzinfo=None)
        ]
        if not available:
            continue
        user = rng.choice(user_ids)
        # Uniform choice among currently-available content (same as the scale-experiment
        # generator): the content catalog's *exact* proportional category allocation above
        # (not a random weighted draw) is what realizes the requested category skew in the
        # labeled interaction rows, without compounding two separate weighted-selection steps.
        content = rng.choice(available)
        strength = .78 if content.category in preferred_categories[user] else .22
        if rng.random() < .1:
            strength = rng.uniform(.1, .9)  # exploration/noisy sessions, same idea as the scale-experiment generator
        duration = rng.randint(15, 180)
        ratio = max(0.0, min(1.5, rng.gauss(.15 + .55 * strength, .28)))
        watch = round(duration * ratio, 2)
        event_type = "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")
        roll = rng.random()
        if roll < .03:
            event_type = "CONTENT_NOT_INTERESTED"
        rows.append(Interaction(
            # "syn-" prefix (not just "cmp-"): app.services.training_service._dataset_source()
            # detects synthetic rows this way -- without it, a 100%-synthetic comparative
            # dataset would be misreported as non-synthetic in training metadata.
            event_id=f"syn-cmp-{definition.experiment_id}-evt-{i + 1}",
            user_id=user, content_id=content.content_id, creator_id=content.creator_id,
            category=content.category, event_type=event_type,
            watch_time_seconds=watch, content_duration_seconds=duration,
            watch_percentage=round(watch / duration * 100, 4) if duration else None,
            liked=rng.random() < .05 * strength, shared=rng.random() < .02 * strength,
            favorited=rng.random() < .02 * strength, commented=rng.random() < .02 * strength,
            creator_followed=rng.random() < .015 * strength,
            timestamp=timestamp,
        ))
        if downsample_target_per_class is None and len(rows) >= 1000:
            db.add_all(rows)
            db.commit()
            rows = []
    if downsample_target_per_class is not None:
        rows = _stratified_downsample(rng, rows, target_per_class=downsample_target_per_class)
    if rows:
        db.add_all(rows)
        db.commit()


def category_distribution(rows: Iterable[object]) -> dict[str, float]:
    """Realized category share of `rows` (e.g. Interaction ORM rows), rounded to 4 decimals.
    Always computed from the actual generated/persisted rows -- never assumed from
    `definition.category_weights`, which is only a generation target."""
    categories = [category for row in rows if (category := getattr(row, "category", None)) is not None]
    counts: Counter[str] = Counter(categories)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {category: round(count / total, 4) for category, count in sorted(counts.items(), key=lambda item: -item[1])}
