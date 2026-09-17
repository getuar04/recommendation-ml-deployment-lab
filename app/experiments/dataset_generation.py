"""Parameterized synthetic VIDEO dataset generation for experiments (spec §45-47).

Deliberately separate from `scripts/generate_synthetic_data.py`, which is the fixed-shape
generator the existing (non-experiment) tests and presentation flow already depend on and
which this session must not change. This generator is driven entirely by an
`ExperimentDefinition` -- seed, user/content/creator/interaction counts, categories -- and
writes into whatever `Session` it's given, so callers control isolation (an isolated SQLite
file per experiment run; never the shared/real database).
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.db.models import Content, Interaction
from app.experiments.definitions import ExperimentDefinition


def generate_experiment_dataset(db: Session, definition: ExperimentDefinition) -> None:
    """Deterministic for a given definition (fixed `random.Random(definition.seed)`).
    Content/Interaction IDs are namespaced with `definition.experiment_id` so they can never
    collide with rows from a different experiment even if a caller mistakenly points two
    runs at the same database."""
    rng = random.Random(definition.seed)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=90)
    window_seconds = 90 * 24 * 3600

    creator_ids = [f"exp-{definition.experiment_id}-creator-{i}" for i in range(1, definition.creators + 1)]
    creator_quality = {c: rng.betavariate(2.5, 2) for c in creator_ids}

    contents: list[Content] = []
    for i in range(definition.contents):
        creator = creator_ids[i % definition.creators]
        created = start + timedelta(seconds=rng.randint(0, window_seconds // 2))
        contents.append(Content(
            content_id=f"exp-{definition.experiment_id}-content-{i + 1}",
            creator_id=creator,
            category=definition.categories[i % len(definition.categories)],
            content_type="VIDEO",
            popularity_score=round(max(0.0, min(1.0, .6 * creator_quality[creator] + .4 * rng.random())), 4),
            is_active=True,
            created_at=created,
            updated_at=created,
        ))
    db.add_all(contents)
    db.flush()

    user_ids = [f"exp-{definition.experiment_id}-user-{i}" for i in range(1, definition.users + 1)]
    preferred_categories = {
        user: rng.sample(definition.categories, min(2, len(definition.categories))) for user in user_ids
    }

    rows: list[Interaction] = []
    for i in range(definition.interactions):
        timestamp = start + timedelta(seconds=int(i * window_seconds / max(1, definition.interactions)) + rng.randint(0, 300))
        # SQLite has no native timezone-aware datetime type: after db.flush(), re-reading
        # content.created_at can come back offset-naive even though it was written
        # offset-aware (the same normalization pattern used elsewhere, e.g.
        # app/api/candidate_routes.py). Compare on naive values to sidestep the mismatch.
        available = [
            content for content in contents
            if (content.created_at.replace(tzinfo=None) if content.created_at.tzinfo else content.created_at) <= timestamp.replace(tzinfo=None)
        ]
        if not available:
            continue
        user = rng.choice(user_ids)
        content = rng.choice(available)
        strength = .78 if content.category in preferred_categories[user] else .22
        if rng.random() < .1:
            strength = rng.uniform(.1, .9)  # exploration/noisy sessions, same idea as scripts/generate_synthetic_data.py
        duration = rng.randint(15, 180)
        ratio = max(0.0, min(1.5, rng.gauss(.15 + .55 * strength, .28)))
        watch = round(duration * ratio, 2)
        event_type = "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")
        roll = rng.random()
        if roll < .03:
            event_type = "CONTENT_NOT_INTERESTED"
        rows.append(Interaction(
            # "syn-" prefix (not just "exp-"): app.services.training_service._dataset_source()
            # detects synthetic rows this way (the same convention scripts/generate_synthetic_data.py
            # uses) -- without it, a 100%-synthetic experiment dataset would be misreported as
            # non-synthetic in the training metadata's datasetSource field.
            event_id=f"syn-exp-{definition.experiment_id}-evt-{i + 1}",
            user_id=user, content_id=content.content_id, creator_id=content.creator_id,
            category=content.category, event_type=event_type,
            watch_time_seconds=watch, content_duration_seconds=duration,
            watch_percentage=round(watch / duration * 100, 4) if duration else None,
            liked=rng.random() < .05 * strength, shared=rng.random() < .02 * strength,
            favorited=rng.random() < .02 * strength, commented=rng.random() < .02 * strength,
            creator_followed=rng.random() < .015 * strength,
            timestamp=timestamp,
        ))
        if len(rows) >= 1000:
            db.add_all(rows)
            db.commit()
            rows = []
    if rows:
        db.add_all(rows)
        db.commit()

    _add_semantic_preference_cohort(db, rng, definition, start)


def _add_semantic_preference_cohort(db: Session, rng: random.Random, definition: ExperimentDefinition, start: datetime) -> None:
    """Additive, isolated cohort teaching genuine per-user semantic-token preference within a
    single category (mirrors the proven pattern in scripts/generate_synthetic_data.py's
    `_semantic_preference_rows`: real positive history with one generic token vs real negative
    history with a DIFFERENT generic token, same category, so the semantic-affinity features
    carry a learnable signal isolated from category_affinity).

    Why this exists: the main population loop above never sets any hashtag/topic/entity/
    subgenre/title on generated content, so hashtag_affinity/topic_affinity/etc. never vary
    across training rows and the model has no basis to learn them -- app.ml.eligibility's
    `semantic` gate (which every candidate algorithm must pass) then fails close to by
    definition, regardless of algorithm or aggregate metrics. This cohort is purely additive
    (new dedicated users/content/creators, appended after the main population is fully
    generated and committed) so it cannot perturb the main population's determinism, category
    distribution, or any already-passing behavioral gate.

    Generic across categories/experiments (Decision 6 pattern, same as the full generator):
    the category is chosen by `rng` from `definition.categories`, and the two tokens are built
    from that category name, not a hardcoded literal category/token/algorithm. Sized
    proportionally to `definition.users` so it scales with experiment size instead of being
    tuned to one specific definition."""
    cohort_size = max(2, definition.users // 5)
    contents: list[Content] = []
    rows: list[Interaction] = []
    for u in range(1, cohort_size + 1):
        user = f"exp-{definition.experiment_id}-sempref-user-{u}"
        creator = f"exp-{definition.experiment_id}-sempref-creator-{u}"
        category = rng.choice(definition.categories)
        token_pos = f"{category}-SEMPREF-A"
        token_neg = f"{category}-SEMPREF-B"
        created_at = start
        for i in range(6):  # positive: strong, noisy, real history with token_pos.
            content_id = f"exp-{definition.experiment_id}-sempref-pos-content-{u}-{i}"
            contents.append(Content(
                content_id=content_id, creator_id=creator, category=category, content_type="VIDEO",
                popularity_score=0.5, is_active=True, created_at=created_at, updated_at=created_at,
                hashtags_json=json.dumps([token_pos]), topics_json=json.dumps([token_pos]),
                entities_json=json.dumps([token_pos]), subgenres_json=json.dumps([token_pos]), title=token_pos,
            ))
            ratio = max(0.0, min(1.2, rng.gauss(0.88, 0.08)))
            duration = rng.randint(15, 180)
            watch = round(duration * ratio, 2)
            rows.append(Interaction(
                event_id=f"syn-exp-{definition.experiment_id}-sempref-pos-{u}-{i}", user_id=user,
                content_id=content_id, creator_id=creator, category=category,
                event_type="VIDEO_COMPLETED" if ratio >= .9 else "VIDEO_WATCHED",
                watch_time_seconds=watch, content_duration_seconds=duration,
                watch_percentage=round(watch / duration * 100, 4),
                liked=rng.random() < 0.6, shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=created_at + timedelta(days=rng.uniform(10, 50)),
            ))
        for i in range(4):  # negative: strong, noisy, real history with the DIFFERENT token_neg.
            content_id = f"exp-{definition.experiment_id}-sempref-neg-content-{u}-{i}"
            contents.append(Content(
                content_id=content_id, creator_id=creator, category=category, content_type="VIDEO",
                popularity_score=0.5, is_active=True, created_at=created_at, updated_at=created_at,
                hashtags_json=json.dumps([token_neg]), topics_json=json.dumps([token_neg]),
                entities_json=json.dumps([token_neg]), subgenres_json=json.dumps([token_neg]), title=token_neg,
            ))
            ratio = max(0.0, min(1.2, rng.gauss(0.08, 0.05)))
            duration = rng.randint(15, 180)
            watch = round(duration * ratio, 2)
            rows.append(Interaction(
                event_id=f"syn-exp-{definition.experiment_id}-sempref-neg-{u}-{i}", user_id=user,
                content_id=content_id, creator_id=creator, category=category, event_type="VIDEO_SKIPPED",
                watch_time_seconds=watch, content_duration_seconds=duration,
                watch_percentage=round(watch / duration * 100, 4),
                liked=False, shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=created_at + timedelta(days=rng.uniform(10, 50)),
            ))
    if contents:
        db.add_all(contents)
        db.flush()
    if rows:
        db.add_all(rows)
        db.commit()
