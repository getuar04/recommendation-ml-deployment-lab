"""Deterministic synthetic LIVE training data generation.

Separated from `app.ml.live_trainer` so the trainer can accept any injected
DataFrame (e.g. a future repository-backed dataset) while this module remains
the single, clearly-labeled source of synthetic rows used for this service/tests.
Generation is deterministic (fixed `random.Random(seed)`), so repeated calls
with the same `size`/`seed` produce byte-identical output.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pandas as pd

from app.ml.live_feature_builder import derive_live_affinities, live_target

DEFAULT_SEED = 20260722
DEFAULT_SIZE = 4000
CATEGORIES = ["GAMING", "MUSIC", "SPORT", "CHAT", "NEWS"]


def generate_synthetic_live_dataset(size: int = DEFAULT_SIZE, *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate a deterministic synthetic LIVE interaction dataset.

    This is not production data: it is a hand-authored simulation used because no
    real LIVE behavior log exists yet for this service. Metrics computed on it describe
    whether the training pipeline works, not real user preference. See
    `app.ml.live_trainer.train_live_model`'s `datasetSource` metadata field.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(size):
        followed = rng.random() < .12
        region = rng.random() < .7
        language = rng.random() < .8
        viewers = int(rng.lognormvariate(5.5, 1))
        growth = rng.gauss(.05, .2)
        age = rng.uniform(1, 240)
        previous = rng.randint(0, 12)
        watch = previous * rng.uniform(10, 100)
        derived = derive_live_affinities(previous, watch)
        affinity = derived["live_category_affinity"]
        creator = derived["creator_affinity"]
        latent = (
            1.5 * affinity + 1.0 * creator + 1.0 * followed + .35 * region + .45 * language
            + .25 * min(1, viewers / 1500) + .3 * growth - .3 * (age / 240) + rng.gauss(0, .65)
        )
        engaged = latent > 1.55
        if rng.random() < .10:
            event = SimpleNamespace(joined=True, watch_time_seconds=30, liked=False, shared=False,
                                     commented=False, gift_sent=False, creator_followed=False, impression=True)
        elif engaged:
            event = SimpleNamespace(joined=True, watch_time_seconds=rng.uniform(60, 600), liked=rng.random() < .2,
                                     shared=False, commented=rng.random() < .15, gift_sent=rng.random() < .04,
                                     creator_followed=followed, impression=True)
        else:
            joined = rng.random() < .25
            event = SimpleNamespace(joined=joined, watch_time_seconds=rng.uniform(0, 9) if joined else 0,
                                     liked=False, shared=False, commented=False, gift_sent=False,
                                     creator_followed=False, impression=True)
        target = live_target(event)
        if target is None:
            continue
        rows.append({
            "category": rng.choice(CATEGORIES),
            "region": "eu-central-1" if region else "us-east-1",
            "language": "sq" if language else "en",
            "live_category_affinity": affinity,
            "creator_affinity": creator,
            "creator_followed": int(followed),
            "previous_live_interaction_count": previous,
            "previous_live_watch_time": watch,
            "average_live_watch_time_for_category": derived["average_live_watch_time_for_category"],
            "recent_live_category_activity": derived["recent_live_category_activity"],
            "current_viewer_count": viewers,
            "viewer_growth_rate": growth,
            "live_age_minutes": age,
            "region_match": int(region),
            "language_match": int(language),
            "hour_of_day": i % 24,
            "already_joined": int(rng.random() < .08),
            "target": target,
            "candidate_group": f"u{i % 100}:window-{i // 500}",
            "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=i),
        })
    return pd.DataFrame(rows)
