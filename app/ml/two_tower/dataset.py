"""Point-in-time (user, content, label) training pairs for the Two-Tower PoC.

Reuses, unmodified:
  - app.ml.feature_builder.target_for -- the exact same positive(1)/negative(0)/ambiguous(None)
    event-semantics the production ranker trains on (strong watch/completion/like/share/
    favorite = positive; fast skip/CONTENT_NOT_INTERESTED = negative), so "respect the existing
    event semantics" holds by construction, not by re-deriving it.
  - app.ml.dataset_builder.FeatureHistory -- the same point-in-time accumulator the ranker's
    own app.ml.dataset_builder.build_dataset() uses: for each interaction row, examples are
    built from `history` BEFORE `history.update(row, ...)` is called for that same row, so a
    user's target-event example never sees its own event (or any later one) -- no leakage.
  - app.ml.splitting.chronological_group_split -- the same chronological, group-preserving
    split the ranker's trainer uses, applied here to a (user_id + day) group per example so no
    single user's day is split across train/eval.

Limitation (explicitly documented per the task's Step 3 instruction): grouping by
"user_id:date" (not a stricter per-session boundary) means two examples from the same user on
the same calendar day always land in the same split -- consistent with, not stricter than, what
app.ml.dataset_builder.build_dataset() already does for the production ranker (same
candidate_group convention), so this PoC's leakage profile matches the existing, already-
reviewed ranker training pipeline rather than inventing a new (untested) one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.ml.dataset_builder import FeatureHistory
from app.ml.feature_builder import target_for
from app.ml.splitting import chronological_group_split
from app.ml.two_tower.features import build_content_vector, build_user_vector


@dataclass
class Example:
    user_id: str
    content_id: str
    creator_id: str
    category: str
    label: int  # 1 = positive pair, 0 = negative pair
    timestamp: Any
    user_vector: np.ndarray
    content_vector: np.ndarray


def build_examples(rows: list[Any], content_by_id: dict[str, Any], categories: list[str]) -> list[Example]:
    """One example per row with a definitive (non-None) target_for(row) label, using only the
    user's strictly-prior history for the user vector. The content vector never depends on
    history at all (see features.build_content_vector), so it carries no leakage risk."""
    history = FeatureHistory()
    examples: list[Example] = []
    for row in sorted(rows, key=lambda item: item.timestamp):
        label = target_for(row)
        content = content_by_id.get(row.content_id)
        if label is not None and content is not None:
            user_vec = build_user_vector(history, row.user_id, categories)
            content_vec = build_content_vector(content, categories)
            examples.append(Example(
                user_id=row.user_id, content_id=row.content_id, creator_id=row.creator_id,
                category=row.category, label=label, timestamp=row.timestamp,
                user_vector=user_vec, content_vector=content_vec,
            ))
        history.update(row, content=content)
    return examples


def chronological_split(examples: list[Example], *, ratios: dict[str, float]) -> dict[str, list[Example]]:
    """Thin adapter onto the existing app.ml.splitting.chronological_group_split: builds the
    minimal (candidate_group, timestamp) frame it needs, then maps the resulting row assignment
    back onto the original Example objects -- no reimplementation of the split algorithm."""
    frame = pd.DataFrame({
        "index": range(len(examples)),
        "candidate_group": [f"{ex.user_id}:{ex.timestamp.date().isoformat()}" for ex in examples],
        "timestamp": [ex.timestamp for ex in examples],
    })
    split_frames = chronological_group_split(frame, ratios=ratios)
    return {name: [examples[i] for i in split_frames[name]["index"]] for name in ratios}
