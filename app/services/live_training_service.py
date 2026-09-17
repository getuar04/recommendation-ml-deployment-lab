"""Orchestrates LIVE training's data-source choice: mirrors
`app.services.training_service`'s VIDEO split (DB access lives in the service layer,
`app.ml.live_trainer` stays a pure-DataFrame function) -- the difference is that LIVE, unlike
VIDEO, still has an explicit synthetic mode, since no real LIVE behavior log existed until
`app.ml.live_dataset_builder` was built.

Mode is always explicit, never inferred: "synthetic" never silently substitutes for a
requested "real" run, and a "real" run that finds too little data fails with the trainer's own
`InsufficientLiveData` rather than falling back (see `train_live`).
"""
from __future__ import annotations

from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Content
from app.db.repositories import interactions
from app.ml.live_dataset_builder import build_live_dataset
from app.ml.live_trainer import InsufficientLiveData, train_live_model

__all__ = ["InsufficientLiveData", "train_live"]


def train_live(db: Session, *, mode: Literal["synthetic", "real"] = "synthetic") -> dict[str, Any]:
    """`mode="synthetic"` (default, unchanged pre-existing behavior): delegates straight to
    `train_live_model()`, which generates its own deterministic synthetic dataset.

    `mode="real"`: builds a dataset from this service's own stored LIVE interactions
    (`app.ml.live_dataset_builder.build_live_dataset`) and trains on that -- never on
    synthetic data, even if the real dataset turns out to be too small: `train_live_model`'s
    own size/class-balance check then raises `InsufficientLiveData` (propagated unchanged),
    which is the correct, clear failure for "requested real training, don't have enough real
    data yet" -- there is no fallback path from here to synthetic.
    """
    if mode == "synthetic":
        return train_live_model()
    rows = interactions(db)
    content_by_id = {item.content_id: item for item in db.scalars(select(Content)).all()}
    live_rows = [row for row in rows if getattr(content_by_id.get(row.content_id), "content_type", "VIDEO") == "LIVE"]
    dataset = build_live_dataset(live_rows, content_by_id)
    return train_live_model(dataset, dataset_provenance="real")
