"""Two-Tower model artifact persistence -- Phase 3.2 shadow integration.

Deliberately separate from `app.ml.model_store` (the production RandomForest artifact):
different file, different directory, different loader, own exception hierarchy. Nothing in
this module is ever imported by `app.services.recommendation_service` or the public
`POST /recommendations` path -- it exists only for the internal/shadow pipeline
(`app.services.two_tower_shadow_service`) and offline scripts.

Single-file `torch.save`/`torch.load` (state_dict + enough metadata to reconstruct the exact
architecture and feature-hashing contract used at training time) -- no atomic tmp-then-swap,
checksum, or schema-version machinery like `model_store` needs, since this artifact is never
served on a public/production request path. Training (`scripts/run_two_tower_poc.py --save-
artifact`) and inference (`load_two_tower`) are strictly separate: nothing in this module ever
trains a model.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from app.core.config import MODEL_DIR
from app.ml.two_tower.model import TwoTowerModel

# Own subdirectory under the existing model root -- never collides with, and is never read by,
# the production RandomForest artifact (recommendation_model.joblib / model_metadata.json).
TWO_TOWER_ARTIFACT_PATH = Path(MODEL_DIR) / "two_tower" / "two_tower_model.pt"

REQUIRED_ARTIFACT_FIELDS = (
    "stateDict", "userInputDim", "contentInputDim", "hiddenDim", "embeddingDim",
    "categories", "semanticHashDim", "creatorHashDim", "categorySignalWeight",
    "semanticHashWeight", "creatorHashWeight", "trainedAt", "randomSeed", "epochs",
)


class TwoTowerArtifactError(Exception):
    """Base class for Two-Tower artifact load failures -- mirrors app.ml.model_store's
    ArtifactError naming/shape so a caller can handle either model's failure the same way."""


class TwoTowerArtifactNotFoundError(TwoTowerArtifactError):
    """No artifact file exists at the expected path (never trained/exported yet)."""


class TwoTowerArtifactCorruptedError(TwoTowerArtifactError):
    """The artifact exists but cannot be trusted: unreadable, missing fields, or describes a
    different feature-hashing contract than the code currently running."""


def save_two_tower(
    model: TwoTowerModel, *, categories: list[str], hidden_dim: int, embedding_dim: int,
    trained_at: str, random_seed: int, epochs: int, path: Path | None = None,
) -> dict[str, Any]:
    """Offline-only: called from the training script after `train_two_tower()` finishes, never
    from a request-serving path."""
    from app.ml.two_tower.features import (
        CATEGORY_SIGNAL_WEIGHT,
        CREATOR_HASH_DIM,
        CREATOR_HASH_WEIGHT,
        SEMANTIC_HASH_DIM,
        SEMANTIC_HASH_WEIGHT,
    )

    path = path or TWO_TOWER_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    user_input_dim = model.user_tower.net[0].in_features
    content_input_dim = model.content_tower.net[0].in_features
    payload: dict[str, Any] = {
        "stateDict": model.state_dict(),
        "userInputDim": user_input_dim,
        "contentInputDim": content_input_dim,
        "hiddenDim": hidden_dim,
        "embeddingDim": embedding_dim,
        "categories": list(categories),
        "semanticHashDim": SEMANTIC_HASH_DIM,
        "creatorHashDim": CREATOR_HASH_DIM,
        "categorySignalWeight": CATEGORY_SIGNAL_WEIGHT,
        "semanticHashWeight": SEMANTIC_HASH_WEIGHT,
        "creatorHashWeight": CREATOR_HASH_WEIGHT,
        "trainedAt": trained_at,
        "randomSeed": random_seed,
        "epochs": epochs,
    }
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # same directory -> atomic on both POSIX and Windows
    return payload


def load_two_tower(path: Path | None = None) -> tuple[TwoTowerModel, dict[str, Any]]:
    """Runtime-only: reconstructs the trained model from a prior `save_two_tower()` artifact.
    Never trains. Raises a specific `TwoTowerArtifactError` subclass instead of ever silently
    fabricating a model or serving a stale/incompatible one -- callers (the shadow service,
    shadow comparison script) must catch these and skip the Two-Tower path, not guess."""
    path = path or TWO_TOWER_ARTIFACT_PATH
    if not path.exists():
        raise TwoTowerArtifactNotFoundError(f"No trained Two-Tower artifact at {path}.")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise TwoTowerArtifactCorruptedError(f"Two-Tower artifact at {path} could not be deserialized: {exc}") from exc

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise TwoTowerArtifactCorruptedError(f"Two-Tower artifact at {path} is missing required fields {missing}.")

    from app.ml.two_tower.features import (
        CATEGORY_SIGNAL_WEIGHT,
        CREATOR_HASH_DIM,
        CREATOR_HASH_WEIGHT,
        SEMANTIC_HASH_DIM,
        SEMANTIC_HASH_WEIGHT,
    )
    if (
        payload["semanticHashDim"] != SEMANTIC_HASH_DIM or payload["creatorHashDim"] != CREATOR_HASH_DIM
        or payload["categorySignalWeight"] != CATEGORY_SIGNAL_WEIGHT
        or payload["semanticHashWeight"] != SEMANTIC_HASH_WEIGHT
        or payload["creatorHashWeight"] != CREATOR_HASH_WEIGHT
    ):
        raise TwoTowerArtifactCorruptedError(
            f"Two-Tower artifact at {path} was trained with a different feature-hashing config "
            "than the currently running code; retrain and re-export to produce a compatible artifact."
        )

    model = TwoTowerModel(
        payload["userInputDim"], payload["contentInputDim"],
        hidden_dim=payload["hiddenDim"], embedding_dim=payload["embeddingDim"],
    )
    try:
        model.load_state_dict(payload["stateDict"])
    except Exception as exc:
        raise TwoTowerArtifactCorruptedError(f"Two-Tower artifact at {path} state_dict is incompatible: {exc}") from exc
    model.eval()
    return model, payload
