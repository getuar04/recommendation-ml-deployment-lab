"""Offline, brute-force Top-K retrieval.

Step 6: no FAISS/vector DB -- the PoC's content universe is small enough that a plain
NumPy similarity matrix (embeddings are already unit-normalized, so a matrix multiply IS
cosine similarity) is both simpler and fast enough. Swappable for an ANN index later without
changing the Two-Tower model or its embeddings at all -- this module is the only thing that
would need to change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from app.ml.two_tower.model import TwoTowerModel


@dataclass
class ContentIndex:
    """Precomputed content embeddings for a fixed content universe -- built once, reused for
    every user's retrieval call (the whole point of the two-tower split: the content side never
    needs to be recomputed per request)."""
    content_ids: list[str]
    embeddings: np.ndarray  # (n_content, embedding_dim), unit-normalized


def build_content_index(model: TwoTowerModel, content_ids: list[str], content_vectors: np.ndarray) -> ContentIndex:
    model.eval()
    with torch.no_grad():
        embeddings = model.embed_content(torch.tensor(content_vectors, dtype=torch.float32)).numpy()
    return ContentIndex(content_ids=list(content_ids), embeddings=embeddings)


def embed_user(model: TwoTowerModel, user_vector: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.embed_user(torch.tensor(user_vector[None, :], dtype=torch.float32)).numpy()[0]


def retrieve_top_k(
    user_embedding: np.ndarray, index: ContentIndex, k: int, *, exclude_content_ids: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Brute-force cosine similarity (dot product of unit-normalized vectors) against every
    indexed content item, returning the K highest-scoring (content_id, similarity) pairs.
    `exclude_content_ids` (already-seen filtering) is applied BEFORE truncating to K, never
    after -- an already-seen item must never silently occupy a Top-K slot."""
    exclude_content_ids = exclude_content_ids or set()
    similarities = index.embeddings @ user_embedding  # (n_content,)
    ranked = sorted(
        (
            (cid, float(score)) for cid, score in zip(index.content_ids, similarities)
            if cid not in exclude_content_ids
        ),
        key=lambda pair: pair[1], reverse=True,
    )
    return ranked[:k]
