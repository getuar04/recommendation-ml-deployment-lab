"""Training loop for the Two-Tower PoC.

OBJECTIVE (revised, Phase 3.1 in-batch-negative fix): a plain pointwise BCE-on-cosine
objective (the original Step 5 design) only ever constrains the exact (user, content) pairs
present in the training set -- every non-interacted pair gets zero gradient. Diagnosed
(read-only investigation) as the dominant cause of "universal attractor" content items: a few
items with sparse/zero explicit-negative labels drift toward whatever direction the positively-
labeled users pull them, with nothing pushing them away from everyone else, especially once
sparse-history user embeddings (e.g. an 11-interaction demo persona) are themselves weakly
differentiated. Fix: for POSITIVE pairs, use an in-batch contrastive (softmax cross-entropy)
loss -- every other content vector already in the same batch acts as an implicit negative for
that user, for free, with no new data pipeline or catalog sampling. Explicit negative labels
(CONTENT_NOT_INTERESTED / fast skips / target_for(...)==0, from app.ml.feature_builder,
reused unchanged) remain real, meaningful signal and are NOT folded into the diagonal-target
contrastive term -- they keep their own BCE term against the same similarity value as before.

    INPUT:    one user feature vector + one content feature vector per example
    TRAINING, per batch of B examples:
      - full [B, B] cosine-similarity matrix (user_embeddings @ content_embeddings.T), scaled
        by the existing fixed temperature
      - positive rows (label==1): softmax cross-entropy with target[i]=i (the matching content
        at the same index is the positive; every other column is an implicit negative) --
        columns belonging to another row for the SAME user that is ALSO a known positive are
        masked out of that row's negative set first, so one real positive is never forced to
        act as a negative for another positive belonging to the same user. (Two different
        users both positively labeling the same content item in one batch is NOT masked --
        accepted as a PoC-scope false-negative rate, documented rather than fixed by
        redesigning the batch loader.)
      - negative rows (label==0): unchanged BCEWithLogitsLoss on that row's own diagonal
        similarity value, exactly as the original pointwise objective did for negatives.
      - total_loss = positive_in_batch_contrastive_loss
                      + EXPLICIT_NEGATIVE_LOSS_WEIGHT * explicit_negative_bce_loss
    OUTPUT:   trained TwoTowerModel producing (user_embedding, content_embedding, similarity).

Deterministic (seed=42, reused from app.ml.trainer.RANDOM_SEED -- the exact seed the existing
RandomForest/LogisticRegression trainer already uses, not a new/different one), CPU-only, small
batch size, limited epochs -- no GPU requirement, no production infrastructure.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from app.ml.trainer import RANDOM_SEED
from app.ml.two_tower.dataset import Example
from app.ml.two_tower.model import TwoTowerModel

SIMILARITY_TEMPERATURE = 5.0  # scales cosine similarity (bounded [-1,1]) into a usable logit range.
EXPLICIT_NEGATIVE_LOSS_WEIGHT = 1.0  # single, simple weight on the explicit-negative BCE term
# alongside the in-batch positive contrastive term -- deliberately not tuned per the task's
# "keep the weighting simple" instruction; both terms are already on a comparable (BCE-like /
# cross-entropy-like) scale since they share the same temperature-scaled logits.
DEFAULT_HIDDEN_DIM = 64
DEFAULT_EMBEDDING_DIM = 32
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 40  # Phase 3.1 (EXP4 final): 20 epochs under-converged with the 1.5/0.6/0.6
# feature scaling (SPORT-heavy retrieval collapsed toward a couple of high-popularity items
# instead of SPORT); 40 was the smallest tested epoch count that reliably resolved it.
DEFAULT_LR = 0.01


@dataclass
class TrainingResult:
    model: TwoTowerModel
    epoch_losses: list[float] = field(default_factory=list)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_batch_loss(
    model: TwoTowerModel, user_x_batch: torch.Tensor, content_x_batch: torch.Tensor,
    labels_batch: torch.Tensor, user_ids_batch: np.ndarray,
) -> torch.Tensor:
    """One batch's combined loss: in-batch contrastive cross-entropy for positive rows (every
    other content embedding already in the batch is a free implicit negative) + explicit-negative
    BCE for label==0 rows (unchanged from the original pointwise objective). Extracted as its own
    function purely so it's directly unit-testable (see tests/test_two_tower.py); called once per
    batch from train_two_tower's loop below."""
    user_emb = model.embed_user(user_x_batch)
    content_emb = model.embed_content(content_x_batch)
    logits = (user_emb @ content_emb.T) * SIMILARITY_TEMPERATURE  # [B, B] cosine sim (unit-normalized embeddings)

    positive_idx = (labels_batch == 1).nonzero(as_tuple=True)[0]
    negative_idx = (labels_batch == 0).nonzero(as_tuple=True)[0]

    if positive_idx.numel() > 0:
        # Mask out, for each positive anchor row, any OTHER column in the batch that belongs to
        # the SAME user and is ALSO a known positive -- that content must not be forced to act
        # as a negative for this row's contrastive target. (Two DIFFERENT users both positively
        # labeling the same content in one batch is intentionally NOT masked -- accepted as a
        # PoC-scope false-negative rate rather than redesigning the batch loader for it.)
        same_user = torch.from_numpy(user_ids_batch[:, None] == user_ids_batch[None, :])
        other_known_positive = same_user & (labels_batch == 1).unsqueeze(0)
        other_known_positive &= ~torch.eye(len(user_ids_batch), dtype=torch.bool)
        ce_logits = logits.masked_fill(other_known_positive, float("-inf"))
        positive_loss = nn.functional.cross_entropy(ce_logits[positive_idx], positive_idx)
    else:
        positive_loss = torch.tensor(0.0)

    if negative_idx.numel() > 0:
        diagonal = torch.diagonal(logits)
        negative_loss = nn.functional.binary_cross_entropy_with_logits(diagonal[negative_idx], labels_batch[negative_idx])
    else:
        negative_loss = torch.tensor(0.0)

    return positive_loss + EXPLICIT_NEGATIVE_LOSS_WEIGHT * negative_loss


def train_two_tower(
    examples: list[Example], *, user_input_dim: int, content_input_dim: int,
    hidden_dim: int = DEFAULT_HIDDEN_DIM, embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    batch_size: int = DEFAULT_BATCH_SIZE, epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR,
    seed: int = RANDOM_SEED,
) -> TrainingResult:
    _set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    user_x = torch.tensor(np.stack([ex.user_vector for ex in examples]), dtype=torch.float32)
    content_x = torch.tensor(np.stack([ex.content_vector for ex in examples]), dtype=torch.float32)
    labels = torch.tensor([float(ex.label) for ex in examples], dtype=torch.float32)
    user_ids = np.array([ex.user_id for ex in examples], dtype=object)

    model = TwoTowerModel(user_input_dim, content_input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    generator = torch.Generator().manual_seed(seed)
    n = len(examples)
    epoch_losses: list[float] = []
    for _epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = permutation[start:start + batch_size]
            loss = compute_batch_loss(
                model, user_x[batch_idx], content_x[batch_idx], labels[batch_idx],
                user_ids[batch_idx.numpy()],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        epoch_losses.append(total_loss / n)

    model.eval()
    return TrainingResult(model=model, epoch_losses=epoch_losses)
