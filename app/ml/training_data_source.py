"""Explicit training-data-source provenance for ranker training runs.

Root problem this addresses (RMS real-data training foundation task): before this module, every
ranker training run in this codebase implicitly meant "trained on app.ml.ranking_groups'
synthetic archetypes" -- there was no type-level distinction between that and a future real-data
run, so a future real-observed artifact's metadata could accidentally end up indistinguishable
from (or literally stamped with) the synthetic generator's own `RANKING_GROUP_VERSION`
("v3-multisignal"). See the RMS training-path reconciliation finding ("CURRENT TRAINING PATH
EQUIVALENCE NOT PROVEN") for why conflating the two is a real, not hypothetical, risk.

Every future ranker training/metadata code path must declare exactly one of these values -- there
is no default and no implicit fallback between them. `app.ml.real_ranker_trainer` enforces this
for the future real-data path (it never imports or falls back to `app.ml.ranking_groups`), and
`app.ml.artifact_metadata.build_real_ranker_metadata` enforces it for future real-data metadata
(hard-rejects reuse of the synthetic generator's own version string).
"""
from __future__ import annotations

from enum import Enum


class TrainingDataSource(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    REAL_OBSERVED = "REAL_OBSERVED"
