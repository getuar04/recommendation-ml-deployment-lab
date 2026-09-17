"""Real-data ranker training path: validates/builds a fit-ready dataset from real ranking
decisions (`prepare_real_ranker_training_input`, unchanged from before this task), then fits
ONE real-data SHADOW-ONLY ranker candidate from it (`fit_real_ranker`, new).

This module never imports `app.ml.ranking_groups` (the synthetic `v3-multisignal` generator) --
that is deliberate and load-bearing, not an oversight: a caller here cannot receive synthetic
groups labeled as real, because this module has no way to reach `build_ranking_groups()` at all
(verified directly: `tests/test_real_ranker_trainer.py` asserts the string "ranking_groups"
never appears in this module's own source). Every reused helper below --
`app.ml.ranker_registry.build_candidate_rankers` (model/hyperparameter config only, no data),
`app.ml.pipeline_builder.build_classifier_pipeline`/`ranker_fit_params` (generic sklearn Pipeline
plumbing), `app.ml.dataset_builder.CATEGORICAL`/`FEATURES`/`NUMERIC` (the one production feature
schema) -- was individually confirmed, by reading its source, to carry zero synthetic-archetype
or `app.ml.ranking_groups` dependency before being reused here.

Not wired into `/model/train`, `app.ml.trainer`, or any other current training entrypoint.
`fit_real_ranker` never calls `app.ml.model_store.save` or `app.ml.artifact_lifecycle.promote`
-- its result is an in-memory-only `RealRankerShadowTrainingResult`, never a versioned or
persisted artifact (see that class's own docstring for why `shadow_status` can only ever be
`SHADOW_ONLY`).

NOT IN SCOPE (deliberately deferred, not overlooked):
  - No train/validation split is performed here. A caller-supplied `decisions` sequence is
    fit on in full -- applying a held-out split would presuppose a validation/shadow-evaluation
    step this task explicitly defers to a future step (see SPLIT NOTE below).
  - No candidate-selection sweep: exactly one named `algorithm` is fit per call, never a
    multi-algorithm comparison.
  - No promotion, no save, no `model_version` assignment -- versioning a real-data artifact is a
    separate, later decision this module does not make.

SPLIT NOTE: `app.ml.real_ranking_dataset_builder.to_split_ready_frame` + the generic, already-
existing `app.ml.splitting.chronological_group_split` COULD produce a real chronological train/
test split today -- that contract is not "incomplete" so much as "not this step's job to invoke"
(see NOT IN SCOPE above): calling it here would silently make a validation-strategy decision this
task's own instructions reserve for a later step.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.ml.dataset_builder import CATEGORICAL, FEATURES, NUMERIC
from app.ml.pipeline_builder import build_classifier_pipeline, ranker_fit_params
from app.ml.ranker_registry import build_candidate_rankers
from app.ml.real_data_readiness import evaluate_readiness
from app.ml.real_data_shadow import ShadowTrainingStatus
from app.ml.real_ranking_dataset_builder import (
    RealRankingDatasetBuildResult,
    build_real_ranking_dataset,
    to_split_ready_frame,
)
from app.ml.real_ranking_decision import RealRankingDecision
from app.ml.training_data_source import TrainingDataSource

__all__ = [
    "InsufficientTrainableGroups",
    "RealDataVolumeNotConfirmed",
    "RealRankerAlgorithmUnavailable",
    "RealRankerInvalidDatasetError",
    "RealRankerShadowTrainingResult",
    "RealRankerWeightRepresentationError",
    "RealRankingDataUnavailable",
    "fit_real_ranker",
    "prepare_real_ranker_training_input",
]

DEFAULT_REAL_RANKER_ALGORITHM = "XGBRanker"  # matches app.ml.ranker_registry.PRODUCTION_RANKER_NAMES

# Placeholder real-data group-CONTRACT version -- the authoritative cross-service ranking-group
# identity contract (who assigns qids, what they mean) is not yet confirmed anywhere in this
# codebase (see the RMS real-data gap analysis). This exists only so a real-data result's
# `ranking_group_version` can never be confused with, or accidentally equal,
# `app.ml.ranking_groups.RANKING_GROUP_VERSION` ("v3-multisignal") --
# `app.ml.artifact_metadata.build_real_ranker_metadata` already hard-rejects that exact
# collision for saved metadata; this is the same discipline applied to this in-memory result.
REAL_RANKING_GROUP_VERSION = "real-observed-v0-unconfirmed-contract"


class RealRankingDataUnavailable(Exception):
    """No real ranking decisions were supplied at all."""


class InsufficientTrainableGroups(Exception):
    """Real ranking decisions were supplied, but none passed group-trainability validation."""


class RealDataVolumeNotConfirmed(Exception):
    """Trainable groups exist, but nobody has confirmed SUFFICIENT_VOLUME_FOR_MODEL_TRAINING.

    That confirmation is a human/product decision informed by `RealDataReadinessReport` -- see
    `app.ml.real_data_readiness` -- and is never inferred from a hardcoded row-count threshold.
    """


class RealRankerAlgorithmUnavailable(Exception):
    """The requested `algorithm` name is not an enabled, available ranker candidate."""


class RealRankerInvalidDatasetError(Exception):
    """The built dataset failed a fit-time structural/finiteness guard."""


class RealRankerWeightRepresentationError(Exception):
    """`resolved_weight` values present on the dataset cannot be safely represented for the
    requested algorithm's weight API (see `_resolve_sample_weight`'s own docstring) -- never
    silently aggregated, dropped, or replaced with a synthetic default."""


def prepare_real_ranker_training_input(
    decisions: Sequence[RealRankingDecision], *, human_confirmed_sufficient_volume: bool,
) -> RealRankingDatasetBuildResult:
    """Validates and builds a real-data ranker-ready dataset, or raises a precise domain error
    before any dataset is returned. Returns the built dataset (rows + group sizes), ready for
    `fit_real_ranker` below."""
    if not decisions:
        raise RealRankingDataUnavailable(
            "no real ranking decisions supplied -- real-data training cannot proceed without "
            "confirmed contracts and collected data (see the RMS cross-service contract package)"
        )

    readiness = evaluate_readiness(decisions)
    if readiness.trainable_group_count == 0:
        raise InsufficientTrainableGroups(
            f"{readiness.ranking_decision_count} ranking decision(s) supplied, but zero are "
            "trainable (all excluded -- see RealDataReadinessReport diagnostics for reasons)"
        )

    if not human_confirmed_sufficient_volume:
        raise RealDataVolumeNotConfirmed(
            "SUFFICIENT_VOLUME_FOR_MODEL_TRAINING has not been confirmed. This is a human/data "
            "decision informed by RealDataReadinessReport, never inferred automatically -- pass "
            "human_confirmed_sufficient_volume=True only after that review."
        )

    return build_real_ranking_dataset(decisions)


@dataclass(frozen=True, slots=True)
class RealRankerShadowTrainingResult:
    """One fitted, in-memory-only, SHADOW-ONLY real-data ranker candidate. Never saved,
    never versioned, never promoted -- `shadow_status` is fixed at construction, mirroring
    `app.ml.real_data_shadow.ShadowChallengerResult`'s own "can only ever be SHADOW_ONLY"
    discipline, extended to a raw fit result that (unlike `ShadowChallengerResult`) has no
    `model_version` yet at all: assigning one is a separate, later decision this result
    deliberately does not make."""

    model: Any  # fitted sklearn Pipeline (app.ml.pipeline_builder.build_classifier_pipeline)
    algorithm: str
    training_data_source: TrainingDataSource
    ranking_group_version: str
    training_row_count: int
    training_group_count: int
    feature_names: tuple[str, ...]
    human_confirmed_sufficient_volume: bool
    shadow_status: ShadowTrainingStatus

    def __post_init__(self) -> None:
        if self.training_data_source is not TrainingDataSource.REAL_OBSERVED:
            raise ValueError(
                f"RealRankerShadowTrainingResult.training_data_source must be REAL_OBSERVED, "
                f"got {self.training_data_source!r}"
            )
        if self.shadow_status is not ShadowTrainingStatus.SHADOW_ONLY:
            raise ValueError(
                f"RealRankerShadowTrainingResult.shadow_status must be SHADOW_ONLY, got {self.shadow_status!r}"
            )
        if not self.human_confirmed_sufficient_volume:
            raise ValueError(
                "RealRankerShadowTrainingResult must never be constructed with "
                "human_confirmed_sufficient_volume=False -- fit_real_ranker() itself already "
                "refuses to fit in that case (via prepare_real_ranker_training_input)"
            )


def _resolve_sample_weight(
    weights: list[float | None], *, weight_granularity: str, algorithm: str,
) -> np.ndarray | None:
    """`resolved_weight` values, used EXACTLY as supplied by the real dataset -- this function
    never computes, invents, or defaults a weight (see `app.ml.real_sample_weight_policy` for
    where a weight value itself comes from).

    `weight_granularity` (`app.ml.ranker_registry.RankerSpec`) is a REAL library constraint, not
    a design choice: XGBoost's ranking objective requires exactly ONE weight per query GROUP
    when its `group` fit-param is set ("per_group"; confirmed by `app.ml.ranker_registry`'s own
    comment, itself citing a real XGBoostError), while LightGBM/CatBoost's ranking objectives
    accept a per-candidate weight ("per_row"). Our real dataset rows only ever carry a PER-ROW
    `resolved_weight` (`app.ml.real_ranking_decision.RealCandidateObservation`) -- there is no
    per-group value anywhere to read. So:
      - "per_row": every row's weight is used directly if ALL are present; if ALL are `None`,
        weighting is omitted entirely (unweighted fit); a MIX of `None` and present values
        cannot be safely represented (a per-row weight array cannot contain a "no weight" hole)
        and is rejected.
      - "per_group": a per-row weight cannot be safely collapsed into the single required
        per-group value without an aggregation policy this function does not own (see
        `app.ml.real_sample_weight_policy`'s own module docstring for the identical stance on
        not inventing one) -- ANY present weight is rejected; only an all-`None` dataset (no
        weighting at all) can be safely fit with a "per_group" algorithm today.
    """
    present = [w for w in weights if w is not None]
    if present and not all(math.isfinite(w) for w in present):
        raise RealRankerInvalidDatasetError("non-finite resolved_weight value -- refusing to fit")

    if weight_granularity == "per_group":
        if present:
            raise RealRankerWeightRepresentationError(
                f"{algorithm!r} requires one sample_weight per query GROUP (a real XGBoost-style "
                "API constraint), but per-row resolved_weight values are present on this real "
                "dataset. Aggregating per-row weights into a per-group value is a policy decision "
                "this trainer does not own -- retrain with a per-row-weighted algorithm "
                "(weight_granularity='per_row', e.g. LGBMRanker/CatBoostRanker) or omit "
                "resolved_weight upstream."
            )
        return None

    if not present:
        return None
    if len(present) != len(weights):
        raise RealRankerWeightRepresentationError(
            f"{len(weights) - len(present)} of {len(weights)} row(s) have resolved_weight=None "
            f"while the rest carry a value -- a per-row weight array cannot represent a missing "
            "weight, and silently defaulting the missing ones would misrepresent what was "
            "actually observed. Resolve every row's weight upstream, or supply none at all."
        )
    return np.asarray(weights, dtype=float)


def _validate_dataset(frame: Any, result: RealRankingDatasetBuildResult) -> None:
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise RealRankerInvalidDatasetError(f"built dataset frame is missing feature column(s): {sorted(missing)}")
    if sum(result.group_sizes) != len(result.rows):
        raise RealRankerInvalidDatasetError(
            f"group_sizes sum ({sum(result.group_sizes)}) does not match row count "
            f"({len(result.rows)}) -- refusing to fit against a misaligned group/row structure"
        )
    numeric_and_label = frame[[*NUMERIC, "relevance"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_and_label).all():
        raise RealRankerInvalidDatasetError(
            "non-finite value found among numeric features or relevance labels -- refusing to fit"
        )


def fit_real_ranker(
    decisions: Sequence[RealRankingDecision],
    *,
    human_confirmed_sufficient_volume: bool,
    algorithm: str = DEFAULT_REAL_RANKER_ALGORITHM,
    random_seed: int = 42,
) -> RealRankerShadowTrainingResult:
    """Fits exactly ONE real-data SHADOW-ONLY ranker candidate on `decisions`.

    Reuses `prepare_real_ranker_training_input` for every existing guard (empty input,
    zero trainable groups, unconfirmed volume) and adds fit-time guards on top: an
    unavailable/unknown `algorithm`, a structurally invalid built dataset, or `resolved_weight`
    values that cannot be safely represented for `algorithm`'s weight API (see
    `_resolve_sample_weight`). No train/validation split, no candidate sweep, no save, no
    promote -- see this module's own docstring for what is deliberately out of scope.
    """
    result = prepare_real_ranker_training_input(
        decisions, human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
    )
    if not result.rows:
        raise RealRankingDataUnavailable(
            "prepare_real_ranker_training_input returned zero trainable rows -- nothing to fit"
        )

    candidates = build_candidate_rankers(random_seed, only=(algorithm,))
    if algorithm not in candidates:
        raise RealRankerAlgorithmUnavailable(
            f"{algorithm!r} is not an enabled/available ranker candidate -- see "
            "app.ml.ranker_registry.unavailable_rankers() for why"
        )
    estimator, group_param_name, weight_granularity = candidates[algorithm]

    frame = to_split_ready_frame(result)
    _validate_dataset(frame, result)

    weights = [row.sample_weight for row in result.rows]
    sample_weight = _resolve_sample_weight(weights, weight_granularity=weight_granularity, algorithm=algorithm)

    if group_param_name == "group":
        group_values: Any = list(result.group_sizes)
    else:  # "group_id": a per-row group-identifier array (CatBoostRanker)
        group_values = [
            qid
            for qid, size in zip(result.ranking_group_ids_in_order, result.group_sizes, strict=True)
            for _ in range(size)
        ]

    pipeline = build_classifier_pipeline(estimator, categorical=CATEGORICAL, numeric=list(NUMERIC), scale_numeric=False)
    fit_params = ranker_fit_params(group_param_name=group_param_name, group_values=group_values, sample_weight=sample_weight)
    pipeline.fit(frame[FEATURES], frame["relevance"], **fit_params)

    return RealRankerShadowTrainingResult(
        model=pipeline,
        algorithm=algorithm,
        training_data_source=TrainingDataSource.REAL_OBSERVED,
        ranking_group_version=REAL_RANKING_GROUP_VERSION,
        training_row_count=len(result.rows),
        training_group_count=len(result.group_sizes),
        feature_names=tuple(FEATURES),
        human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
        shadow_status=ShadowTrainingStatus.SHADOW_ONLY,
    )
