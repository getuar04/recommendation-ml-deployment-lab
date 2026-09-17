"""Offline real-data shadow evaluation: does a ranker trained on real observed ranking data
beat the current baseline on held-out real ranking groups?

NOT a promotion mechanism. `evaluate_real_shadow_candidate` never saves, versions, or promotes
anything -- its own return type documents a comparison result only (see
`RealShadowEvaluationResult`), never a promotion-ready artifact. `try_load_baseline_model` is
the ONLY function in this module that imports `app.ml.model_store`, and it calls exclusively
`load_validated()` -- a pure read (schema/checksum validation + `joblib.load`, confirmed by
reading `app.ml.model_store`'s own source; no write path is reachable from it). Neither function
imports `app.ml.artifact_lifecycle` or invokes any save/promote call anywhere (see
`tests/test_real_shadow_evaluation.py`'s structural checks).

REUSE, NOT REINVENTION:
  - Ranking metrics: `app.ml.evaluator._rank_groups`/`_ndcg_at`/`_precision_at` -- the exact
    same per-group DCG/precision computation already used everywhere else in this codebase
    (`app.ml.evaluator.evaluate`). Not reimplemented here. NDCG@10 is computed on the real,
    GRADED [0,4] relevance directly (exactly what NDCG is designed for); Precision@5 is
    computed on a BINARIZED (relevance > 0) view of the same labels, since Precision@K's
    standard IR definition is "fraction of top-K that are relevant" -- feeding it the same
    graded array `evaluate()`'s classifier-oriented callers do would silently redefine it into
    a non-standard "average relevance / K" metric this task did not ask for.
  - Chronological, group-preserving splitting: `app.ml.splitting.chronological_group_split` --
    already generic (no synthetic dependency), already proven column-compatible with this
    dataset's shape (`app.ml.real_ranking_dataset_builder.to_split_ready_frame`, itself already
    tested against this exact splitter). Guarantees, for the returned train/eval partitions,
    `train["timestamp"].max() < eval["timestamp"].min()` AND that no `ranking_group_id` is ever
    split across the two -- true point-in-time safety, not merely group-level safety, because
    every real ranking group already carries a real `ranking_timestamp`
    (`app.ml.real_ranking_decision.RealRankingDecision`) -- sufficient temporal information for
    a real split, not a fabricated one.
  - Training: `app.ml.real_ranker_trainer.fit_real_ranker`, called unmodified on exactly the
    TRAIN-group-filtered `RealRankingDecision` list. Every existing guard (empty input, zero
    trainable groups, unconfirmed volume, unavailable algorithm, the XGBRanker per-group weight
    rejection) still applies unchanged and is never caught here -- see module docstring's
    "does not swallow" note below.

NOT WIRED into `/training`, any HTTP route, Kafka, or the DB -- this module has no import of
any of those. `app.ml.ranking_groups` (the synthetic generator) is never imported either.

EXISTING-GUARD PROPAGATION vs. NEW INSUFFICIENT_DATA CASES: `RealRankingDataUnavailable`,
`InsufficientTrainableGroups`, and `RealDataVolumeNotConfirmed` (raised by the reused
`prepare_real_ranker_training_input`/`fit_real_ranker`) are pre-existing, deliberate, precise
domain errors from an earlier task -- this module does NOT catch or convert them into a
returned result; they propagate exactly as before. Only conditions NEW to this evaluation step
(a chronological split that cannot produce two non-empty partitions; a held-out split with zero
groups clearing `evaluator.py`'s own >=2-candidate ranking-eligibility floor) produce a returned
`INSUFFICIENT_DATA` result instead of raising, since fabricating a comparison from an empty/
trivial split would misrepresent what was actually evaluated.

ACCEPTANCE RULE: no existing repository contract defines a "candidate vs. persisted-baseline"
regression/promotion threshold for ANY model family (searched `app.ml.quality_scorer`,
`app.ml.eligibility_policy`, `app.benchmark.*` -- all of those score/gate candidates against
each other or against fixed behavioral constraints, never against a specific prior artifact's
metrics). `_DECISION_RULE` below is therefore the task's own explicitly-sanctioned conservative
fallback ("candidate must not regress the authoritative primary metrics") -- NOT an existing
repository-authoritative contract. `RealShadowEvaluationResult.decision_rule` says so verbatim
on every result, and PASS here must never be read as "ready to promote" -- an acceptance-policy
contract for real-data shadow candidates still does not exist.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml import model_store
from app.ml.dataset_builder import FEATURES
from app.ml.evaluator import _ndcg_at, _precision_at, _rank_groups
from app.ml.real_data_shadow import ShadowTrainingStatus
from app.ml.real_ranker_trainer import (
    DEFAULT_REAL_RANKER_ALGORITHM,
    fit_real_ranker,
    prepare_real_ranker_training_input,
)
from app.ml.real_ranking_dataset_builder import to_split_ready_frame
from app.ml.real_ranking_decision import RealRankingDecision
from app.ml.splitting import InsufficientSplitDataError, chronological_group_split
from app.ml.training_data_source import TrainingDataSource

__all__ = [
    "PRIMARY_METRICS",
    "RealShadowEvaluationResult",
    "RealShadowEvaluationStatus",
    "evaluate_real_shadow_candidate",
    "try_load_baseline_model",
]

# Matches app.ml.ranker_trainer.TRAIN_GROUP_FRACTION exactly -- the one train/holdout ratio this
# codebase already uses for a group-preserving ranker split, not a newly-invented number.
DEFAULT_TRAIN_RATIO = 0.70

# The two metrics this task requires at minimum, in the exact order deltas/decisions are
# reported. "ndcgAt10" mirrors app.ml.quality_scorer.QUALITY_WEIGHTS' own naming convention and
# is that module's single highest-weighted component (0.30) among its ranking-quality signals --
# already this codebase's own primary ranking-quality metric, not a new choice.
PRIMARY_METRICS: tuple[str, ...] = ("ndcgAt10", "precisionAt5")

# A held-out split must clear evaluator.py's OWN pre-existing ranking-eligibility floor (a group
# needs >=2 candidates to express a ranking at all -- see _rank_groups) for at least this many
# groups before a comparison is meaningful. Not a new number: 1 is the minimum that floor can
# ever produce a non-vacuous evaluatedGroups count.
MIN_EVALUATED_EVAL_GROUPS = 1

_DECISION_RULE = (
    "Conservative fallback rule (see module docstring: no existing repository-authoritative "
    "real-data-shadow acceptance threshold was found) -- PASS requires the candidate to not "
    "regress the baseline on ANY primary metric (ndcgAt10, precisionAt5); a regression on "
    "either is FAIL. This is NOT an existing repository contract and must not be read as "
    "promotion readiness."
)


class RealShadowEvaluationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    # Distinct from INSUFFICIENT_DATA (which is about training/eval DATA volume): there is
    # plenty of data, but no baseline model was supplied/loadable to compare against -- the
    # honest report per this task's own instruction ("return a comparison result without
    # pretending promotion readiness") rather than forcing this into INSUFFICIENT_DATA for a
    # reason that has nothing to do with data volume.
    BASELINE_UNAVAILABLE = "BASELINE_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class RealShadowEvaluationResult:
    status: RealShadowEvaluationStatus
    reason: str
    algorithm: str
    training_data_source: TrainingDataSource
    # Always ShadowTrainingStatus.SHADOW_ONLY on every result this module ever constructs -- a
    # structural property of the PIPELINE (fit_real_ranker's own return type,
    # RealRankerShadowTrainingResult, can only ever declare SHADOW_ONLY; see its __post_init__),
    # not a fact derived per-instance from a specific fitted candidate. Reported explicitly
    # (rather than left implicit) so a downstream reader -- notably the real shadow readiness
    # gate -- never has to trust an unstated upstream invariant to verify "shadow-only".
    shadow_status: ShadowTrainingStatus
    # Echoes the caller-supplied parameter of the same name. Always True on any result that
    # reached a `return` here: `prepare_real_ranker_training_input` already raises
    # RealDataVolumeNotConfirmed before this function can construct anything when False -- so,
    # like shadow_status above, this is a structural guarantee once reached, stored explicitly
    # for the same auditability reason.
    human_confirmed_sufficient_volume: bool
    train_group_count: int
    eval_group_count: int
    train_row_count: int
    eval_row_count: int
    candidate_metrics: dict[str, float] | None
    baseline_metrics: dict[str, float] | None
    absolute_deltas: dict[str, float] | None
    relative_deltas: dict[str, float | None] | None
    group_isolation_verified: bool
    decision_rule: str


def try_load_baseline_model(
    *, model_path: Path | None = None, metadata_path: Path | None = None,
) -> tuple[Any, dict[str, Any]] | None:
    """Read-only attempt to load the real active-serving artifact as this evaluation's
    baseline. `model_store.load_validated` validates schema/feature/checksum compatibility and
    calls `joblib.load` -- it writes nothing, ever (confirmed by reading `app.ml.model_store`'s
    own source: `save()` is a structurally separate function this call never reaches). Returns
    `None` (never raises) on ANY `ArtifactError` -- not found (the common case in a checkout
    with no artifact on disk), incompatible feature contract, corrupted, or busy -- so a caller
    can honestly report `BASELINE_UNAVAILABLE` instead of crashing.
    """
    try:
        return model_store.load_validated(list(FEATURES), model_path=model_path, metadata_path=metadata_path)
    except model_store.ArtifactError:
        return None


def _score(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Minimal duck-typed score adapter -- normalizes the two score interfaces already used
    across this codebase (`predict_proba` for a classifier or an `app.ml.ranker_adapter.
    RankerScorer`-wrapped ranker; bare `predict` for the raw ranker Pipeline
    `app.ml.real_ranker_trainer.fit_real_ranker` returns) into one array. Ranking metrics only
    need relative ordering, so no calibration/normalization decision is made here -- this is
    interface plumbing, not a policy."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(frame))[:, 1]
    return np.asarray(model.predict(frame))


def _rank_metrics(relevance: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> tuple[dict[str, float], int]:
    """`_rank_groups` sorts purely by `scores`/`groups` -- calling it twice with the same
    scores/groups but different `y` (graded vs. binarized relevance) yields the identical
    per-group sort order both times, so this is a correct (if slightly redundant) reuse, not a
    second splitting implementation."""
    graded_ranked, diagnostics = _rank_groups(relevance, scores, groups)
    binary_ranked, _ = _rank_groups((relevance > 0).astype(int), scores, groups)
    metrics = {
        "ndcgAt10": _ndcg_at(graded_ranked, 10),
        "precisionAt5": _precision_at(binary_ranked, 5),
    }
    return metrics, diagnostics["evaluatedGroups"]


def _decide_status(
    candidate_metrics: dict[str, float], baseline_metrics: dict[str, float],
) -> tuple[RealShadowEvaluationStatus, str]:
    regressed = [name for name in PRIMARY_METRICS if candidate_metrics[name] < baseline_metrics[name]]
    if regressed:
        return RealShadowEvaluationStatus.FAIL, f"candidate regressed {regressed} relative to baseline"
    return RealShadowEvaluationStatus.PASS, f"candidate did not regress any of {list(PRIMARY_METRICS)} relative to baseline"


def evaluate_real_shadow_candidate(
    decisions: Sequence[RealRankingDecision],
    *,
    human_confirmed_sufficient_volume: bool,
    algorithm: str = DEFAULT_REAL_RANKER_ALGORITHM,
    random_seed: int = 42,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    baseline_model: Any | None = None,
    baseline_label: str = "baseline",
) -> RealShadowEvaluationResult:
    """Fits `algorithm` on the TRAIN groups only, scores it (and `baseline_model`, if supplied)
    on the disjoint, strictly-later HELD-OUT groups, and returns a structured comparison. Never
    saves, versions, or promotes anything (see module docstring). `baseline_model` must expose
    `predict_proba` or `predict` (see `_score`); pass the result of `try_load_baseline_model`
    (its `[0]`) when a real artifact is available, or `None` to receive `BASELINE_UNAVAILABLE`
    with candidate-only metrics still reported.
    """
    full_result = prepare_real_ranker_training_input(
        decisions, human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
    )
    full_frame = to_split_ready_frame(full_result)

    try:
        splits = chronological_group_split(
            full_frame, ratios={"train": train_ratio, "eval": round(1 - train_ratio, 10)},
            group_col="candidate_group", timestamp_col="timestamp",
        )
    except InsufficientSplitDataError as exc:
        return RealShadowEvaluationResult(
            status=RealShadowEvaluationStatus.INSUFFICIENT_DATA,
            reason=f"chronological group split could not produce non-empty train/eval partitions: {exc}",
            algorithm=algorithm, training_data_source=TrainingDataSource.REAL_OBSERVED,
            shadow_status=ShadowTrainingStatus.SHADOW_ONLY,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            train_group_count=0, eval_group_count=0, train_row_count=0, eval_row_count=0,
            candidate_metrics=None, baseline_metrics=None, absolute_deltas=None, relative_deltas=None,
            group_isolation_verified=True, decision_rule=_DECISION_RULE,
        )

    train_frame, eval_frame = splits["train"], splits["eval"]
    train_group_ids = set(train_frame["candidate_group"])
    eval_group_ids = set(eval_frame["candidate_group"])
    # Re-asserted explicitly, not merely trusted: chronological_group_split already guarantees
    # this (one group -> one block -> one split), but a caller-visible invariant check on the
    # actual returned data costs nothing and is exactly what group_isolation_verified reports.
    group_isolation_verified = train_group_ids.isdisjoint(eval_group_ids)
    assert group_isolation_verified, "chronological_group_split violated group isolation"

    train_decisions = [decision for decision in decisions if decision.ranking_group_id in train_group_ids]

    candidate_result = fit_real_ranker(
        train_decisions, human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
        algorithm=algorithm, random_seed=random_seed,
    )

    eval_relevance = eval_frame["relevance"].to_numpy()
    eval_groups = eval_frame["candidate_group"].to_numpy()
    candidate_scores = _score(candidate_result.model, eval_frame[list(FEATURES)])
    candidate_metrics, evaluated_groups = _rank_metrics(eval_relevance, candidate_scores, eval_groups)

    train_group_count = len(train_group_ids)
    eval_group_count = len(eval_group_ids)
    train_row_count = len(train_frame)
    eval_row_count = len(eval_frame)

    if evaluated_groups < MIN_EVALUATED_EVAL_GROUPS:
        return RealShadowEvaluationResult(
            status=RealShadowEvaluationStatus.INSUFFICIENT_DATA,
            reason=(
                f"only {evaluated_groups} held-out group(s) had >=2 candidates (evaluator.py's own "
                "ranking-metric floor) -- no meaningful ranking comparison possible"
            ),
            algorithm=algorithm, training_data_source=TrainingDataSource.REAL_OBSERVED,
            shadow_status=ShadowTrainingStatus.SHADOW_ONLY,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            train_group_count=train_group_count, eval_group_count=eval_group_count,
            train_row_count=train_row_count, eval_row_count=eval_row_count,
            candidate_metrics=candidate_metrics, baseline_metrics=None, absolute_deltas=None, relative_deltas=None,
            group_isolation_verified=group_isolation_verified, decision_rule=_DECISION_RULE,
        )

    if baseline_model is None:
        return RealShadowEvaluationResult(
            status=RealShadowEvaluationStatus.BASELINE_UNAVAILABLE,
            reason=(
                f"no baseline model supplied/loadable ({baseline_label!r}) -- candidate metrics "
                "computed, but no evidence-backed PASS/FAIL decision is possible without a baseline"
            ),
            algorithm=algorithm, training_data_source=TrainingDataSource.REAL_OBSERVED,
            shadow_status=ShadowTrainingStatus.SHADOW_ONLY,
            human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
            train_group_count=train_group_count, eval_group_count=eval_group_count,
            train_row_count=train_row_count, eval_row_count=eval_row_count,
            candidate_metrics=candidate_metrics, baseline_metrics=None, absolute_deltas=None, relative_deltas=None,
            group_isolation_verified=group_isolation_verified, decision_rule=_DECISION_RULE,
        )

    baseline_scores = _score(baseline_model, eval_frame[list(FEATURES)])
    baseline_metrics, _ = _rank_metrics(eval_relevance, baseline_scores, eval_groups)

    absolute_deltas = {name: round(candidate_metrics[name] - baseline_metrics[name], 6) for name in PRIMARY_METRICS}
    relative_deltas: dict[str, float | None] = {
        name: (round(absolute_deltas[name] / baseline_metrics[name], 6) if baseline_metrics[name] else None)
        for name in PRIMARY_METRICS
    }
    status, reason = _decide_status(candidate_metrics, baseline_metrics)

    return RealShadowEvaluationResult(
        status=status, reason=reason,
        algorithm=algorithm, training_data_source=TrainingDataSource.REAL_OBSERVED,
        shadow_status=ShadowTrainingStatus.SHADOW_ONLY,
        human_confirmed_sufficient_volume=human_confirmed_sufficient_volume,
        train_group_count=train_group_count, eval_group_count=eval_group_count,
        train_row_count=train_row_count, eval_row_count=eval_row_count,
        candidate_metrics=candidate_metrics, baseline_metrics=baseline_metrics,
        absolute_deltas=absolute_deltas, relative_deltas=relative_deltas,
        group_isolation_verified=group_isolation_verified, decision_rule=_DECISION_RULE,
    )
