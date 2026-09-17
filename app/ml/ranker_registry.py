"""Ranking-native challenger algorithm registry (Task 7) -- mirrors app.ml.algorithm_registry's
shape and conventions exactly, but for the 3 ranking-objective estimators this task adds as
CHALLENGERS ONLY (never inserted into app.ml.algorithm_registry's classifier candidate pool;
see app.ml.ranker_trainer). RandomForest/LogisticRegression have no natural ranking-native
equivalent in this stack and are deliberately not given one here (spec section 6).

Fixed, small, conservative-default hyperparameters per candidate -- this is an objective-type
comparison, not a hyperparameter search (same "no GridSearchCV/Optuna" spec every other
algorithm registry in this codebase already follows).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.core import config

# Same fixed tree-size/depth/learning-rate choices app.ml.algorithm_registry already uses for
# the classifier counterparts, so any behavioral difference measured between a classifier and
# its ranking counterpart is attributable to the OBJECTIVE, not to a different model capacity.
RANKER_N_ESTIMATORS = 200
RANKER_MAX_DEPTH = 4
RANKER_NUM_LEAVES = 31
RANKER_LEARNING_RATE = 0.1

# One objective per library, chosen for stability/support rather than swept (spec section 17:
# "one small objective comparison is acceptable" -- not a search). rank:ndcg directly optimizes
# the metric this whole project already reports (NDCG@K); lambdarank is LightGBM's standard,
# well-supported listwise ranking loss; YetiRank is CatBoost's own recommended default ranking
# loss (its docs list it ahead of the pairwise-only alternatives for listwise-style relevance).
XGBOOST_RANKER_OBJECTIVE = "rank:ndcg"
LIGHTGBM_RANKER_OBJECTIVE = "lambdarank"
CATBOOST_RANKER_OBJECTIVE = "YetiRank"


@dataclass(frozen=True)
class RankerSpec:
    name: str
    build_estimator: Callable[[int], Any]
    group_param_name: str  # "group" (XGBoost/LightGBM: group SIZES) or "group_id" (CatBoost: per-row group IDs)
    # XGBoost's ranking objectives interpret `sample_weight` as ONE WEIGHT PER QUERY GROUP when
    # `group` is set (a real library constraint, confirmed empirically -- XGBoostError "Size of
    # weight must equal to the number of query groups"), not per-candidate like every other
    # estimator in this codebase. LightGBM/CatBoost's ranking objectives both accept a per-row
    # weight. This means the explicit-rejection-vs-generic-skip distinction (app.ml.
    # ranking_groups.row_weight) can only be preserved at candidate granularity for the
    # libraries where `weight_granularity == "per_row"` -- see app.ml.ranker_trainer.
    weight_granularity: str  # "per_row" | "per_group"
    enabled: bool
    unavailable_reason: str | None


def _xgboost_ranker_estimator(random_seed: int) -> Any:
    from xgboost import XGBRanker
    return XGBRanker(
        objective=XGBOOST_RANKER_OBJECTIVE, n_estimators=RANKER_N_ESTIMATORS, max_depth=RANKER_MAX_DEPTH,
        learning_rate=RANKER_LEARNING_RATE, subsample=0.9, colsample_bytree=0.9, random_state=random_seed, n_jobs=1,
    )


def _lightgbm_ranker_estimator(random_seed: int) -> Any:
    from lightgbm import LGBMRanker
    return LGBMRanker(
        objective=LIGHTGBM_RANKER_OBJECTIVE, n_estimators=RANKER_N_ESTIMATORS, num_leaves=RANKER_NUM_LEAVES,
        learning_rate=RANKER_LEARNING_RATE, random_state=random_seed, n_jobs=1, verbose=-1,
        # Fast-follow fix: LightGBM's sklearn wrapper does not subsample rows by default, so
        # `random_state` had no effective randomness to act on (every estimator seed produced
        # a byte-identical fitted model -- confirmed empirically, not a real robustness result).
        # `subsample`/`subsample_freq` are LGBMRanker's own constructor aliases for LightGBM's
        # native bagging_fraction/bagging_freq -- 0.9 matches XGBRanker's own subsample=0.9
        # immediately above (same conservative magnitude, not tuned), and subsample_freq=1
        # (bag every iteration) is required for `subsample` to have any effect at all in
        # LightGBM -- a subsample fraction with subsample_freq=0 is silently ignored.
        subsample=0.9, subsample_freq=1,
    )


def _catboost_ranker_estimator(random_seed: int) -> Any:
    from catboost import CatBoostRanker
    return CatBoostRanker(
        loss_function=CATBOOST_RANKER_OBJECTIVE, iterations=RANKER_N_ESTIMATORS, depth=6,
        learning_rate=RANKER_LEARNING_RATE, random_seed=random_seed, verbose=False, thread_count=1,
        allow_writing_files=False,
    )


def _unavailable_reason(build_estimator: Callable[[int], Any]) -> str | None:
    """Mirrors app.ml.algorithm_registry._unavailable_reason: catches OSError alongside
    ImportError, since a present-but-broken native extension (e.g. LightGBM's libgomp.so.1
    missing from the runtime image) fails with OSError at import time, not ImportError -- and
    this runs eagerly at module import (ALL_RANKER_NAMES), so an uncaught OSError here would
    crash application startup entirely instead of just marking one ranker unavailable."""
    try:
        build_estimator(0)
    except (ImportError, OSError) as exc:
        return str(exc)
    return None


def _ranker_specs() -> dict[str, RankerSpec]:
    return {
        "XGBRanker": RankerSpec(
            name="XGBRanker", build_estimator=_xgboost_ranker_estimator, group_param_name="group",
            weight_granularity="per_group",
            enabled=config.TRAINING_ENABLE_XGBOOST, unavailable_reason=_unavailable_reason(_xgboost_ranker_estimator),
        ),
        "LGBMRanker": RankerSpec(
            name="LGBMRanker", build_estimator=_lightgbm_ranker_estimator, group_param_name="group",
            weight_granularity="per_row",
            enabled=config.TRAINING_ENABLE_LIGHTGBM, unavailable_reason=_unavailable_reason(_lightgbm_ranker_estimator),
        ),
        "CatBoostRanker": RankerSpec(
            name="CatBoostRanker", build_estimator=_catboost_ranker_estimator, group_param_name="group_id",
            weight_granularity="per_row",
            enabled=config.TRAINING_ENABLE_CATBOOST, unavailable_reason=_unavailable_reason(_catboost_ranker_estimator),
        ),
    }


ALL_RANKER_NAMES: tuple[str, ...] = tuple(_ranker_specs())

# XGBRanker promotion task, item 3 (initial production ranker scope): of the 3 ranker
# challengers above, only XGBRanker graduates into app.ml.trainer.train_and_select_cross_family's
# production candidate pool. LGBMRanker remains ineligible from prior robustness work (Task 7
# measured it as the least stable of the three across seeds); CatBoostRanker stays
# research/benchmark-only -- neither has been through the same behavioral-eligibility scrutiny
# XGBRanker has. Deliberately a short, explicit allowlist rather than "every available ranker"
# -- expanding it is a separate, deliberate decision, not a side effect of a library becoming
# importable.
PRODUCTION_RANKER_NAMES: tuple[str, ...] = ("XGBRanker",)

# Classifier -> its ranking-objective counterpart, for the required pairwise comparison
# (spec: "Required classifier vs ranker comparison" / "Required pairwise comparison").
CLASSIFIER_TO_RANKER: dict[str, str] = {
    "XGBoostClassifier": "XGBRanker",
    "LightGBMClassifier": "LGBMRanker",
    "CatBoostClassifier": "CatBoostRanker",
}


def build_candidate_rankers(
    random_seed: int, *, only: tuple[str, ...] | None = None,
) -> dict[str, tuple[Any, str, str]]:
    """Every enabled, available ranker candidate -> (unfitted estimator, group_param_name,
    weight_granularity). Mirrors app.ml.algorithm_registry.build_candidate_pipelines's
    availability/enabled semantics exactly (operator-enabled AND dependency-importable).

    `only`, when given, further restricts the pool to names in `only` (still subject to the
    same enabled/available checks -- unlike app.ml.algorithm_registry's `only`, this is a
    filter, not an override, since production selection never wants an operator-disabled or
    unavailable ranker forced in). Used by app.ml.ranker_trainer.train_ranking_challengers to
    train just PRODUCTION_RANKER_NAMES for the cross-family production pool while leaving the
    full-challenger-set default behavior (`only=None`) unchanged for the existing benchmark/
    research entrypoint."""
    specs = _ranker_specs()
    names = set(only) if only is not None else None
    return {
        name: (spec.build_estimator(random_seed), spec.group_param_name, spec.weight_granularity)
        for name, spec in specs.items()
        if spec.enabled and spec.unavailable_reason is None and (names is None or name in names)
    }


def unavailable_rankers() -> dict[str, str]:
    return {name: spec.unavailable_reason for name, spec in _ranker_specs().items() if spec.unavailable_reason}
