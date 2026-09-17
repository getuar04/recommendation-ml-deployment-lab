"""Central candidate-algorithm registry for VIDEO model training (app.ml.trainer).

Adding a new candidate algorithm means adding one entry to `_algorithm_specs()` below --
never editing the trainer, the eligibility gates, the evaluator, or the artifact lifecycle,
all of which already work generically against any `Pipeline` exposing `predict_proba`.

Optional algorithms (XGBoost, LightGBM, CatBoost) degrade gracefully: an uninstalled or
operator-disabled optional algorithm is simply absent from the pool `build_candidate_pipelines`
returns, with the reason recorded by `unavailable_algorithms()` -- never a crash at import
time or training time. Tree/boosted-tree models skip numeric scaling (RobustScaler is only
useful to a linear model like LogisticRegression); see `app.ml.pipeline_builder`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.core import config
from app.ml.pipeline_builder import build_classifier_pipeline

# Fixed, small, production-minded hyperparameters per candidate -- this is algorithm
# comparison, not a hyperparameter search (spec: no GridSearchCV/Optuna/AutoML here).
XGBOOST_N_ESTIMATORS = 200
XGBOOST_MAX_DEPTH = 4
LIGHTGBM_N_ESTIMATORS = 200
LIGHTGBM_NUM_LEAVES = 31
CATBOOST_ITERATIONS = 200
CATBOOST_DEPTH = 6
LEARNING_RATE = 0.1


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    build_estimator: Callable[[int], Any]  # random_seed -> unfitted sklearn-compatible estimator.
    scale_numeric: bool  # True for linear models (LogisticRegression); False for tree/boosted-tree models.
    enabled: bool  # Operator kill switch (config flag); independent of whether the dependency is importable.
    unavailable_reason: str | None  # None if this algorithm's dependency imports cleanly.


def _logistic_regression_estimator(random_seed: int) -> Any:
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed)


def _random_forest_estimator(random_seed: int) -> Any:
    return RandomForestClassifier(n_estimators=120, class_weight="balanced", random_state=random_seed, n_jobs=1)


def _xgboost_estimator(random_seed: int) -> Any:
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=XGBOOST_N_ESTIMATORS, max_depth=XGBOOST_MAX_DEPTH, learning_rate=LEARNING_RATE,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=random_seed, n_jobs=1,
    )


def _lightgbm_estimator(random_seed: int) -> Any:
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=LIGHTGBM_N_ESTIMATORS, num_leaves=LIGHTGBM_NUM_LEAVES, learning_rate=LEARNING_RATE,
        class_weight="balanced", random_state=random_seed, n_jobs=1, verbose=-1,
    )


def _catboost_estimator(random_seed: int) -> Any:
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=CATBOOST_ITERATIONS, depth=CATBOOST_DEPTH, learning_rate=LEARNING_RATE,
        auto_class_weights="Balanced", random_seed=random_seed, verbose=False, thread_count=1,
        allow_writing_files=False,
    )


def _unavailable_reason(build_estimator: Callable[[int], Any]) -> str | None:
    """None if `build_estimator`'s dependency imports and constructs cleanly; otherwise the
    concrete error, so an operator/log reader sees exactly why an algorithm is missing
    instead of it silently vanishing from the candidate pool. Catches OSError alongside
    ImportError: a present-but-broken native extension (e.g. LightGBM's libgomp.so.1 missing
    from the runtime image) fails with OSError at import time, not ImportError -- and this
    runs eagerly at module import (ALL_ALGORITHM_NAMES), so an uncaught OSError here would
    crash application startup entirely instead of just marking one algorithm unavailable."""
    try:
        build_estimator(0)
    except (ImportError, OSError) as exc:
        return str(exc)
    return None


def _algorithm_specs() -> dict[str, AlgorithmSpec]:
    return {
        "LogisticRegression": AlgorithmSpec(
            name="LogisticRegression", build_estimator=_logistic_regression_estimator,
            scale_numeric=True, enabled=True, unavailable_reason=None,
        ),
        "RandomForestClassifier": AlgorithmSpec(
            name="RandomForestClassifier", build_estimator=_random_forest_estimator,
            scale_numeric=False, enabled=True, unavailable_reason=None,
        ),
        "XGBoostClassifier": AlgorithmSpec(
            name="XGBoostClassifier", build_estimator=_xgboost_estimator, scale_numeric=False,
            enabled=config.TRAINING_ENABLE_XGBOOST, unavailable_reason=_unavailable_reason(_xgboost_estimator),
        ),
        "LightGBMClassifier": AlgorithmSpec(
            name="LightGBMClassifier", build_estimator=_lightgbm_estimator, scale_numeric=False,
            enabled=config.TRAINING_ENABLE_LIGHTGBM, unavailable_reason=_unavailable_reason(_lightgbm_estimator),
        ),
        "CatBoostClassifier": AlgorithmSpec(
            name="CatBoostClassifier", build_estimator=_catboost_estimator, scale_numeric=False,
            enabled=config.TRAINING_ENABLE_CATBOOST, unavailable_reason=_unavailable_reason(_catboost_estimator),
        ),
    }


ALL_ALGORITHM_NAMES: tuple[str, ...] = tuple(_algorithm_specs())


def build_candidate_pipelines(
    random_seed: int, *, categorical: list[str], numeric: list[str], only: str | None = None,
) -> dict[str, Pipeline]:
    """Every enabled, available candidate's fit-ready `Pipeline`, keyed by algorithm name.

    `only`, when given, restricts the pool to exactly that one named algorithm -- used by
    `app.ml.trainer._candidate_models`'s `restrict_algorithm`/`TRAINING_ALGORITHM_LOCK` for
    "same algorithm, different dataset" controlled experiments -- regardless of that
    algorithm's own enabled/available state (an explicit request for a specific algorithm is
    a stronger signal than the general-purpose default pool), so a locked run still fits
    that one algorithm; raises if the name is not a real algorithm at all. Without `only`,
    the pool is every algorithm that is both operator-enabled and whose dependency actually
    imported (see `unavailable_algorithms()`), never a partially-broken candidate.
    """
    specs = _algorithm_specs()
    if only is not None:
        if only not in specs:
            raise ValueError(f"restrict_algorithm must be one of {sorted(specs)} (got {only!r})")
        names = [only]
    else:
        names = [name for name, spec in specs.items() if spec.enabled and spec.unavailable_reason is None]
    return {
        name: build_classifier_pipeline(
            specs[name].build_estimator(random_seed),
            categorical=categorical, numeric=numeric, scale_numeric=specs[name].scale_numeric,
        )
        for name in names
    }


def unavailable_algorithms() -> dict[str, str]:
    """name -> reason, for every algorithm whose dependency did not import cleanly. Never
    raises -- a missing optional library is a normal, expected deployment state (see module
    docstring), not an error condition."""
    return {name: spec.unavailable_reason for name, spec in _algorithm_specs().items() if spec.unavailable_reason}
