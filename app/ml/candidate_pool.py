"""Common candidate representation for cross-family production model selection (XGBRanker
promotion task): the smallest shared shape a classifier candidate (app.ml.trainer) and a
ranking-native candidate (app.ml.ranker_trainer) can both be normalized into, WITHOUT pretending
they share identical training metrics.

Deliberately not a big framework: `TrainedModelCandidate` only carries what selection actually
needs (identity, the fitted model itself, family/objective/score-semantics labels, the PHASE 1
eligibility decision, and the PHASE 2 cross-family quality score), plus each candidate's own
family-native diagnostics (`metrics` for a classifier, `native_quality` for either family) kept
around for reporting -- never fabricated for the family it doesn't apply to (see module
docstring on `metrics`/`native_quality` below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrainedModelCandidate:
    name: str
    model: Any
    model_family: str  # "classifier" | "ranker"
    objective: str
    score_semantics: str  # "probability" (classifier) | "relative_relevance_score" (ranker)
    eligible: bool
    eligibility: dict[str, Any]  # app.ml.eligibility_policy.decide() result
    # app.ml.quality_scorer.quality_score_cross_family(...) -- the ONLY score cross-family
    # SELECTION may use (see select_cross_family_winner below): a pure function of this
    # candidate's own independent end-to-end benchmark result, defined and computed identically
    # regardless of family, so neither family's own training-time metrics can tilt selection.
    cross_family_quality: dict[str, Any]
    # This candidate's own family-native quality score (app.ml.quality_scorer.quality_score for
    # a classifier, quality_score_for_ranker for a ranker) -- a useful diagnostic, but NEVER
    # comparable in absolute magnitude across families (different weight normalization; see
    # app.ml.quality_scorer's own docstring) and never used for cross-family selection.
    native_quality: dict[str, Any]
    # A classifier's modelSelection-split binary/ranking evaluation (app.ml.evaluator.evaluate).
    # None for a ranker -- it has no such split, and PR-AUC/precision/recall/logLoss must never
    # be fabricated for a ranking-native model (spec: use None/absent, not 0.0, where a metric
    # genuinely does not apply).
    metrics: dict[str, Any] | None
    training_duration_seconds: float
    # Family-specific extra fields the eventual artifact metadata needs (e.g. a ranker's
    # normalizationStrategy/normalizationScale/rankingGroupVersion/groupParamName) -- empty for
    # a classifier, whose full metadata is already assembled by the existing classifier tail.
    metadata: dict[str, Any] = field(default_factory=dict)


def _tie_break_key(candidate: TrainedModelCandidate) -> tuple[Any, ...]:
    """Deterministic total order, best-first, mirroring app.ml.model_selection's tie-break
    style: 1) higher cross-family quality score, 2) higher criticalPassRate (the sharpest
    behavioral-safety signal both families produce identically), 3) higher HARD-difficulty
    final NDCG (HARD/ADVERSARIAL matter more than MEDIUM -- same priority app.ml.quality_scorer
    already encodes), 4) shorter training duration, 5) name, alphabetically -- final,
    always-deterministic fallback."""
    components = candidate.cross_family_quality.get("components", {})
    return (
        -candidate.cross_family_quality["score"],
        -float(components.get("criticalPassRate") or 0.0),
        -float(components.get("hardFinalNdcg") or 0.0),
        float(candidate.training_duration_seconds or 0.0),
        candidate.name,
    )


def select_cross_family_winner(candidates: dict[str, TrainedModelCandidate]) -> dict[str, Any]:
    """PHASE 1 (already decided per-candidate into `candidate.eligible`): restrict the pool to
    eligible candidates, of EITHER family. If none are eligible, fall back to ranking every
    candidate anyway (so a caller can still see/report a ranking) -- mirroring
    app.ml.model_selection.select_winner_v2's own no-eligible-candidate fallback semantics --
    but `eligibleSelection=False` is returned so the caller must refuse to promote it.
    PHASE 2: rank the pool by cross-family quality score, deterministic tie-break above.

    Returns {"selected", "eligibleSelection", "eligibleCandidates", "rankedOrder",
    "crossFamilyScores", "modelFamilies"} -- enough to audit why a candidate won across
    families without hiding disagreement."""
    if not candidates:
        raise ValueError("select_cross_family_winner requires at least one candidate.")
    eligible_names = [name for name, c in candidates.items() if c.eligible]
    eligible_selection = bool(eligible_names)
    pool = eligible_names if eligible_selection else list(candidates)
    ranked = sorted(pool, key=lambda name: _tie_break_key(candidates[name]))
    return {
        "selected": ranked[0],
        "eligibleSelection": eligible_selection,
        "eligibleCandidates": eligible_names,
        "rankedOrder": ranked,
        "crossFamilyScores": {name: candidates[name].cross_family_quality["score"] for name in pool},
        "modelFamilies": {name: candidates[name].model_family for name in candidates},
    }
