"""HARD vs. SOFT severity classification for app.ml.eligibility's raw ModelBehavior gates.

Task 5 finding: measuring every gate's pass/margin for all 5 candidate algorithms on this
project's real synthetic dataset (seed=42) showed two clearly different failure regimes:

    gate                              worst-case margin among the 4 non-LR algorithms
    negative / notInterested          -0.41 / -0.42  (large, decisive direction reversal)
    longTerm / creator / coldStart    -0.13 / -0.01 / -0.16  (small, noise-scale)
    recent / session / semantic       currently 0 failures across all 5 algorithms

`negative` and `notInterested` failing means the model scores a just-explicitly-rejected (or
just-fast-skipped) category HIGHER than an unsuppressed baseline by nearly half the probability
range -- a real, structural, product-dangerous behavior (serving MORE of what a user just
rejected), not measurement noise. `recent`/`session`/`semantic` currently pass everywhere with
healthy (0.17-0.41) margins, so requiring them is free insurance against a future regression on
an easy, well-learned signal. All five are therefore HARD: failing any of them must keep
rejecting a candidate outright, regardless of how good its end-to-end ranking quality is.

`longTerm`/`creator`/`coldStart`/`subthemeRejectionLocalization`, by contrast, show only small
-magnitude, second-order-feature-interaction failures -- real quality signal, but not the kind
of failure that actively harms a user the way re-serving rejected content does. These are SOFT:
they lower a candidate's quality score but never by themselves reject it, so "one weak non-
safety monotonicity probe" (Task 5 spec) cannot eliminate a model that is clearly superior
end-to-end.

This module only classifies and summarizes; it never changes gate logic, tolerances, or
`app.ml.eligibility.evaluate_eligibility`'s own all-gates-must-pass `eligible` flag.
"""
from __future__ import annotations

from typing import Any

from app.ml.eligibility import GATE_NAMES, evaluate_eligibility

HARD_GATES: tuple[str, ...] = ("negative", "notInterested", "recent", "session", "semantic")
SOFT_GATES: tuple[str, ...] = ("longTerm", "creator", "coldStart", "subthemeRejectionLocalization")

if sorted(HARD_GATES + SOFT_GATES) != sorted(GATE_NAMES):
    raise AssertionError("HARD_GATES + SOFT_GATES must exactly partition app.ml.eligibility.GATE_NAMES")


def severity_report(model: Any) -> dict[str, Any]:
    """Runs `evaluate_eligibility` once and reclassifies the result by severity. Returns:

        {"hardPassed": int, "hardTotal": int, "hardFailedGates": [str, ...],
         "softPassed": int, "softTotal": int, "softFailedGates": [str, ...],
         "hardEligible": bool,   # every HARD gate passed
         "gateMargins": {gate: float, ...}}
    """
    report = evaluate_eligibility(model)
    return severity_report_from_eligibility(report)


def severity_report_from_eligibility(report: dict[str, Any]) -> dict[str, Any]:
    """Same as `severity_report`, but reclassifies an already-computed `evaluate_eligibility()`
    report -- avoids re-scoring every probe a second time when the caller already has it.
    Deliberately does not re-embed `report` itself: a caller persisting this alongside the
    original `eligibility` report (e.g. app.services.training_service's metadata) should not
    store the same gate pass/margin data twice (Task 5 spec: keep metadata structured, not
    unnecessarily large)."""
    gates = report["gates"]
    hard_failed = [name for name in HARD_GATES if not gates[name]["pass"]]
    soft_failed = [name for name in SOFT_GATES if not gates[name]["pass"]]
    return {
        "hardPassed": len(HARD_GATES) - len(hard_failed),
        "hardTotal": len(HARD_GATES),
        "hardFailedGates": hard_failed,
        "softPassed": len(SOFT_GATES) - len(soft_failed),
        "softTotal": len(SOFT_GATES),
        "softFailedGates": soft_failed,
        "hardEligible": not hard_failed,
        "gateMargins": {name: g["margin"] for name, g in gates.items()},
    }
