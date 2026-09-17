"""Central, safe experiment-failure classification.

Raw exception text (`str(exc)`) can legitimately contain a database URL, a credential, an
absolute filesystem path, or other internal detail depending on where in the pipeline it
came from -- that's fine for server-side logs, tied to a request/run id, but it must never
reach a persisted report or the read-only `/experiments` API. This module maps "which stage
of the run failed" (never "what the raw exception said") to one of a small, stable set of
public codes and a fixed -- or, for the one case where it's safe, numerically-parameterized
-- message.
"""
from __future__ import annotations

from dataclasses import dataclass

INSUFFICIENT_EXPERIMENT_DATA = "INSUFFICIENT_EXPERIMENT_DATA"
EXPERIMENT_CONFIGURATION_INVALID = "EXPERIMENT_CONFIGURATION_INVALID"
EXPERIMENT_GENERATION_FAILED = "EXPERIMENT_GENERATION_FAILED"
EXPERIMENT_TRAINING_FAILED = "EXPERIMENT_TRAINING_FAILED"
EXPERIMENT_ARTIFACT_FAILED = "EXPERIMENT_ARTIFACT_FAILED"
EXPERIMENT_REPORT_FAILED = "EXPERIMENT_REPORT_FAILED"

# Fixed, safe messages -- never built from exception text. Keyed by pipeline stage.
_STAGE_TO_FAILURE = {
    "CONFIGURATION": (EXPERIMENT_CONFIGURATION_INVALID, "The experiment definition or requested configuration was invalid."),
    "GENERATION": (EXPERIMENT_GENERATION_FAILED, "Synthetic dataset generation failed before training could start."),
    "TRAINING": (EXPERIMENT_TRAINING_FAILED, "Model training failed during this experiment run."),
    "ARTIFACT": (EXPERIMENT_ARTIFACT_FAILED, "The trained candidate artifact failed validation and was not promoted."),
    "REPORT": (EXPERIMENT_REPORT_FAILED, "The experiment completed but its report could not be persisted."),
}

__all__ = [
    "EXPERIMENT_ARTIFACT_FAILED",
    "EXPERIMENT_CONFIGURATION_INVALID",
    "EXPERIMENT_GENERATION_FAILED",
    "EXPERIMENT_REPORT_FAILED",
    "EXPERIMENT_TRAINING_FAILED",
    "INSUFFICIENT_EXPERIMENT_DATA",
    "ExperimentStageError",
    "SafeFailure",
    "insufficient_data_failure",
    "stage_failure",
]


@dataclass(frozen=True)
class SafeFailure:
    code: str
    message: str


class ExperimentStageError(Exception):
    """Wraps an underlying exception with a known pipeline stage tag, so the caller can map
    it to a safe, stable failure code without ever inspecting/persisting the exception's own
    text. The original exception is kept only for server-side logging (`logger.exception`),
    never for the persisted report or any API response.

    `safe_failure` lets the raiser supply an already-computed `SafeFailure` (used for
    insufficient-data, where the safe message is parameterized from trusted numeric fields
    the raiser already has in hand); when omitted, the caller derives one from `stage` via
    `stage_failure()`."""

    def __init__(self, stage: str, original: BaseException, *, safe_failure: SafeFailure | None = None) -> None:
        super().__init__(stage)
        self.stage = stage
        self.original = original
        self.safe_failure = safe_failure or stage_failure(stage)


def insufficient_data_failure(*, labelled: int, positive: int, negative: int, minimum: int = 100) -> SafeFailure:
    """The one parameterized message -- built entirely from our own trusted numeric
    dataset-summary fields, never from exception text."""
    return SafeFailure(
        code=INSUFFICIENT_EXPERIMENT_DATA,
        message=(
            f"Only {labelled} labelled rows with {positive} positive / {negative} negative "
            f"samples (need at least {minimum} labelled rows containing both classes)."
        ),
    )


def stage_failure(stage: str) -> SafeFailure:
    code, message = _STAGE_TO_FAILURE.get(stage, _STAGE_TO_FAILURE["TRAINING"])
    return SafeFailure(code=code, message=message)
