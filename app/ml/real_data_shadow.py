"""Baseline protection and shadow-only status for the future real-data challenger.

Deliberately domain-invariant only -- no serving code, no promotion mechanism, and no change to
`app.ml.model_store`'s existing active/previous/candidate promotion machinery. The RMS training-
path reconciliation task found that training-path reproducibility for the active artifact is NOT
proven from this checkout; this module makes sure nothing in the future real-data workflow can
paper over that gap by silently overwriting, relabeling, or bypassing the trusted baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class TrustedBaselineReference:
    model_version: str


# The trusted, frozen serving baseline (see the verified-baseline context carried across the RMS
# real-data readiness tasks). Not re-derived from any file in this checkout -- the reconciliation
# task found no model_metadata.json for this artifact present locally -- this is a reference
# value only, to be updated exclusively when a new artifact is explicitly promoted through the
# existing, separate `app.ml.model_store` promotion path, never as a side effect of real-data
# training work.
ACTIVE_SERVING_BASELINE = TrustedBaselineReference(model_version="recommendation-prod-20260831095040")


class ShadowTrainingStatus(str, Enum):
    SHADOW_ONLY = "SHADOW_ONLY"


@dataclass(frozen=True, slots=True)
class ShadowChallengerResult:
    """A future real-data challenger's evaluation result. Can only ever be constructed as
    SHADOW_ONLY through this type -- there is no method here that produces any other status, and
    promotion (making a challenger the active serving artifact) is a separate, explicit action
    entirely outside this module, unaffected by anything here."""

    model_version: str
    baseline: TrustedBaselineReference
    status: ShadowTrainingStatus = ShadowTrainingStatus.SHADOW_ONLY

    def __post_init__(self) -> None:
        if not self.model_version or not self.model_version.strip():
            raise ValueError("model_version must be a non-blank string")
        if self.model_version == self.baseline.model_version:
            raise ValueError(
                f"a shadow challenger must never claim the trusted baseline's own modelVersion "
                f"({self.baseline.model_version!r}) -- this would silently relabel or overwrite "
                "the baseline reference"
            )
        if self.status is not ShadowTrainingStatus.SHADOW_ONLY:
            raise ValueError(
                "ShadowChallengerResult can only be constructed as SHADOW_ONLY -- promotion is a "
                "separate, explicit action outside this module"
            )


def assert_not_baseline_overwrite(
    candidate_model_version: str, *, baseline: TrustedBaselineReference = ACTIVE_SERVING_BASELINE,
) -> None:
    """Domain-invariant guard for a future real-data artifact save/promotion call site to call
    before ever writing an artifact. Not wired into any current save path (no real training
    exists yet) and never called from serving."""
    if candidate_model_version == baseline.model_version:
        raise ValueError(f"refusing to write over trusted baseline modelVersion {baseline.model_version!r}")
