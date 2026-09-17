"""Adaptation latency (Step 11: research/debug metric, explicitly NOT a production SLA): how
many strong same-category session signals are needed before session_category_affinity for a
newly emerging interest exceeds session_category_affinity for the previously-dominant one.

Deliberately measured directly from `FeatureHistory.features()`'s session_category_affinity,
not through a trained model's score: the model's score also depends on long-term/recent/
semantic/popularity signals and on which algorithm training happened to select at any given
moment (see tests/test_session_switch_production.py's module docstring for a detailed,
investigated account of why this repo's bulk synthetic trainer is not currently a reliable,
deterministic basis for a score-based measurement). This metric isolates and measures the
*responsiveness of the session feature engineering itself*, which is what "how fast does
current-session intent adapt" is actually asking -- and it is 100% reproducible, unlike a
score derived from an unseeded, non-deterministic training run.

No I/O, no database access, no model loading -- reusable directly in a debug script, a
notebook, or an ad hoc investigation.
"""
from __future__ import annotations

from typing import Any

from app.ml.dataset_builder import FeatureHistory


def measure_adaptation_latency(
    history: FeatureHistory, *, user_id: str, previous_category: str, new_category: str,
    new_category_events: list[Any],
) -> dict[str, Any]:
    """Replays `new_category_events` (chronologically ordered rows in `new_category`) one at a
    time through `history.update()`, recording session_category_affinity for both
    `previous_category` and `new_category` after each event.

    Returns `{"crossoverAtEventNumber": int | None, "trace": [...]}` -- the 1-based index of
    the first event after which `new_category` outranks `previous_category` on
    session_category_affinity (None if it never does within the given events), plus the full
    per-event trace for inspection.
    """
    trace: list[dict[str, Any]] = []
    crossover_at_event_number: int | None = None
    for event_number, event in enumerate(new_category_events, start=1):
        history.update(event)
        as_of = event.timestamp
        previous_snapshot = history.features(
            user_id=user_id, category=previous_category, creator_id="latency-probe",
            content_id="latency-probe-previous", timestamp=as_of, content_popularity_score=0.5,
            content_created_at=as_of,
        )
        new_snapshot = history.features(
            user_id=user_id, category=new_category, creator_id="latency-probe",
            content_id="latency-probe-new", timestamp=as_of, content_popularity_score=0.5,
            content_created_at=as_of,
        )
        previous_affinity = float(previous_snapshot["session_category_affinity"])
        new_affinity = float(new_snapshot["session_category_affinity"])
        new_outranks_previous = new_affinity > previous_affinity
        if new_outranks_previous and crossover_at_event_number is None:
            crossover_at_event_number = event_number
        trace.append({
            "eventNumber": event_number,
            "previousCategorySessionAffinity": previous_affinity,
            "newCategorySessionAffinity": new_affinity,
            "newCategoryIntentConfidence": float(new_snapshot["session_intent_confidence"]),
            "newOutranksPrevious": new_outranks_previous,
        })
    return {"crossoverAtEventNumber": crossover_at_event_number, "trace": trace}
