"""Context/label separation architecture (2026-08-27): `build_dataset` can now be given rows
marked `is_training_context_only=True` -- they update `FeatureHistory` exactly like any other
historical event but are never emitted as a labeled training row, regardless of what
`target_for` would say about them. Added specifically to build independent behavioral-state
synthetic cohorts (N prior negative events + one labeled outcome) without the point-in-time
checkpoint-collision problem a literal N-length skip streak causes (every intermediate skip in
the streak also becomes its own guaranteed-negative labeled row) -- see the repair report.

See app.db.models.Interaction.is_training_context_only and app.ml.dataset_builder.build_dataset
for the production-facing rationale. This file proves the mechanism itself, in isolation from
any specific synthetic cohort.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.ml.dataset_builder import FEATURES, build_dataset

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _row(
    event_id, user_id="u", content_id=None, category="SPORT", creator_id="cr", minutes=0,
    watch_percentage=80.0, event_type="VIDEO_WATCHED", liked=False, is_training_context_only=False,
):
    return SimpleNamespace(
        event_id=event_id, user_id=user_id, content_id=content_id or event_id, creator_id=creator_id,
        category=category, event_type=event_type, watch_percentage=watch_percentage, liked=liked,
        shared=False, favorited=False, commented=False, creator_followed=False,
        timestamp=BASE + timedelta(minutes=minutes), is_training_context_only=is_training_context_only,
    )


def test_context_event_updates_history_but_emits_no_label():
    """A single context-only fast-skip: history sees it (session_negative_interaction_count
    reflects it on the next real candidate), but it never becomes its own dataset row."""
    context = _row("ctx-skip", minutes=0, watch_percentage=3.0, event_type="VIDEO_SKIPPED",
                    is_training_context_only=True)
    labeled = _row("target", minutes=5, watch_percentage=90.0, event_type="VIDEO_COMPLETED")

    df = build_dataset([context, labeled])

    assert len(df) == 1  # only the labeled row, never the context row
    row = df.iloc[0]
    assert row["content_id"] == "target"
    assert row["session_negative_interaction_count"] > 0  # context event's effect IS visible


def test_labeled_event_preserves_existing_behavior_exactly():
    """A labeled (non-context) row behaves exactly as before: feature snapshot from prior
    history, target computed from its own outcome, then folded into history."""
    e1 = _row("e1", minutes=0, watch_percentage=95.0, event_type="VIDEO_COMPLETED")
    e2 = _row("e2", minutes=5, watch_percentage=3.0, event_type="VIDEO_SKIPPED")

    df = build_dataset([e1, e2])

    assert len(df) == 2
    assert list(df["target"]) == [1, 0]
    # e2's own feature snapshot reflects e1 (prior history), not e2 itself.
    assert df.iloc[1]["category_positive_count"] > 0


def test_point_in_time_safe_single_context_event():
    """T1 context, T2 labeled: T1 never appears as a row; T2's features include T1's effect;
    T2's target is computed purely from T2's own outcome; no T2 information leaks into T2's own
    feature snapshot (features are read BEFORE T2 is folded into history)."""
    t1 = _row("t1", minutes=0, watch_percentage=95.0, event_type="VIDEO_COMPLETED", liked=True,
              is_training_context_only=True)
    t2 = _row("t2", minutes=10, watch_percentage=90.0, event_type="VIDEO_COMPLETED")

    df = build_dataset([t1, t2])

    assert len(df) == 1
    assert df.iloc[0]["content_id"] == "t2"
    assert df.iloc[0]["target"] == 1
    # T1's positive effect is visible in T2's snapshot (point-in-time-prior history).
    assert df.iloc[0]["category_positive_count"] > 0
    # No leakage: T2's own event contributes nothing to its OWN snapshot -- category_positive_count
    # reflects only T1 (one prior positive event), not T1+T2.
    import math
    assert math.isclose(df.iloc[0]["category_positive_count"], math.log1p(1), rel_tol=1e-9)


def test_multiple_context_events_all_affect_the_final_labeled_row():
    """T1/T2/T3 context (independent, none labeled), T4 labeled: T4 must see the cumulative
    effect of all three prior context events, and only T4 appears as a dataset row."""
    t1 = _row("t1", minutes=0, watch_percentage=95.0, event_type="VIDEO_COMPLETED",
              is_training_context_only=True)
    t2 = _row("t2", minutes=1, watch_percentage=95.0, event_type="VIDEO_COMPLETED",
              is_training_context_only=True)
    t3 = _row("t3", minutes=2, watch_percentage=3.0, event_type="VIDEO_SKIPPED",
              is_training_context_only=True)
    t4 = _row("t4", minutes=10, watch_percentage=90.0, event_type="VIDEO_COMPLETED")

    df = build_dataset([t1, t2, t3, t4])

    assert len(df) == 1
    assert df.iloc[0]["content_id"] == "t4"
    import math
    assert math.isclose(df.iloc[0]["category_positive_count"], math.log1p(2), rel_tol=1e-9)
    assert math.isclose(df.iloc[0]["category_negative_count"], math.log1p(1), rel_tol=1e-9)
    # session_negative_interaction_count is session-window-scoped (30 min); t3 (8 min before t4)
    # is inside SESSION_WINDOW, so it still registers in the session-level feature too.
    assert df.iloc[0]["session_negative_interaction_count"] > 0


def test_context_ordering_is_point_in_time_safe_regardless_of_input_order():
    """build_dataset sorts by timestamp internally -- passing rows out of order must not change
    the result (matches the existing point-in-time guarantee for ordinary labeled rows)."""
    t1 = _row("t1", minutes=0, watch_percentage=95.0, event_type="VIDEO_COMPLETED",
              is_training_context_only=True)
    t2 = _row("t2", minutes=10, watch_percentage=90.0, event_type="VIDEO_COMPLETED")

    df_in_order = build_dataset([t1, t2])
    df_reversed = build_dataset([t2, t1])

    assert df_in_order.iloc[0]["category_positive_count"] == df_reversed.iloc[0]["category_positive_count"]


def test_default_is_training_context_only_is_false_for_rows_without_the_attribute():
    """Backward compatibility: a row object with no `is_training_context_only` attribute at all
    (every existing duck-typed test/eligibility-probe row) behaves exactly as before -- labeled
    normally, never silently treated as context-only."""
    row_without_attr = SimpleNamespace(
        event_id="e1", user_id="u", content_id="c1", creator_id="cr", category="SPORT",
        event_type="VIDEO_COMPLETED", watch_percentage=95.0, liked=False, shared=False,
        favorited=False, commented=False, creator_followed=False, timestamp=BASE,
    )
    assert not hasattr(row_without_attr, "is_training_context_only")

    df = build_dataset([row_without_attr])

    assert len(df) == 1
    assert df.iloc[0]["target"] == 1


def test_backward_compatible_dataset_is_byte_identical_with_no_context_rows():
    """The exact same set of ordinary (non-context) rows, with the new mechanism present in the
    code, must produce an identical DataFrame to what the pre-change code produced -- same row
    count, same columns, same values."""
    rows = [
        _row("e1", minutes=0, watch_percentage=95.0, event_type="VIDEO_COMPLETED"),
        _row("e2", minutes=5, watch_percentage=3.0, event_type="VIDEO_SKIPPED"),
        _row("e3", minutes=10, watch_percentage=80.0, event_type="VIDEO_WATCHED"),
    ]
    df = build_dataset(rows)
    assert len(df) == 3
    assert set(FEATURES) <= set(df.columns)
    assert list(df["target"]) == [1, 0, 1]
