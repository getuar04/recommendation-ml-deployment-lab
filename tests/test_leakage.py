import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.ml.dataset_builder import (
    MAX_CONTENT_AGE_HOURS,
    RECENT_WINDOW,
    FeatureHistory,
    build_dataset,
)


def _event(user_id="u", content_id="c1", category="FOOD", creator_id="cr", minutes=0,
           watch_percentage=80.0, liked=False, shared=False, favorited=False, creator_followed=False,
           commented=False, event_type="VIDEO_WATCHED", base=None):
    base = base or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SimpleNamespace(
        event_id=f"{content_id}-{minutes}", user_id=user_id, content_id=content_id, creator_id=creator_id,
        category=category, event_type=event_type, watch_percentage=watch_percentage, liked=liked, shared=shared,
        favorited=favorited, commented=commented, creator_followed=creator_followed, timestamp=base + timedelta(minutes=minutes),
    )


def test_current_event_does_not_contribute_to_its_own_features():
    row = _event(minutes=0, watch_percentage=95, liked=True)
    df = build_dataset([row])
    assert len(df) == 1
    features = df.iloc[0]
    # A single event with no prior history must look like a cold-start row, regardless of its own label.
    assert features["has_category_history"] == 0
    assert features["has_creator_history"] == 0
    assert features["category_affinity"] == 0.5
    assert features["recent_category_affinity"] == 0.5
    assert features["average_category_watch_percentage"] == 0.0


def test_future_events_do_not_affect_past_rows():
    early = _event(content_id="c1", minutes=0, watch_percentage=80)
    later = _event(content_id="c2", minutes=10, watch_percentage=95, liked=True)
    df_with_future = build_dataset([early, later])
    df_without_future = build_dataset([early])
    early_row_with = df_with_future[df_with_future.content_id == "c1"].iloc[0]
    early_row_without = df_without_future.iloc[0]
    for column in ("category_affinity", "recent_category_affinity", "average_category_watch_percentage", "has_category_history"):
        assert early_row_with[column] == early_row_without[column]


# --- Phase 1.5 issue 3: explicit CONTENT_NOT_INTERESTED point-in-time correctness ---

def test_not_interested_event_gets_its_own_negative_label_without_using_its_own_state():
    """The explicit-rejection row's label must be 0 (see app.ml.feature_builder.target_for),
    but -- like every other row -- its features must still reflect history strictly BEFORE
    it, never its own negative state (same invariant test_current_event_does_not_contribute_
    to_its_own_features already proves for the ordinary case)."""
    row = _event(minutes=0, watch_percentage=40, event_type="CONTENT_NOT_INTERESTED")
    df = build_dataset([row])
    assert len(df) == 1
    features = df.iloc[0]
    assert features["target"] == 0
    assert features["has_category_history"] == 0
    assert features["category_affinity"] == 0.5
    assert features["category_negative_count"] == 0.0


def test_not_interested_state_only_affects_subsequent_rows():
    """The explicit-rejection row's negative state must be invisible to itself but visible to
    a later row in the same category -- point-in-time correctness cuts both ways."""
    rejection = _event(content_id="c1", minutes=0, watch_percentage=40, event_type="CONTENT_NOT_INTERESTED")
    later_positive = _event(content_id="c2", minutes=5, watch_percentage=95, event_type="VIDEO_COMPLETED", liked=True)
    df = build_dataset([rejection, later_positive])
    assert len(df) == 2
    rejection_row = df[df.content_id == "c1"].iloc[0]
    later_row = df[df.content_id == "c2"].iloc[0]
    assert rejection_row["target"] == 0
    assert rejection_row["has_category_history"] == 0  # unaffected by its own rejection
    assert later_row["target"] == 1
    assert later_row["has_category_history"] == 1  # now sees the prior rejection
    assert later_row["category_negative_count"] > 0.0
    assert later_row["category_affinity"] < 0.5  # the prior rejection pulled it below neutral


# --- recent_category_affinity: recent/session vs. long-term category behavior ---

def test_recent_category_affinity_ignores_events_outside_recent_window():
    """A strong positive event older than RECENT_WINDOW must still raise long-term
    category_affinity but must NOT contribute to recent_category_affinity at all."""
    history = FeatureHistory()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    old_event_time = base - RECENT_WINDOW - timedelta(days=1)
    history.update(_event(category="SPORT", watch_percentage=95, liked=True, shared=True,
                           event_type="VIDEO_COMPLETED", base=old_event_time, minutes=0))
    features = history.features(
        user_id="u", category="SPORT", creator_id="cr", content_id="new", timestamp=base,
        content_popularity_score=.5, content_created_at=base,
    )
    assert features["category_affinity"] > 0.5
    assert features["recent_category_affinity"] == 0.5


def test_recent_category_affinity_reflects_recent_positive_session():
    """A recent, strong positive session (completed+liked+shared) inside RECENT_WINDOW must
    push recent_category_affinity well above neutral, even with zero older history."""
    history = FeatureHistory()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session_time = base - timedelta(hours=2)
    history.update(_event(category="MUSIC", watch_percentage=95, liked=True, shared=True,
                           event_type="VIDEO_COMPLETED", base=session_time, minutes=0))
    features = history.features(
        user_id="u", category="MUSIC", creator_id="cr", content_id="new", timestamp=base,
        content_popularity_score=.5, content_created_at=base,
    )
    assert features["recent_category_affinity"] > 0.7
    # No older history at all: long-term and recent affinity agree in this specific case
    # (the user's entire category history happens to be inside the window).
    assert features["category_affinity"] == features["recent_category_affinity"]


def test_recent_category_affinity_reflects_recent_negative_session():
    """A recent fast-skip (the documented negative weighting signal) inside RECENT_WINDOW
    must push recent_category_affinity below neutral."""
    history = FeatureHistory()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session_time = base - timedelta(hours=1)
    history.update(_event(category="SPORT", watch_percentage=8, event_type="VIDEO_SKIPPED",
                           base=session_time, minutes=0))
    features = history.features(
        user_id="u", category="SPORT", creator_id="cr", content_id="new", timestamp=base,
        content_popularity_score=.5, content_created_at=base,
    )
    assert features["recent_category_affinity"] < 0.5


def test_recent_category_affinity_long_term_and_recent_can_diverge_in_opposite_directions():
    """The SPORT -> MUSIC scenario in miniature: strong long-term SPORT history plus a
    recent SPORT fast-skip keeps category_affinity(SPORT) high while dropping
    recent_category_affinity(SPORT); a category with weak/negative long-term history plus a
    strong recent session shows the opposite split. Neither long-term nor recent history is
    a duplicate of the other -- they must be able to disagree."""
    history = FeatureHistory()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    long_term = base - RECENT_WINDOW - timedelta(days=30)
    for i in range(4):
        history.update(_event(content_id=f"sport-long-{i}", category="SPORT", watch_percentage=95, liked=True,
                               event_type="VIDEO_COMPLETED", base=long_term, minutes=i))
    history.update(_event(content_id="sport-recent", category="SPORT", watch_percentage=8, event_type="VIDEO_SKIPPED",
                           base=base - timedelta(hours=1), minutes=0))
    for i in range(4):
        history.update(_event(content_id=f"music-long-{i}", category="MUSIC", watch_percentage=8, event_type="VIDEO_SKIPPED",
                               base=long_term, minutes=i))
    history.update(_event(content_id="music-recent", category="MUSIC", watch_percentage=95, liked=True, shared=True,
                           event_type="VIDEO_COMPLETED", base=base - timedelta(hours=2), minutes=0))

    sport = history.features(user_id="u", category="SPORT", creator_id="cr", content_id="ns",
                              timestamp=base, content_popularity_score=.5, content_created_at=base)
    music = history.features(user_id="u", category="MUSIC", creator_id="cr", content_id="nm",
                              timestamp=base, content_popularity_score=.5, content_created_at=base)

    assert sport["category_affinity"] > 0.7  # long-term SPORT interest preserved
    assert sport["recent_category_affinity"] < 0.5  # recent SPORT session was negative
    assert music["category_affinity"] < 0.5  # long-term MUSIC interest stayed weak
    assert music["recent_category_affinity"] > 0.7  # recent MUSIC session was strong


def test_recent_category_affinity_window_boundary_is_inclusive_and_exclusive_correctly():
    """Matches the existing recent_category_watch_percentage/recent_category_completion_rate
    boundary convention (`ts - t <= RECENT_WINDOW`): an event exactly RECENT_WINDOW old still
    counts; one day further back does not."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    at_boundary = FeatureHistory()
    at_boundary.update(_event(category="SPORT", watch_percentage=95, liked=True,
                               event_type="VIDEO_COMPLETED", base=base - RECENT_WINDOW, minutes=0))
    just_outside = FeatureHistory()
    just_outside.update(_event(category="SPORT", watch_percentage=95, liked=True,
                                event_type="VIDEO_COMPLETED", base=base - RECENT_WINDOW - timedelta(days=1), minutes=0))

    boundary_features = at_boundary.features(user_id="u", category="SPORT", creator_id="cr", content_id="ns",
                                              timestamp=base, content_popularity_score=.5, content_created_at=base)
    outside_features = just_outside.features(user_id="u", category="SPORT", creator_id="cr", content_id="ns",
                                              timestamp=base, content_popularity_score=.5, content_created_at=base)
    assert boundary_features["recent_category_affinity"] > 0.5
    assert outside_features["recent_category_affinity"] == 0.5


def test_recent_category_affinity_is_not_a_banned_leakage_field():
    from app.ml.dataset_builder import FEATURES
    assert "recent_category_affinity" in FEATURES


def test_feature_generation_is_deterministic():
    rows = [_event(content_id=f"c{i}", minutes=i, watch_percentage=70 + i) for i in range(5)]
    first = build_dataset(rows)
    second = build_dataset(rows)
    assert first.drop(columns=["timestamp"]).equals(second.drop(columns=["timestamp"]))


def test_target_derived_fields_are_not_model_inputs():
    # "creator_followed" is a legitimate FEATURE (prior-history-derived, computed by
    # FeatureHistory.features from state accumulated strictly before the candidate
    # timestamp); it must not be confused with the raw current-event field of the same
    # name, which build_dataset never passes into features() -- verified structurally by
    # test_current_event_does_not_contribute_to_its_own_features above. The raw,
    # target-defining fields themselves must never appear as column names in FEATURES.
    from app.ml.dataset_builder import FEATURES
    banned = {"target", "watch_percentage", "liked", "shared", "favorited", "commented", "event_type"}
    assert not (set(FEATURES) & banned)


def test_unknown_history_indicators_flip_after_first_interaction():
    history = FeatureHistory()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cold = history.features(
        user_id="u", category="FOOD", creator_id="cr", content_id="c1", timestamp=base,
        content_popularity_score=.5, content_created_at=base,
    )
    assert cold["has_category_history"] == 0 and cold["has_creator_history"] == 0
    history.update(_event(minutes=0, base=base))
    warm = history.features(
        user_id="u", category="FOOD", creator_id="cr", content_id="c2", timestamp=base + timedelta(minutes=5),
        content_popularity_score=.5, content_created_at=base,
    )
    assert warm["has_category_history"] == 1 and warm["has_creator_history"] == 1


def test_unbounded_counts_are_log_scaled():
    history = FeatureHistory()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(50):
        history.update(_event(content_id=f"c{i}", minutes=i, base=base, watch_percentage=95))
    features = history.features(
        user_id="u", category="FOOD", creator_id="cr", content_id="new", timestamp=base + timedelta(minutes=51),
        content_popularity_score=.5, content_created_at=base,
    )
    # log1p(50) ~= 3.93, far below the raw count -- confirms log-scaling rather than raw counts.
    assert features["category_interaction_count"] == math.log1p(50)
    assert features["category_interaction_count"] < 10


def test_content_age_hours_is_capped():
    history = FeatureHistory()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    features = history.features(
        user_id="u", category="FOOD", creator_id="cr", content_id="c", timestamp=base,
        content_popularity_score=.5, content_created_at=base - timedelta(days=3650),
    )
    assert features["content_age_hours"] == MAX_CONTENT_AGE_HOURS


def test_unseen_category_is_handled_by_the_trained_pipeline():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from app.ml.dataset_builder import CATEGORICAL, FEATURES, NUMERIC
    from app.ml.pipeline_builder import build_classifier_pipeline

    rows = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = FeatureHistory()
    for i in range(20):
        category = "FOOD" if i % 2 == 0 else "SPORT"
        features = history.features(
            user_id="u", category=category, creator_id="cr", content_id=f"c{i}", timestamp=base + timedelta(minutes=i),
            content_popularity_score=.5, content_created_at=base,
        )
        rows.append({**features, "target": i % 2})
        history.update(_event(content_id=f"c{i}", category=category, minutes=i, base=base, watch_percentage=90 if i % 2 == 0 else 5))
    df = pd.DataFrame(rows)
    pipeline = build_classifier_pipeline(
        LogisticRegression(max_iter=200), categorical=CATEGORICAL, numeric=NUMERIC, scale_numeric=True,
    )
    pipeline.fit(df[FEATURES], df["target"])

    unseen_row = dict(df.iloc[0])
    unseen_row["category"] = "NEVER_SEEN_BEFORE"
    prediction = pipeline.predict_proba(pd.DataFrame([unseen_row])[FEATURES])
    assert prediction.shape == (1, 2)
    assert 0.0 <= prediction[0, 1] <= 1.0
