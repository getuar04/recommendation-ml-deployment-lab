"""Real-data LIVE dataset construction: stored Interaction rows -> reconstructed sessions
(`app.ml.live_session_builder`) -> point-in-time historical aggregates -> one labeled row per
session -> a `pandas.DataFrame` ready for `app.ml.live_trainer.train_live_model`.

Conceptually equivalent to VIDEO's `app.ml.dataset_builder.build_dataset`, but LIVE-specific:
one row per RECONSTRUCTED SESSION rather than one row per raw event (a LIVE viewing is
naturally session-shaped -- see `app.ml.live_session_builder`), and a much smaller, LIVE-only
aggregate state (`LiveHistory`) rather than VIDEO's `FeatureHistory`/replay-saturation
machinery, which models a different phenomenon (repeated exposure to static content) and is
deliberately not reused here.

Feature-contract honesty (spec: "if a feature cannot currently be derived reliably from local
DB data, do not fake it"): eight of the active LIVE model's 17 features
(`region`, `language`, `region_match`, `language_match`, `current_viewer_count`,
`viewer_growth_rate`, `live_age_minutes`, `already_joined`) describe ephemeral, request-time
state that this service has never persisted anywhere (no LIVE event schema field captures
them, no Content column stores them) -- they only ever exist on the live serving-time
`LiveCandidate` payload. `UNAVAILABLE_FROM_HISTORY_FEATURES` below names them explicitly, and
`build_live_dataset` fills them with documented neutral placeholders rather than inventing
plausible-looking values, so a model trained on this dataset is never misled into treating a
placeholder as a real signal.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from itertools import groupby
from typing import Any

import pandas as pd

from app.ml.live_feature_builder import (
    LIVE_FEATURES,
    LIVE_MEANINGFUL_STAY_SECONDS,
    LIVE_SHORT_STAY_SECONDS,
    derive_live_affinities,
)
from app.ml.live_session_builder import LiveSession, reconstruct_live_sessions

# LIVE streams are ephemeral and recency-sensitive by nature -- a shorter recent-activity
# window than VIDEO's 30-day RECENT_WINDOW (app.ml.dataset_builder) is a deliberate, LIVE-
# specific choice, not a copy-then-forgot-to-change value.
LIVE_RECENT_WINDOW = timedelta(days=7)

# See module docstring: these eight LIVE_FEATURES columns describe request-time-only state
# this service has never stored anywhere, so `build_live_dataset` cannot derive them from
# real history. Kept as an explicit, importable/testable list rather than a scattered set of
# inline literals.
UNAVAILABLE_FROM_HISTORY_FEATURES = (
    "region", "language", "region_match", "language_match",
    "current_viewer_count", "viewer_growth_rate", "live_age_minutes", "already_joined",
)


class LiveHistory:
    """Point-in-time LIVE history state, folded in one reconstructed session at a time.

    Much simpler than VIDEO's `FeatureHistory` on purpose: LIVE has no semantic/session-decay/
    replay-saturation concepts yet (see this repo's LIVE audit) -- this only tracks the plain
    counts/sums section 5 of the foundation spec asks for, at user/category/creator scope,
    plus a bounded recent-activity window.
    """

    def __init__(self) -> None:
        self._user: dict[str, dict[str, float]] = {}
        self._category: dict[tuple[str, str], dict[str, float]] = {}
        self._creator: dict[tuple[str, str], dict[str, Any]] = {}
        self._recent: dict[str, list[tuple[datetime, str | None, float]]] = {}

    def snapshot(self, *, user_id: str, category: str | None, creator_id: str | None, at: datetime) -> dict[str, Any]:
        user = self._user.get(user_id, {"sessions": 0, "watch_seconds": 0.0})
        cat = self._category.get((user_id, category or ""), {
            "sessions": 0, "watch_seconds": 0.0, "joined_sessions": 0, "positive_engagement": 0,
        })
        creator = self._creator.get((user_id, creator_id or ""), {
            "sessions": 0, "watch_seconds": 0.0, "positive_engagement": 0, "followed": False,
        })
        recent_all = self._recent.get(user_id, [])
        recent = [entry for entry in recent_all if (at - entry[0]) <= LIVE_RECENT_WINDOW]
        recent_category = [entry for entry in recent if entry[1] == category]
        return {
            "previous_live_interaction_count": user["sessions"],
            "previous_live_watch_time": user["watch_seconds"],
            "average_live_watch_time": (user["watch_seconds"] / user["sessions"]) if user["sessions"] else None,
            "previous_live_category_interaction_count": cat["sessions"],
            "previous_live_category_watch_time": cat["watch_seconds"],
            "average_live_watch_time_for_category": (cat["watch_seconds"] / cat["sessions"]) if cat["sessions"] else None,
            "category_join_rate": (cat["joined_sessions"] / cat["sessions"]) if cat["sessions"] else None,
            "category_positive_engagement_count": cat["positive_engagement"],
            "previous_creator_live_interaction_count": creator["sessions"],
            "previous_creator_live_watch_time": creator["watch_seconds"],
            "average_creator_live_watch_time": (creator["watch_seconds"] / creator["sessions"]) if creator["sessions"] else None,
            "creator_positive_engagement_count": creator["positive_engagement"],
            "creator_followed": creator["followed"],
            "recent_live_interaction_count": len(recent),
            "recent_live_watch_time": sum(entry[2] for entry in recent),
            "recent_category_live_activity": len(recent_category),
        }

    def top_categories(self, user_id: str, *, limit: int) -> list[str]:
        """Categories this user has real LIVE session history in, ranked by session count
        (ties broken by category name for determinism), most-active first. Used by LIVE
        candidate retrieval (`app.services.providers.live_candidate_provider`) to bucket
        candidates toward categories this user has actually engaged with -- a ranking-layer
        read of the same aggregate `snapshot()` already exposes for feature construction,
        never a second, independently-computed affinity."""
        entries = [
            (category, agg["sessions"])
            for (uid, category), agg in self._category.items()
            if uid == user_id and category and agg["sessions"] > 0
        ]
        entries.sort(key=lambda entry: (-entry[1], entry[0]))
        return [category for category, _ in entries[:limit]]

    def followed_creator_ids(self, user_id: str) -> list[str]:
        """Creators this user has a recorded `LIVE_CREATOR_FOLLOWED` event with, in no
        particular order (candidate bucketing only needs set membership, not ranking)."""
        return [
            creator_id for (uid, creator_id), agg in self._creator.items()
            if uid == user_id and creator_id and agg.get("followed")
        ]

    def user_session_count(self, user_id: str) -> int:
        """Total real reconstructed LIVE sessions this user has, across every category/
        creator -- the LIVE-native evidence count for lifecycle/strategy reporting
        (app.services.live_recommendation_service.recommend_live), read from the same
        per-user aggregate `snapshot()` already exposes as `previous_live_interaction_count`
        for a specific category/creator scope, just without that scoping."""
        return int(self._user.get(user_id, {}).get("sessions", 0))

    def update(self, session: LiveSession) -> None:
        positive = any((
            session.like_count, session.share_count, session.comment_count,
            session.gift_count, session.creator_follow_count,
        ))
        user = self._user.setdefault(session.user_id, {"sessions": 0, "watch_seconds": 0.0})
        user["sessions"] += 1
        user["watch_seconds"] += session.total_watch_seconds

        cat_key = (session.user_id, session.category or "")
        cat = self._category.setdefault(cat_key, {
            "sessions": 0, "watch_seconds": 0.0, "joined_sessions": 0, "positive_engagement": 0,
        })
        cat["sessions"] += 1
        cat["watch_seconds"] += session.total_watch_seconds
        if session.joined:
            cat["joined_sessions"] += 1
        if positive:
            cat["positive_engagement"] += 1

        if session.creator_id:
            creator_key = (session.user_id, session.creator_id)
            creator = self._creator.setdefault(creator_key, {
                "sessions": 0, "watch_seconds": 0.0, "positive_engagement": 0, "followed": False,
            })
            creator["sessions"] += 1
            creator["watch_seconds"] += session.total_watch_seconds
            if positive:
                creator["positive_engagement"] += 1
            if session.creator_follow_count > 0:
                creator["followed"] = True

        self._recent.setdefault(session.user_id, []).append(
            (session.ended_at, session.category, session.total_watch_seconds),
        )


def live_session_target(session: LiveSession) -> int | None:
    """Session-level equivalent of `app.ml.live_feature_builder.live_target`, refactored to
    work on a reconstructed `LiveSession` instead of one raw event, and extended with the
    explicit LIVE_NOT_INTERESTED precedence rule the foundation spec calls for.

    Positive (1): a strong explicit action (like/share/comment/gift/creator-follow) at any
    point in the session, OR joined with total_watch_seconds >= LIVE_MEANINGFUL_STAY_SECONDS.
    Negative (0): a LIVE_NOT_INTERESTED anywhere in the session (see precedence note below),
    OR an impression with no join, OR joined with total_watch_seconds < LIVE_SHORT_STAY_SECONDS
    and no positive action.
    Neutral (excluded, None): everything else, INCLUDING a structurally malformed/ambiguous
    session (`exclusion_reason` set, e.g. an open-ended join with no LEFT and no explicit
    signal) -- a session with no positive/negative signal of its own must never be labeled
    from an ambiguous duration.

    LIVE_NOT_INTERESTED precedence (spec requirement): an explicit rejection ALWAYS wins over
    any positive engagement present in the same logical session, mirroring this codebase's
    existing VIDEO precedent (`app.ml.feature_builder.target_for`: "explicit rejection takes
    precedence over everything else, including... a liked/shared/... flag also present on the
    same row") -- extended here from "same row" to "same logical session".
    """
    if session.not_interested_count > 0:
        return 0
    positive_action = any((
        session.like_count, session.share_count, session.comment_count,
        session.gift_count, session.creator_follow_count,
    ))
    if positive_action:
        return 1
    if session.exclusion_reason is not None:
        return None
    if session.joined and session.total_watch_seconds >= LIVE_MEANINGFUL_STAY_SECONDS:
        return 1
    if (session.impression and not session.joined) or (
        session.joined and session.total_watch_seconds < LIVE_SHORT_STAY_SECONDS
    ):
        return 0
    return None


def live_feature_snapshot(history: LiveHistory, *, user_id: str, category: str | None,
                          creator_id: str | None, at: datetime,
                          video_bootstrap: dict[str, float] | None = None) -> dict[str, Any]:
    """Bridges real reconstructed-session history onto the same fields the active LIVE
    model's caller-trusted path derives via `derive_live_affinities` -- reuses that function's
    exact formulas (never re-hardcodes its constants), just fed by properly separated real
    category-level/creator-level counts instead of the one caller-supplied scalar the original
    design used to stand in for both.

    `previous_live_interaction_count`/`previous_live_watch_time` resolve at CREATOR scope
    (their own feature definitions read "prior interactions/watch seconds with this
    creator/live context"), matching `creator_affinity`'s scope; `average_live_watch_time_for_
    category`/`recent_live_category_activity` stay CATEGORY-scoped, matching
    `live_category_affinity`'s scope.

    `video_bootstrap` (cross-format LIVE-relevance-bootstrap architecture requirement, keys
    `"live_category_affinity"`/`"creator_affinity"`, see
    `app.services.providers.live_personalization_provider.video_bootstrap_affinities`):
    substituted in place of `derive_live_affinities`' flat 0.5 baseline ONLY when this specific
    candidate's category/creator has ZERO real LIVE session evidence (`sessions == 0`) --
    real LIVE evidence, however small, always wins; this never overrides genuine LIVE-specific
    signal, it only replaces an uninformative neutral placeholder with real (VIDEO) evidence
    RMS already has locally. `previous_live_interaction_count`/`previous_live_watch_time`
    deliberately stay LIVE-only/untouched below -- those fields mean "prior LIVE interactions",
    or bootstrapping them from VIDEO would fabricate LIVE history that never happened; later
    real LIVE-specific signals are what is meant to refine/replace the bootstrap over time.
    """
    agg = history.snapshot(user_id=user_id, category=category, creator_id=creator_id, at=at)
    category_view = derive_live_affinities(
        int(agg["previous_live_category_interaction_count"]), agg["previous_live_category_watch_time"],
    )
    creator_view = derive_live_affinities(
        int(agg["previous_creator_live_interaction_count"]), agg["previous_creator_live_watch_time"],
    )
    category_affinity = category_view["live_category_affinity"]
    creator_affinity = creator_view["creator_affinity"]
    if video_bootstrap is not None:
        if agg["previous_live_category_interaction_count"] == 0 and "live_category_affinity" in video_bootstrap:
            category_affinity = video_bootstrap["live_category_affinity"]
        if agg["previous_creator_live_interaction_count"] == 0 and "creator_affinity" in video_bootstrap:
            creator_affinity = video_bootstrap["creator_affinity"]
    return {
        "live_category_affinity": category_affinity,
        "creator_affinity": creator_affinity,
        "average_live_watch_time_for_category": category_view["average_live_watch_time_for_category"],
        "recent_live_category_activity": agg["recent_category_live_activity"],
        "previous_live_interaction_count": agg["previous_creator_live_interaction_count"],
        "previous_live_watch_time": agg["previous_creator_live_watch_time"],
        "creator_followed": agg["creator_followed"],
    }


def _session_batches(sessions: Iterable[LiveSession]) -> Iterable[list[LiveSession]]:
    ordered = sorted(sessions, key=lambda session: session.ended_at)
    for _, batch in groupby(ordered, key=lambda session: session.ended_at):
        yield list(batch)


def build_live_dataset(rows: Iterable[Any], content_by_id: dict[str, Any] | None = None) -> pd.DataFrame:
    """Real-data equivalent of `app.ml.dataset_builder.build_dataset` for LIVE: reconstructs
    sessions from `rows` (real `Interaction` rows, already filtered to LIVE content by the
    caller -- mirrors `app.services.training_service.train`'s own VIDEO/LIVE split), then
    emits one labeled row per session using only history strictly before that session's own
    `ended_at` (sessions are folded into `LiveHistory` in `ended_at` batches, so simultaneous
    sessions cannot observe each other, exactly like VIDEO's `_timestamp_batches`).

    `content_by_id` is accepted for interface symmetry with `build_dataset` but is currently
    unused: nothing in the real LIVE feature contract needs a `Content` lookup today (see
    `app.ml.live_session_builder`'s docstring) -- kept for forward compatibility once LIVE
    content-understanding features exist.
    """
    del content_by_id  # unused today; see docstring
    history = LiveHistory()
    result: list[dict[str, Any]] = []
    for batch in _session_batches(reconstruct_live_sessions(list(rows))):
        for session in batch:
            label = live_session_target(session)
            if label is not None:
                snapshot = live_feature_snapshot(
                    history, user_id=session.user_id, category=session.category,
                    creator_id=session.creator_id, at=session.ended_at,
                )
                result.append({
                    "category": (session.category or "UNKNOWN").upper(),
                    # UNAVAILABLE_FROM_HISTORY_FEATURES: never stored, so a documented neutral
                    # placeholder stands in rather than a fabricated value (see module docstring).
                    "region": "unknown", "language": "unknown",
                    "live_category_affinity": snapshot["live_category_affinity"],
                    "creator_affinity": snapshot["creator_affinity"],
                    "creator_followed": int(snapshot["creator_followed"]),
                    "previous_live_interaction_count": snapshot["previous_live_interaction_count"],
                    "previous_live_watch_time": snapshot["previous_live_watch_time"],
                    "average_live_watch_time_for_category": snapshot["average_live_watch_time_for_category"],
                    "recent_live_category_activity": snapshot["recent_live_category_activity"],
                    "current_viewer_count": 0, "viewer_growth_rate": 0.0, "live_age_minutes": 0.0,
                    "region_match": 0, "language_match": 0,
                    "hour_of_day": session.ended_at.hour, "already_joined": 0,
                    "target": label,
                    "user_id": session.user_id, "content_id": session.content_id,
                    "creator_id": session.creator_id,
                    "candidate_group": f"{session.user_id}:{session.ended_at.date().isoformat()}",
                    "timestamp": session.ended_at,
                })
        for session in batch:
            history.update(session)
    # Always return the full expected column set, even with zero rows: an empty
    # `pd.DataFrame(result)` from an empty `result` list has NO columns at all, which would
    # make `dataset.target`/`.nunique()` raise instead of the trainer's own, correct
    # `InsufficientLiveData` (see app.services.live_training_service.train_live) -- an empty
    # real dataset must fail clearly, not crash.
    columns = [*LIVE_FEATURES, "target", "user_id", "content_id", "creator_id", "candidate_group", "timestamp"]
    return pd.DataFrame(result, columns=columns)
