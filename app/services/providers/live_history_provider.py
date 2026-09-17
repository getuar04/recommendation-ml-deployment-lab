"""LIVE real-history data-adapter boundary: loads this user's own stored LIVE interactions,
reconstructs sessions (`app.ml.live_session_builder`), and folds them into a `LiveHistory`
(`app.ml.live_dataset_builder`) for `app.services.live_recommendation_service.recommend_live`
to score against.

Mirrors `app.services.providers.user_behavior_provider`'s LOCAL_DB path, but LIVE has no
REAL/UBS-adapter equivalent yet (see this repo's LIVE audit) -- this is currently the only
history source for LIVE serving.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import RECOMMENDATION_HISTORY_MAX_INTERACTIONS
from app.db.models import Content
from app.db.repositories import recent_interactions_for_ranking
from app.ml.live_dataset_builder import LiveHistory
from app.ml.live_session_builder import reconstruct_live_sessions


def load_live_history_for_user(db: Session, user_id: str) -> LiveHistory | None:
    """Returns a `LiveHistory` folded with every reconstructed real LIVE session this user
    has, or `None` when this user has no real LIVE interaction rows stored at all. Callers use
    `None` to fall back to caller-supplied history fields (see
    `app.services.live_recommendation_service.recommend_live`) -- a brand-new/unknown-locally
    user must not be treated as having "real history of zero", which would be indistinguishable
    from "we checked and this user genuinely has none"."""
    raw_rows = recent_interactions_for_ranking(db, user_id, limit=RECOMMENDATION_HISTORY_MAX_INTERACTIONS)
    if not raw_rows:
        return None
    content_ids = {row.content_id for row in raw_rows}
    content_by_id = {
        item.content_id: item
        for item in db.scalars(select(Content).where(Content.content_id.in_(content_ids))).all()
    }
    # LIVE-only local history: mirrors app.services.providers.user_behavior_provider's inverse
    # VIDEO-only filter -- a row PROVABLY LIVE (its Content row exists and is explicitly
    # content_type=="LIVE") is kept; anything else (VIDEO, or a missing/unavailable Content
    # row -- "unavailable" is not evidence of being LIVE) is excluded.
    live_rows = [
        row for row in raw_rows
        if getattr(content_by_id.get(row.content_id), "content_type", "VIDEO") == "LIVE"
    ]
    if not live_rows:
        return None
    history = LiveHistory()
    for session in reconstruct_live_sessions(live_rows):
        history.update(session)
    return history
