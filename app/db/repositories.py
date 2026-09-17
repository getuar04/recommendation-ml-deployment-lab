from collections.abc import Iterable
from datetime import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.db.models import Interaction


def find_event(db: Session, event_id: str):
    return db.scalar(select(Interaction).where(Interaction.event_id == event_id))

def interactions(db: Session, user_id: str | None = None):
    """Eventual-consistency audit (Task: content_pending evidence-leak audit): excludes
    `content_pending` rows -- RMS has no authoritative Content metadata for them yet (see
    `Interaction.content_pending`'s own comment, app.db.models), so they must not contribute
    to the long-term behavior profile, VIDEO/LIVE training datasets, or cohort aggregation,
    every one of which reads through this function. A pending row becomes visible here again
    automatically once `app.api.content_routes.create_content`'s reconciliation clears the
    flag -- no caller of this function needs its own awareness of pending state."""
    q = select(Interaction).where(Interaction.content_pending.is_(False))
    if user_id:
        q = q.where(Interaction.user_id == user_id)
    return list(db.scalars(q.order_by(Interaction.timestamp)).all())


def recent_interactions_for_ranking(db: Session, user_id: str, *, limit: int) -> list[Interaction]:
    """Phase A: bounded variant used only by the VIDEO recommendation hot path
    (app.services.recommendation_service) -- deliberately separate from `interactions()`
    above, which stays unbounded and unchanged for every other caller (user_profile_service,
    candidate_routes). Selects the `limit` most recent rows for `user_id` at the database
    level (ORDER BY timestamp DESC, id DESC LIMIT N, using the composite
    ix_interactions_user_id_timestamp index) and returns them in ascending order, matching
    what `interactions()` already returns, so callers built on top of either function see the
    same row ordering. `id` (monotonically increasing insert order) breaks ties between rows
    sharing an identical timestamp so which N rows land at the boundary is deterministic
    across database engines, not merely "whatever order the engine happens to return".

    Eventual-consistency audit: excludes `content_pending` rows -- see `interactions()`'s own
    comment above for why. This is the VIDEO/LIVE serving-time ranking/candidate/session-
    affinity history source, so an unresolved-content row must not shape a live score either."""
    rows = db.scalars(
        select(Interaction)
        .where(Interaction.user_id == user_id, Interaction.content_pending.is_(False))
        .order_by(Interaction.timestamp.desc(), Interaction.id.desc())
        .limit(limit)
    ).all()
    return list(reversed(rows))


def recent_interactions_for_drift(db: Session, *, limit: int) -> list[Interaction]:
    """Drift monitoring's only local VIDEO observation source (app.api.drift_routes,
    GET /model/drift): the `limit` most recent interactions across ALL users -- unlike
    `recent_interactions_for_ranking()` above, this is not scoped to one user_id, since a
    drift sample needs a cross-population slice of recent activity, not one user's history.
    Uses the existing single-column `ix_interactions_timestamp` index (no WHERE clause, so
    the per-user composite index doesn't apply here); `id` breaks ties on identical
    timestamps for the same deterministic-boundary reason as `recent_interactions_for_ranking`.
    Returned in ascending (chronological) order, matching every other row-returning function
    in this module, since `app.ml.dataset_builder.build_feature_rows()` expects that order.

    Eventual-consistency audit: excludes `content_pending` rows -- see `interactions()`'s own
    comment above for why. A drift observation must reflect the same evidence contract as
    training, not a superset of it."""
    rows = db.scalars(
        select(Interaction)
        .where(Interaction.content_pending.is_(False))
        .order_by(Interaction.timestamp.desc(), Interaction.id.desc())
        .limit(limit)
    ).all()
    return list(reversed(rows))


def interactions_for_content(db: Session, content_id: str) -> list[Interaction]:
    """All real-engagement interactions for one piece of content, across every user --
    the source of truth for `app.ml.feature_builder.content_popularity_score()`
    (app.services.event_service.store_event recomputes `Content.popularity_score` from
    this after every new VIDEO interaction). Excludes `is_training_context_only` rows,
    matching the same precedent app.services.cohort_aggregation_service.rebuild() already
    established: synthetic training-context-only rows exist solely to seed
    FeatureHistory for model training and must not be counted as real production
    engagement. Order is irrelevant to the caller (an aggregate over all rows), so no
    ORDER BY is applied.

    Eventual-consistency audit: also excludes `content_pending` rows -- see `interactions()`'s
    own comment above for why. Without this, a stale pending row recorded for this content_id
    before its Content row existed could inflate `Content.popularity_score` the first time
    ANY new (already-resolved) interaction on the same content triggers a recompute, even
    though that pending row was never validated against this Content at all."""
    rows = db.scalars(
        select(Interaction).where(
            Interaction.content_id == content_id,
            Interaction.is_training_context_only.is_(False),
            Interaction.content_pending.is_(False),
        )
    ).all()
    return list(rows)


def warmup_interactions_for_drift(
    db: Session, *, user_ids: Iterable[str], before_timestamp: datetime, before_id: int, limit: int,
) -> list[Interaction]:
    """Corrective pass (Finding 2): bounded prior-history warm-up for GET /model/drift's
    observation-window reconstruction (app.services.drift_service). Most VIDEO features are
    history-dependent (see app.ml.dataset_builder.build_feature_rows's docstring for the
    full audit) -- without this, every user in the observation window would be silently
    reconstructed as an artificial cold start for those features, regardless of their real
    prior history.

    One single query for every user represented in the window (never one query per user --
    no N+1), restricted to rows strictly before the window's own boundary using the exact
    same `(timestamp, id)` lexicographic tie-break `recent_interactions_for_drift()` uses to
    pick that boundary, so warm-up and the observation window never overlap and never leave
    a gap at a tied timestamp. Bounded by `limit` in total across all requested users, not
    per-user -- see `app.core.config.DRIFT_WARMUP_MAX_ROWS`. An empty `user_ids` returns an
    empty list without querying at all.

    Eventual-consistency audit: excludes `content_pending` rows -- see `interactions()`'s own
    comment above for why. Warm-up primes the same FeatureHistory accumulators the
    observation window itself feeds, so it must honor the same evidence contract."""
    user_id_list = list(user_ids)
    if not user_id_list or limit <= 0:
        return []
    rows = db.scalars(
        select(Interaction)
        .where(
            Interaction.user_id.in_(user_id_list),
            Interaction.content_pending.is_(False),
            or_(
                Interaction.timestamp < before_timestamp,
                and_(Interaction.timestamp == before_timestamp, Interaction.id < before_id),
            ),
        )
        .order_by(Interaction.timestamp.desc(), Interaction.id.desc())
        .limit(limit)
    ).all()
    return list(reversed(rows))

