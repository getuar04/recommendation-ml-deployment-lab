"""Repository PORT for internal ranking-decision snapshots (see `app.ml.ranking_snapshot`).

No implementation here assumes external persistence ownership. Who durably stores these
snapshots, for how long, and keyed by which authoritative cross-service identity is a
decision that depends on contracts RMS does not have yet (Feed Backend / Candidate Service /
Event Tracking Service / real request-identity ownership) -- see
`app.ml.ranking_snapshot_capture` for the construction boundary this repository sits behind.

`NullRankingSnapshotRepository` is the only adapter wired anywhere in production today: it
accepts and immediately discards every snapshot. That is a deliberate placeholder, not an
oversight -- it lets `app.services.recommendation_service.recommend()` exercise the real
capture code path against real production scoring output (proving the shape is correct)
without RMS silently committing itself to a storage contract nobody has approved yet.
`InMemoryRankingSnapshotRepository` exists for tests/local inspection only and must never be
treated as durable.
"""
from __future__ import annotations

from typing import Protocol

from app.ml.ranking_snapshot import RankedCandidateSnapshot

__all__ = [
    "NULL_RANKING_SNAPSHOT_REPOSITORY",
    "InMemoryRankingSnapshotRepository",
    "NullRankingSnapshotRepository",
    "RankingSnapshotRepository",
]


class RankingSnapshotRepository(Protocol):
    def record(self, snapshots: list[RankedCandidateSnapshot]) -> None: ...


class NullRankingSnapshotRepository:
    """Production default. Accepts and discards -- see module docstring."""

    def record(self, snapshots: list[RankedCandidateSnapshot]) -> None:
        return None


class InMemoryRankingSnapshotRepository:
    """Test/local-inspection adapter only. Unbounded in-process list; never durable, never
    shared across requests/processes -- do not wire this into any real deployment."""

    def __init__(self) -> None:
        self.recorded: list[RankedCandidateSnapshot] = []

    def record(self, snapshots: list[RankedCandidateSnapshot]) -> None:
        self.recorded.extend(snapshots)


# Stateless (holds nothing, discards everything -- see NullRankingSnapshotRepository above), so
# one shared module-level instance is safe as a default-argument singleton (avoids constructing
# a fresh one on every `recommend()` call/definition).
NULL_RANKING_SNAPSHOT_REPOSITORY = NullRankingSnapshotRepository()
