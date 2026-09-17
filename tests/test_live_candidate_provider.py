"""Local LIVE candidate sourcing (app.services.providers.live_candidate_provider):
recommendation-eligible LIVE streams read from this service's own local Content
projection, independent of viewer session state.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.db.database import SessionLocal
from app.db.models import Content, Interaction
from app.services.providers.live_candidate_provider import load_active_live_candidates

FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _content(db, content_id, *, content_type="LIVE", is_active=True, category="GAMING",
             creator_id="creator-1", created_at=FIXED_NOW):
    db.add(Content(
        content_id=content_id, creator_id=creator_id, category=category, content_type=content_type,
        popularity_score=0.5, is_active=is_active, created_at=created_at, updated_at=created_at,
    ))


# --- 1/4: active LIVE content becomes a local candidate, multiple streams -----------------

def test_active_live_content_becomes_a_local_candidate(db):
    _content(db, "live-1")
    db.commit()
    candidates = load_active_live_candidates(db, "any-user", limit=10)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.stream_id == "live-1"
    assert candidate.creator_id == "creator-1"
    assert candidate.category == "GAMING"
    assert candidate.status.value == "ACTIVE"


def test_multiple_active_live_streams_are_all_returned(db):
    _content(db, "live-1")
    _content(db, "live-2", creator_id="creator-2", category="MUSIC")
    _content(db, "live-3", creator_id="creator-3", category="SPORT")
    db.commit()
    candidates = load_active_live_candidates(db, "any-user", limit=10)
    assert {c.stream_id for c in candidates} == {"live-1", "live-2", "live-3"}


# --- 2/3: inactive LIVE excluded, VIDEO excluded -------------------------------------------

def test_inactive_live_content_is_excluded(db):
    _content(db, "live-active")
    _content(db, "live-inactive", is_active=False)
    db.commit()
    candidates = load_active_live_candidates(db, "any-user", limit=10)
    assert {c.stream_id for c in candidates} == {"live-active"}


def test_video_content_is_excluded(db):
    _content(db, "live-1")
    _content(db, "video-1", content_type="VIDEO")
    db.commit()
    candidates = load_active_live_candidates(db, "any-user", limit=10)
    assert {c.stream_id for c in candidates} == {"live-1"}


# --- 5: duplicate stream ids cannot appear -------------------------------------------------

def test_no_duplicate_stream_ids_across_the_pool(db):
    _content(db, "live-1")
    _content(db, "live-2")
    db.commit()
    candidates = load_active_live_candidates(db, "any-user", limit=10)
    stream_ids = [c.stream_id for c in candidates]
    assert len(stream_ids) == len(set(stream_ids))


# --- 10/16: empty state, filter scope ------------------------------------------------------

def test_no_active_live_streams_returns_an_empty_list(db):
    assert load_active_live_candidates(db, "any-user", limit=10) == []


def test_only_active_content_type_live_rows_are_ever_considered(db):
    _content(db, "live-active")
    _content(db, "live-inactive", is_active=False)
    _content(db, "video-active", content_type="VIDEO")
    db.commit()
    candidates = load_active_live_candidates(db, "any-user", limit=10)
    assert [c.stream_id for c in candidates] == ["live-active"]


def test_limit_is_respected(db):
    for i in range(5):
        _content(db, f"live-{i}", created_at=FIXED_NOW + timedelta(minutes=i))
    db.commit()
    candidates = load_active_live_candidates(db, "any-user", limit=2)
    assert len(candidates) == 2


def test_retrieval_is_deterministic_for_identical_db_state(db):
    for i in range(4):
        _content(db, f"live-{i}", created_at=FIXED_NOW + timedelta(minutes=i))
    db.commit()
    first = [c.stream_id for c in load_active_live_candidates(db, "u", limit=10)]
    second = [c.stream_id for c in load_active_live_candidates(db, "u", limit=10)]
    assert first == second


# --- 5/19: honest dynamic-signal placeholders, live age from content lifecycle ------------

def test_live_age_minutes_is_derived_from_content_created_at_not_from_viewer_join(db):
    # load_active_live_candidates derives liveAgeMinutes from real wall-clock "now" (it is not
    # given an injectable clock), so this anchor must be real recent wall-clock time too --
    # unlike other tests in this module, it cannot use the arbitrary historical FIXED_NOW.
    real_now = datetime.now(timezone.utc)
    started_30_minutes_ago = real_now - timedelta(minutes=30)
    _content(db, "live-1", created_at=started_30_minutes_ago)
    # A viewer JOIN with no matching LEFT exists for this stream -- must have zero influence
    # on liveAgeMinutes (stream lifecycle and viewer session state are different concepts).
    db.add(Interaction(
        event_id="join-1", user_id="viewer-1", content_id="live-1", creator_id="creator-1",
        category="GAMING", event_type="LIVE_JOINED", timestamp=real_now - timedelta(minutes=1),
    ))
    db.commit()
    candidates = load_active_live_candidates(db, "any-user", limit=10)
    assert len(candidates) == 1
    age = candidates[0].live_age_minutes
    assert 29.0 <= age <= 31.0


def test_current_viewer_count_and_growth_rate_are_neutral_not_fabricated(db):
    _content(db, "live-1")
    # Real, active viewer sessions exist -- but current_viewer_count/viewer_growth_rate must
    # still be the documented neutral placeholder, never derived/fabricated from them (no
    # staleness policy exists to safely count "currently open" sessions -- see module docstring).
    for i in range(5):
        db.add(Interaction(
            event_id=f"join-{i}", user_id=f"viewer-{i}", content_id="live-1", creator_id="creator-1",
            category="GAMING", event_type="LIVE_JOINED", timestamp=FIXED_NOW,
        ))
    db.commit()
    candidate = load_active_live_candidates(db, "any-user", limit=10)[0]
    assert candidate.current_viewer_count == 0
    assert candidate.viewer_growth_rate == 0.0


def test_region_and_language_are_documented_neutral_placeholders(db):
    _content(db, "live-1")
    db.commit()
    candidate = load_active_live_candidates(db, "any-user", limit=10)[0]
    assert candidate.region == "unknown"
    assert candidate.language == "unknown"
    assert candidate.region_match is False
    assert candidate.language_match is False


# --- 17: viewer/session events alone never mark an inactive stream active -----------------

def test_viewer_session_events_do_not_make_an_inactive_stream_eligible(db):
    _content(db, "live-inactive", is_active=False)
    for event_type, minutes in (("LIVE_JOINED", 0), ("LIVE_WATCHED", 1), ("LIVE_LEFT", 2)):
        db.add(Interaction(
            event_id=f"{event_type}-{minutes}", user_id="viewer-1", content_id="live-inactive",
            creator_id="creator-1", category="GAMING", event_type=event_type,
            timestamp=FIXED_NOW + timedelta(minutes=minutes),
        ))
    db.commit()
    assert load_active_live_candidates(db, "any-user", limit=10) == []
