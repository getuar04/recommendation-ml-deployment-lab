"""DEV/TEST-ONLY: seeds the 4 UUID-based lead-demo users' interaction history into whatever
Recommendation ML Service is running at --base-url, so GET /api/v1/recommendation-ml-service/users/{userId}/behaviour-
profile returns real, non-COLD_START data for them.

This does NOT invent a new persistence/ownership model. It uses only the two ingestion
endpoints this service already exposes and already tags as demo scaffolding (not the
production Content Service / Event Tracking Service contract -- see app/api/content_routes.py
and app/api/event_routes.py's router `tags=`):
    POST /api/v1/recommendation-ml-service/contents   (creates the historical content each interaction references)
    POST /api/v1/recommendation-ml-service/events     (creates the interaction itself; idempotent by eventId)
GET /api/v1/recommendation-ml-service/users/{userId}/behaviour-profile (app/services/user_profile_service.py) already
reads its answer live from the same local `interactions` table these two endpoints write --
this script does not touch that read path at all.

Determinism: every id is uuid5(NAMESPACE, slug) -- rerunning this script produces byte-
identical ids/timestamps-are-relative-to-run-time, and reposting the same eventId is a no-op
(the existing idempotency in app.services.event_service.store_event).

Scope note: the 4 users' RECOMMENDATION requests (POST /api/v1/recommendation-ml-service/recommendations) use a
separate, stateless, caller-supplied `userProfile` snapshot (the Phase A contract) that does
not read this persisted store at all -- that is what lets Users D2 (CONFIRMED) and D3
(REJECTED) exist as two simultaneous, independent hypothetical states for the SAME user in the
same demo. This script seeds exactly one real, persisted state for User D (the shared BEFORE
base: GAMING + COMEDY, zero TRAVEL) since a real interactions table cannot hold two
contradictory histories for one user at once -- see docs/...postman_collection.json's
D2/D3 requests for the CONFIRMED/REJECTED demonstration instead.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://recommendation-ml-service.local/demo")
RECENT_WINDOW_DAYS = 30


def stable_uuid(kind: str, slug: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{kind}:{slug}"))


def target_label(wp, liked, shared, favorited, creator_followed):
    positive = (wp is not None and wp >= 70) or liked or shared or favorited or creator_followed
    negative = wp is not None and wp < 20 and not any((liked, shared, favorited, creator_followed))
    return 1 if positive else (0 if negative else None)


def is_completed(event_type, wp):
    return event_type == "VIDEO_COMPLETED" or (wp or 0) >= 90


class EventBuilder:
    """Accumulates one user's raw interaction events; mirrors exactly the weighting/label
    formulas in app.ml.dataset_builder / app.ml.feature_builder so a profile built from these
    events (whether read back live from the DB via behaviour-profile, or hand-aggregated into
    a caller-supplied userProfile snapshot for a recommendation request) tells one consistent
    story."""

    def __init__(self, user_id: str, now: datetime, *, content_index_start: int = 0):
        self.user_id = user_id
        self.now = now
        self.events: list[dict] = []
        self._creator_pool: dict[str, list[str]] = {}
        # Content-id index namespace offset -- lets a SECOND, independent EventBuilder for the
        # SAME user_id (e.g. User C's live shift-events builder, separate from the persisted
        # BEFORE-state builder) guarantee its content_ids never collide with the first
        # builder's, even when both happen to use the same category (see build_user_c_shift_
        # events: without this, its SPORT rows would silently reuse content_ids already
        # assigned to build_user_c_before's long-term SPORT history).
        self._content_index_start = content_index_start

    def _creator_for(self, category: str, index: int, pool_size: int) -> str:
        pool = self._creator_pool.setdefault(category, [
            stable_uuid("demo2-creator", f"{self.user_id}:{category}:{i}") for i in range(pool_size)
        ])
        return pool[index % pool_size]

    def add(self, *, category: str, days_ago: float, watch_percentage: float, event_type: str = "VIDEO_WATCHED",
            liked=False, shared=False, favorited=False, commented=False, creator_followed=False,
            creator_pool_size=4, creator_index=None, duration_seconds=100.0,
            creator_id_override: str | None = None, subtopic: str | None = None):
        """`creator_id_override` picks a specific, stable creator identity instead of the
        pool-index scheme below (needed for a deliberate preferred-vs-weak creator split).
        `subtopic`, when given, tags the history content's title/hashtags/topics/entities/
        subgenres with that one generic token (Decision 6: never a real-world name) -- this is
        what lets a within-category semantic-preference narrative (e.g. two different GAMING
        subtopics) actually populate app.ml.dataset_builder's semantic-affinity features,
        instead of leaving them neutral like every other event in this file."""
        category = category.upper()
        idx = len(self.events)
        creator_index = creator_index if creator_index is not None else idx
        creator_id = creator_id_override or self._creator_for(category, creator_index, creator_pool_size)
        content_id = stable_uuid("demo2-history-content", f"{self.user_id}:{category}:{self._content_index_start + idx}")
        timestamp = self.now - timedelta(days=days_ago)
        wp = round(watch_percentage, 2)
        if event_type == "VIDEO_WATCHED" and wp >= 90:
            event_type = "VIDEO_COMPLETED"
        event = {
            # Bug fix: event_id must be namespaced by `_content_index_start` too, exactly like
            # content_id already is above -- without this, a SECOND independent EventBuilder
            # for the SAME user_id (e.g. build_user_c_shift_events, separate from
            # build_user_c_before) mints event_ids that collide with the first builder's
            # (both start their own local `idx` at 0), so app.services.event_service.
            # store_event's idempotency check silently treats every "new" shift event as a
            # duplicate of an unrelated earlier event and never stores it -- discovered via a
            # live end-to-end demo run where User C's post-shift recommendation scores were
            # byte-identical to the pre-shift ones. Every OTHER builder in this file passes
            # the default content_index_start=0, so this is a no-op for them (identical ids
            # to before this fix); only build_user_c_shift_events' ids change.
            "event_id": stable_uuid("demo2-event", f"{self.user_id}:{self._content_index_start + idx}"),
            "user_id": self.user_id, "content_id": content_id, "creator_id": creator_id,
            "category": category, "event_type": event_type,
            "watch_time_seconds": round(wp / 100 * duration_seconds, 2),
            "content_duration_seconds": duration_seconds, "watch_percentage": wp,
            "liked": liked, "shared": shared, "favorited": favorited, "commented": commented,
            "creator_followed": creator_followed, "timestamp": timestamp.isoformat(),
        }
        if subtopic:
            event.update(title=subtopic, hashtags=[subtopic], topics=[subtopic], entities=[subtopic], subgenres=[subtopic])
        self.events.append(event)
        return event


def build_user_a(now):
    user_id = stable_uuid("demo2-user", "user-a-longterm-sport")
    b = EventBuilder(user_id, now)
    for i, wp in enumerate([96, 93, 91, 88, 86, 84, 82, 79, 77, 76]):
        b.add(category="SPORT", days_ago=3 + i * 2.6, watch_percentage=wp,
              liked=(i % 3 == 0), creator_followed=(i == 0), creator_pool_size=5, creator_index=i % 5)
    for i, wp in enumerate([62, 58, 55]):
        b.add(category="SPORT", days_ago=30 + i * 4, watch_percentage=wp, creator_pool_size=5, creator_index=i)
    for i, wp in enumerate([14, 9, 22]):
        b.add(category="SPORT", days_ago=40 + i * 3, watch_percentage=wp, event_type="VIDEO_SKIPPED", creator_pool_size=5, creator_index=i + 2)
    # Demo-data-only tweak: pushed past RECENT_WINDOW (30 days) so this stays genuine
    # "neutral/small activity in another category" (the persona's own intent) rather than
    # accidentally reading as recent engagement -- at days_ago=8-28 every one of these events
    # sat INSIDE the recent window, which was strong/consistent enough on its own (RECENT_
    # CATEGORY_ACTIVITY) to outscore SPORT's HIGH_CATEGORY_AFFINITY candidates, undermining the
    # "User A clearly leans SPORT" comparison this persona exists to demonstrate.
    for i, wp in enumerate([72, 68, 65, 74, 60]):
        b.add(category="COMEDY", days_ago=45 + i * 6, watch_percentage=wp, liked=(i == 0), creator_pool_size=3, creator_index=i % 3)
    for i, wp in enumerate([48, 55, 42, 51]):
        b.add(category="MUSIC", days_ago=12 + i * 6, watch_percentage=wp, creator_pool_size=3, creator_index=i % 3)
    for i, wp in enumerate([71, 66, 74]):
        b.add(category="TRAVEL", days_ago=20 + i * 5, watch_percentage=wp, liked=(i == 2), creator_pool_size=2, creator_index=i % 2)
    b.add(category="GAMING", days_ago=25, watch_percentage=6, event_type="VIDEO_SKIPPED", creator_pool_size=2, creator_index=0)
    b.add(category="GAMING", days_ago=35, watch_percentage=11, event_type="CONTENT_NOT_INTERESTED", creator_pool_size=2, creator_index=1)
    assert len(b.events) == 30
    return user_id, b


def build_user_b(now):
    user_id = stable_uuid("demo2-user", "user-b-balanced-multi")
    b = EventBuilder(user_id, now)
    for i, wp in enumerate([88, 84, 79, 91, 73, 81, 69, 86]):
        b.add(category="SPORT", days_ago=6 + i * 3, watch_percentage=wp, liked=(i % 4 == 0), creator_pool_size=4, creator_index=i % 4)
    # Demo-data-only tweak: like/share rate raised above User A's SPORT pattern (more likes,
    # one share) so MUSIC clearly separates from SPORT in category_affinity for this user --
    # the two categories' comparable watch-percentage/event-count alone left them within
    # rounding distance of each other (0.7311 == 0.7311), which does not visually demonstrate
    # "this user's top category is MUSIC" in a live comparison against User A's SPORT lead.
    for i, wp in enumerate([90, 76, 82, 68, 87, 71, 79]):
        b.add(category="MUSIC", days_ago=2 + i * 1.5, watch_percentage=wp, liked=(i % 2 == 0),
              shared=(i == 0), favorited=(i == 4), creator_pool_size=4, creator_index=i % 4)
    for i, wp in enumerate([66, 58, 70, 61, 54, 67]):
        b.add(category="COMEDY", days_ago=10 + i * 3, watch_percentage=wp, liked=(i == 2), creator_pool_size=3, creator_index=i % 3)
    for i, wp in enumerate([78, 63, 72, 59, 69]):
        b.add(category="TRAVEL", days_ago=22 + i * 4, watch_percentage=wp, liked=(i == 0), creator_pool_size=3, creator_index=i % 3)
    for i, wp in enumerate([52, 44, 38, 15]):
        et = "VIDEO_SKIPPED" if wp < 20 else "VIDEO_WATCHED"
        b.add(category="GAMING", days_ago=14 + i * 5, watch_percentage=wp, event_type=et, creator_pool_size=3, creator_index=i % 3)
    assert len(b.events) == 30
    return user_id, b


USER_C_SLUG = "user-c-sport-to-music-shift"


def build_user_c_before(now):
    """Persisted BEFORE state only (long-term SPORT + neutral-ish COMEDY) -- deliberately does
    NOT include the recent MUSIC-shift/SPORT-rejection events. Those are staged separately
    (see `build_user_c_shift_events`) so the lead demo can POST /recommendations once against
    this state, then send the shift live via real /events calls, then POST the identical
    request again and see the ranking change -- without seeding both states at once and losing
    the "before" moment."""
    user_id = stable_uuid("demo2-user", USER_C_SLUG)
    b = EventBuilder(user_id, now)
    for i, wp in enumerate([92, 89, 85, 95, 81, 78, 90, 83, 76, 87, 79, 94, 73, 88, 75]):
        b.add(category="SPORT", days_ago=15 + i * 2.7, watch_percentage=wp, liked=(i % 3 == 0),
              creator_followed=(i == 0), creator_pool_size=5, creator_index=i % 5)
    # Demo-data-only tweak (same fix/reason as User A's COMEDY loop): pushed past
    # RECENT_WINDOW so it stays neutral background activity instead of outscoring SPORT's
    # HIGH_CATEGORY_AFFINITY lead in the BEFORE-state comparison.
    for i, wp in enumerate([64, 58, 70, 55, 61]):
        b.add(category="COMEDY", days_ago=45 + i * 5, watch_percentage=wp, creator_pool_size=3, creator_index=i % 3)
    assert len(b.events) == 20
    return user_id, b


def build_user_c_shift_events(now):
    """The live, staged behavior-shift step: strong recent MUSIC watches (liked/shared/
    favorited where appropriate) plus recent SPORT fast-skips AND one explicit
    CONTENT_NOT_INTERESTED (both implicit and explicit rejection, per the current event
    contract). `now` here is meant to be REAL wall-clock time at demo time (not the seed
    script's original `now`) -- timestamps are spaced minutes apart, newest last, so both
    SESSION_WINDOW (30 min) and RECENT_WINDOW (30 days) pick them up when POSTed live."""
    user_id = stable_uuid("demo2-user", USER_C_SLUG)
    b = EventBuilder(user_id, now, content_index_start=100)  # namespaced away from build_user_c_before's SPORT idx 0-14 / COMEDY idx 0-4
    for i, wp in enumerate([97, 92, 100, 88, 95, 90, 99]):
        b.add(category="MUSIC", days_ago=(28 - i * 4) / (24 * 60), watch_percentage=wp, event_type="VIDEO_COMPLETED",
              liked=(i % 2 == 0), shared=(i == 2), favorited=(i == 5), creator_pool_size=3, creator_index=i % 3)
    for i, wp in enumerate([6, 4]):
        b.add(category="SPORT", days_ago=(20 - i * 4) / (24 * 60), watch_percentage=wp, event_type="VIDEO_SKIPPED", creator_pool_size=5, creator_index=i)
    b.add(category="SPORT", days_ago=2 / (24 * 60), watch_percentage=9, event_type="CONTENT_NOT_INTERESTED", creator_pool_size=5, creator_index=2)
    assert len(b.events) == 10
    return user_id, b


def build_user_d_before(now):
    user_id = stable_uuid("demo2-user", "user-d-exploration-feedback")
    b = EventBuilder(user_id, now)
    for i, wp in enumerate([82, 78, 90, 75, 86, 71, 88, 79, 68, 92, 74, 81]):
        b.add(category="GAMING", days_ago=4 + i * 2.2, watch_percentage=wp, liked=(i % 4 == 0),
              creator_followed=(i == 0), creator_pool_size=5, creator_index=i % 5)
    for i, wp in enumerate([69, 74, 61, 77, 65, 71, 58, 80, 63, 72]):
        b.add(category="COMEDY", days_ago=6 + i * 2.5, watch_percentage=wp, liked=(i % 3 == 0), creator_pool_size=4, creator_index=i % 4)
    assert len(b.events) == 22
    return user_id, b


def add_user_d_semantic_creator_scenario(b: EventBuilder) -> None:
    """Additive-only: appends a genuine within-category semantic + creator preference
    narrative to User D's existing GAMING history (never a new category, so
    tests/test_seed_demo_users.py's `categories == {"GAMING", "COMEDY"}` / zero-TRAVEL
    invariant for `build_user_d_before` is untouched -- this is called separately, only from
    `build_all_users`, against the SAME builder instance, after `build_user_d_before` already
    populated it). Mirrors the exact proven shape of app.ml.eligibility's `semantic`/`creator`
    gates: real positive history with one generic subtopic + creator, real weak/negative
    history with a DIFFERENT subtopic + creator, same category -- so semantic_affinity and
    creator_interaction_count/creator_completion_rate carry a genuine, non-neutral signal
    instead of the 0.5/0-count default every other event in this file leaves them at."""
    preferred_creator = stable_uuid("demo2-creator", f"{b.user_id}:GAMING:preferred")
    weak_creator = stable_uuid("demo2-creator", f"{b.user_id}:GAMING:weak")
    for i, wp in enumerate([94, 90, 97, 88, 92, 85]):  # preferred subtopic: strong, noisy, real positive history.
        b.add(category="GAMING", days_ago=3 + i * 2.3, watch_percentage=wp, event_type="VIDEO_COMPLETED",
              liked=(i % 2 == 0), shared=(i == 0), creator_id_override=preferred_creator, subtopic="BATTLE_ROYALE")
    for i, wp in enumerate([9, 12, 6, 15]):  # weak subtopic: real negative/low-completion history.
        b.add(category="GAMING", days_ago=5 + i * 3.1, watch_percentage=wp, event_type="VIDEO_SKIPPED",
              creator_id_override=weak_creator, subtopic="RACING_SIM")


def build_user_e(now):
    """Presentation-only 5th demo user: an EXTREME long-term SPORT persona, deliberately
    stronger and narrower than User A (User A is a mixed SPORT-leads-but-has-real-other-
    interests persona; User E exists to contrast against that -- same model, same candidate
    pool, but personalization strength should visibly scale with preference strength). SPORT
    engagement is spread across 5 real subthemes (Football/Basketball/Formula1/UFC/Tennis, one
    stable creator "channel" each) so the affinity isn't an artifact of one repeated item, and
    the total completion/like/share weight is calibrated (not asserted) to land the resulting
    category_affinity in the ~0.85-0.95 band per feature_builder.affinity_score's sigmoid
    (verified live against the running service, not hand-assumed). The other 3 categories get
    only a few negative-leaning fast-skips, pushed past RECENT_WINDOW so they read as
    background disinterest rather than a recent session shift -- no likes/shares/favorites
    anywhere outside SPORT, by design, so nothing here contradicts a "no other real interest"
    persona the way User A's moderate COMEDY/MUSIC/TRAVEL affinities intentionally do."""
    user_id = stable_uuid("demo2-user", "user-e-strong-sport")
    b = EventBuilder(user_id, now)
    subthemes = [
        ("FOOTBALL", [92, 80, 75], "liked"),
        ("BASKETBALL", [85, 78, 70], "liked"),
        ("FORMULA1", [93, 82, 77], "liked"),
        ("UFC", [88, 81, 74], "shared"),
        ("TENNIS", [86, 79, 72], "liked"),
    ]
    for si, (subtopic, wps, engagement) in enumerate(subthemes):
        creator_id = stable_uuid("demo2-creator", f"{user_id}:SPORT:{subtopic}")
        for i, wp in enumerate(wps):
            b.add(category="SPORT", days_ago=4 + si * 6 + i * 2, watch_percentage=wp,
                  liked=(engagement == "liked" and i == 0), shared=(engagement == "shared" and i == 0),
                  creator_id_override=creator_id, subtopic=subtopic)
    # Weak/background categories only -- pushed well past RECENT_WINDOW (30 days) for the same
    # reason User A's own background categories are (see build_user_a's comment): a fast-skip
    # sitting INSIDE the recent window would register as recent negative activity rather than
    # genuine long-term neutral-to-negative disinterest.
    for category, wps in (("MUSIC", [15, 12, 18]), ("COMEDY", [17, 11, 14]), ("TRAVEL", [13, 19, 10])):
        for i, wp in enumerate(wps):
            b.add(category=category, days_ago=50 + i * 5, watch_percentage=wp, event_type="VIDEO_SKIPPED",
                  creator_pool_size=2, creator_index=i % 2)
    assert len(b.events) == 24
    return user_id, b


def build_all_users(now: datetime):
    a_id, a = build_user_a(now)
    b_id, b = build_user_b(now)
    c_id, c = build_user_c_before(now)  # persisted seed = BEFORE state only; see build_user_c_shift_events for the live demo step
    d_id, d = build_user_d_before(now)
    add_user_d_semantic_creator_scenario(d)
    e_id, e = build_user_e(now)
    return {"A": (a_id, a), "B": (b_id, b), "C": (c_id, c), "D": (d_id, d), "E": (e_id, e)}


# ------------------------------------------------------------- recommendation candidate pools
# POST /api/v1/recommendation-ml-service/recommendations candidates are fully self-contained in the request body (no DB
# lookup by candidate.content_id -- app.services.recommendation_service only joins Content for
# the CALLER's own interaction history, never for candidates), so these ids don't strictly need
# a persisted Content row to make a recommendation request work. They are still deterministic
# uuid5 ids (not the old free-form "demo-cand-sport-1" strings) so the demo Postman bodies match
# the same contract every other id in this file uses, and `seed_candidate_pool` below still
# creates real Content rows for them so they read back like genuine catalog entries.
#
# One 40-item pool, shared verbatim by all 5 demo users (A/B/C/D/E), spanning 8 categories with
# real-looking titles/hashtags/topics/entities/subgenres (not generic placeholders) and a wide,
# deliberate popularity/age spread per item -- from high-popularity/fresh (e.g. Nadal 0.9/1h) to
# low-popularity/old (e.g. UFC recap 0.3/500h) to near-zero-popularity/very-fresh (the two ART
# items, 0.1/1h and 0.05/0.5h) -- so the same fixed pool can also demonstrate cold-start/
# exploration and recency-vs-popularity tradeoffs, not just category personalization. Each
# (slug, creator_slug) pair is deterministic uuid5(NAMESPACE, "demo2-candidate-*:<slug>"); items
# sharing a creator_slug share one creator identity, mirroring how a handful of prolific
# creators dominate a real catalog.
CANDIDATE_POOL_SPEC = [
    # slug, category, creator_slug, title, hashtags, topics, entities, subgenres, popularity, age_hours
    ("sport-barcelona-ucl", "SPORT", "creator-football-1", "Barcelona wins dramatic Champions League match",
     ["barcelona", "football", "championsleague"], ["Football", "Champions League"], ["Barcelona"], ["Football"], 0.85, 2),
    ("sport-real-madrid-laliga", "SPORT", "creator-football-1", "Real Madrid stuns rivals with last-minute winner",
     ["realmadrid", "football", "laliga"], ["Football", "La Liga"], ["Real Madrid"], ["Football"], 0.55, 48),
    ("sport-lakers-buzzer", "SPORT", "creator-nba-1", "Lakers clinch playoff spot with buzzer-beater",
     ["nba", "basketball", "lakers"], ["NBA", "Basketball"], ["Lakers"], ["Basketball"], 0.7, 5),
    ("sport-warriors-celtics", "SPORT", "creator-nba-1", "Warriors vs Celtics: full NBA game recap",
     ["nba", "basketball", "warriors"], ["NBA", "Basketball"], ["Golden State Warriors"], ["Basketball"], 0.4, 100),
    ("sport-nadal-roland-garros", "SPORT", "creator-tennis-1", "Nadal wins epic five-set thriller at Roland Garros",
     ["nadal", "tennis", "rolandgarros"], ["Tennis", "Roland Garros"], ["Rafael Nadal"], ["Tennis"], 0.9, 1),
    ("sport-djokovic-practice", "SPORT", "creator-tennis-1", "Djokovic's practice session ahead of Wimbledon",
     ["djokovic", "tennis", "wimbledon"], ["Tennis"], ["Novak Djokovic"], ["Tennis"], 0.35, 200),
    ("sport-verstappen-monaco", "SPORT", "creator-f1-1", "Verstappen dominates Monaco Grand Prix",
     ["f1", "formula1", "monaco"], ["Formula 1", "Monaco Grand Prix"], ["Max Verstappen"], ["Formula 1"], 0.6, 10),
    ("sport-hamilton-comeback", "SPORT", "creator-f1-1", "Hamilton's stunning comeback drive analyzed",
     ["f1", "formula1", "hamilton"], ["Formula 1"], ["Lewis Hamilton"], ["Formula 1"], 0.45, 300),
    ("sport-ufc-knockouts", "SPORT", "creator-ufc-1", "UFC fight night: top knockout compilation",
     ["ufc", "mma", "knockout"], ["UFC", "Mixed Martial Arts"], ["UFC"], ["MMA"], 0.75, 3),
    ("sport-ufc-title-recap", "SPORT", "creator-ufc-2", "UFC title bout: full fight recap",
     ["ufc", "mma", "titlefight"], ["UFC", "Mixed Martial Arts"], ["UFC"], ["MMA"], 0.3, 500),
    ("music-dualipa-single", "MUSIC", "creator-pop-1", "Dua Lipa drops surprise new pop single",
     ["pop", "newmusic", "dualipa"], ["Pop Music"], ["Dua Lipa"], ["Pop"], 0.8, 2),
    ("music-taylorswift-live", "MUSIC", "creator-pop-1", "Taylor Swift live performance highlights",
     ["pop", "taylorswift", "live"], ["Pop Music", "Live Performance"], ["Taylor Swift"], ["Pop"], 0.5, 60),
    ("music-drake-freestyle", "MUSIC", "creator-hiphop-1", "Drake freestyle session goes viral",
     ["hiphop", "drake", "freestyle"], ["Hip-Hop"], ["Drake"], ["Hip-Hop"], 0.65, 8),
    ("music-kendrick-album", "MUSIC", "creator-hiphop-1", "Kendrick Lamar's new album breakdown",
     ["hiphop", "kendricklamar", "album"], ["Hip-Hop"], ["Kendrick Lamar"], ["Hip-Hop"], 0.42, 150),
    ("music-tomorrowland-mainstage", "MUSIC", "creator-edm-1", "Tomorrowland festival main stage highlights",
     ["edm", "electronic", "tomorrowland"], ["Electronic Music", "Festival"], ["Tomorrowland"], ["Electronic"], 0.88, 1),
    ("music-foofighters-concert", "MUSIC", "creator-rock-1", "Foo Fighters rock arena with electric concert",
     ["rock", "foofighters", "concert"], ["Rock Music"], ["Foo Fighters"], ["Rock"], 0.55, 20),
    ("music-coldplay-tour", "MUSIC", "creator-rock-2", "Coldplay stadium tour: fan-favorite moments",
     ["concert", "coldplay", "live"], ["Live Performance"], ["Coldplay"], ["Concert"], 0.33, 400),
    ("gaming-minecraft-build", "GAMING", "creator-sandbox-1", "Epic Minecraft survival base build timelapse",
     ["minecraft", "gaming", "survival"], ["Minecraft"], ["Minecraft"], ["Minecraft"], 0.82, 2),
    ("gaming-fortnite-victory", "GAMING", "creator-sandbox-1", "Fortnite victory royale montage",
     ["fortnite", "gaming", "battleroyale"], ["Fortnite"], ["Fortnite"], ["Fortnite"], 0.48, 90),
    ("gaming-gtav-heist", "GAMING", "creator-openworld-1", "GTA V heist walkthrough: full mission",
     ["gta", "gtav", "heist"], ["Grand Theft Auto"], ["Grand Theft Auto"], ["GTA"], 0.7, 6),
    ("gaming-valorant-clutch", "GAMING", "creator-openworld-1", "Valorant ranked clutch play of the week",
     ["valorant", "esports", "clutch"], ["Valorant", "Esports"], ["Valorant"], ["Valorant"], 0.38, 250),
    ("gaming-lol-worlds", "GAMING", "creator-esports-1", "League of Legends World Championship recap",
     ["esports", "lol", "worlds"], ["Esports", "League of Legends"], ["League of Legends"], ["Esports"], 0.6, 15),
    ("gaming-minecraft-redstone", "GAMING", "creator-tutorial-1", "Minecraft redstone tutorial: automatic farm",
     ["minecraft", "redstone", "tutorial"], ["Minecraft"], ["Minecraft"], ["Minecraft"], 0.32, 350),
    ("comedy-kevinhart-standup", "COMEDY", "creator-standup-1", "Kevin Hart's new stand-up special clip",
     ["standup", "comedy", "kevinhart"], ["Stand-up Comedy"], ["Kevin Hart"], ["Stand-up"], 0.78, 3),
    ("comedy-public-pranks", "COMEDY", "creator-standup-1", "Hilarious public prank compilation",
     ["pranks", "funny", "comedy"], ["Pranks"], ["Candid Camera"], ["Pranks"], 0.45, 70),
    ("comedy-snl-sketch", "COMEDY", "creator-sketch-1", "SNL sketch: best moments of the season",
     ["snl", "sketch", "comedy"], ["Sketch Comedy"], ["Saturday Night Live"], ["Sketch"], 0.62, 9),
    ("comedy-funny-fails", "COMEDY", "creator-sketch-1", "Best fails and funny moments this week",
     ["funny", "fails", "comedy"], ["Funny Moments"], ["America's Funniest Videos"], ["Funny Moments"], 0.4, 180),
    ("comedy-chappelle-clip", "COMEDY", "creator-standup-2", "Dave Chappelle's latest comedy clip",
     ["standup", "chappelle", "comedy"], ["Stand-up Comedy"], ["Dave Chappelle"], ["Stand-up"], 0.58, 25),
    ("comedy-prank-war", "COMEDY", "creator-pranks-1", "Prank war compilation: roommates edition",
     ["pranks", "comedy", "prankwar"], ["Pranks"], ["Prank Wars"], ["Pranks"], 0.29, 450),
    ("travel-tokyo-guide", "TRAVEL", "creator-citytravel-1", "48 hours in Tokyo: ultimate city guide",
     ["travel", "tokyo", "citytrip"], ["City Travel"], ["Tokyo"], ["City Trip"], 0.65, 4),
    ("travel-bali-beaches", "TRAVEL", "creator-citytravel-1", "Best beaches in Bali you must visit",
     ["travel", "bali", "beach"], ["Beach Travel"], ["Bali"], ["Beach"], 0.3, 200),
    ("travel-swiss-alps", "TRAVEL", "creator-naturetravel-1", "Hiking the Swiss Alps: breathtaking views",
     ["travel", "alps", "hiking"], ["Mountain Travel"], ["Swiss Alps"], ["Mountains"], 0.5, 12),
    ("travel-eiffel-tower", "TRAVEL", "creator-naturetravel-1", "Exploring the Eiffel Tower at sunset",
     ["travel", "paris", "eiffeltower"], ["Landmarks"], ["Eiffel Tower"], ["Landmarks"], 0.2, 600),
    ("food-neapolitan-pizza", "FOOD", "creator-recipe-1", "Authentic Neapolitan pizza recipe",
     ["food", "pizza", "italian"], ["Pizza"], ["Naples"], ["Pizza"], 0.55, 10),
    ("food-carbonara-tutorial", "FOOD", "creator-recipe-1", "Creamy carbonara pasta tutorial",
     ["food", "pasta", "carbonara"], ["Pasta"], ["Carbonara"], ["Pasta"], 0.35, 150),
    ("food-bangkok-streetfood", "FOOD", "creator-streetfood-1", "Bangkok street food tour: must-try dishes",
     ["food", "streetfood", "bangkok"], ["Street Food"], ["Bangkok"], ["Street Food"], 0.6, 5),
    ("news-ai-breakthrough", "NEWS", "creator-technews-1", "Latest AI breakthrough explained",
     ["news", "technology", "ai"], ["Technology News"], ["Artificial Intelligence"], ["Technology"], 0.7, 1),
    ("news-g20-summit", "NEWS", "creator-worldnews-1", "Global summit highlights: what to know",
     ["news", "worldnews", "summit"], ["World News"], ["G20 Summit"], ["World News"], 0.25, 300),
    ("art-digital-speedpaint", "ART", "creator-digitalart-1", "Digital art speedpaint timelapse",
     ["art", "digitalart", "speedpaint"], ["Digital Art"], ["Procreate"], ["Digital Art"], 0.1, 1),
    ("art-landscape-photography", "ART", "creator-photography-1", "Landscape photography tips for beginners",
     ["art", "photography", "landscape"], ["Photography"], ["Canon"], ["Photography"], 0.05, 0.5),
]


def _candidate(slug, category, creator_slug, title, hashtags, topics, entities, subgenres, popularity, age_hours):
    return {
        "contentId": stable_uuid("demo2-candidate-content", slug),
        "creatorId": stable_uuid("demo2-candidate-creator", creator_slug),
        "category": category, "popularityScore": popularity, "ageHours": age_hours,
        "title": title, "hashtags": hashtags, "topics": topics, "entities": entities, "subgenres": subgenres,
    }


def build_shared_candidate_pool() -> list[dict]:
    """The 40-item candidate pool shared verbatim by all 5 demo users (A/B/C/D/E), spanning
    SPORT/MUSIC/GAMING/COMEDY/TRAVEL/FOOD/NEWS/ART with real-looking titles/hashtags/topics/
    entities/subgenres and a deliberate popularity/age spread."""
    return [_candidate(*spec) for spec in CANDIDATE_POOL_SPEC]


def seed_candidate_pool(base_url: str, api_key: str | None, *, dry_run: bool = False):
    """Creates the Content rows the shared candidate pool above references (idempotent, content
    only -- candidates are never interaction history, so no /events calls here)."""
    headers = {"X-Internal-API-Key": api_key} if api_key else {}
    pool = build_shared_candidate_pool()
    created, skipped = 0, 0
    for entry in pool:
        payload = {
            "contentId": entry["contentId"], "creatorId": entry["creatorId"], "contentType": "VIDEO",
            "category": entry["category"], "durationSeconds": 100.0, "popularityScore": entry["popularityScore"],
        }
        for field in ("title", "hashtags", "topics", "entities", "subgenres"):
            if field in entry:
                payload[field] = entry[field]
        if dry_run:
            continue
        status, body = _post(base_url, "/api/v1/recommendation-ml-service/contents", payload, headers)
        if status == 201:
            created += 1
        elif status == 409:
            skipped += 1
        else:
            raise RuntimeError(f"POST /contents failed ({status}): {body}")
    print(f"candidate pool content created={created} already-existed={skipped}")
    return pool


# ---------------------------------------------------------------- HTTP seeding (real API only)
def _post(base_url, path, body, headers):
    req = urllib.request.Request(f"{base_url}{path}", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def seed(base_url: str, api_key: str | None, *, dry_run: bool = False):
    now = datetime.now(timezone.utc)
    headers = {"X-Internal-API-Key": api_key} if api_key else {}
    users = build_all_users(now)

    created_content = 0
    skipped_content = 0
    stored_events = 0
    duplicate_events = 0

    for label, (user_id, builder) in users.items():
        print(f"--- seeding User {label} ({user_id}), {len(builder.events)} interactions ---")
        for event in builder.events:
            content_payload = {
                "contentId": event["content_id"], "creatorId": event["creator_id"],
                "contentType": "VIDEO", "category": event["category"],
                "durationSeconds": event["content_duration_seconds"], "popularityScore": 0.5,
            }
            # Optional semantic tags (Phase D semantic/creator scenario) -- only present on
            # events built with `subtopic=`; every other event leaves content semantically
            # neutral exactly as before.
            for field in ("title", "hashtags", "topics", "entities", "subgenres"):
                if field in event:
                    content_payload[field] = event[field]
            event_payload = {
                "eventId": event["event_id"], "userId": event["user_id"], "contentId": event["content_id"],
                "creatorId": event["creator_id"], "category": event["category"], "eventType": event["event_type"],
                "watchTimeSeconds": event["watch_time_seconds"], "contentDurationSeconds": event["content_duration_seconds"],
                "liked": event["liked"], "shared": event["shared"], "favorited": event["favorited"],
                "commented": event["commented"], "creatorFollowed": event["creator_followed"],
                "timestamp": event["timestamp"],
            }
            if dry_run:
                continue
            status, body = _post(base_url, "/api/v1/recommendation-ml-service/contents", content_payload, headers)
            if status == 201:
                created_content += 1
            elif status == 409:
                skipped_content += 1  # already exists from a prior run -- expected on reseed
            else:
                raise RuntimeError(f"POST /contents failed ({status}): {body}")

            status, body = _post(base_url, "/api/v1/recommendation-ml-service/events", event_payload, headers)
            if status != 201:
                raise RuntimeError(f"POST /events failed ({status}): {body}")
            if body.get("stored"):
                stored_events += 1
            else:
                duplicate_events += 1

    print()
    print(f"content created={created_content} already-existed={skipped_content}")
    print(f"events stored={stored_events} idempotent-duplicates={duplicate_events}")
    return users


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:3500", help="Running Recommendation ML Service base URL.")
    parser.add_argument("--api-key", default=None, help="X-Internal-API-Key, if INTERNAL_API_KEY is configured on the target.")
    parser.add_argument("--dump-json", default=None, help="Write generated event data as JSON instead of (or in addition to) seeding, for offline inspection.")
    parser.add_argument("--dry-run", action="store_true", help="Generate data and print the summary without making any HTTP calls.")
    args = parser.parse_args()

    print("DEV/TEST-ONLY seed script -- populates demo interaction history for 4 fixed UUID")
    print(f"users at {args.base_url} via the existing POST /api/v1/recommendation-ml-service/contents + POST /api/v1/recommendation-ml-service/events")
    print("ingestion endpoints. Safe to rerun (idempotent). Not a production data path.\n")

    now = datetime.now(timezone.utc)
    if args.dump_json:
        users = build_all_users(now)
        dump = {label: {"userId": uid, "events": b.events} for label, (uid, b) in users.items()}
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(dump, f, indent=2)
        print(f"wrote {args.dump_json}")
        if not args.dry_run:
            seed(args.base_url, args.api_key)
            seed_candidate_pool(args.base_url, args.api_key)
        return

    seed(args.base_url, args.api_key, dry_run=args.dry_run)
    seed_candidate_pool(args.base_url, args.api_key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
