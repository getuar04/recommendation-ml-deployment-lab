"""Deterministic ranking-benchmark scenarios (EASY/MEDIUM/HARD/ADVERSARIAL, plus the
temporal-shift and NOT_INTERESTED-localization paired scenarios).

Deliberately independent of scripts/generate_synthetic_data.py (the TRAINING data generator):
these scenario IDs/patterns must never be fed into training data, or "the benchmark looks
perfect" would just mean "the model memorized the benchmark" -- see
app.benchmark.runner/scripts/run_ranking_benchmark.py for how training and benchmarking stay
on physically separate databases.

Every history event and candidate below is genuinely synthetic, generic content
(no named real-world entities beyond ordinary football-club-style placeholders already used
elsewhere in this codebase's own eligibility-gate probes/synthetic generator).
"""
from __future__ import annotations

from datetime import timedelta

from app.benchmark.scenario_types import (
    BenchmarkCandidate,
    BenchmarkScenario,
    HistoryEvent,
    RelativeConstraint,
)

_DAY = timedelta(days=1)
_HOUR = timedelta(hours=1)
_MIN = timedelta(minutes=1)


def _event(content_id, creator_id, category, *, watch_percentage, when, event_type=None,
           liked=False, shared=False, favorited=False, creator_followed=False,
           topics=None, hashtags=None, entities=None, subgenres=None, title=None) -> HistoryEvent:
    if event_type is None:
        event_type = "VIDEO_COMPLETED" if watch_percentage >= 90 else (
            "VIDEO_SKIPPED" if watch_percentage < 20 else "VIDEO_WATCHED")
    return HistoryEvent(
        content_id=content_id, creator_id=creator_id, category=category, event_type=event_type,
        watch_percentage=watch_percentage, when=when, liked=liked, shared=shared, favorited=favorited,
        creator_followed=creator_followed, topics=topics or [], hashtags=hashtags or [],
        entities=entities or [], subgenres=subgenres or [], title=title,
    )


# ============================================================================================
# EASY: fundamental category preference.
# ============================================================================================

def easy_scenario() -> BenchmarkScenario:
    history = [
        *[_event(f"easy-hist-sport-{i}", "coach-easy", "SPORT", watch_percentage=92, liked=True,
                 when=-(30 + i) * _DAY, topics=["FOOTBALL"], hashtags=["FOOTBALL"])
          for i in range(10)],
        # Weak (not absent) MUSIC/COMEDY/NEWS history: a couple of lukewarm, unremarkable watches.
        _event("easy-hist-music-1", "dj-easy", "MUSIC", watch_percentage=40, when=-20 * _DAY),
        _event("easy-hist-comedy-1", "fun-easy", "COMEDY", watch_percentage=35, when=-18 * _DAY),
        _event("easy-hist-news-1", "anchor-easy", "NEWS", watch_percentage=30, when=-15 * _DAY),
    ]
    candidates = [
        BenchmarkCandidate("easy-sport-football-1", "SPORT", "coach-easy", 4, content_popularity_score=0.6,
                            topics=["FOOTBALL"], hashtags=["FOOTBALL"], note="Strong long-term category + semantic match, unseen"),
        BenchmarkCandidate("easy-sport-football-2", "SPORT", "coach-easy-2", 4, content_popularity_score=0.55,
                            topics=["FOOTBALL"], hashtags=["FOOTBALL"], note="Same as above, different creator"),
        BenchmarkCandidate("easy-sport-generic", "SPORT", "coach-easy-3", 3, content_popularity_score=0.5,
                            note="Plain SPORT, no semantic boost"),
        BenchmarkCandidate("easy-music-secondary", "MUSIC", "dj-easy-2", 1, content_popularity_score=0.5,
                            note="Weak secondary interest"),
        BenchmarkCandidate("easy-comedy-popular", "COMEDY", "fun-easy-2", 1, content_popularity_score=0.9,
                            note="Popular but weak-interest category"),
        BenchmarkCandidate("easy-news-trending", "NEWS", "anchor-easy-2", 1, content_popularity_score=0.85,
                            note="Trending but weak-interest category"),
    ]
    constraints = [
        RelativeConstraint("football_beats_music", "easy-sport-football-1", "easy-music-secondary", critical=True,
                            description="Relevant SPORT/Football must clearly outrank weak-interest MUSIC"),
        RelativeConstraint("football_beats_comedy_popularity", "easy-sport-football-1", "easy-comedy-popular",
                            description="Relevance must beat raw popularity in an unambiguous case"),
        RelativeConstraint("football_2_beats_news", "easy-sport-football-2", "easy-news-trending"),
        RelativeConstraint("generic_sport_beats_music", "easy-sport-generic", "easy-music-secondary"),
    ]
    return BenchmarkScenario("easy-fundamental-preference", "EASY",
                              "Strong long-term SPORT/Football preference vs. weak MUSIC/COMEDY/NEWS interest.",
                              "bench-easy-user", history, candidates, constraints)


# ============================================================================================
# Category-identity preference (2026-09-04 root-cause addition): a real production XGBRanker
# candidate was found to pass every EASY/MEDIUM/HARD/ADVERSARIAL constraint above (including
# easy_scenario's own "football_beats_music") while still ranking a plain, genuinely never-seen
# MUSIC candidate above a plain, strongly-preferred long-term SPORT candidate -- because every
# candidate above carries topic/hashtag semantic metadata on its SPORT side, which supplies
# extra reranker-side reinforcement (semantic affinity boost) a bare category-only candidate
# never gets. This scenario isolates that exact gap: candidates carry NO semantic metadata at
# all (matching the plain {contentId, creatorId, category, contentPopularityScore,
# contentAgeHours} shape production candidate pools commonly send), so category identity +
# behavioral affinity is the ONLY signal available -- exactly what app.ml.eligibility's generic
# CATEGORY_A/CATEGORY_B-placeholder probes intend to test, but (per that module's own docstring)
# cannot, since OneHotEncoder(handle_unknown="ignore") encodes an out-of-vocabulary placeholder
# category as all-zero for both sides of every probe there, hiding any real, named-category-
# identity bias a candidate learned. Deliberately excluded from all_named_scenarios() (never
# folded into SELECTION_DIFFICULTIES' byDifficulty aggregates) -- see
# app.ml.candidate_evaluation.FUNDAMENTAL_PREFERENCE_SCENARIO_ID for how it is force-included in
# every candidate's eligibility check regardless, the same mechanism already used for
# "adversarial-seen-vs-unseen".
# ============================================================================================

def category_identity_preference_scenario() -> BenchmarkScenario:
    history = [_event(f"catid-hist-{i}", "coach-catid", "SPORT", watch_percentage=95, liked=i < 8,
                       when=-(50 + i) * _DAY)
               for i in range(14)]
    candidates = [
        BenchmarkCandidate("catid-sport", "SPORT", "coach-catid-2", 4,
                            note="Plain, no semantic metadata -- strong real long-term category history"),
        BenchmarkCandidate("catid-music", "MUSIC", "dj-catid", 0,
                            note="Plain, no semantic metadata -- genuinely never-seen category"),
    ]
    constraints = [
        RelativeConstraint("sport_beats_music_by_identity_alone", "catid-sport", "catid-music", critical=True,
                            description="A strong, real, long-term-preferred category must beat a genuinely "
                                        "never-seen one when neither candidate carries any semantic metadata"),
    ]
    return BenchmarkScenario("fundamental-category-identity-preference", "EASY",
                              "Strong long-term SPORT preference vs. genuinely never-seen MUSIC, both bare "
                              "(no topics/hashtags/title) -- category identity + behavioral affinity only.",
                              "bench-catid-user", history, candidates, constraints)


# ============================================================================================
# MEDIUM: competing signals (recent shift, semantic subtheme, creator, popularity).
# ============================================================================================

def medium_scenario() -> BenchmarkScenario:
    history = [
        *[_event(f"med-hist-sport-{i}", "coach-med-a", "SPORT", watch_percentage=88, liked=(i % 2 == 0),
                 when=-(35 + i) * _DAY, topics=["FOOTBALL"], hashtags=["FOOTBALL"])
          for i in range(8)],
        *[_event(f"med-hist-music-{i}", "dj-med", "MUSIC", watch_percentage=60, when=-(20 + i) * _DAY)
          for i in range(3)],
        # Recent MUSIC uptick (hours-scale, well outside SESSION_WINDOW but inside RECENT_WINDOW).
        *[_event(f"med-hist-music-recent-{i}", "dj-med", "MUSIC", watch_percentage=93, liked=True,
                 when=-(2 * _HOUR + i * 30 * _MIN))
          for i in range(3)],
        # Strong prior positive history with Creator A specifically.
        *[_event(f"med-hist-creatorA-{i}", "creator-med-a", "SPORT", watch_percentage=90, liked=True,
                  when=-(25 + i) * _DAY, topics=["FOOTBALL"])
          for i in range(4)],
    ]
    candidates = [
        BenchmarkCandidate("med-football-creatorA", "SPORT", "creator-med-a", 4, topics=["FOOTBALL"], hashtags=["FOOTBALL"],
                            note="Strong semantic match + strong creator history"),
        BenchmarkCandidate("med-football-creatorB", "SPORT", "creator-med-b-unknown", 3, topics=["FOOTBALL"], hashtags=["FOOTBALL"],
                            note="Strong semantic match, unknown creator"),
        BenchmarkCandidate("med-music-recent-match", "MUSIC", "dj-med", 3, content_popularity_score=0.6,
                            note="Matches the recent MUSIC uptick"),
        BenchmarkCandidate("med-sport-tennis", "SPORT", "coach-med-b", 1, topics=["TENNIS"], hashtags=["TENNIS"],
                            note="SPORT category but a subtheme with no positive history"),
        BenchmarkCandidate("med-comedy-extreme-popularity", "COMEDY", "fun-med", 1, content_popularity_score=0.97,
                            note="Extremely popular, irrelevant category"),
        BenchmarkCandidate("med-football-old", "SPORT", "coach-med-a", 3, content_age_hours=200,
                            topics=["FOOTBALL"], hashtags=["FOOTBALL"], note="Old but still relevant"),
        BenchmarkCandidate("med-football-fresh-weak-semantic", "SPORT", "coach-med-c", 2, content_age_hours=1,
                            note="Fresh but weaker semantic match (no topic tag)"),
    ]
    constraints = [
        RelativeConstraint("football_remains_top", "med-football-creatorA", "med-sport-tennis", critical=True,
                            description="Football must remain highly ranked over a no-history subtheme"),
        RelativeConstraint("tennis_not_boosted_by_category_alone", "med-football-creatorB", "med-sport-tennis",
                            description="SPORT category alone must not lift Tennis to the top"),
        RelativeConstraint("popularity_alone_insufficient", "med-football-creatorA", "med-comedy-extreme-popularity",
                            critical=True, description="Extreme popularity alone must not beat strong relevance"),
        RelativeConstraint("creator_helps_but_not_absolute", "med-football-creatorA", "med-football-fresh-weak-semantic",
                            description="Strong creator + semantic match should still lead a weaker semantic match"),
        RelativeConstraint("recent_music_competitive", "med-music-recent-match", "med-comedy-extreme-popularity",
                            description="Recent-interest MUSIC should be competitive with irrelevant popularity"),
    ]
    return BenchmarkScenario("medium-competing-signals", "MEDIUM",
                              "Long-term SPORT/Football + strong creator, vs. a recent MUSIC uptick, a no-history "
                              "subtheme, and extreme irrelevant popularity.",
                              "bench-medium-user", history, candidates, constraints)


# ============================================================================================
# HARD: hard negatives, explicit rejection, seen/unseen, wrong-subtheme-but-followed-creator.
# ============================================================================================

def hard_scenario() -> BenchmarkScenario:
    history = [
        # Very strong long-term SPORT/Football.
        *[_event(f"hard-hist-football-{i}", "creator-hard-a", "SPORT", watch_percentage=93, liked=True,
                  when=-(30 + i) * _DAY, topics=["FOOTBALL"], hashtags=["FOOTBALL"], entities=["REAL_MADRID"])
          for i in range(10)],
        # Weak/moderate long-term MUSIC.
        *[_event(f"hard-hist-music-{i}", "dj-hard", "MUSIC", watch_percentage=55, when=-(25 + i) * _DAY)
          for i in range(3)],
        # Recent/session MUSIC rapidly increasing (minute-scale, inside SESSION_WINDOW).
        *[_event(f"hard-hist-music-session-{i}", "dj-hard", "MUSIC", watch_percentage=94, liked=True,
                  when=-(10 - i * 2) * _MIN)
          for i in range(4)],
        # Explicit, repeated NOT_INTERESTED on Tennis specifically.
        *[_event(f"hard-hist-tennis-reject-{i}", "creator-hard-tennis", "SPORT", watch_percentage=4,
                  event_type="CONTENT_NOT_INTERESTED", when=-(12 + i * 3) * _DAY, topics=["TENNIS"], hashtags=["TENNIS"])
          for i in range(4)],
        # Creator B: strong completion history (not followed).
        *[_event(f"hard-hist-creatorB-{i}", "creator-hard-b", "SPORT", watch_percentage=91, liked=True,
                  when=-(20 + i) * _DAY, topics=["FOOTBALL"])
          for i in range(5)],
        # Creator A followed explicitly.
        _event("hard-hist-follow", "creator-hard-a", "SPORT", watch_percentage=95, liked=True,
               creator_followed=True, when=-28 * _DAY, topics=["FOOTBALL"]),
        # Already-seen Football content (same content_id reused as candidate G below).
        _event("hard-seen-football", "creator-hard-a", "SPORT", watch_percentage=85, when=-5 * _DAY, topics=["FOOTBALL"]),
    ]
    candidates = [
        BenchmarkCandidate("hard-A-football-realmadrid", "SPORT", "creator-hard-a", 4,
                            topics=["FOOTBALL"], entities=["REAL_MADRID"], hashtags=["FOOTBALL"],
                            note="A: unseen, strong semantic match, preferred creator"),
        BenchmarkCandidate("hard-B-football-ucl", "SPORT", "creator-hard-b", 4,
                            topics=["FOOTBALL"], entities=["CHAMPIONS_LEAGUE"], hashtags=["FOOTBALL"],
                            note="B: unseen, strong semantic match, strong-completion creator"),
        BenchmarkCandidate("hard-C-tennis-djokovic", "SPORT", "creator-hard-tennis", 0, content_popularity_score=0.8,
                            topics=["TENNIS"], entities=["DJOKOVIC"], hashtags=["TENNIS"],
                            note="C: popular but explicitly-rejected subtheme"),
        BenchmarkCandidate("hard-D-tennis-wimbledon", "SPORT", "creator-hard-tennis-2", 0, content_popularity_score=0.9,
                            topics=["TENNIS"], entities=["WIMBLEDON"], hashtags=["TENNIS"],
                            note="D: very popular but explicitly-rejected subtheme"),
        BenchmarkCandidate("hard-E-music-session", "MUSIC", "dj-hard", 3, content_popularity_score=0.5,
                            note="E: matches the recent/session MUSIC signal"),
        BenchmarkCandidate("hard-F-comedy-popular", "COMEDY", "fun-hard", 1, content_popularity_score=0.95,
                            note="F: globally popular, irrelevant"),
        BenchmarkCandidate("hard-G-football-seen", "SPORT", "creator-hard-a", 2, already_seen=True,
                            topics=["FOOTBALL"], hashtags=["FOOTBALL"], note="G: same profile as A, already seen"),
        BenchmarkCandidate("hard-H-wrong-subtheme-followed", "SPORT", "creator-hard-a", 1,
                            topics=["TENNIS"], hashtags=["TENNIS"], creator_followed=True,
                            note="H: followed creator, but wrong (rejected) subtheme"),
        BenchmarkCandidate("hard-I-music-weak-semantic-strong-recent", "MUSIC", "dj-hard-2", 3,
                            note="I: weak semantic match but strong recent-category signal"),
        BenchmarkCandidate("hard-J-news-trending", "NEWS", "anchor-hard", 1, content_popularity_score=0.8,
                            note="J: trending, irrelevant"),
        BenchmarkCandidate("hard-K-football-older", "SPORT", "creator-hard-a", 3, content_age_hours=300,
                            topics=["FOOTBALL"], hashtags=["FOOTBALL"], note="K: older Football content"),
        BenchmarkCandidate("hard-L-generic-strong-creator", "SPORT", "creator-hard-b", 2,
                            note="L: strong creator, weaker user semantic match (no topic tag)"),
    ]
    constraints = [
        RelativeConstraint("football_beats_tennis_rejected", "hard-A-football-realmadrid", "hard-C-tennis-djokovic",
                            critical=True, description="Unseen relevant Football must beat explicitly-rejected Tennis"),
        RelativeConstraint("football_B_beats_tennis_D", "hard-B-football-ucl", "hard-D-tennis-wimbledon",
                            critical=True, description="Same, second Football/Tennis pair (very popular Tennis)"),
        RelativeConstraint("unseen_beats_seen", "hard-A-football-realmadrid", "hard-G-football-seen", critical=True,
                            description="Unseen must beat an equivalent already-seen candidate"),
        RelativeConstraint("recent_music_beats_irrelevant_popularity", "hard-E-music-session", "hard-F-comedy-popular",
                            critical=True, description="Recent/session MUSIC must beat irrelevant global popularity"),
        RelativeConstraint("semantic_beats_wrong_subtheme_followed", "hard-A-football-realmadrid", "hard-H-wrong-subtheme-followed",
                            critical=True, description="Strong semantic match must beat a followed-but-wrong-subtheme candidate"),
        RelativeConstraint("tennis_rejection_visible_despite_sport_strength", "hard-A-football-realmadrid", "hard-D-tennis-wimbledon",
                            description="Tennis rejection must remain visible even though SPORT overall is very strong"),
        RelativeConstraint("weak_semantic_recent_beats_irrelevant", "hard-I-music-weak-semantic-strong-recent", "hard-J-news-trending",
                            description="Recent-category signal should outweigh a weak semantic match against pure irrelevance"),
    ]
    return BenchmarkScenario("hard-mixed-signals", "HARD",
                              "Strong long-term SPORT/Football, rising recent MUSIC, repeated explicit Tennis "
                              "rejection, mixed creator history, and an already-seen duplicate.",
                              "bench-hard-user", history, candidates, constraints)


# ============================================================================================
# ADVERSARIAL: six intentional conflicts, each its own small scenario.
# ============================================================================================

def _adversarial_popularity_vs_relevance() -> BenchmarkScenario:
    history = [_event(f"adv-pop-hist-{i}", "creator-adv-pop", "SPORT", watch_percentage=90, liked=True,
                       when=-(30 + i) * _DAY, topics=["FOOTBALL"]) for i in range(8)]
    candidates = [
        BenchmarkCandidate("adv-pop-relevant-unpopular", "SPORT", "creator-adv-pop", 4, content_popularity_score=0.15,
                            topics=["FOOTBALL"], note="Perfect match, low popularity"),
        BenchmarkCandidate("adv-pop-irrelevant-popular", "COMEDY", "fun-adv-pop", 0, content_popularity_score=0.98,
                            note="Weak relevance, extremely popular"),
    ]
    constraints = [
        RelativeConstraint("relevance_beats_popularity", "adv-pop-relevant-unpopular", "adv-pop-irrelevant-popular",
                            critical=True, description="Strong relevance should usually beat pure popularity"),
    ]
    return BenchmarkScenario("adversarial-popularity-vs-relevance", "ADVERSARIAL",
                              "Conflict A: perfect low-popularity match vs. weak-relevance viral content.",
                              "bench-adv-pop-user", history, candidates, constraints)


def _adversarial_creator_vs_not_interested() -> BenchmarkScenario:
    history = [
        *[_event(f"adv-cvn-hist-follow-{i}", "creator-adv-cvn", "SPORT", watch_percentage=90, liked=True,
                  creator_followed=(i == 0), when=-(25 + i) * _DAY, topics=["FOOTBALL"])
          for i in range(6)],
        *[_event(f"adv-cvn-hist-reject-{i}", "creator-adv-cvn", "SPORT", watch_percentage=4,
                  event_type="CONTENT_NOT_INTERESTED", when=-(10 + i * 2) * _DAY, topics=["TENNIS"])
          for i in range(4)],
    ]
    candidates = [
        BenchmarkCandidate("adv-cvn-followed-rejected-subtheme", "SPORT", "creator-adv-cvn", 0,
                            creator_followed=True, topics=["TENNIS"], note="Followed creator, rejected subtheme"),
        BenchmarkCandidate("adv-cvn-followed-good-subtheme", "SPORT", "creator-adv-cvn", 4,
                            creator_followed=True, topics=["FOOTBALL"], note="Followed creator, good subtheme"),
    ]
    constraints = [
        RelativeConstraint("good_subtheme_beats_rejected_subtheme", "adv-cvn-followed-good-subtheme",
                            "adv-cvn-followed-rejected-subtheme", critical=True,
                            description="Followed creator must not override an explicitly-rejected subtheme"),
    ]
    return BenchmarkScenario("adversarial-creator-vs-not-interested", "ADVERSARIAL",
                              "Conflict B: same followed creator, one candidate in a rejected subtheme.",
                              "bench-adv-cvn-user", history, candidates, constraints)


def _adversarial_longterm_vs_session() -> BenchmarkScenario:
    history = [
        *[_event(f"adv-lvs-hist-sport-{i}", "coach-adv-lvs", "SPORT", watch_percentage=95, liked=True,
                  when=-(30 + i) * _DAY, topics=["FOOTBALL"]) for i in range(12)],
        *[_event(f"adv-lvs-hist-music-session-{i}", "dj-adv-lvs", "MUSIC", watch_percentage=96, liked=True,
                  shared=(i == 2), when=-(8 - i * 2) * _MIN) for i in range(4)],
    ]
    candidates = [
        BenchmarkCandidate("adv-lvs-sport", "SPORT", "coach-adv-lvs", 3, topics=["FOOTBALL"], note="Long-term dominant category"),
        BenchmarkCandidate("adv-lvs-music", "MUSIC", "dj-adv-lvs", 4, note="Very strong, very recent session signal"),
    ]
    constraints = [
        RelativeConstraint("session_music_materially_competitive", "adv-lvs-music", "adv-lvs-sport",
                            critical=True, description="A very strong, very recent session signal should materially rise"),
    ]
    return BenchmarkScenario("adversarial-longterm-vs-session", "ADVERSARIAL",
                              "Conflict C: extremely strong long-term SPORT vs. a very strong, very recent MUSIC session.",
                              "bench-adv-lvs-user", history, candidates, constraints)


def _adversarial_seen_vs_unseen() -> BenchmarkScenario:
    history = [_event(f"adv-svu-hist-{i}", "creator-adv-svu", "SPORT", watch_percentage=90, liked=True,
                       when=-(20 + i) * _DAY, topics=["FOOTBALL"]) for i in range(8)]
    candidates = [
        BenchmarkCandidate("adv-svu-unseen", "SPORT", "creator-adv-svu", 4, topics=["FOOTBALL"], note="Unseen"),
        BenchmarkCandidate("adv-svu-seen", "SPORT", "creator-adv-svu", 4, already_seen=True, topics=["FOOTBALL"], note="Seen, otherwise identical"),
    ]
    constraints = [
        RelativeConstraint("unseen_beats_seen", "adv-svu-unseen", "adv-svu-seen", critical=True,
                            description="Two near-identical relevant candidates: unseen must beat seen"),
    ]
    return BenchmarkScenario("adversarial-seen-vs-unseen", "ADVERSARIAL",
                              "Conflict D: two otherwise-identical relevant candidates, one already seen.",
                              "bench-adv-svu-user", history, candidates, constraints)


def _adversarial_semantic_neighbor() -> BenchmarkScenario:
    history = [
        *[_event(f"adv-sn-hist-ucl-{i}", "creator-adv-sn", "SPORT", watch_percentage=93, liked=True,
                  when=-(25 + i) * _DAY, topics=["FOOTBALL"], entities=["CHAMPIONS_LEAGUE"]) for i in range(6)],
    ]
    candidates = [
        BenchmarkCandidate("adv-sn-ucl", "SPORT", "creator-adv-sn", 4, topics=["FOOTBALL"], entities=["CHAMPIONS_LEAGUE"],
                            note="Champions League -- matches prior entity history"),
        BenchmarkCandidate("adv-sn-domestic", "SPORT", "creator-adv-sn-2", 2, topics=["FOOTBALL"],
                            note="Domestic league -- same topic, no entity match"),
        BenchmarkCandidate("adv-sn-tennis", "SPORT", "creator-adv-sn-3", 0, topics=["TENNIS"],
                            note="Tennis -- same broad category, unrelated subtheme"),
    ]
    constraints = [
        RelativeConstraint("ucl_beats_domestic", "adv-sn-ucl", "adv-sn-domestic",
                            description="Exact entity match should outrank same-topic-only content"),
        RelativeConstraint("domestic_beats_tennis", "adv-sn-domestic", "adv-sn-tennis",
                            description="Same broad category alone should still beat an unrelated subtheme"),
    ]
    return BenchmarkScenario("adversarial-semantic-neighbor", "ADVERSARIAL",
                              "Conflict E: Champions League vs. domestic league vs. Tennis, all nominally SPORT.",
                              "bench-adv-sn-user", history, candidates, constraints)


def _adversarial_creator_shortcut_prevention() -> BenchmarkScenario:
    history = [
        *[_event(f"adv-csp-hist-creator-{i}", "creator-adv-csp", "SPORT", watch_percentage=92, liked=True,
                  when=-(25 + i) * _DAY) for i in range(8)],  # strong creator history, no topic tags at all
        *[_event(f"adv-csp-hist-session-{i}", "dj-adv-csp", "MUSIC", watch_percentage=95, liked=True,
                  when=-(6 - i * 2) * _MIN) for i in range(3)],
    ]
    candidates = [
        BenchmarkCandidate("adv-csp-preferred-creator-weak-semantic", "SPORT", "creator-adv-csp", 2,
                            topics=["TENNIS"], note="Strongly preferred creator, but low-relevance content"),
        BenchmarkCandidate("adv-csp-unknown-creator-strong-session", "MUSIC", "dj-adv-csp", 4,
                            note="Unknown-ish creator, strong current session relevance"),
    ]
    constraints = [
        RelativeConstraint("session_relevance_beats_creator_shortcut", "adv-csp-unknown-creator-strong-session",
                            "adv-csp-preferred-creator-weak-semantic", critical=True,
                            description="Creator preference must not act as an absolute override of low relevance"),
    ]
    return BenchmarkScenario("adversarial-creator-shortcut-prevention", "ADVERSARIAL",
                              "Conflict F: strongly preferred creator with weak relevance vs. strong session relevance.",
                              "bench-adv-csp-user", history, candidates, constraints)


def adversarial_scenarios() -> list[BenchmarkScenario]:
    return [
        _adversarial_popularity_vs_relevance(),
        _adversarial_creator_vs_not_interested(),
        _adversarial_longterm_vs_session(),
        _adversarial_seen_vs_unseen(),
        _adversarial_semantic_neighbor(),
        _adversarial_creator_shortcut_prevention(),
    ]


# ============================================================================================
# TEMPORAL: same user/candidates, two history states (T0 -> T1), to prove personalization
# responds to a user-state change without retraining.
# ============================================================================================

def temporal_shift_scenarios() -> tuple[BenchmarkScenario, BenchmarkScenario]:
    """Returns (state_t0, state_t1) -- identical user_id and candidate pool; only the seeded
    history differs. The runner compares MUSIC candidates' rank/score between the two."""
    shared_candidates = [
        BenchmarkCandidate("temporal-sport", "SPORT", "coach-temporal", 3, topics=["FOOTBALL"], note="Long-term dominant"),
        BenchmarkCandidate("temporal-music-1", "MUSIC", "dj-temporal", 2, note="MUSIC candidate 1"),
        BenchmarkCandidate("temporal-music-2", "MUSIC", "dj-temporal-2", 2, note="MUSIC candidate 2"),
        BenchmarkCandidate("temporal-comedy", "COMEDY", "fun-temporal", 1, content_popularity_score=0.6, note="Filler"),
    ]
    sport_history = [_event(f"temporal-hist-sport-{i}", "coach-temporal", "SPORT", watch_percentage=92, liked=True,
                             when=-(30 + i) * _DAY, topics=["FOOTBALL"]) for i in range(10)]
    t0 = BenchmarkScenario("temporal-shift-t0", "TEMPORAL",
                            "T0: SPORT long-term dominant, MUSIC low/absent.",
                            "bench-temporal-user", sport_history, shared_candidates, [])
    t1_extra = [
        _event("temporal-hist-music-shift-1", "dj-temporal", "MUSIC", watch_percentage=95, liked=True, when=-5 * _MIN),
        _event("temporal-hist-music-shift-2", "dj-temporal", "MUSIC", watch_percentage=90, shared=True, when=-3 * _MIN),
    ]
    t1 = BenchmarkScenario("temporal-shift-t1", "TEMPORAL",
                            "T1: same as T0, plus a recent strong MUSIC watch+like+share burst.",
                            "bench-temporal-user", sport_history + t1_extra, shared_candidates, [])
    return t0, t1


# ============================================================================================
# LOCALIZATION: same-subtheme-repeat vs. diverse-subtheme rejection.
# ============================================================================================

def not_interested_localization_scenarios() -> tuple[BenchmarkScenario, BenchmarkScenario]:
    """Returns (history_a, history_b) -- same candidate pool; history_a repeatedly rejects
    ONE subtheme (Tennis), history_b rejects several DIFFERENT subthemes. Expected: Football
    stays strong in both, but broad SPORT relevance should hold up better in (a) than (b)."""
    shared_candidates = [
        BenchmarkCandidate("localization-football", "SPORT", "coach-localization", 3, topics=["FOOTBALL"],
                            note="Never-rejected subtheme"),
        BenchmarkCandidate("localization-basketball-unrejected", "SPORT", "coach-localization-2", 2,
                            topics=["BASKETBALL"], note="Broad-category probe: never directly rejected in EITHER history"),
    ]
    base_history = [_event(f"localization-hist-base-{i}", "coach-localization", "SPORT", watch_percentage=90,
                            liked=True, when=-(30 + i) * _DAY, topics=["FOOTBALL"]) for i in range(6)]
    same_subtheme_rejections = [
        _event(f"localization-hist-tennis-{i}", "coach-localization-tennis", "SPORT", watch_percentage=4,
               event_type="CONTENT_NOT_INTERESTED", when=-(10 + i * 2) * _DAY, topics=["TENNIS"])
        for i in range(4)
    ]
    diverse_subtheme_rejections = [
        _event(f"localization-hist-diverse-{i}", f"coach-localization-{token.lower()}", "SPORT", watch_percentage=4,
               event_type="CONTENT_NOT_INTERESTED", when=-(10 + i * 2) * _DAY, topics=[token])
        # Deliberately excludes BASKETBALL/FOOTBALL: both are used as "never directly
        # rejected in either history" probe candidates below, isolating the broad-category
        # spillover effect from direct-rejection effects on the same subtheme.
        for i, token in enumerate(["TENNIS", "RUGBY", "CRICKET", "FORMULA1"])
    ]
    history_a = BenchmarkScenario("localization-same-subtheme-repeat", "LOCALIZATION",
                                   "Repeated NOT_INTERESTED on the SAME subtheme (Tennis) only.",
                                   "bench-localization-user", base_history + same_subtheme_rejections,
                                   shared_candidates, [])
    history_b = BenchmarkScenario("localization-diverse-subthemes", "LOCALIZATION",
                                   "NOT_INTERESTED spread across several DIFFERENT subthemes.",
                                   "bench-localization-user", base_history + diverse_subtheme_rejections,
                                   shared_candidates, [])
    return history_a, history_b


# ============================================================================================
# DOMINANT-CATEGORY DIVERSITY: a legitimately SPORT-dominant user with many strong SPORT
# candidates. The reranker's diversity decay must not forcibly interleave categories the
# user has no real interest in just to avoid repetition -- see app.benchmark.metrics.
# diversity_diagnostics and app.benchmark.runner for how this is measured (diagnostic only,
# never a pass/fail constraint, per spec).
# ============================================================================================

def dominant_category_scenario() -> BenchmarkScenario:
    history = [_event(f"dom-hist-{i}", f"coach-dom-{i % 3}", "SPORT", watch_percentage=93, liked=True,
                       when=-(20 + i) * _DAY, topics=["FOOTBALL"]) for i in range(15)]
    candidates = [
        BenchmarkCandidate(f"dom-sport-{i}", "SPORT", f"coach-dom-{i % 4}", 4 if i < 5 else 3,
                            topics=["FOOTBALL"], hashtags=["FOOTBALL"], note=f"Dominant-category candidate {i}")
        for i in range(10)
    ] + [
        BenchmarkCandidate("dom-music", "MUSIC", "dj-dom", 1, note="Secondary interest"),
        BenchmarkCandidate("dom-comedy", "COMEDY", "fun-dom", 1, content_popularity_score=0.7, note="Filler"),
    ]
    return BenchmarkScenario("dominant-category-diversity", "HARD",
                              "A legitimately SPORT-dominant user with 10 strong SPORT candidates: Top-10 should "
                              "still reflect that dominance, not be forcibly diversified.",
                              "bench-dominant-user", history, candidates, [])


def all_named_scenarios() -> list[BenchmarkScenario]:
    """EASY/MEDIUM/HARD/ADVERSARIAL only -- the paired temporal/localization scenarios (and
    the SOCIAL scenarios below) are evaluated separately, never through this function -- see
    each group's own module docstring for why. Never wired into model selection."""
    return [easy_scenario(), medium_scenario(), hard_scenario(), *adversarial_scenarios()]


# ============================================================================================
# SOCIAL: related-user (SOCIAL/COLLABORATIVE candidateSource) candidate generation. Diagnostic
# only, exactly like temporal-shift/localization above -- NEVER returned by
# all_named_scenarios(), never fed into app.ml.candidate_evaluation.SELECTION_DIFFICULTIES, so
# these can never affect model eligibility/selection (spec: "do not train on these"). Exists to
# prove app.ml.reranker.social_relevance's real behavior end to end through the actual
# recommendation_service.recommend() path (app.benchmark.runner.run_scenario), not just at the
# unit level tests/test_social_relevance.py already covers.
# ============================================================================================

def social_scenarios() -> list[BenchmarkScenario]:
    def _social(interest_similarity, relationship_strength, source_user_engagement, *, mutual_follow=False):
        return {
            "interestSimilarity": interest_similarity, "relationshipStrength": relationship_strength,
            "sourceUserEngagement": source_user_engagement, "mutualFollow": mutual_follow,
        }

    scenarios = []

    # 1. Mutual-friend relevant candidate: high relationship_strength (mutual follow) + high
    # similarity, candidate is in the user's own real long-term-preferred category/subtheme.
    history = [_event(f"social-mutual-hist-{i}", "coach-social-mutual", "SPORT", watch_percentage=92,
                       liked=True, when=-(30 + i) * _DAY, topics=["FOOTBALL"]) for i in range(8)]
    candidates = [
        BenchmarkCandidate("social-mutual-target", "SPORT", "friend-mutual", 4, topics=["FOOTBALL"],
                            candidate_source="SOCIAL", social_context=_social(0.85, 0.9, 0.8, mutual_follow=True),
                            note="Mutual-friend candidate, matches real long-term SPORT/Football preference"),
        BenchmarkCandidate("social-mutual-filler", "COMEDY", "fun-social", 1, note="Unrelated filler"),
    ]
    scenarios.append(BenchmarkScenario(
        "social-mutual-friend-relevant", "SOCIAL",
        "Mutual-follow friend with high interest similarity engaged a candidate matching the "
        "user's own real SPORT/Football preference -- must rank well.",
        "bench-social-mutual-user", history, candidates,
        [RelativeConstraint("mutual_friend_beats_filler", "social-mutual-target", "social-mutual-filler", critical=True)],
    ))

    # 2. Similar-user collaborative candidate: NO relationship at all (relationship_strength=0,
    # mutual_follow=False) -- pure behavior/interest similarity. COLLABORATIVE source.
    history = [_event(f"social-collab-hist-{i}", "coach-social-collab", "GAMING", watch_percentage=90,
                       liked=True, when=-(25 + i) * _DAY, topics=["STRATEGY"]) for i in range(8)]
    candidates = [
        BenchmarkCandidate("social-collab-target", "GAMING", "collab-source-user", 4, topics=["STRATEGY"],
                            candidate_source="COLLABORATIVE", social_context=_social(0.9, 0.0, 0.75, mutual_follow=False),
                            note="No direct relationship at all -- pure interest-similarity match"),
        BenchmarkCandidate("social-collab-filler", "TRAVEL", "trav-social", 1, note="Unrelated filler"),
    ]
    scenarios.append(BenchmarkScenario(
        "social-similar-user-collaborative", "SOCIAL",
        "Behaviorally similar but unconnected user (no follow relationship) engaged a "
        "candidate matching the target user's own real GAMING/Strategy preference.",
        "bench-social-collab-user", history, candidates,
        [RelativeConstraint("collaborative_beats_filler", "social-collab-target", "social-collab-filler", critical=True)],
    ))

    # 3. Social-vs-negative-feedback conflict: user explicitly, repeatedly rejected this exact
    # subtheme; a mutual friend with maximal social evidence engaged the same subtheme -- the
    # social boost (bounded, multiplicative on model_score) must NOT resurrect it.
    history = [
        *[_event(f"social-neg-hist-pos-{i}", "coach-social-neg", "SPORT", watch_percentage=90, liked=True,
                  when=-(60 + i) * _DAY, topics=["FOOTBALL"]) for i in range(6)],
        *[_event(f"social-neg-hist-rej-{i}", "coach-social-neg-tennis", "SPORT", watch_percentage=4,
                  event_type="CONTENT_NOT_INTERESTED", when=-(10 + i * 2) * _DAY, topics=["TENNIS"]) for i in range(4)],
    ]
    candidates = [
        BenchmarkCandidate("social-neg-rejected", "SPORT", "friend-neg", 0, topics=["TENNIS"],
                            candidate_source="SOCIAL", social_context=_social(1.0, 1.0, 1.0, mutual_follow=True),
                            note="Explicitly, repeatedly rejected subtheme -- maximal social evidence must not override"),
        BenchmarkCandidate("social-neg-preferred", "SPORT", "coach-social-neg", 4, topics=["FOOTBALL"],
                            note="Genuinely preferred subtheme, no social boost"),
    ]
    scenarios.append(BenchmarkScenario(
        "social-vs-negative-feedback-conflict", "SOCIAL",
        "Maximal social evidence on a subtheme the user explicitly, repeatedly rejected must "
        "not outrank the user's genuinely preferred subtheme.",
        "bench-social-neg-user", history, candidates,
        [RelativeConstraint("preferred_beats_rejected_despite_social", "social-neg-preferred", "social-neg-rejected",
                             critical=True, description="Explicit rejection must survive maximal social pressure")],
    ))

    # 4. Social-vs-semantic-interest conflict: strong real preference for one GAMING subtheme;
    # social candidate is a DIFFERENT subtheme, same broad category, no direct rejection of it.
    # Semantic match should still legitimately outrank a same-category-but-off-subtheme social
    # candidate with only moderate social evidence.
    history = [_event(f"social-sem-hist-{i}", "coach-social-sem", "GAMING", watch_percentage=93, liked=True,
                       when=-(20 + i) * _DAY, topics=["BATTLE_ROYALE"]) for i in range(8)]
    candidates = [
        BenchmarkCandidate("social-sem-preferred", "GAMING", "coach-social-sem", 4, topics=["BATTLE_ROYALE"],
                            note="Real preferred subtheme, no social boost"),
        BenchmarkCandidate("social-sem-offtheme", "GAMING", "friend-sem", 2, topics=["RACING_SIM"],
                            candidate_source="SOCIAL", social_context=_social(0.4, 0.5, 0.4, mutual_follow=True),
                            note="Same broad category, different (never-preferred) subtheme, moderate social evidence"),
    ]
    scenarios.append(BenchmarkScenario(
        "social-vs-semantic-interest-conflict", "SOCIAL",
        "Moderate social evidence for an off-subtheme candidate must not outrank the user's "
        "own genuinely preferred subtheme in the same broad category.",
        "bench-social-sem-user", history, candidates,
        [RelativeConstraint("preferred_subtheme_beats_offtheme_social", "social-sem-preferred", "social-sem-offtheme",
                             critical=True)],
    ))

    # 5. Already-seen social candidate: social_relevance() is defined to be a hard zero once
    # already_seen=True (app.ml.reranker.social_relevance) -- prove that holds through the real
    # end-to-end path, not just the unit-level check.
    history = [_event(f"social-seen-hist-{i}", "coach-social-seen", "MUSIC", watch_percentage=91, liked=True,
                       when=-(15 + i) * _DAY, topics=["POP"]) for i in range(6)]
    candidates = [
        # Both candidates deliberately use a creator with ZERO prior history (neither
        # "coach-social-seen", the creator the history events themselves are attributed to) --
        # isolates the already_seen/social-boost comparison from any real, learned creator-
        # affinity edge, which would otherwise confound this scenario's one intended variable.
        BenchmarkCandidate("social-seen-target", "MUSIC", "friend-seen", 3, topics=["POP"], already_seen=True,
                            candidate_source="SOCIAL", social_context=_social(0.9, 0.9, 0.9, mutual_follow=True),
                            note="Maximal social evidence but already seen -- must not get the social boost"),
        BenchmarkCandidate("social-seen-unseen-equivalent", "MUSIC", "friend-seen-2", 3, topics=["POP"],
                            already_seen=False, note="Same relevance, unseen, no social context at all"),
    ]
    scenarios.append(BenchmarkScenario(
        "social-already-seen-suppressed", "SOCIAL",
        "An already-seen social candidate must not outrank an equally-relevant unseen "
        "candidate purely because of social evidence.",
        "bench-social-seen-user", history, candidates,
        [RelativeConstraint("unseen_beats_seen_social", "social-seen-unseen-equivalent", "social-seen-target",
                             critical=True, description="alreadySeen must dominate a social boost, not the reverse")],
    ))

    # 6. Irrelevant friend's popular content: a friend's social engagement on content in a
    # category the user has zero real interest in, propped up by high raw popularity, must not
    # beat genuinely relevant content just because it is both popular and socially sourced.
    history = [_event(f"social-irrel-hist-{i}", "coach-social-irrel", "SPORT", watch_percentage=92, liked=True,
                       when=-(20 + i) * _DAY, topics=["FOOTBALL"]) for i in range(8)]
    candidates = [
        BenchmarkCandidate("social-irrel-relevant", "SPORT", "coach-social-irrel", 4, topics=["FOOTBALL"],
                            note="Genuinely relevant, no social boost needed"),
        BenchmarkCandidate("social-irrel-popular", "ART", "friend-irrel", 1, content_popularity_score=0.95,
                            candidate_source="SOCIAL", social_context=_social(0.3, 0.6, 0.5, mutual_follow=True),
                            note="Popular + socially sourced, but a category with zero real user interest"),
    ]
    scenarios.append(BenchmarkScenario(
        "social-irrelevant-friend-popular-content", "SOCIAL",
        "A friend's popular content in a category the user has no real interest in must not "
        "outrank genuinely relevant content just from popularity + a social tag.",
        "bench-social-irrel-user", history, candidates,
        [RelativeConstraint("relevant_beats_irrelevant_popular_social", "social-irrel-relevant", "social-irrel-popular",
                             critical=True)],
    ))

    return scenarios
