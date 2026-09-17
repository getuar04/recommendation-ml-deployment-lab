"""Generate deterministic, noisy behavior data (safe to rerun).

Determinism (Step 10): `rng=random.Random(SEED)` already seeds every random draw below, but
the original version anchored all timestamps to the real `datetime.now(timezone.utc)` at
generation time -- harmless for the *relative* ordering/structure of the dataset (every
timestamp is `start + offset`, and content_age_hours/split-by-time only ever depend on
relative differences), but it does change the absolute wall-clock `hour_of_day` feature
(`ts.hour`) run to run, which is enough to make `train_models()`'s automatic
LogisticRegression-vs-RandomForest selection non-deterministic across regenerations (see
tests/test_session_switch_production.py's module docstring for the investigated failure mode
this caused). `reference_timestamp`, when given, replaces that `datetime.now()` anchor so
`generate(reference_timestamp=X)` produces a bit-for-bit identical dataset every time for the
same `X` -- mirrors the exact pattern already proven in
app.experiments.comparative_dataset_generation.generate_comparative_dataset. Defaults to None
(real wall-clock time, the original behavior) so every existing caller that does not pass it
is unaffected.
"""
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from app.db.database import Base, SessionLocal, engine
from app.db.models import Content, Interaction

CATEGORIES=["FOOD","SPORT","MUSIC","TECH","GAMING","TRAVEL","COMEDY","NEWS","FASHION","FITNESS"]
SEED=20260721
# Step 12: cold-start users get 0-3 interactions total (never more) so they can never
# approach the ~120 interactions/user a bulk user accumulates -- "do not allow cold-start
# rows to dominate training".
COLD_START_USER_COUNT=20
SESSION_SWITCH_USER_COUNT=30
# Phase 1.5 issue 2 (negative eligibility gate / StandardScaler fragility fix): see
# _extra_synthetic_rows' session-switch cohort comment for why this cohort's anchor must be
# spread across the historical horizon rather than real "now".
SESSION_SWITCH_ANCHOR_MIN_DAYS_AGO=40
SESSION_SWITCH_ANCHOR_MAX_DAYS_AGO=90
# Phase 1.5 issue 2C (fast-skip gate residual-value fix): see the stage-3 coin-flip comment
# in _extra_synthetic_rows for why this exists. Recovery (0.6) outweighs permanent pivot (0.4)
# deliberately, not just to clear a threshold: a single bad session (a handful of fast-skips)
# is product-realistically more often a temporary mood/context dip than a genuine, permanent
# category abandonment -- the ORIGINAL session-switch cohort exists specifically to teach the
# rarer, more notable "genuine pivot" case (see its own docstring, "the COMEDY->MUSIC production
# scenario shape"), which is exactly why it should stay the minority outcome, not the default.
SESSION_SWITCH_RECOVERY_PROBABILITY=0.6
# Phase 1.5 issue 1 (recent-vs-stale extrapolation fix): see _recent_shift_rows docstring.
RECENT_SHIFT_USER_COUNT=5
# Finalization spec Step 11: dedicated same-category, cross-subtopic preference cohort (see
# _semantic_preference_rows) -- kept small relative to the ~12000-row bulk population, same
# proportions as the existing cold-start/session-switch cohorts above.
SEMANTIC_PREFERENCE_USER_COUNT=40
# Finalization spec Step 11 (final model selection + creator affinity validation): dedicated
# same-category, cross-creator preference cohort (see _creator_preference_rows).
#
# Final training-coverage repair (fast-skip/creator) -- investigated, reverted: raising this
# count (40 -> 50/60/65) DID improve the controlled preferred-vs-unknown creator margin in
# isolation (e.g. count=60 alone measured margin +0.0194 vs the +0.012 baseline), consistent
# with the diagnosed root cause (creator identity is orthogonal to outcome in the bulk
# population -- see _creator_preference_rows' own docstring -- so this dedicated cohort is the
# only source of genuine creator-preference signal, and it is proportionally small). However,
# combined with the _fast_skip_streak_rows cohort added in the same repair pass, the SAME
# increase flipped the controlled creator margin NEGATIVE (LogisticRegression is one global
# linear model fit on the whole dataset, so the two cohorts' coefficient effects interact, not
# just add). Kept at the original value: creator-affinity training-data coverage remains a
# real, documented, KNOWN LIMITATION (see the repair report) for a dedicated, isolated future
# pass -- not bundled with any other synthetic-data change, so its effect can be measured
# without this exact interaction.
CREATOR_PREFERENCE_USER_COUNT=40
# Final training-coverage repair (fast-skip): teaches that a longer same-category fast-skip
# streak (rising session_intent_confidence/session_category_streak -- both valence-agnostic,
# magnitude-only features) does NOT by itself imply a positive outcome -- see
# _fast_skip_streak_rows for the full root-cause explanation and design.
FAST_SKIP_STREAK_USER_COUNT=18
FAST_SKIP_STREAK_ANCHOR_MIN_DAYS_AGO=40
FAST_SKIP_STREAK_ANCHOR_MAX_DAYS_AGO=90

# Finalization spec Step 11 (Decision 5/6): a generic, multi-category subtopic taxonomy used
# ONLY to tag synthetic training/candidate Content rows -- never referenced by any ranking or
# reranker code (app.ml.reranker has no knowledge of this module at all). Two subtopics per
# category so a user can have a real, isolated PREFERENCE WITHIN a category (e.g. loves
# FOOTBALL, dislikes BASKETBALL, both still "SPORT") -- category_affinity alone cannot explain
# that difference, only the semantic-token features can, which is exactly what forces the
# model to actually learn them instead of leaving them at zero importance.
#
# Decorrelation (Step 11 controlled iteration): an earlier version of this taxonomy tagged
# every field (hashtags/topics/entities/subgenres) with the exact same token(s) for every
# video in a subtopic -- realistic for any ONE video, but across a user's whole history it
# made hashtag_affinity/topic_affinity/entity_affinity/subgenre_affinity/average_semantic_
# affinity near-perfectly correlated (measured r=0.84-1.00, topic/entity at r=1.000), which is
# a synthetic multicollinearity artifact, not a real property of how creators actually tag
# content. Real creators are inconsistent: one video's title carries the story, another's
# hashtags carry the trend, not every video names a specific entity, and topic vs subgenre
# operate at genuinely different levels of granularity (broad concept vs specific content
# type). This taxonomy now models that: each field has its own pool of plausible values (a
# canonical topic, a small hashtag pool that overlaps with but isn't identical to it, a named-
# entity pool, a narrower subgenre pool) and `_tag_content` independently decides, per video,
# whether each field is populated at all and which subset of its pool to use -- title remains
# natural free text and is always present when a video is tagged, so title-token evidence
# never disappears entirely. The underlying CONCEPT per subtopic is unchanged (still exactly
# the same 10 categories x 2 subtopics, still the same real-world pairs) -- only how
# consistently each field surfaces that concept per individual video has changed.
SEMANTIC_TAXONOMY = {
    "SPORT": [
        {"topic": "FOOTBALL", "subgenre_pool": ["FOOTBALL_HIGHLIGHTS", "TACTICAL_ANALYSIS", "MATCH_RECAP"],
         "hashtag_pool": ["FOOTBALL", "SOCCER", "CHAMPIONSLEAGUE"], "entity_pool": ["BARCELONA", "REAL_MADRID", "MESSI"],
         "titles": ["Barcelona wins dramatic match tonight", "Amazing football skills compilation", "Football transfer news update", "Champions League tactical breakdown"]},
        {"topic": "BASKETBALL", "subgenre_pool": ["BASKETBALL_HIGHLIGHTS", "GAME_ANALYSIS", "DUNK_REEL"],
         "hashtag_pool": ["BASKETBALL", "NBA", "HOOPS"], "entity_pool": ["LAKERS", "WARRIORS", "LEBRON"],
         "titles": ["Lakers dominate in overtime thriller", "Basketball dunk compilation reel", "NBA playoff highlights tonight", "Basketball game strategy breakdown"]},
    ],
    "MUSIC": [
        {"topic": "HIPHOP", "subgenre_pool": ["HIPHOP_TRACK", "FREESTYLE_SESSION", "CYPHER"],
         "hashtag_pool": ["HIPHOP", "RAP", "FREESTYLE"], "entity_pool": ["DRAKE", "KENDRICK", "TRAVIS_SCOTT"],
         "titles": ["Drake drops surprise hiphop single", "Kendrick Lamar freestyle session", "Hiphop cypher battle highlights", "Rap album review breakdown"]},
        {"topic": "POP", "subgenre_pool": ["POP_TRACK", "CONCERT_FOOTAGE", "CHART_RECAP"],
         "hashtag_pool": ["POP", "POPMUSIC", "CHARTS"], "entity_pool": ["DUA_LIPA", "TAYLOR_SWIFT", "ARIANA_GRANDE"],
         "titles": ["Dua Lipa new pop single release", "Taylor Swift concert tour highlights", "Pop chart countdown this week", "Pop music trend analysis"]},
    ],
    "GAMING": [
        {"topic": "GTA", "subgenre_pool": ["OPEN_WORLD", "HEIST_GAMEPLAY", "STUNT_MONTAGE"],
         "hashtag_pool": ["GTA", "GTAONLINE", "ROCKSTARGAMES"], "entity_pool": ["GTA", "ROCKSTAR_GAMES", "LOS_SANTOS"],
         "titles": ["GTA online heist gameplay walkthrough", "Grand Theft Auto funny moments compilation", "GTA five epic stunts montage", "Open world game exploration highlights"]},
        {"topic": "MINECRAFT", "subgenre_pool": ["SANDBOX", "REDSTONE_BUILD", "SURVIVAL_MODE"],
         "hashtag_pool": ["MINECRAFT", "MCBUILD", "SURVIVAL"], "entity_pool": ["MINECRAFT", "MOJANG", "CREEPER"],
         "titles": ["Minecraft survival base building tutorial", "Minecraft redstone contraption showcase", "Minecraft epic castle build timelapse", "Sandbox building game highlights"]},
    ],
    "FOOD": [
        {"topic": "BAKING", "subgenre_pool": ["BAKING_TUTORIAL", "PASTRY_TECHNIQUE", "BREAD_RECIPE"],
         "hashtag_pool": ["BAKING", "SOURDOUGH", "HOMEBAKING"], "entity_pool": ["SOURDOUGH", "PASTRY", "CROISSANT"],
         "titles": ["Sourdough bread baking tutorial", "Pastry chef technique masterclass", "Homemade baking recipe walkthrough", "Bread recipe step by step guide"]},
        {"topic": "GRILLING", "subgenre_pool": ["GRILLING_TUTORIAL", "BBQ_RECIPE", "COOKOUT_GUIDE"],
         "hashtag_pool": ["GRILLING", "BBQ", "COOKOUT"], "entity_pool": ["BARBECUE", "STEAK", "SMOKER"],
         "titles": ["Backyard barbecue grilling techniques", "Steak grilling perfect temperature guide", "Grilling recipe weekend cookout", "Smoker recipe low and slow guide"]},
    ],
    "TECH": [
        {"topic": "SMARTPHONES", "subgenre_pool": ["DEVICE_REVIEW", "CAMERA_TEST", "UNBOXING"],
         "hashtag_pool": ["SMARTPHONE", "TECHREVIEW", "MOBILETECH"], "entity_pool": ["IPHONE", "ANDROID", "SAMSUNG"],
         "titles": ["iPhone camera review comparison", "Android smartphone unboxing review", "Smartphone battery life test", "Mobile tech buying guide"]},
        {"topic": "LAPTOPS", "subgenre_pool": ["DEVICE_REVIEW", "BENCHMARK_TEST", "DURABILITY_TEST"],
         "hashtag_pool": ["LAPTOP", "TECHREVIEW", "PORTABLETECH"], "entity_pool": ["MACBOOK", "THINKPAD", "DELL_XPS"],
         "titles": ["Macbook performance benchmark review", "Thinkpad laptop durability test", "Laptop buying guide comparison", "Portable computing device roundup"]},
    ],
    "TRAVEL": [
        {"topic": "BEACH", "subgenre_pool": ["TRAVEL_VLOG", "RESORT_TOUR", "PHOTOGRAPHY_GUIDE"],
         "hashtag_pool": ["BEACH", "ISLAND", "TROPICAL"], "entity_pool": ["BALI", "MALDIVES", "PHUKET"],
         "titles": ["Bali beach vacation travel guide", "Maldives resort island tour", "Beach sunset photography tips", "Tropical island travel diary"]},
        {"topic": "MOUNTAIN", "subgenre_pool": ["TRAVEL_VLOG", "TREKKING_GUIDE", "GEAR_REVIEW"],
         "hashtag_pool": ["MOUNTAIN", "HIKING", "TREKKING"], "entity_pool": ["ALPS", "HIMALAYAS", "ANDES"],
         "titles": ["Alps mountain hiking adventure", "Himalayas trekking expedition vlog", "Mountain climbing gear review", "Trekking route planning guide"]},
    ],
    "COMEDY": [
        {"topic": "STANDUP", "subgenre_pool": ["STANDUP_SPECIAL", "CROWD_WORK", "SET_BREAKDOWN"],
         "hashtag_pool": ["STANDUP", "COMEDY", "COMEDYSPECIAL"], "entity_pool": ["OPEN_MIC", "COMEDY_CLUB", "COMEDY_TOUR"],
         "titles": ["Standup comedy special highlights", "Open mic comedy night compilation", "Comedy club best moments reel", "Crowd work comedy compilation"]},
        {"topic": "PRANKS", "subgenre_pool": ["PRANK_COMPILATION", "REACTION_VIDEO", "HIDDEN_CAM"],
         "hashtag_pool": ["PRANK", "FUNNY", "REACTIONS"], "entity_pool": ["PRANK_WARS", "HIDDEN_CAMERA", "CANDID_CAMERA"],
         "titles": ["Prank wars epic compilation", "Hidden camera prank reactions", "Funny prank fails compilation", "Reaction video prank highlights"]},
    ],
    "NEWS": [
        {"topic": "POLITICS", "subgenre_pool": ["NEWS_ANALYSIS", "DEBATE_RECAP", "ELECTION_COVERAGE"],
         "hashtag_pool": ["POLITICS", "ELECTION", "POLICY"], "entity_pool": ["ELECTION", "CONGRESS", "SENATE"],
         "titles": ["Election night coverage analysis", "Congress hearing highlights today", "Political debate recap tonight", "Policy analysis roundtable discussion"]},
        {"topic": "TECH_NEWS", "subgenre_pool": ["NEWS_BRIEFING", "INDUSTRY_ANALYSIS", "FUNDING_ROUNDUP"],
         "hashtag_pool": ["TECHNEWS", "STARTUP", "INNOVATION"], "entity_pool": ["STARTUP", "SILICON_VALLEY", "VENTURE_CAPITAL"],
         "titles": ["Startup funding news roundup", "Silicon valley tech industry update", "Tech news weekly briefing", "Innovation industry analysis report"]},
    ],
    "FASHION": [
        {"topic": "STREETWEAR", "subgenre_pool": ["FIT_CHECK", "SNEAKER_REVIEW", "DROP_ALERT"],
         "hashtag_pool": ["STREETWEAR", "SNEAKERS", "HYPE"], "entity_pool": ["SNEAKERS", "HYPEBEAST", "SUPREME"],
         "titles": ["Sneaker collection unboxing haul", "Hypebeast streetwear fit check", "Sneaker release drop review", "Streetwear trend forecast video"]},
        {"topic": "LUXURY", "subgenre_pool": ["LUXURY_HAUL", "RUNWAY_RECAP", "BRAND_SPOTLIGHT"],
         "hashtag_pool": ["LUXURY", "FASHION", "DESIGNER"], "entity_pool": ["GUCCI", "LOUIS_VUITTON", "CHANEL"],
         "titles": ["Gucci fashion show recap", "Louis vuitton unboxing haul", "Luxury fashion trend forecast", "Designer brand spotlight review"]},
    ],
    "FITNESS": [
        {"topic": "WEIGHTLIFTING", "subgenre_pool": ["STRENGTH_TRAINING", "FORM_TUTORIAL", "PR_ATTEMPT"],
         "hashtag_pool": ["WEIGHTLIFTING", "GYM", "STRENGTHTRAINING"], "entity_pool": ["DEADLIFT", "POWERLIFTING", "SQUAT"],
         "titles": ["Deadlift form technique tutorial", "Powerlifting competition highlights", "Weightlifting personal record attempt", "Strength training program guide"]},
        {"topic": "YOGA", "subgenre_pool": ["MIND_BODY", "FLOW_TUTORIAL", "BEGINNER_GUIDE"],
         "hashtag_pool": ["YOGA", "MEDITATION", "MINDFULNESS"], "entity_pool": ["VINYASA", "MEDITATION", "ASHTANGA"],
         "titles": ["Vinyasa yoga flow tutorial", "Meditation and yoga session guide", "Yoga for beginners full routine", "Mindfulness practice session guide"]},
    ],
}
# Fraction of generated Content rows that get semantic-tagged at all (Decision 6: "realistic",
# not every single row). Per-field inclusion probabilities below are independent of each other
# and of this rate -- title is always present once a video is tagged (guaranteed minimum
# signal); the other four fields are each included by an independent coin flip so a video's
# exact combination of populated fields varies realistically instead of mirroring in lockstep.
SEMANTIC_TAG_RATE = 0.8
_FIELD_INCLUSION_RATE = {"topics": 0.80, "hashtags": 0.85, "entities": 0.55, "subgenres": 0.40}


def _tag_content(rng: random.Random, category: str) -> dict | None:
    subtopics = SEMANTIC_TAXONOMY.get(category)
    if not subtopics or rng.random() >= SEMANTIC_TAG_RATE:
        return None
    subtopic = rng.choice(subtopics)
    fields = {"title": rng.choice(subtopic["titles"])}

    def _sample_pool(pool: Sequence[str]) -> list[str]:
        k = min(len(pool), rng.choice([1, 1, 2]))
        return rng.sample(pool, k)

    fields["topics_json"] = json.dumps([subtopic["topic"]] if rng.random() < _FIELD_INCLUSION_RATE["topics"] else [])
    fields["hashtags_json"] = json.dumps(_sample_pool(subtopic["hashtag_pool"]) if rng.random() < _FIELD_INCLUSION_RATE["hashtags"] else [])
    fields["entities_json"] = json.dumps(_sample_pool(subtopic["entity_pool"]) if rng.random() < _FIELD_INCLUSION_RATE["entities"] else [])
    fields["subgenres_json"] = json.dumps(_sample_pool(subtopic["subgenre_pool"])[:1] if rng.random() < _FIELD_INCLUSION_RATE["subgenres"] else [])
    return fields


# Step 11 (final model selection + creator affinity validation): timestamp-window fix. The
# dedicated semantic-/creator-preference cohorts originally used `now - timedelta(days=rng.
# randint(5, 60))` -- confirmed via direct inspection of app.ml.split_lifecycle's chronological
# 5-way split boundaries to place 0% of BOTH cohorts' rows in the `train` split (train covered
# the dataset's EARLIEST ~50%, e.g. day 0-58 of a 120-day range; the cohorts' 5-60-day-ago
# window fell entirely in the LATEST ~50%, e.g. day 60-115) -- so the base model literally never
# fit on a single row from either cohort; all the real signal only reached modelSelection/
# calibration/thresholdTuning/test, none of which retrain the underlying trees/coefficients.
# `_cohort_timestamp` widens the window to the full historical horizon (5-115 days ago, mirroring
# the bulk population's own `start = now - 120 days` span) while staying point-in-time safe: the
# random offset is bounded below by the SPECIFIC content item's own `created_at` (a video can
# never be watched before it exists), not just by the cohort's nominal day range.
COHORT_TIMESTAMP_MIN_DAYS_AGO = 5
COHORT_TIMESTAMP_MAX_DAYS_AGO = 115


def _cohort_timestamp(rng: random.Random, content: Content, now: datetime) -> datetime:
    latest_possible = now - timedelta(days=COHORT_TIMESTAMP_MIN_DAYS_AGO)
    earliest_possible = max(content.created_at, now - timedelta(days=COHORT_TIMESTAMP_MAX_DAYS_AGO))
    if earliest_possible >= latest_possible:
        return latest_possible
    span_seconds = (latest_possible - earliest_possible).total_seconds()
    return earliest_possible + timedelta(seconds=rng.uniform(0, span_seconds))


def _extra_synthetic_rows(rng: random.Random, contents: list[Content], now: datetime, seed: int = SEED) -> list[Interaction]:
    """Step 11/12: realistic session-switch and cold-start cohorts, layered on top of the
    bulk stable-interest population `generate()` already produces. Deliberately noisy (gaussian
    jitter, randomized like/share rates, randomized event counts) rather than hand-tuned clean
    signals -- see module docstrings on app.ml.dataset_builder and
    tests/test_session_switch_production.py for why this matters: a bulk population where every
    user has *some* affinity (0.20-0.82) for *every* category never teaches the model what
    genuine "no history yet" looks like, and a population with no recent-vs-long-term conflict
    never teaches it to trust a strong recent/session signal over a middling long-term one.
    Kept small relative to the ~12000-row bulk population (see COLD_START_USER_COUNT/
    SESSION_SWITCH_USER_COUNT) so neither cohort dominates training."""
    by_category: dict[str, list[Content]] = defaultdict(list)
    for c in contents:
        by_category[c.category].append(c)
    rows: list[Interaction] = []
    # Phase 1.5 issue 2: independent RNG for the session-switch anchor draw only (added below)
    # -- every other draw in this function stays on the shared `rng` stream, at the exact same
    # position/count as before, so _semantic_preference_rows/_creator_preference_rows (called
    # after this function with the same shared `rng`) are not perturbed by this fix, exactly
    # the same isolation principle already applied to _recent_shift_rows.
    anchor_rng = random.Random(seed + 2)

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    # --- Cold start (Step 12): no/near-no history, first-session-only. Roughly a third get
    # zero interactions at all (a genuinely new user); the rest get 1-3, all clustered in the
    # last couple of hours (their one and only session so far), noisy/ambiguous outcomes. ---
    for u in range(1, COLD_START_USER_COUNT + 1):
        user = f"coldstart-user-{u}"
        num_events = rng.choice([0, 0, 1, 1, 2, 3])
        if num_events == 0:
            continue
        category = rng.choice(CATEGORIES)
        pool = [c for c in by_category[category] if c.created_at <= now]
        if not pool:
            continue
        for i in range(num_events):
            c = rng.choice(pool)
            ratio = rng.gauss(0.45, 0.3)  # no established preference -- wide, noisy spread
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-cs-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.08,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=now - timedelta(minutes=rng.randint(1, 90)),
            ))

    # --- Session switch (Step 11): stable long-term interest in one category, a recent
    # same-category reinforcement burst, a fast-skip rejection burst of that same category,
    # then genuine discovery of a *different* category -- the COMEDY->MUSIC production
    # scenario shape, generalized across randomized category pairs and randomized noisy
    # outcomes (not one hand-crafted clean example) so the model sees many instances of
    # "trust the recent signal over the middling long-term one".
    #
    # Phase 1.5 issue 2 (found while root-causing the `negative` eligibility gate's
    # near-zero-variance StandardScaler fragility, not guessed): this cohort is the ONLY
    # training data with real session_negative_interaction_count/session_category_affinity
    # examples reflecting a genuine within-session rejection, but every stage was anchored to
    # `now - <minutes>` -- always the very end of the 120-day dataset, structurally outside
    # `train` (app.ml.split_lifecycle's earliest ~40-56% chronological slice; confirmed
    # directly: session_negative_interaction_count was nonzero in only 3 of 3137 `train` rows).
    # `train` is the only split that actually refits the candidate models' coefficients/trees,
    # so this cohort's real session-negative examples never reached it, leaving that feature
    # near-degenerate for StandardScaler to fit a std from -- which is what turns the `negative`
    # eligibility gate's own (realistic) session-negative probe value into an extreme,
    # logit-saturating outlier. Same fix as `_recent_shift_rows`: anchor the whole per-user
    # narrative to a point spread across the full historical horizon instead of real `now`,
    # keeping the same minute-scale spacing BETWEEN the three session stages (so they stay
    # within each other's SESSION_WINDOW, which is what makes them session features at all) --
    # this makes real session-negative examples in-distribution for `train`, addressing the
    # root cause (data sparsity) rather than patching StandardScaler or the gate. ---
    for u in range(1, SESSION_SWITCH_USER_COUNT + 1):
        user = f"session-switch-user-{u}"
        old_category, new_category = rng.sample(CATEGORIES, 2)
        anchor_days_ago = anchor_rng.uniform(SESSION_SWITCH_ANCHOR_MIN_DAYS_AGO, SESSION_SWITCH_ANCHOR_MAX_DAYS_AGO)
        anchor = now - timedelta(days=anchor_days_ago)
        old_pool = [c for c in by_category[old_category] if c.created_at <= anchor - timedelta(days=30)]
        new_pool = [c for c in by_category[new_category] if c.created_at <= anchor]
        if not old_pool or not new_pool:
            continue
        for i in range(rng.randint(4, 7)):  # long-term: moderate-to-strong, noisy, 40-90 days before the anchor.
            c = rng.choice(old_pool)
            ratio = rng.gauss(0.75, 0.2)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-ss-lt-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.4,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(days=rng.randint(40, 90)),
            ))
        for i in range(rng.randint(1, 2)):  # session stage 1: same-category reinforcement, ~20-30 min before the anchor.
            c = rng.choice(old_pool)
            ratio = rng.gauss(0.93, 0.05)
            duration, watch, watch_pct = _watched(ratio, min_duration=60)
            rows.append(Interaction(
                event_id=f"syn-ss-burst-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type="VIDEO_COMPLETED", watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.5,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(minutes=rng.randint(20, 30)),
            ))
        for i in range(rng.randint(2, 3)):  # session stage 2: fast-skip rejection, ~8-18 min before the anchor.
            c = rng.choice(old_pool)
            ratio = rng.gauss(0.08, 0.05)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-ss-skip-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type="VIDEO_SKIPPED", watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=False,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(minutes=rng.randint(8, 18)),
            ))
        # Phase 1.5 issue 2C: stage 3 is a coin flip between the existing "pivot to a new
        # category" narrative and a "recover in the OLD category" one, instead of always
        # pivoting. Root cause this addresses (measured directly, not guessed): of the `train`
        # rows resembling "strong long-term category_affinity + a negative session dip in that
        # same category" (app.ml.eligibility's `negative` gate probe shape), 6 of 6 were
        # labeled target=0 -- every existing cohort teaches "a fast-skip burst predicts a
        # negative outcome" but NONE teaches "residual long-term value can still produce a
        # genuinely positive outcome after a dip in the SAME category" -- a real, common user
        # behavior (a bad session doesn't always mean permanent abandonment) that was simply
        # absent from the data, not something the model was ever positioned to learn. The coin
        # flip is decided on the independent `anchor_rng` (not the shared `rng` stream) so it
        # doesn't perturb draw counts/positions for _semantic_preference_rows/
        # _creator_preference_rows, matching every other isolation fix in this module; the
        # branch itself still draws the same number of values from the shared `rng` either way.
        recovers = anchor_rng.random() < SESSION_SWITCH_RECOVERY_PROBABILITY
        stage3_pool = old_pool if recovers else new_pool
        stage3_prefix = "syn-ss-recover" if recovers else "syn-ss-new"
        for i in range(rng.randint(2, 4)):  # session stage 3: recovery (old category) or discovery (new category), 1-7 min before the anchor, noisy.
            c = rng.choice(stage3_pool)
            ratio = rng.gauss(0.85, 0.15)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"{stage3_prefix}-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.6,
                shared=rng.random() < 0.3, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(minutes=rng.randint(1, 7)),
            ))
    return rows


# Final training-coverage repair: dedicated fast-skip-streak cohort (production-readiness
# repair report). Root cause (proven by feature-ablation on the already-trained model, not
# guessed): scoring the SAME controlled candidate after 0/1/2/3/4 accumulated same-category
# fast-skips produced raw probabilities 0.930 -> 0.703 -> 0.654 -> 0.657 -> 0.709 -- a
# non-monotonic uptick from the 2nd skip onward. Zeroing session_intent_confidence alone
# (keeping every other feature real) nearly restored a clean monotonic decline; zeroing it
# plus session_category_streak restored it exactly. Both features are valence-agnostic (they
# measure evidence MAGNITUDE/certainty, not direction -- see
# app.ml.dataset_builder.FeatureHistory._session_snapshot), so a longer streak raises them
# regardless of whether the streak is positive or negative. The existing session-switch cohort
# (_extra_synthetic_rows) only ever produces a fixed 2-3-event skip burst -- so training data
# never taught the model "session_intent_confidence/session_category_streak can be HIGH while
# the very next same-category outcome is STILL negative".
#
# FINAL, ACCEPTED design (depths (1, 2) only, no pivot branch) -- narrower than first
# attempted, by direct experimental necessity, not preference: an earlier version covered
# depths 1-4 (matching the controlled probe's own anomaly depths more directly), with a
# deterministic negative continuation and a same-category-recovery-or-different-category-
# pivot coin flip. That version reliably broke `app.ml.eligibility._gate_negative` outright
# (NoEligibleModel) -- depth-3/4 same-category fast-skip streaks with strong long-term history
# is EXACTLY that gate's own probe shape, so training on more of them shifts the model's
# response to that exact input pattern, structurally in tension with the gate's own
# requirement (suppression must be real but must not erase residual long-term value below a
# cold floor). Depths (1, 2), a probabilistic (not deterministic) continuation, and dropping
# the pivot branch (always recover in the SAME category, preserving residual value) left every
# eligibility gate intact while still measurably reducing the skip-2/3/4 reversal magnitude --
# LogisticRegression is one global linear model, so shifting its learned coefficient using
# depth-1/2 examples generalizes to the unseen depth-3/4 inputs the controlled probe checks,
# without training directly on the exact shape the mandatory gate also probes.
# Depths kept to (1, 2) only, not (1, 2, 3, 4): a data-only experiment adding depth-3/4
# streaks (closely matching app.ml.eligibility._gate_negative's own 4-fast-skip probe shape)
# reliably broke that mandatory gate outright (NoEligibleModel) even at a small cohort size
# and a low, depth-gated continuation probability -- confirmed by direct isolation testing,
# not guessed. Depths 1-2 alone left every eligibility gate intact (including `negative`) and
# still measurably improved the controlled skip-2/3/4 raw-probability trace (see the repair
# report) -- LogisticRegression is a single global linear model, so shifting its learned
# coefficient for session_intent_confidence/session_category_streak away from "more evidence
# always leans positive" using depth-1/2 examples generalizes to unseen depth-3/4 inputs at
# inference time, without needing (and without safely being able to afford) direct depth-3/4
# training coverage that collides with the gate's own probe shape.
FAST_SKIP_STREAK_DEPTHS = (1, 2)

# Training-coverage repair (this task): dedicated CONTENT_NOT_INTERESTED-burst cohort.
# Root cause (diagnosed directly against this project's real trained candidates, not
# guessed): `_fast_skip_streak_rows` above already teaches a minute-scale, same-category
# recent-rejection burst -- but ONLY using VIDEO_SKIPPED. app.ml.eligibility's dedicated
# `notInterested` gate (see its own module) probes the SAME temporal shape using
# CONTENT_NOT_INTERESTED specifically -- a materially different, stronger-weighted event
# (app.ml.dataset_builder.CONTENT_NOT_INTERESTED_CATEGORY_PENALTY=8 vs. a fast-skip's 4) that
# no existing cohort places at minute-scale recency against strong long-term history. This
# left every candidate algorithm generalizing to that exact gate probe shape from fast-skip
# examples alone, which measurably does not transfer as well as direct coverage would.
#
# Mirrors `_fast_skip_streak_rows` exactly (same depths, same probabilistic -- never
# deterministic -- continuation, same same-category recovery, same isolated RNG-stream
# convention): depths kept to (1, 2) only, NOT the gate's own probe depth (4), for the exact
# reason documented above `FAST_SKIP_STREAK_DEPTHS` -- training directly on a mandatory gate's
# own exact probe depth previously broke that gate outright (NoEligibleModel), confirmed by
# isolation testing at the time. The same risk applies here without the same restraint.
NOT_INTERESTED_STREAK_USER_COUNT = 18
NOT_INTERESTED_STREAK_ANCHOR_MIN_DAYS_AGO = 40
NOT_INTERESTED_STREAK_ANCHOR_MAX_DAYS_AGO = 90
NOT_INTERESTED_STREAK_DEPTHS = (1, 2)


def _not_interested_streak_rows(rng: random.Random, contents: list[Content], now: datetime) -> list[Interaction]:
    by_category: dict[str, list[Content]] = defaultdict(list)
    for c in contents:
        by_category[c.category].append(c)
    rows: list[Interaction] = []

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    for u in range(1, NOT_INTERESTED_STREAK_USER_COUNT + 1):
        user = f"notinterested-streak-user-{u}"
        depth = NOT_INTERESTED_STREAK_DEPTHS[(u - 1) % len(NOT_INTERESTED_STREAK_DEPTHS)]
        category = rng.choice(CATEGORIES)
        anchor_days_ago = rng.uniform(NOT_INTERESTED_STREAK_ANCHOR_MIN_DAYS_AGO, NOT_INTERESTED_STREAK_ANCHOR_MAX_DAYS_AGO)
        anchor = now - timedelta(days=anchor_days_ago)
        pool = [c for c in by_category[category] if c.created_at <= anchor - timedelta(days=30)]
        if len(pool) < depth + 3:
            continue

        # Long-term positive baseline: a genuine established interest, not a cold-start user.
        for i in range(rng.randint(6, 9)):
            c = rng.choice(pool)
            ratio = rng.gauss(0.85, 0.1)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-nis-lt-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.5,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(days=rng.randint(40, 90)),
            ))

        # The CONTENT_NOT_INTERESTED burst: exactly `depth` events, minute-scale spacing
        # (inside SESSION_WINDOW), each an explicit rejection -- not an implicit fast-skip.
        streak_start_offset = 20
        for i in range(depth):
            c = rng.choice(pool)
            ratio = rng.gauss(0.05, 0.03)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-nis-reject-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type="CONTENT_NOT_INTERESTED", watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=False,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(minutes=streak_start_offset - i * 3),
            ))

        # Probabilistic (never deterministic) same-category recovery -- same rationale as
        # `_fast_skip_streak_rows`: residual long-term value must survive, matching what the
        # `negative`/`notInterested`/`longTerm` eligibility gates themselves require.
        recovery_steps = [(0.90, False, False), (0.94, True, False), (0.97, True, True)]  # watch -> LIKE -> SHARE
        continuation_offset = streak_start_offset - depth * 3
        for i, (ratio, liked, shared) in enumerate(recovery_steps):
            c = rng.choice(pool)
            duration, watch, watch_pct = _watched(ratio, min_duration=60)
            rows.append(Interaction(
                event_id=f"syn-nis-recover-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=liked, shared=shared,
                favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(minutes=max(0.5, continuation_offset - (i + 1) * 2)),
            ))
    return rows


def _fast_skip_streak_rows(rng: random.Random, contents: list[Content], now: datetime) -> list[Interaction]:
    by_category: dict[str, list[Content]] = defaultdict(list)
    for c in contents:
        by_category[c.category].append(c)
    rows: list[Interaction] = []

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    for u in range(1, FAST_SKIP_STREAK_USER_COUNT + 1):
        user = f"fastskip-streak-user-{u}"
        # Stratified, not random, so both depths in FAST_SKIP_STREAK_DEPTHS are guaranteed real
        # coverage regardless of how the rest of `rng` happens to draw.
        depth = FAST_SKIP_STREAK_DEPTHS[(u - 1) % len(FAST_SKIP_STREAK_DEPTHS)]
        category = rng.choice(CATEGORIES)
        anchor_days_ago = rng.uniform(FAST_SKIP_STREAK_ANCHOR_MIN_DAYS_AGO, FAST_SKIP_STREAK_ANCHOR_MAX_DAYS_AGO)
        anchor = now - timedelta(days=anchor_days_ago)
        pool = [c for c in by_category[category] if c.created_at <= anchor - timedelta(days=30)]
        if len(pool) < depth + 3:
            continue

        # Long-term positive baseline: a genuine established interest, not a cold-start user --
        # the streak below is a real session dip against real prior positive history.
        for i in range(rng.randint(6, 9)):
            c = rng.choice(pool)
            ratio = rng.gauss(0.85, 0.1)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-fss-lt-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.5,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(days=rng.randint(40, 90)),
            ))

        # The fast-skip streak itself: exactly `depth` events, minute-scale spacing (inside
        # SESSION_WINDOW), each independently negative-labeled (watch% < 20, VIDEO_SKIPPED).
        streak_start_offset = 20
        for i in range(depth):
            c = rng.choice(pool)
            ratio = rng.gauss(0.05, 0.03)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-fss-skip-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type="VIDEO_SKIPPED", watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=False,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(minutes=streak_start_offset - i * 3),
            ))

        # The key example: roughly half the time, the row immediately following the streak
        # (SAME category) is ALSO still negative -- teaching "high accumulated session
        # evidence at THIS depth != positive" without making it a deterministic rule (a
        # deterministic 100%-negative continuation, tried first, over-corrected: it fully
        # erased the residual long-term value the `negative`/`notInterested`/`longTerm`
        # eligibility gates require to survive a fast-skip burst, breaking them outright).
        # Kept probabilistic and noisy, like every other cohort in this module.
        continuation_offset = streak_start_offset - depth * 3
        if depth >= 2 and rng.random() < 0.35:
            c = rng.choice(pool)
            ratio = rng.gauss(0.06, 0.03)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-fss-continue-{user}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type="VIDEO_SKIPPED", watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=False,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(minutes=continuation_offset),
            ))

        # Recovery, same category (unlike the session-switch cohort above, which already
        # covers the category-pivot narrative -- this cohort's own job is narrower: prove a
        # longer streak still recovers rather than permanently suppressing the category, so
        # residual long-term value survives, matching what the `negative` eligibility gate
        # itself requires).
        recovery_pool = pool
        if not recovery_pool:
            continue
        recovery_steps = [(0.90, False, False), (0.94, True, False), (0.97, True, True)]  # watch -> LIKE -> SHARE
        for i, (ratio, liked, shared) in enumerate(recovery_steps):
            c = rng.choice(recovery_pool)
            duration, watch, watch_pct = _watched(ratio, min_duration=60)
            rows.append(Interaction(
                event_id=f"syn-fss-recover-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=liked, shared=shared,
                favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(minutes=max(0.5, continuation_offset - (i + 1) * 2)),
            ))
    return rows


# Phase 1.5 issue 1 (second root cause, found after the first fix didn't survive a correct
# content_by_id re-check): the shift moment itself must be spread across the full historical
# horizon, not anchored to real "now" -- see _recent_shift_rows docstring "Second root cause".
RECENT_SHIFT_ANCHOR_MIN_DAYS_AGO = 40
RECENT_SHIFT_ANCHOR_MAX_DAYS_AGO = 90
RECENT_SHIFT_LONG_TERM_EXTRA_MIN_DAYS = 35
RECENT_SHIFT_LONG_TERM_EXTRA_MAX_DAYS = 50


def _recent_shift_rows(rng: random.Random, contents: list[Content], now: datetime) -> list[Interaction]:
    """Phase 1.5 issue 1: recent (hours-scale), session-DECOUPLED shift cohort.

    First root cause this addresses (diagnosed via tests/test_recent_session_behavior.py's
    failing scenario, not guessed): `_extra_synthetic_rows`'s session-switch cohort is the only
    training data teaching "trust a recent signal over a stale one", but every one of its shift
    events (reinforcement/rejection/discovery) is timestamped 1-30 minutes ago -- always inside
    SESSION_WINDOW (30 min, app.ml.dataset_builder.SESSION_WINDOW). That means
    recent_category_affinity and session_category_affinity always move together in training
    data: the model never sees a case where recent_category_affinity swings hard (RECENT_WINDOW
    = 30 days) while session_category_affinity shows no activity at all. This cohort instead
    times the rejection-of-old/discovery-of-new bursts in the 1-6 HOUR range -- well outside
    SESSION_WINDOW, still well inside RECENT_WINDOW -- and lets the old category's long-term
    history be strong (not moderate) with a genuine negative recent flip (including a
    CONTENT_NOT_INTERESTED event), so the model sees the harder "both categories invert sign
    between long-term and recent" pattern, not just a one-sided shift.

    Second root cause (found empirically: after the first fix, a direct fit-and-score check
    against the real content_by_id -- not an empty one -- showed the ranking was STILL wrong
    for both candidate algorithms): `_extra_synthetic_rows`'s session-switch cohort -- and my
    first version of this cohort -- both anchor their shift burst to `now - <small offset>`.
    `now` is the very end of the dataset's 120-day historical range, so the shift burst always
    lands in the LATEST slice of the timeline -- structurally outside `train`
    (app.ml.split_lifecycle's earliest ~40-56% chronological slice, confirmed by direct
    inspection: `train`'s measured boundary is ~day 56 of a 120-day range). `train` is the only
    split whose rows actually refit the candidate models' coefficients/trees (modelSelection/
    calibration/thresholdTuning/test only select/calibrate/threshold an already-fit model) -- so
    a shift burst anchored to `now` can NEVER be learned by the base model, no matter how much
    of it exists. Mirrors the fix already proven for `_semantic_preference_rows`/
    `_creator_preference_rows` (see their "timestamp-window fix" docstrings): the whole
    long-term-then-shift narrative is anchored to a per-user point spread across the full
    historical horizon (RECENT_SHIFT_ANCHOR_MIN/MAX_DAYS_AGO), not to real `now`, so a
    meaningful fraction of anchors fall inside `train`'s boundary. The long-term events stay
    genuinely outside RECENT_WINDOW *relative to the anchor* (RECENT_SHIFT_LONG_TERM_EXTRA_MIN/
    MAX_DAYS_AGO, both > RECENT_WINDOW's 30 days), which is what point-in-time correctness
    actually requires -- not proximity to real wall-clock time."""
    by_category: dict[str, list[Content]] = defaultdict(list)
    for c in contents:
        by_category[c.category].append(c)
    rows: list[Interaction] = []

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    for u in range(1, RECENT_SHIFT_USER_COUNT + 1):
        user = f"recent-shift-user-{u}"
        old_category, new_category = rng.sample(CATEGORIES, 2)
        anchor_days_ago = rng.uniform(RECENT_SHIFT_ANCHOR_MIN_DAYS_AGO, RECENT_SHIFT_ANCHOR_MAX_DAYS_AGO)
        anchor = now - timedelta(days=anchor_days_ago)
        long_term_extra_days = rng.uniform(RECENT_SHIFT_LONG_TERM_EXTRA_MIN_DAYS, RECENT_SHIFT_LONG_TERM_EXTRA_MAX_DAYS)
        long_term_base = anchor - timedelta(days=long_term_extra_days)
        old_pool = [c for c in by_category[old_category] if c.created_at <= long_term_base]
        new_pool = [c for c in by_category[new_category] if c.created_at <= anchor]
        if not old_pool or not new_pool:
            continue
        for i in range(rng.randint(5, 8)):  # long-term: strong, noisy, well outside RECENT_WINDOW of the anchor.
            c = rng.choice(old_pool)
            ratio = rng.gauss(0.90, 0.08)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-rs-lt-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.6,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=long_term_base - timedelta(days=rng.uniform(0, 10)),
            ))
        for i in range(rng.randint(2, 4)):  # recent rejection of OLD category: 1-6h before the anchor.
            c = rng.choice(old_pool)
            ratio = rng.gauss(0.08, 0.05)
            duration, watch, watch_pct = _watched(ratio)
            event_type = "CONTENT_NOT_INTERESTED" if rng.random() < 0.3 else _event_type(ratio)
            rows.append(Interaction(
                event_id=f"syn-rs-reject-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=event_type, watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=False,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(hours=rng.uniform(1, 6)),
            ))
        for i in range(rng.randint(3, 6)):  # recent embrace of NEW category: 1-6h before the anchor.
            c = rng.choice(new_pool)
            ratio = rng.gauss(0.90, 0.08)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-rs-embrace-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.6,
                shared=rng.random() < 0.3, favorited=rng.random() < 0.15, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(hours=rng.uniform(1, 6)),
            ))
    return rows


def _semantic_preference_rows(rng: random.Random, contents: list[Content], now: datetime) -> list[Interaction]:
    """Finalization spec Step 11: a dedicated cohort proving genuine WITHIN-CATEGORY subtopic
    preference, isolated from category_affinity -- each user gets real positive history with
    one subtopic of a category (e.g. FOOTBALL) and real negative history with a *different*
    subtopic of the SAME category (e.g. BASKETBALL, both "SPORT"). Since both subtopics
    contribute to the same category's aggregate, category_affinity alone ends up roughly mixed
    for these users -- only the semantic-token features (hashtag/topic/entity/subgenre/title
    affinity) can explain the large outcome difference between the two subtopics, which is
    what forces the model to actually learn them instead of leaving them at zero importance.

    Generic and randomized (Decision 6: "no hardcoded content-specific outcome logic"): which
    category and which of its two subtopics is "preferred" vs "disliked" is chosen by `rng`
    per user, using the SAME weighting/outcome formula for every subtopic pair -- nothing here
    is specific to football/basketball, hiphop/pop, or gta/minecraft; those pairs simply exist
    because SEMANTIC_TAXONOMY defines them, exactly like every other category pair does."""
    by_subtopic: dict[tuple[str, str], list[Content]] = defaultdict(list)
    for c in contents:
        if c.created_at <= now and c.topics:
            by_subtopic[(c.category, c.topics[0])].append(c)

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    categories_with_pairs = [cat for cat, subtopics in SEMANTIC_TAXONOMY.items() if len(subtopics) >= 2]
    rows: list[Interaction] = []
    for u in range(1, SEMANTIC_PREFERENCE_USER_COUNT + 1):
        user = f"semantic-pref-user-{u}"
        category = rng.choice(categories_with_pairs)
        preferred, disliked = rng.sample(SEMANTIC_TAXONOMY[category], 2)  # random order -> random direction per user
        preferred_pool = by_subtopic.get((category, str(preferred["topic"])), [])
        disliked_pool = by_subtopic.get((category, str(disliked["topic"])), [])
        if not preferred_pool or not disliked_pool:
            continue
        for i in range(rng.randint(10, 14)):  # positive: strong, noisy, spread across the full historical horizon.
            c = rng.choice(preferred_pool)
            ratio = rng.gauss(0.88, 0.12)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-sem-pos-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct,
                liked=rng.random() < 0.6, shared=rng.random() < 0.15, favorited=rng.random() < 0.1,
                commented=False, creator_followed=False,
                timestamp=_cohort_timestamp(rng, c, now),
            ))
        for i in range(rng.randint(6, 10)):  # negative: strong, noisy, same window.
            c = rng.choice(disliked_pool)
            ratio = rng.gauss(0.10, 0.08)
            duration, watch, watch_pct = _watched(ratio)
            event_type = "CONTENT_NOT_INTERESTED" if rng.random() < 0.4 else _event_type(ratio)
            rows.append(Interaction(
                event_id=f"syn-sem-neg-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=event_type, watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct,
                liked=False, shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=_cohort_timestamp(rng, c, now),
            ))
    return rows


def _creator_preference_rows(rng: random.Random, contents: list[Content], now: datetime) -> list[Interaction]:
    """Step 11 (final model selection + creator affinity validation): a dedicated cohort
    proving genuine WITHIN-CATEGORY creator preference, isolated from category_affinity --
    mirrors `_semantic_preference_rows` exactly, keyed by creator instead of subtopic.

    Root cause this addresses (diagnosed before writing this function, not guessed): the
    content catalog has only 3 creators per category (300 content rows / 10 categories / 30
    creators), so `has_creator_history` is already true for ~75% of bulk-population rows --
    but creator identity is otherwise ORTHOGONAL to interaction outcome in the bulk loop
    (`generate()` below only conditions the outcome ratio on category-level `strength`, never
    on which of a category's 3 creators the content belongs to), so creator_interaction_count/
    creator_completion_rate end up correlating with target at roughly zero (measured -0.007 to
    +0.13 -- real but weak, and two of the four creator features were measured slightly
    NEGATIVE). This cohort gives a subset of users a REAL, deliberate creator-level preference
    within one category (Creator A: strong positive history; Creator B: little/negative
    history; same category, similar candidate quality/popularity/freshness -- only creator
    identity differs) so the model has genuine signal to learn from, exactly like
    `_semantic_preference_rows` did for semantic tokens. Generic and randomized: which category
    and which of its 3 creators is "preferred" vs "weak" is chosen by `rng` per user -- no
    creator ID is ever hardcoded, and reversing the draw (Decision 6's mandatory A>B / B>A
    acceptance) happens naturally across the cohort's population, not as a special case."""
    by_category_creator: dict[tuple[str, str], list[Content]] = defaultdict(list)
    for c in contents:
        if c.created_at <= now:
            by_category_creator[(c.category, c.creator_id)].append(c)

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    creators_by_category: dict[str, list[str]] = defaultdict(list)
    for category, creator_id in by_category_creator:
        creators_by_category[category].append(creator_id)
    categories_with_pairs = [cat for cat, creators in creators_by_category.items() if len(creators) >= 2]

    rows: list[Interaction] = []
    for u in range(1, CREATOR_PREFERENCE_USER_COUNT + 1):
        user = f"creator-pref-user-{u}"
        category = rng.choice(categories_with_pairs)
        preferred_creator, weak_creator = rng.sample(creators_by_category[category], 2)
        preferred_pool = by_category_creator.get((category, preferred_creator), [])
        weak_pool = by_category_creator.get((category, weak_creator), [])
        if not preferred_pool or not weak_pool:
            continue
        for i in range(rng.randint(10, 14)):  # positive: strong, noisy, spread across the full historical horizon.
            c = rng.choice(preferred_pool)
            ratio = rng.gauss(0.88, 0.12)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-cr-pos-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct,
                liked=rng.random() < 0.6, shared=rng.random() < 0.15, favorited=rng.random() < 0.1,
                commented=False, creator_followed=rng.random() < 0.2,
                timestamp=_cohort_timestamp(rng, c, now),
            ))
        for i in range(rng.randint(6, 10)):  # weak/negative: strong, noisy, same window.
            c = rng.choice(weak_pool)
            ratio = rng.gauss(0.10, 0.08)
            duration, watch, watch_pct = _watched(ratio)
            event_type = "CONTENT_NOT_INTERESTED" if rng.random() < 0.4 else _event_type(ratio)
            rows.append(Interaction(
                event_id=f"syn-cr-neg-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=event_type, watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct,
                liked=False, shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=_cohort_timestamp(rng, c, now),
            ))
    return rows


# Hard-negative cohort (this task, Step 5): "followed creator but wrong content preference" --
# a user follows a creator (a real, explicit positive action) early on, then has genuinely
# negative outcomes with OTHER content from that SAME creator later. Teaches that
# creator_followed=1 must not be read as an unconditional positive signal, the same way
# `_creator_preference_rows` already teaches genuine cross-creator preference within a
# category -- this cohort instead varies OUTCOME within a single creator's own catalog.
CREATOR_FOLLOW_MISMATCH_USER_COUNT = 18


def _creator_follow_mismatch_rows(rng: random.Random, contents: list[Content], now: datetime) -> list[Interaction]:
    by_creator: dict[str, list[Content]] = defaultdict(list)
    for c in contents:
        by_creator[c.creator_id].append(c)
    eligible_creators = [creator_id for creator_id, items in by_creator.items() if len(items) >= 6]

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    rows: list[Interaction] = []
    for u in range(1, CREATOR_FOLLOW_MISMATCH_USER_COUNT + 1):
        if not eligible_creators:
            break
        user = f"creator-mismatch-user-{u}"
        creator = rng.choice(eligible_creators)
        pool = [c for c in by_creator[creator] if c.created_at <= now - timedelta(days=35)]
        if len(pool) < 6:
            continue
        category = pool[0].category

        # Early: genuine positive engagement, including an explicit CREATOR_FOLLOWED action.
        for i in range(rng.randint(5, 8)):
            c = rng.choice(pool)
            ratio = rng.gauss(0.88, 0.1)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-cfm-follow-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=creator,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.5,
                shared=False, favorited=False, commented=False, creator_followed=(i == 0),
                timestamp=now - timedelta(days=rng.uniform(40, 90)),
            ))
        # Later: genuinely negative outcomes with OTHER content from the SAME (already-
        # followed) creator -- the follow itself never repeats, only the creator identity.
        for i in range(rng.randint(4, 7)):
            c = rng.choice(pool)
            ratio = rng.gauss(0.10, 0.08)
            duration, watch, watch_pct = _watched(ratio)
            event_type = "CONTENT_NOT_INTERESTED" if rng.random() < 0.3 else "VIDEO_SKIPPED"
            rows.append(Interaction(
                event_id=f"syn-cfm-reject-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=creator,
                category=category, event_type=event_type, watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct,
                liked=False, shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=now - timedelta(days=rng.uniform(5, 25)),
            ))
    return rows


# --- has_category_history=0 coverage repair (root-cause investigation, 2026-08-27) ---
#
# Root cause (measured, not guessed): has_category_history=0 rows were only 1.6% of the
# labeled dataset before this repair (17/1088-shaped gap: see the two cohorts below), yet
# still landed at ~72.7% positive vs. ~65.3% for has_category_history=1 rows -- large enough
# that a LogisticRegression fit a strongly negative coefficient (~-1.10) on the flag itself,
# meaning the model learned "no history in this category" as an intrinsically GOOD sign
# rather than a neutral "no evidence either way" signal disambiguated by the accompanying
# affinity/count features (exactly how has_category_history is meant to work, mirroring
# category_affinity's own "0.5 for both neutral and no history, disambiguate with
# has_category_history" convention already documented in app.ml.dataset_builder).
#
# Traced to two causes, both addressed here:
# 1. A handful of small pre-existing cohorts' FIRST event in a new category (e.g.
#    _extra_synthetic_rows' session-switch "discovery" stage, _fast_skip_streak_rows/
#    _not_interested_streak_rows/_recent_shift_rows' own baseline events) happen to be
#    has_category_history=0 by construction and are individually high-quality/positive by
#    that cohort's own design -- legitimate on their own, but with no larger has_category_
#    history=0 population to sit alongside, they dominate the small sample.
# 2. This repair's own _cross_category_session_rows/_stale_category_history_rows cohorts
#    (added for the earlier saturation fix) used an unconditional `liked` probability on
#    their labeled events -- since target_for() treats liked=True as positive regardless of
#    watch_percentage, that alone pushed their outcome distribution well above what their
#    intended "broad, realistic, not a fixed direction" watch-ratio spread produces on its
#    own. Reduced to the same low background `liked` rate app.generate()'s own bulk
#    population already uses for a low-affinity interaction, and a new, explicitly
#    negative-biased _no_history_irrelevant_rows cohort is added alongside them so
#    has_category_history=0 is no longer disproportionately positive -- NOT by suppressing
#    the legitimate cohorts above (Step 12's cold-start users, or the session-switch
#    discovery stage still teach real, intended "no history yet, but this specific new
#    interest is genuine" scenarios), but by adding enough realistic negative/irrelevant
#    no-history examples that the flag itself stops correlating with the label.
CROSS_CATEGORY_SESSION_USER_COUNT = 150
STALE_HISTORY_USER_COUNT = 200
NO_HISTORY_IRRELEVANT_USER_COUNT = 150


def _cross_category_session_rows(rng: random.Random, contents: list[Content], now: datetime) -> list[Interaction]:
    """A real, active, recent same-category session -- then 1-2 labeled events in a genuinely
    different category the user has NO history in at all. Broad randomized outcome
    (rng.gauss(0.5, 0.32), low background `liked` rate) so has_category_history=0 rows from
    this cohort land on a REALISTIC MIX of positive and negative labels, not a fixed
    direction -- this is the "active session elsewhere, zero history in the candidate's own
    category" shape (see app.ml.dataset_builder's last_interaction_category_match/
    session_category_streak_valence design docstring), previously almost entirely absent
    from training data."""
    by_category: dict[str, list[Content]] = defaultdict(list)
    for c in contents:
        by_category[c.category].append(c)

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    rows: list[Interaction] = []
    for u in range(1, CROSS_CATEGORY_SESSION_USER_COUNT + 1):
        user = f"covfix-crosscat-user-{u}"
        chosen = rng.sample(CATEGORIES, 3)
        session_category, candidate_categories = chosen[0], chosen[1:]
        session_pool = [c for c in by_category.get(session_category, []) if c.created_at <= now]
        if not session_pool:
            continue
        anchor = now - timedelta(minutes=rng.randint(15, 25))
        for i in range(rng.randint(2, 4)):  # real, recent, same-category session (mixed outcomes).
            c = rng.choice(session_pool)
            ratio = rng.gauss(0.6, 0.3)
            duration, watch, watch_pct = _watched(ratio, min_duration=40)
            rows.append(Interaction(
                event_id=f"syn-covfix-cc-sess-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.4,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor + timedelta(minutes=i * 3),
            ))
        # One labeled row per DISTINCT candidate category (each a genuine first-time exposure,
        # so it actually lands in has_category_history=0). Low background `liked` rate
        # (matches the bulk population's own low-affinity `.015` baseline order of magnitude)
        # so the label comes from the broad watch-ratio spread, not an unconditional bonus.
        for j, candidate_category in enumerate(candidate_categories):
            candidate_pool = [c for c in by_category.get(candidate_category, []) if c.created_at <= now]
            if not candidate_pool:
                continue
            c = rng.choice(candidate_pool)
            ratio = rng.gauss(0.5, 0.32)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-covfix-cc-cand-{user}-{j}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.05,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=now - timedelta(minutes=rng.randint(1, 10)) + timedelta(seconds=j),
            ))
    return rows


def _stale_category_history_rows(rng: random.Random, contents: list[Content], now: datetime) -> list[Interaction]:
    """Real category history 80-125 days old, no activity since -- then ONE labeled 'return'
    event, recent, in the SAME category, with days_since_last_category_interaction landing
    genuinely in the 90+ range (has_category_history=1, not a no-history sentinel). Broad
    randomized outcome, low background `liked` rate (see _cross_category_session_rows) so the
    model learns "known but long ago" is not automatically positive or negative."""
    by_category: dict[str, list[Content]] = defaultdict(list)
    for c in contents:
        by_category[c.category].append(c)

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    rows: list[Interaction] = []
    for u in range(1, STALE_HISTORY_USER_COUNT + 1):
        user = f"covfix-stale-user-{u}"
        category = rng.choice(CATEGORIES)
        # Content in this generator is created at most ~150 days before `now` (see generate()'s
        # own `start = now - timedelta(days=120)` plus its own +/-30-day jitter) -- bounded to
        # what the actual catalog age distribution supports while still landing well past the
        # "stale" (>=90-day) threshold this cohort targets.
        gap_days = rng.uniform(80, 125)
        pool = [c for c in by_category.get(category, []) if c.created_at <= now - timedelta(days=gap_days + 10)]
        if not pool:
            continue
        old_anchor = now - timedelta(days=gap_days + rng.uniform(2, 8))
        for i in range(rng.randint(2, 4)):  # real old history, mixed outcomes.
            c = rng.choice(pool)
            ratio = rng.gauss(0.5, 0.3)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-covfix-stale-old-{user}-{i}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
                category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
                content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.05,
                shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=old_anchor - timedelta(days=i * 3),
            ))
        return_pool = [c for c in by_category.get(category, []) if c.created_at <= now]
        if not return_pool:
            continue
        c = rng.choice(return_pool)
        ratio = rng.gauss(0.5, 0.32)
        duration, watch, watch_pct = _watched(ratio)
        rows.append(Interaction(
            event_id=f"syn-covfix-stale-return-{user}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
            category=c.category, event_type=_event_type(ratio), watch_time_seconds=watch,
            content_duration_seconds=duration, watch_percentage=watch_pct, liked=rng.random() < 0.05,
            shared=False, favorited=False, commented=False, creator_followed=False,
            timestamp=now - timedelta(minutes=rng.randint(1, 30)),
        ))
    return rows


def _no_history_irrelevant_rows(rng: random.Random, contents: list[Content], now: datetime) -> list[Interaction]:
    """Deliberately negative-biased has_category_history=0 rows: a user with real history in
    ONE category is shown ONE candidate in a genuinely different, unrelated category and
    dislikes it (low watch, no like/share/favorite/follow -- a real, common "this recommendation
    missed" outcome). Exists specifically so has_category_history=0 is not disproportionately
    positive relative to has_category_history=1 (measured pre-repair: ~72.7% vs ~65.3%) --
    the flag itself must not become a usable shortcut for the label; the model should rely on
    the accompanying affinity/count features instead, exactly as has_category_history's own
    "disambiguate a neutral 0.5 from no history at all" design intent already requires."""
    by_category: dict[str, list[Content]] = defaultdict(list)
    for c in contents:
        by_category[c.category].append(c)

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    rows: list[Interaction] = []
    for u in range(1, NO_HISTORY_IRRELEVANT_USER_COUNT + 1):
        user = f"covfix-nohist-neg-user-{u}"
        own_category, candidate_category = rng.sample(CATEGORIES, 2)
        own_pool = [c for c in by_category.get(own_category, []) if c.created_at <= now - timedelta(days=20)]
        candidate_pool = [c for c in by_category.get(candidate_category, []) if c.created_at <= now]
        if not own_pool or not candidate_pool:
            continue
        for i in range(rng.randint(3, 6)):  # real, unrelated-category history (not this row's category).
            c = rng.choice(own_pool)
            ratio = rng.gauss(0.6, 0.25)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-covfix-nohistneg-own-{user}-{i}", user_id=user, content_id=c.content_id,
                creator_id=c.creator_id, category=c.category, event_type="VIDEO_COMPLETED" if ratio >= .9 else "VIDEO_WATCHED",
                watch_time_seconds=watch, content_duration_seconds=duration, watch_percentage=watch_pct,
                liked=rng.random() < 0.1, shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=now - timedelta(days=rng.uniform(10, 60) + i),
            ))
        c = rng.choice(candidate_pool)
        ratio = rng.gauss(0.25, 0.2)  # deliberately negative-skewed: a real "irrelevant to me" outcome.
        duration, watch, watch_pct = _watched(ratio)
        rows.append(Interaction(
            event_id=f"syn-covfix-nohistneg-cand-{user}", user_id=user, content_id=c.content_id, creator_id=c.creator_id,
            category=c.category, event_type="VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED",
            watch_time_seconds=watch, content_duration_seconds=duration, watch_percentage=watch_pct,
            liked=False, shared=False, favorited=False, commented=False, creator_followed=False,
            timestamp=now - timedelta(minutes=rng.randint(1, 15)),
        ))
    return rows


# FINAL DATASET-BUILDER CONTEXT/LABEL SEPARATION (negative-gate coverage task, take 2):
# independent behavioral-state cohort, built on app.ml.dataset_builder.build_dataset's new
# is_training_context_only mechanism. Unlike the reverted literal-streak attempt (see the
# prior repair report), the N prior negative events here are marked context-only: they update
# FeatureHistory exactly like real history but are NEVER emitted as their own labeled row, so
# there is no intermediate-skip-checkpoint contamination of shallower depth buckets -- each
# depth's ONLY labeled contribution is its own single continuation candidate, from an
# INDEPENDENT user with an INDEPENDENT RNG draw, never a byproduct of a longer streak.
INDEPENDENT_STATE_USER_COUNT_PER_DEPTH = 100
INDEPENDENT_STATE_DEPTHS = (0, 1, 3, 5)
INDEPENDENT_STATE_ANCHOR_MIN_DAYS_AGO = 40
INDEPENDENT_STATE_ANCHOR_MAX_DAYS_AGO = 90
# Positive probability for the single same-category labeled continuation, by depth: gentle
# decay, floored well above the dataset's own cold-category baseline rate (~0.67-0.70) even at
# the deepest depth -- calibrated by direct measurement (see the repair report), matching the
# official negative gate's own "reduce, never erase below a cold floor" semantics. Raised from
# an initial {0.80,0.76,0.74,0.72} after direct measurement showed the deepest bucket (N=5,
# smallest natural sample) landing below its target under real RNG variance; both user count
# and headroom above the floor were increased together.
INDEPENDENT_STATE_POSITIVE_PROBABILITY = {0: 0.85, 1: 0.81, 3: 0.79, 5: 0.78}
# The unrelated-category control's own outcome probability is depth-INDEPENDENT (same value
# regardless of the SAME user's SPORT-side N) -- this is exactly what "unrelated distribution
# stays stable across N" requires; if it varied with N, that would itself be a leak.
INDEPENDENT_STATE_UNRELATED_POSITIVE_PROBABILITY = 0.55


def _independent_negative_session_state_rows(
    rng: random.Random, contents: list[Content], now: datetime,
) -> list[Interaction]:
    by_category: dict[str, list[Content]] = defaultdict(list)
    for c in contents:
        by_category[c.category].append(c)

    def _watched(ratio: float, *, min_duration=15, max_duration=180) -> tuple[float, float, float]:
        duration = rng.randint(min_duration, max_duration)
        watch = round(duration * max(0.0, min(1.2, ratio)), 2)
        return duration, watch, round(watch / duration * 100, 4)

    def _event_type(ratio: float) -> str:
        return "VIDEO_COMPLETED" if ratio >= .9 else ("VIDEO_SKIPPED" if ratio < .2 else "VIDEO_WATCHED")

    rows: list[Interaction] = []
    user_counter = 0
    for depth in INDEPENDENT_STATE_DEPTHS:
        positive_probability = INDEPENDENT_STATE_POSITIVE_PROBABILITY[depth]
        for _ in range(1, INDEPENDENT_STATE_USER_COUNT_PER_DEPTH + 1):
            user_counter += 1
            user = f"indepstate-user-{user_counter}"
            category, unrelated_category = rng.sample(CATEGORIES, 2)
            anchor_days_ago = rng.uniform(INDEPENDENT_STATE_ANCHOR_MIN_DAYS_AGO, INDEPENDENT_STATE_ANCHOR_MAX_DAYS_AGO)
            anchor = now - timedelta(days=anchor_days_ago)
            pool = [c for c in by_category.get(category, []) if c.created_at <= anchor - timedelta(days=30)]
            unrelated_pool = [c for c in by_category.get(unrelated_category, []) if c.created_at <= now]
            if len(pool) < depth + 4 or not unrelated_pool:
                continue

            # HISTORY: 8-12 established positive events, 40-90 days old -- context-only, never
            # labeled rows (matches how real accumulated history is never itself a supervised
            # target).
            for i in range(rng.randint(8, 12)):
                c = rng.choice(pool)
                ratio = rng.gauss(0.87, 0.08)
                duration, watch, watch_pct = _watched(ratio)
                rows.append(Interaction(
                    event_id=f"syn-indepstate-lt-{user}-{i}", user_id=user, content_id=c.content_id,
                    creator_id=c.creator_id, category=c.category, event_type=_event_type(ratio),
                    watch_time_seconds=watch, content_duration_seconds=duration, watch_percentage=watch_pct,
                    liked=rng.random() < 0.55, shared=rng.random() < 0.1, favorited=False, commented=False,
                    creator_followed=False, timestamp=anchor - timedelta(days=rng.uniform(40, 90) + i),
                    is_training_context_only=True,
                ))

            # SESSION CONTEXT: exactly `depth` implicit fast-skips, minute-scale spacing, SAME
            # category -- context-only. These update session_negative_interaction_count/
            # category_negative_count for the eventual candidate below, but are NEVER
            # themselves emitted as labeled rows, eliminating the checkpoint-collision problem
            # a literal labeled streak causes.
            streak_start_offset = 20
            for i in range(depth):
                c = rng.choice(pool)
                ratio = rng.gauss(0.04, 0.02)
                duration, watch, watch_pct = _watched(ratio)
                rows.append(Interaction(
                    event_id=f"syn-indepstate-ctx-{user}-{i}", user_id=user, content_id=c.content_id,
                    creator_id=c.creator_id, category=c.category, event_type="VIDEO_SKIPPED",
                    watch_time_seconds=watch, content_duration_seconds=duration, watch_percentage=watch_pct,
                    liked=False, shared=False, favorited=False, commented=False, creator_followed=False,
                    timestamp=anchor - timedelta(minutes=streak_start_offset - i * 3),
                    is_training_context_only=True,
                ))

            # TARGET: the ONE independently-sampled, labeled next-SPORT-event outcome for this
            # state -- the only row this state contributes to the dataset in this category.
            continuation_offset = streak_start_offset - depth * 3
            c = rng.choice(pool)
            if rng.random() < positive_probability:
                ratio = rng.gauss(0.88, 0.07)
            else:
                ratio = rng.gauss(0.10, 0.05)
            duration, watch, watch_pct = _watched(ratio)
            rows.append(Interaction(
                event_id=f"syn-indepstate-target-{user}", user_id=user, content_id=c.content_id,
                creator_id=c.creator_id, category=c.category, event_type=_event_type(ratio),
                watch_time_seconds=watch, content_duration_seconds=duration, watch_percentage=watch_pct,
                liked=ratio >= 0.85 and rng.random() < 0.4, shared=False, favorited=False, commented=False,
                creator_followed=False, timestamp=anchor - timedelta(minutes=max(0.5, continuation_offset)),
            ))

            # UNRELATED CONTROL: a labeled candidate in a DIFFERENT category, outcome
            # probability independent of `depth` -- proves SPORT's N context events must not
            # bleed into GAMING's own label distribution.
            uc = rng.choice(unrelated_pool)
            if rng.random() < INDEPENDENT_STATE_UNRELATED_POSITIVE_PROBABILITY:
                uratio = rng.gauss(0.80, 0.1)
            else:
                uratio = rng.gauss(0.30, 0.15)
            uduration, uwatch, uwatch_pct = _watched(uratio)
            rows.append(Interaction(
                event_id=f"syn-indepstate-unrel-{user}", user_id=user, content_id=uc.content_id,
                creator_id=uc.creator_id, category=uc.category, event_type=_event_type(uratio),
                watch_time_seconds=uwatch, content_duration_seconds=uduration, watch_percentage=uwatch_pct,
                liked=False, shared=False, favorited=False, commented=False, creator_followed=False,
                timestamp=anchor - timedelta(minutes=max(0.5, continuation_offset - 1)),
            ))
    return rows


def generate(
    count=12000, *, reference_timestamp: datetime | None = None, include_hard_negatives: bool = True,
    seed: int = SEED, db=None, db_engine=None,
):
    """`seed` (negative-feedback feature representation task): overrides every internal RNG
    stream's derivation below (all of which were already offsets of the single module-level
    SEED constant -- `SEED+1`/`SEED+2`/... for each independently-isolated cohort, see the
    comments on each call site). The default `seed=SEED` reproduces today's exact dataset
    byte-for-byte -- every existing caller (including every test) is unaffected. An explicit
    `seed` produces a deterministic, materially different alternate synthetic distribution
    (same generation LOGIC, different random draws throughout), for dataset-seed robustness
    validation -- see scripts/run_negative_feature_experiment.py.

    `db`/`db_engine` (Task 7: multi-dataset-seed robustness without touching the app's
    singleton engine/session): both default to None, in which case this function behaves
    EXACTLY as before -- schema created on, and rows written through, the module-level
    `engine`/`SessionLocal` every existing caller already gets. Passing an isolated
    `db`/`db_engine` pair (e.g. `app.db.database.make_engine("sqlite:///:memory:")` plus a
    fresh sessionmaker -- the same isolation convention app.benchmark.runner.isolated_session
    already uses) lets a caller generate several independent datasets within one process
    without ever touching global DB state; see
    scripts/run_ranking_challenger_dataset_seeds.py."""
    owns_session = db is None
    session_engine = db_engine or engine
    rng=random.Random(seed); Base.metadata.create_all(session_engine); db=db or SessionLocal()
    try:
        if db.query(Interaction).filter(Interaction.event_id.like("syn-%")).first():
            print("Synthetic data already exists; nothing inserted."); return
        now=reference_timestamp if reference_timestamp is not None else datetime.now(timezone.utc)
        start=now-timedelta(days=120); contents=[]
        creator_quality={f"creator-{i}":rng.betavariate(2.5,2) for i in range(1,31)}
        for i in range(300):
            creator=f"creator-{i%30+1}"; quality=creator_quality[creator]
            created=start-timedelta(days=rng.randint(1,30)) if i<20 else start+timedelta(days=rng.randint(0,119))
            category=CATEGORIES[i%10]
            c=Content(content_id=f"video-{i+1}",creator_id=creator,category=category,popularity_score=round(max(0,min(1,.65*quality+.35*rng.random())),4),created_at=created,
                      **(_tag_content(rng, category) or {}))
            contents.append(c)
        db.add_all(contents); db.flush(); interests={}
        for u in range(1,101):
            order=rng.sample(CATEGORIES,len(CATEGORIES)); interests[f"user-{u}"]=(order,{cat:(.82 if cat in order[:2] else .55 if cat in order[2:4] else .20) for cat in CATEGORIES})
        rows=[]
        for i in range(count):
            timestamp=start+timedelta(seconds=i*850+rng.randint(0,600)); available=[item for item in contents if item.created_at<=timestamp]
            user=f"user-{rng.randint(1,100)}"; c=rng.choice(available); order,base=interests[user]; progress=i/count
            strength=base[c.category]
            # A subset of users gradually shifts from an old strong interest to a new one.
            if int(user.split("-")[1]) % 3 == 0:
                if c.category==order[0]: strength-=.35*progress
                if c.category==order[4]: strength+=.40*progress
            if rng.random()<.12: strength=rng.uniform(.1,.9)  # exploration/noisy sessions
            duration=rng.randint(15,180); quality=creator_quality[c.creator_id]
            ratio=max(0,min(1.5,rng.gauss(.08+.68*strength+.18*quality,.24))); watch=round(duration*ratio,2)
            liked=rng.random()<(.015+.18*strength); shared=rng.random()<(.005+.07*strength); favorite=rng.random()<(.008+.09*strength); followed=rng.random()<(.003+.035*strength)
            typ="VIDEO_COMPLETED" if ratio>=.9 else ("VIDEO_SKIPPED" if ratio<.2 else "VIDEO_WATCHED")
            roll=rng.random()
            if roll<.025: typ="CONTENT_NOT_INTERESTED"
            elif roll<.05 and liked: typ="CONTENT_LIKED"
            elif roll<.06 and shared: typ="CONTENT_SHARED"
            elif roll<.07 and favorite: typ="CONTENT_FAVORITED"
            rows.append(Interaction(event_id=f"syn-{i+1}",user_id=user,content_id=c.content_id,creator_id=c.creator_id,category=c.category,event_type=typ,watch_time_seconds=watch,content_duration_seconds=duration,watch_percentage=round(watch/duration*100,4),liked=liked,shared=shared,favorited=favorite,commented=rng.random()<.04*strength,creator_followed=followed,timestamp=timestamp))
            if len(rows)>=1000: db.add_all(rows); db.commit(); rows=[]
        extra_rows = _extra_synthetic_rows(rng, contents, now, seed)
        # Phase 1.5 issue 1: an independent RNG (not the shared `rng` stream), so adding this
        # cohort never perturbs the exact rows _semantic_preference_rows/_creator_preference_rows
        # below draw -- this fix is then purely additive, isolated from every other cohort, and
        # a before/after comparison of any other cohort's output is never confounded by it.
        recent_shift_rows = _recent_shift_rows(random.Random(seed + 1), contents, now)
        # Same isolation convention as recent_shift_rows above (independent RNG stream, SEED+3
        # -- SEED+2 is already used internally by _extra_synthetic_rows' anchor_rng): purely
        # additive, never perturbs semantic_rows/creator_rows' draws from the shared `rng`.
        fast_skip_streak_rows = _fast_skip_streak_rows(random.Random(seed + 3), contents, now)
        semantic_rows = _semantic_preference_rows(rng, contents, now)
        creator_rows = _creator_preference_rows(rng, contents, now)
        rows.extend(extra_rows); rows.extend(recent_shift_rows); rows.extend(fast_skip_streak_rows)
        rows.extend(semantic_rows); rows.extend(creator_rows)
        # This task's hard-negative cohorts (Step 5), gated by `include_hard_negatives` so the
        # exact prior dataset can still be reproduced for a controlled before/after comparison
        # (`generate(include_hard_negatives=False)`). Same isolation convention as every cohort
        # above: independent RNG streams (SEED+4/SEED+5 -- +1/+2/+3 already used), purely
        # additive, never perturbs any other cohort's draws.
        not_interested_streak_rows: list[Interaction] = []
        creator_mismatch_rows: list[Interaction] = []
        if include_hard_negatives:
            not_interested_streak_rows = _not_interested_streak_rows(random.Random(seed + 4), contents, now)
            creator_mismatch_rows = _creator_follow_mismatch_rows(random.Random(seed + 5), contents, now)
            rows.extend(not_interested_streak_rows); rows.extend(creator_mismatch_rows)
        # has_category_history=0 coverage repair (root-cause investigation, 2026-08-27): purely
        # additive, independent RNG streams (seed+6/+7/+8, the next unused offsets after +5
        # above), never perturbing any earlier cohort's draws -- see the constants/docstrings
        # above _cross_category_session_rows for the full root-cause rationale.
        cross_category_rows = _cross_category_session_rows(random.Random(seed + 6), contents, now)
        stale_history_rows = _stale_category_history_rows(random.Random(seed + 7), contents, now)
        no_history_irrelevant_rows = _no_history_irrelevant_rows(random.Random(seed + 8), contents, now)
        rows.extend(cross_category_rows); rows.extend(stale_history_rows); rows.extend(no_history_irrelevant_rows)
        # FINAL DATASET-BUILDER CONTEXT/LABEL SEPARATION (this task): seed+9, the next unused
        # offset -- purely additive, independent RNG stream, never perturbing any earlier
        # cohort's draws. See the cohort's own docstring above for the full mechanism.
        independent_state_rows = _independent_negative_session_state_rows(random.Random(seed + 9), contents, now)
        rows.extend(independent_state_rows)
        db.add_all(rows); db.commit()
        print(f"Generated {count} interactions, 100 users, 300 contents, 30 creators and 10 categories, plus "
              f"{len(extra_rows)} cold-start/session-switch rows across up to {COLD_START_USER_COUNT + SESSION_SWITCH_USER_COUNT} extra users, plus "
              f"{len(recent_shift_rows)} recent-shift rows across up to {RECENT_SHIFT_USER_COUNT} extra users, plus "
              f"{len(fast_skip_streak_rows)} fast-skip-streak rows across up to {FAST_SKIP_STREAK_USER_COUNT} extra users, plus "
              f"{len(semantic_rows)} semantic-preference rows across up to {SEMANTIC_PREFERENCE_USER_COUNT} extra users, plus "
              f"{len(creator_rows)} creator-preference rows across up to {CREATOR_PREFERENCE_USER_COUNT} extra users, plus "
              f"{len(not_interested_streak_rows)} not-interested-streak rows across up to {NOT_INTERESTED_STREAK_USER_COUNT} extra users, plus "
              f"{len(creator_mismatch_rows)} creator-follow-mismatch rows across up to {CREATOR_FOLLOW_MISMATCH_USER_COUNT} extra users, plus "
              f"{len(cross_category_rows)} cross-category-session rows across up to {CROSS_CATEGORY_SESSION_USER_COUNT} extra users, plus "
              f"{len(stale_history_rows)} stale-category-history rows across up to {STALE_HISTORY_USER_COUNT} extra users, plus "
              f"{len(no_history_irrelevant_rows)} no-history-irrelevant rows across up to {NO_HISTORY_IRRELEVANT_USER_COUNT} extra users, plus "
              f"{len(independent_state_rows)} independent-negative-session-state rows across up to "
              f"{INDEPENDENT_STATE_USER_COUNT_PER_DEPTH * len(INDEPENDENT_STATE_DEPTHS)} extra users.")
    finally:
        if owns_session:
            db.close()

if __name__=="__main__": generate()
