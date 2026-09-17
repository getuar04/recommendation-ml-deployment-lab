"""Point-in-time feature generation shared by training and online prediction.

Semantic features (design note, VIDEO only): candidate hashtags/topics/entities/subgenres/
title-tokens are folded into **aggregated per-user affinity features** (hashtag_affinity,
topic_affinity, ..., title_affinity, semantic_positive_match_count, ...) rather than one-hot/
multi-hot encoding each token. This is a deliberate tradeoff: token identity never becomes a
model input column, so the feature space stays fixed-width regardless of catalog vocabulary
size -- no retraining is required just because a brand-new hashtag (or title word) appears,
and there is no uncontrolled one-hot cardinality explosion. The cost is that the model cannot
learn a token-specific nonlinear interaction (e.g. "MESSI specifically, independent of the
user's general SPORT affinity") -- only the aggregate signal, which is what
app.ml.reranker's semantic boost/penalty complements with.

Title tokens: a candidate/content `title` is free text, not a pre-tagged list, so it is
deterministically word-tokenized by `app.ml.semantic_tokens.extract_title_tokens`
(stopword-filtered, bounded, deduplicated) and then folded into the exact same per-user
token-affinity machinery as hashtags/topics/entities/subgenres -- title is simply a fifth
entry in `SEMANTIC_FIELDS` with its own `title_affinity` feature.

Semantic features on the `UserProfile` snapshot path (finalization spec Decision 5): unlike
category/creator affinity, `UserProfile.categories[].recentWatchEvents` carries no `content_id`,
so there is no candidate-content reference to *recompute* semantic affinity from raw events in
`FeatureHistory.from_profile`. Instead, the supplying service pre-aggregates per-(field, token)
affinity directly (`UserProfile.semantic_affinities`, mirroring how category/creator counters
are already pre-aggregated) and `from_profile` loads it straight into `self.tokens`. A profile
that omits `semantic_affinities` (every pre-Decision-5 caller) leaves semantic features neutral
for that request, exactly as before -- this is additive, not a behavior change for existing
callers.

Session features (current-session productionization, moved in from the PoC after the PoC
validated the design -- see PoC repo app/ml/dataset_builder.py session_features() for the
original comparison work; NOT blindly copied, evaluated feature-by-feature below):

- **Window**: SESSION_WINDOW=30min lookback, capped to SESSION_MAX_EVENTS=12 -- unchanged
  from the PoC. Evaluated for production and kept as-is: 30 minutes matches a typical
  short-form video session's dwell time (long enough to capture a real session, short enough
  that "current session" doesn't blur into "today"), 12 events bounds per-request compute for
  the read-time streak/decay scan without truncating a real session prematurely, and neither
  constant depends on data volume/scale -- they describe *user session shape*, which is the
  same whether the deployment sees ten or ten million requests/day. Kept.
- **Decay**: identical formula to the PoC, `exp(-ln(2)/SESSION_DECAY_HALF_LIFE_MINUTES *
  age_minutes)`, half-life 5 minutes, applied ONLY to session features -- category_affinity/
  recent_category_affinity above remain fully undecayed, so the three horizons stay distinct.
- **Formula asymmetry vs. category/recent_category_affinity (deliberate)**: the DB-row path
  (`update()`) has full event fields (favorited/commented/creator_followed/event_type); the
  `UserProfile` snapshot path (`from_profile()`, via `RecentWatchEvent`) only ever has
  watch_percentage/completed/liked/shared(/notInterested). Session evidence uses ONLY those
  fields in BOTH paths (`_session_event_delta`/`_session_label`), sacrificing some signal
  richness in the DB-row/training path specifically so training and inference compute session
  features identically -- train/serve parity was judged more valuable here than squeezing every
  available training-time signal in, since this feature only ever needs to work correctly
  against what production inference (UserProfile-driven) can actually supply.
  Negative-feedback session bug fix: `notInterested` (RecentWatchEvent) / explicit-rejection
  event_type (DB row) is now part of that shared four/five-field contract -- omitted, this used
  to let a high watch_percentage (naturally set by a user who watched most of a video before
  explicitly rejecting it) launder an explicit CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED/
  VIDEO_SKIPPED rejection into a POSITIVE session_category_affinity/SESSION_INTEREST reading for
  other same-category candidates, even though the long-term category layer already correctly
  penalized the exact same event (see CONTENT_NOT_INTERESTED_CATEGORY_PENALTY). `notInterested`
  defaults to False (safe default, same convention as `liked`/`shared`/`explicit_negative_count`
  above), so a caller that predates this fix keeps sending valid requests -- it just does not yet
  benefit from the fix until it starts populating the new field.
- **Feature selection** (spec: "do not add everything blindly"), evaluated one by one against
  the PoC's 9 candidates:
  REQUIRED: session_category_affinity (the core "what do they want right now" signal),
  session_category_streak + last_interaction_category_match (order-awareness -- the PRANK-then-
  MUSIC acceptance scenario is meaningless without these), session_intent_confidence (explicit
  spec requirement, "one accidental interaction must not flip the feed").
  DIAGNOSTIC-ONLY after the Variant F rising-tide repair: has_session_activity and
  session_average_watch_percentage are candidate-independent (identical for every candidate
  in a request), so they remain computed for observability/reranker use but are not selected
  into NUMERIC/FEATURES. USEFUL, included: session_positive_interaction_count/
  session_negative_interaction_count (mirrors category_positive_count/category_negative_count's
  existing log1p convention).
  REDUNDANT, dropped: time_since_last_positive_interaction -- its recency signal is already
  carried by session_category_affinity's decay and by session_intent_confidence's own recency
  term; keeping the production feature-vector footprint minimal for a first, lowest-risk
  session rollout outweighed the marginal, mostly-overlapping information this one feature
  added on top of the eight above.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from itertools import groupby
from typing import Any, NamedTuple

import pandas as pd

from app.ml.feature_builder import (
    COMPLETION_WATCH_PERCENTAGE_THRESHOLD,
    EXPLICIT_NEGATIVE_EVENT_TYPES,
    FAST_SKIP_WATCH_PERCENTAGE_THRESHOLD,
    affinity_score,
    interaction_signals,
    target_for,
)
from app.ml.replay_saturation_policy import (
    is_saturation_controlled_interaction,
    repeat_influence_weight,
)
from app.ml.semantic_tokens import extract_title_tokens

RECENT_WINDOW = timedelta(days=30)
MAX_CONTENT_AGE_HOURS = 24 * 365.0  # Cap outlier catalog ages at one year, matching the 365-day recency cap below.
SEMANTIC_FIELDS = ("hashtag", "topic", "entity", "subgenre", "title")

# CONTENT_NOT_INTERESTED weighting (see FeatureHistory.update()): the category-level penalty
# and the per-semantic-token penalty are deliberately different magnitudes. Fixing a forensic
# trace where the full category-level weight was also applied independently to every matched
# semantic token, turning one click into many simultaneous maximally-negative model features.
CONTENT_NOT_INTERESTED_CATEGORY_PENALTY = 8
# Calibration fix: was 2, which made an explicit CONTENT_NOT_INTERESTED's own per-token
# suppression WEAKER than a plain implicit VIDEO_SKIPPED's (base_delta's fast-skip term is -4,
# applied unmodified to a skip's own tokens; a 2-point explicit-rejection penalty could never
# exceed that). 5 is the smallest value that makes one explicit rejection suppress its own
# semantic tokens strictly more than one fast skip (measured: sigmoid(-5/10)=0.3775 <
# sigmoid(-4/10)=0.4013), matching the product intent that CONTENT_NOT_INTERESTED is the
# stronger, more deliberate signal at every granularity, not just the category level.
CONTENT_NOT_INTERESTED_SEMANTIC_PENALTY = 5
# Localization fix (independent-audit finding): repeated CONTENT_NOT_INTERESTED rejection of
# the SAME semantic subtheme (e.g. several different Tennis videos) used to apply the FULL
# category-level penalty (-8) on every single event -- since this accumulates onto the SAME
# (user, category) "raw" total each time, 3-4 rejections of one narrow subtheme collapsed the
# ENTIRE broad category's affinity (measured 0.881 -> 0.769 -> 0.599 -> 0.401 for SPORT after
# 3 Tennis-only rejections), even though the semantic layer had already correctly localized
# the dislike to Tennis-specific tokens. CONTENT_NOT_INTERESTED_CATEGORY_REPEAT_PENALTY applies
# instead of the full penalty when this event's content shares a semantic token with content
# the SAME user has ALREADY net-negatively rejected in this category (see
# FeatureHistory._is_repeat_negative_subtheme) -- i.e. the category-level signal from a 2nd/3rd/
# 4th rejection of the SAME subtheme is treated as mostly redundant with what the semantic layer
# already captured, while a rejection of a genuinely DIFFERENT subtheme (no token overlap with
# any prior rejection) still applies the full category-level penalty every time. Rejecting
# several DISTINCT subthemes within one category (Tennis, then Football, then Basketball, ...)
# therefore still drives a real, material category-level decline -- only same-subtheme repeats
# are dampened. Never hardcodes a category/subtheme name; purely mechanical, general token-
# overlap logic reusing the SAME per-(user, token) state the semantic-affinity features already
# maintain -- no new tracking structure. A rejection on content with no semantic metadata at
# all (content=None) cannot be localized, so it conservatively falls back to the full penalty,
# unchanged from before this fix.
#
# Calibration fix (VIDEO negative-feedback sweep): was 2, which measurably regressed two
# independent, real-trained-model behavioral checks that this task's own safety criteria forbid
# breaking -- tests/test_step11_rf_candidate_regression.py (RandomForest candidate: a real
# MAX_CONSECUTIVE_CATEGORY diversity violation, a run of 3 same-category picks at full-pool
# limit) and tests/test_finalization_scenario_matrix.py (a basketball-positive user's
# Lakers/BASKETBALL candidate no longer outranking Barcelona/FOOTBALL -- direction reverses).
# Isolated by binary sweep (2/4 fail, 5/6/8 pass) holding CONTENT_NOT_INTERESTED_SEMANTIC_PENALTY
# fixed at 5: dampening the repeat-subtheme category penalty too aggressively measurably shifts
# the trained model's real category-affinity/target relationship across the FULL synthetic
# dataset (not just the isolated Tennis/Nadal probe this constant was tuned against), because
# scripts/generate_synthetic_data.py's not-interested-streak rows are themselves same-subtheme
# repeats -- so this constant's value has training-set-wide reach, not just the one-user probe.
# 5 is the smallest tested value with zero regression against both checks; it still provides
# real (if more modest than 2's) localization -- 4 same-subtheme rejections leave SPORT affinity
# at ~0.43 instead of collapsing to ~0.23 under the unlocalized full penalty (8) -- while keeping
# the trained model's broader category/target relationship intact. See
# tests/test_not_interested_gradual_suppression.py for the locked-in regression coverage.
CONTENT_NOT_INTERESTED_CATEGORY_REPEAT_PENALTY = 5

# --- Explicit-rejection feature representation (negative-feedback feature task) ---
# Diagnosis: `category_negative_count`/`category_affinity`/`semantic_negative_match_count`
# already exist, but every one of them conflates an EXPLICIT CONTENT_NOT_INTERESTED/
# LIVE_NOT_INTERESTED event with an ambiguous implicit fast-skip -- both simply increment the
# same "negative" counter and fold into the same weighted "raw" scalar (see `update()` below).
# LogisticRegression's smooth global coefficients can still separate the two cases through the
# combined magnitude of several correlated aggregates; a tree/boosted model splitting on
# individual scalar thresholds has no single feature that directly answers "was this candidate's
# category/subtheme ever EXPLICITLY rejected" -- exactly the gap app.ml.eligibility's `negative`/
# `notInterested` gates probe (see app.ml.gate_severity for the measured evidence). The features
# below are additive, computed unconditionally by `.features()` from EXISTING accumulator state
# plus one new counter/timestamp per (user, category) and per (user, semantic token) -- reusing
# EXPLICIT_NEGATIVE_EVENT_TYPES, the exact same definition app.ml.feature_builder.target_for and
# app.ml.sample_weight_policy already use for "explicit", so there is exactly one definition of
# "explicit rejection" anywhere in this codebase. NOT yet part of production NUMERIC/FEATURES --
# see app.ml.negative_feedback_features for the controlled-experiment feature groups and
# scripts/run_negative_feature_experiment.py for the evidence gate before permanent adoption.
#
# `recent_explicit_rejection_strength` reuses the exact bounded-exponential-decay convention
# `_session_decay_weight` below already established (`exp(-ln(2)/half_life * age)`), just
# parameterized in days instead of minutes -- RECENT_WINDOW (30 days) is reused as the half-life
# so the feature still carries real signal at the 40-90-day recency
# scripts/generate_synthetic_data.py's NOT_INTERESTED_STREAK cohort actually generates, rather
# than inventing an unrelated decay constant.
EXPLICIT_REJECTION_DECAY_HALF_LIFE_DAYS = RECENT_WINDOW.days

# --- Session (current-session horizon) constants -- see module docstring for the full design
# rationale and the PoC-vs-production evaluation of each constant/feature. ---
SESSION_WINDOW = timedelta(minutes=30)
SESSION_MAX_EVENTS = 12
SESSION_DECAY_HALF_LIFE_MINUTES = 5.0
# Evidence mass at which session_intent_confidence's evidence_score saturates to 1.0.
# Calibrated (unchanged from the PoC) so a single plain positive watch (magnitude ~=1, the
# SESSION_LABEL_EVIDENCE_FLOOR) yields "weak/moderate" evidence (~0.25), while ~3 strong
# same-category interactions (completed, or completed+liked/shared) saturate it.
SESSION_CONFIDENCE_SATURATION_WEIGHT = 4.0
# Base evidence magnitude granted to any session event with a definitive positive/negative
# label, on top of its raw |delta| -- without this floor, a positive watch with no completion/
# like/share bonus (delta == 0, e.g. a single 80%-watched video with nothing else) would
# contribute zero evidence despite being a real, labeled signal.
SESSION_LABEL_EVIDENCE_FLOOR = 1.0
# Negative-feedback session bug fix: magnitude for an explicit rejection's session-level delta
# (_session_event_delta), overriding completed/liked/shared entirely rather than merely
# offsetting them (see that function's docstring). 6 is strictly larger in magnitude than any
# single positive term it could otherwise combine with (completed=4, liked=3, shared=5, and
# fast_skip's own -4), matching the same "strictly exceeds the largest opposing signal"
# convention already used by CONTENT_NOT_INTERESTED_CATEGORY_PENALTY (8, vs. long-term positive
# terms) and CONTENT_NOT_INTERESTED_SEMANTIC_PENALTY (5, vs. fast_skip's -4).
SESSION_NOT_INTERESTED_PENALTY = 6
CATEGORICAL = ["category"]
NUMERIC = [
    "category_affinity",
    "recent_category_affinity",
    "average_category_watch_percentage",
    "recent_category_watch_percentage",
    "category_completion_rate",
    "recent_category_completion_rate",
    "category_positive_count",
    "category_negative_count",
    "has_creator_history",
    "creator_interaction_count",
    "creator_completion_rate",
    "creator_followed",
    "hashtag_affinity",
    "topic_affinity",
    "entity_affinity",
    "subgenre_affinity",
    "title_affinity",
    "semantic_positive_match_count",
    "semantic_negative_match_count",
    "has_semantic_history",
    "strongest_semantic_affinity",
    "average_semantic_affinity",
    "user_total_interaction_count",
    "content_popularity_score",
    "content_age_hours",
    "already_seen",
    "hour_of_day",
    "session_category_affinity",
    "session_positive_interaction_count",
    "session_negative_interaction_count",
    "session_category_streak_valence_matched",
    "last_interaction_category_match",
    "session_intent_confidence",
]
FEATURES = CATEGORICAL + NUMERIC
# Columns carried alongside FEATURES purely for offline diagnostics (rerank evaluation,
# baselines). Never selected into model input (see FEATURES) and never required online.
DIAGNOSTIC_COLUMNS = ["user_id", "creator_id", "content_id", "candidate_group", "timestamp"]
# The row's OWN raw event fields -- exactly the fields app.ml.feature_builder.target_for()
# already reads to derive `target` itself, so carrying them adds no leakage beyond what the
# label already encodes. Used only by app.ml.sample_weight_policy (dataset diagnostics /
# training-importance weighting) and by dataset-diagnostic reporting -- NEVER selected into
# FEATURES/model input. Point-in-time safe: these describe the row's own already-happened
# event, never a future or history-derived value. "event_"-prefixed (except event_type,
# already unambiguous) so none of these can ever collide with a same-named FEATURES column
# (e.g. the "creator_followed" *feature* means "was the candidate's creator already followed
# before this scoring moment" -- a materially different question from this row's OWN
# creator_followed flag).
TRAINING_METADATA_COLUMNS = [
    "event_type", "event_watch_percentage", "event_liked", "event_shared", "event_favorited", "event_creator_followed",
]
FEATURE_DEFINITIONS = {
    "category": "Candidate content category, one-hot encoded.",
    "category_affinity": "Sigmoid-normalized weighted behavior score using only prior events. "
                          "0.5 for both a genuinely neutral history and no history at all; disambiguate with has_category_history.",
    "recent_category_affinity": "Sigmoid-normalized weighted behavior score using only prior events within the previous "
                                 "RECENT_WINDOW (30 days) -- the same weighting formula as category_affinity, applied to a "
                                 "recent-only subset instead of full history. 0.5 for both a neutral recent history and no "
                                 "recent history at all (including a user whose only history predates the window).",
    "has_category_history": "Whether the user has any prior interaction in this category (distinguishes cold-start 0.5 affinity from a real neutral signal). Computed unconditionally (still available via FeatureHistory/diagnostics/API) but NOT part of production NUMERIC/FEATURES as of the root-cause investigation (2026-08-27): when this is 0, category_positive_count/category_negative_count/category_interaction_count are structurally exactly 0 and category_affinity/recent_category_affinity collapse to exactly 0.5 -- the flag carries no information those features don't already encode. A LogisticRegression fit a large, unstable coefficient on this structurally-redundant flag regardless of dataset composition (proven: fixing a real, measured dataset-driven correlation from a large magnitude to ~-0.02 left the coefficient itself essentially unchanged), consistent with classic multicollinearity, not a real independent signal.",
    "average_category_watch_percentage": "Mean watch percentage for all prior category watches.",
    "recent_category_watch_percentage": "Mean category watch percentage in the previous 30 days.",
    "category_completion_rate": "Completed category watches divided by prior category watches.",
    "recent_category_completion_rate": "Completion rate for category watches in the previous 30 days.",
    "category_positive_count": "log1p of prior labeled positive interactions in the category (log-scaled: unbounded raw counts are highly skewed).",
    "category_negative_count": "log1p of prior negative or not-interested interactions in the category (log-scaled).",
    "category_interaction_count": "log1p of all prior interactions (any type) in the category, independent of label (log-scaled). Computed unconditionally (still available via FeatureHistory/diagnostics) but NOT part of production NUMERIC/FEATURES as of the root-cause investigation (2026-08-27): its zero/nonzero boundary matches has_category_history==0 at exactly 100% in the real training population (it is 0 if and only if there is no category history at all, by construction), so it silently re-introduced the exact same structural redundancy as the already-removed has_category_history flag -- and its own correlation with target was measured weakly NEGATIVE (-0.056) despite nominally being an 'evidence strength' signal, confounded by counting neutral/unlabeled events alongside labeled ones. category_positive_count/category_negative_count (label-aware subsets of this same total, only 84%/62% redundant with history-presence) carry the genuine signal instead.",
    "has_creator_history": "Whether the user has any prior interaction with the candidate creator.",
    "creator_interaction_count": "log1p of prior interactions between the user and candidate creator (log-scaled).",
    "creator_completion_rate": "Prior completion rate for the candidate creator.",
    "creator_followed": "Whether the user followed the candidate creator before scoring.",
    "hashtag_affinity": "Mean sigmoid-normalized affinity across the candidate's hashtags that the user has prior history with; 0.5 if none matched.",
    "topic_affinity": "Mean sigmoid-normalized affinity across the candidate's topics that the user has prior history with; 0.5 if none matched.",
    "entity_affinity": "Mean sigmoid-normalized affinity across the candidate's entities (e.g. team/artist names) that the user has prior history with; 0.5 if none matched.",
    "subgenre_affinity": "Mean sigmoid-normalized affinity across the candidate's subgenres that the user has prior history with; 0.5 if none matched.",
    "title_affinity": "Mean sigmoid-normalized affinity across the candidate title's deterministically-extracted word tokens (app.ml.semantic_tokens.extract_title_tokens) that the user has prior history with; 0.5 if none matched or no title given.",
    "semantic_positive_match_count": "Count of candidate hashtags/topics/entities/subgenres/title-tokens for which the user's prior history skews positive.",
    "semantic_negative_match_count": "Count of candidate hashtags/topics/entities/subgenres/title-tokens for which the user's prior history skews negative.",
    "has_semantic_history": "Whether the user has any prior history with any of the candidate's hashtags/topics/entities/subgenres/title-tokens.",
    "strongest_semantic_affinity": "The single matched semantic-token affinity furthest from neutral (0.5) in either direction; 0.5 if none matched.",
    "average_semantic_affinity": "Mean affinity across all matched semantic tokens (hashtags/topics/entities/subgenres/title combined); 0.5 if none matched.",
    "user_total_interaction_count": "log1p of the user's total prior interactions across all categories (log-scaled).",
    "content_popularity_score": "Candidate catalog popularity in [0,1], identical online and offline.",
    "content_age_hours": f"Hours from catalog creation to scoring timestamp, capped at {MAX_CONTENT_AGE_HOURS:.0f} to bound outliers.",
    "already_seen": "Whether the user interacted with this content before scoring.",
    "hour_of_day": "Hour of the scoring timestamp.",
    "days_since_last_category_interaction": "Days since prior category activity, capped at 365; 0.0 (not the 365 cap) when there is no prior category activity at all -- disambiguate a real, long-ago interaction from no history using has_category_history. Computed unconditionally (still available via FeatureHistory/diagnostics) but NOT part of production NUMERIC/FEATURES as of the root-cause investigation (2026-08-27): its zero value matches has_category_history==0 at 99.5% in the real training population (structurally near-identical to category_interaction_count's own redundancy, see that feature's note) while its own standalone correlation with target measured near-zero (-0.016) -- contribution tracing on the longTerm eligibility probe showed this feature contributing a large, spurious negative delta (~-0.88) toward a strong-long-term-history candidate purely because a real 59-day-old interaction and a genuinely-never-seen category sit far apart on this one axis, in a direction the model had no reliable signal to calibrate correctly. Removing this (together with category_interaction_count) was measured to move the longTerm eligibility gate from a narrow failure (-0.006) to a clear, robust pass (+0.20) with no metric regression.",
    "session_category_affinity": "Sigmoid-normalized, decay-weighted behavior score for the candidate's category using only current-session-window events (same formula as category_affinity, windowed to SESSION_WINDOW and decayed). 0.5 for no session evidence or genuinely neutral session evidence; disambiguate with has_session_activity.",
    "has_session_activity": "Whether the user has any event at all in the current session window (any category). Computed unconditionally for diagnostics/reranker use but NOT part of production NUMERIC/FEATURES: it is candidate-independent and the Variant F ablation proved that feeding it to the model contributes to unrelated-category score inflation.",
    "session_average_watch_percentage": "Decay-weighted average watch percentage across session-window events (any category). Computed unconditionally for diagnostics but NOT part of production NUMERIC/FEATURES: the same global watch-quality value was passed to unrelated candidates, and the Variant F ablation proved that removing it together with has_session_activity sharply reduces rising-tide behavior without ranking-quality regression.",
    "session_positive_interaction_count": "log1p of session-window events in the candidate's category with a positive session label (watch_percentage >= 70, or liked/shared).",
    "session_negative_interaction_count": "log1p of session-window events in the candidate's category with a negative session label (watch_percentage < 20 and not liked/shared).",
    "session_category_streak": "Length of the trailing run of same-category session-window events ending at the most recent event (candidate-independent; 0 if no session activity). Valence-agnostic by design -- a run of positive engagement and a run of explicit rejections both count as a streak; see session_category_streak_valence for the signed companion. Computed unconditionally but NOT part of production NUMERIC/FEATURES -- see session_category_streak_valence_matched/_unmatched below for why: this raw value is candidate-independent (identical whether or not the candidate's own category is the one streaking), which a linear model cannot conditionally gate on last_interaction_category_match, and was the proven mechanism behind a global (candidate-independent) score-inflation defect (root-cause investigation, 2026-08-27).",
    "session_category_streak_valence": "Signed companion to session_category_streak (Phase 2.7): the trailing run of same-category session-window events ending at the most recent event, counted only while every event in the run shares the SAME session label (all positive, or all negative) -- positive for an all-positive run, negative for an all-negative run, 0 if the most recent session-window event is neutral (or there is no session activity at all). A neutral event or a sign flip stops the run (matches session_category_streak's own category-mismatch stop rule), so a neutral interaction can never be laundered into an artificial positive or negative streak. Candidate-independent, same scope as session_category_streak. Computed unconditionally but NOT part of production NUMERIC/FEATURES -- see session_category_streak_valence_matched/_unmatched, its candidate-aware split, for why.",
    "session_category_streak_valence_matched": "session_category_streak_valence when the trailing streak's category equals the CANDIDATE's own category, else 0.0 (root-cause investigation, 2026-08-27: replaces the candidate-independent session_category_streak/session_category_streak_valence in NUMERIC/FEATURES). Lets the model learn a separate weight for 'this candidate continues an active same-category streak' instead of one shared coefficient forced to explain both the matching and non-matching case identically -- the raw candidate-independent magnitude alone was proven (real frozen-artifact coefficient inspection) to inflate every candidate in a scored pool uniformly, regardless of category, whenever ANY active session streak existed.",
    "session_category_streak_valence_unmatched": "session_category_streak_valence when the trailing streak's category does NOT equal the candidate's own category, else 0.0. Computed unconditionally (still available via FeatureHistory/diagnostics) but NOT part of production NUMERIC/FEATURES as of the root-cause investigation (2026-08-27): a signed session-momentum-elsewhere value fed directly into an unrelated candidate's own score is, by construction, a candidate-independent term forced through a single linear coefficient, which cannot represent 'positive momentum elsewhere is a mild discovery signal' and 'negative momentum elsewhere is neutral, not evidence' at the same time -- proven (contribution tracing) to be ~100% of an unrelated candidate's score inflation during a purely negative session elsewhere with zero evidence of its own. A positive-only clamp and a signed positive/negative sub-component split were both tried and measured to regress the positive-session cross-category discount and/or worsen the negative-session leak -- removed from model input entirely rather than continuing to patch its sign. The product intent this fed (relevant candidates should rank higher; unrelated candidates should not be directly, numerically pushed up or down by another category's momentum) is now expected to emerge from session_category_streak_valence_matched/session_category_affinity/session_intent_confidence making the MATCHING candidate stronger, not from a separate signed penalty/bonus term on every unrelated candidate.",
    "last_interaction_category_match": "Whether the category of that trailing streak equals the candidate's category (the order-sensitive, candidate-specific half of the streak signal; also the gating condition for session_category_streak_valence_matched/_unmatched above).",
    "session_intent_confidence": "evidence_score * consistency_score over session-window events in the candidate's category -- how much the session_category_affinity reading (in either direction) should be trusted; requires repeated, recent, consistent, strong evidence, not one interaction.",
    # Negative-feedback feature representation task: additive, computed unconditionally by
    # `.features()` but NOT part of production NUMERIC/FEATURES -- see
    # app.ml.negative_feedback_features for the controlled-experiment feature groups these
    # belong to and why they exist (category/semantic explicit-rejection counters the model
    # currently has no direct access to; see this module's own constants section above).
    "category_explicit_rejection_count": "log1p of prior EXPLICIT CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED events in the category (log-scaled) -- unlike category_negative_count, never includes an ambiguous implicit fast-skip.",
    "recent_explicit_rejection_strength": "Bounded [0,1] exponential-decay recency signal for the category's most recent EXPLICIT rejection (half-life EXPLICIT_REJECTION_DECAY_HALF_LIFE_DAYS, same decay formula as the session_* features); 0.0 if the category has never been explicitly rejected.",
    "semantic_explicit_rejection_match_count": "Count of the candidate's own hashtags/topics/entities/subgenres/title-tokens that the user has EXPLICITLY rejected (CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED) at least once before -- unlike semantic_negative_match_count, never includes an ambiguous implicit fast-skip.",
    "candidate_negative_semantic_match": "Single bounded [0,1] scalar: the strongest (max, not averaged -- a single explicitly-rejected concept must not be diluted by other neutral matched tokens) explicit-rejection strength across the candidate's matched semantic tokens; 0.0 if none of the candidate's tokens were ever explicitly rejected.",
    "category_positive_minus_negative": "category_positive_count minus category_negative_count (both already log1p-scaled) -- a signed balance feature, purely derived from two existing features, no new state.",
    "semantic_positive_minus_negative": "semantic_positive_match_count minus semantic_negative_match_count -- a signed balance feature, purely derived from two existing features, no new state.",
}


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _completed(row: Any) -> bool:
    return row.event_type == "VIDEO_COMPLETED" or (row.watch_percentage or 0) >= COMPLETION_WATCH_PERCENTAGE_THRESHOLD


def _positive(row: Any) -> bool:
    return target_for(row) == 1


def _negative(row: Any) -> bool:
    # Phase 1.5 issue 3: `target_for(row) == 0` already covers CONTENT_NOT_INTERESTED/
    # LIVE_NOT_INTERESTED unconditionally now (see app.ml.feature_builder.target_for's
    # explicit-rejection precedence) -- the separate `or row.event_type == ...` clause this
    # used to need (back when target_for only recognized implicit low-watch-percentage
    # negatives) is now redundant and has been removed.
    return target_for(row) == 0


def _session_label(*, watch_percentage: float | None, liked: bool, shared: bool, not_interested: bool = False) -> int | None:
    """Session-scoped positive/negative label, restricted to the fields available on BOTH
    the DB-row path and the UserProfile.RecentWatchEvent path (see module docstring's
    "formula asymmetry" note) -- deliberately not `target_for` itself, which also uses
    favorited/creator_followed that RecentWatchEvent does not carry. Same structure and
    thresholds as target_for otherwise: positive at watch_percentage>=70 (or liked/shared),
    negative at watch_percentage<20 with no positive signal, else neutral (None).

    `not_interested` (negative-feedback session bug fix): an explicit rejection
    (CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED, or the equivalent RecentWatchEvent.notInterested
    flag) takes precedence over everything else, exactly like `target_for`'s own explicit-
    rejection precedence rule -- a user who watched most of a video before rejecting it (a high
    watch_percentage, which alone would read as strongly positive here) must not have that
    watch_percentage launder the rejection into a positive session label.

    Comment/no-watch-telemetry session bug fix: `watch_percentage is None` (no watch data at all
    -- e.g. a bare CONTENT_COMMENTED/CONTENT_LIKED event with no attached watch session) is NOT
    the same fact as a genuinely measured 0% watch, and must not be silently coerced into one.
    The `negative` branch below now requires an ACTUAL measured watch_percentage before applying
    the <20 threshold -- `positive` needs no equivalent guard: `wp=None` already fails `>=70`
    identically to `wp=0`, so liked/shared alone still correctly decide it either way."""
    if not_interested:
        return 0
    wp = watch_percentage or 0
    positive = wp >= 70 or liked or shared
    negative = watch_percentage is not None and wp < 20 and not (liked or shared)
    return 1 if positive else (0 if negative else None)


def _session_event_delta(
    *, watch_percentage: float | None, completed: bool, liked: bool, shared: bool, not_interested: bool = False,
) -> int:
    """Single weighting formula for session evidence strength -- restricted to
    watch_percentage/completed/liked/shared (see module docstring) so it can be computed
    identically from a DB Interaction row and from a UserProfile.RecentWatchEvent.

    `not_interested` (negative-feedback session bug fix -- root cause: a user who watches most
    of a video, which alone naturally sets completed=True/a high watch_percentage, and THEN
    explicitly rejects it produced a strongly POSITIVE session_category_affinity/SESSION_INTEREST
    reason for other same-category candidates, because this formula previously had no way to see
    the rejection at all -- confirmed by direct reproduction against FeatureHistory.update()/
    .features()). Mirrors `_session_label`'s and `target_for`'s explicit-rejection precedence:
    overrides completed/liked/shared entirely rather than merely offsetting them, and is a fixed
    magnitude (SESSION_NOT_INTERESTED_PENALTY) strictly larger than any single positive term
    below, the same "strictly exceeds the largest positive/ambiguous signal" convention already
    used by CONTENT_NOT_INTERESTED_CATEGORY_PENALTY/CONTENT_NOT_INTERESTED_SEMANTIC_PENALTY.

    Comment/no-watch-telemetry session bug fix (root cause: `(watch_percentage or 0) < 20`
    silently treated "no watch data at all" -- e.g. a bare CONTENT_COMMENTED event with no
    attached watch session -- identically to a genuinely measured 0% watch, fabricating a
    fast-skip penalty session evidence never actually observed). `watch_percentage is None` now
    means exactly what `_session_label` above already means by it: unknown, not a measured zero
    -- `fast_skip` can only ever be True when a real watch_percentage was supplied AND it falls
    below the threshold. A genuinely measured low/zero watch_percentage (VIDEO_SKIPPED at 0% or
    5%, etc.) is completely unaffected -- this only changes the `None` case."""
    if not_interested:
        return -SESSION_NOT_INTERESTED_PENALTY
    fast_skip = watch_percentage is not None and watch_percentage < 20
    return int(completed) * 4 + int(liked) * 3 + int(shared) * 5 - int(fast_skip) * 4


def _session_decay_weight(age_minutes: float) -> float:
    """Exponential recency decay for session-window events; see module docstring."""
    return math.exp(-math.log(2) / SESSION_DECAY_HALF_LIFE_MINUTES * max(0.0, age_minutes))


def _explicit_rejection_decay_weight(age_days: float) -> float:
    """Same exponential-decay shape as `_session_decay_weight`, parameterized in days for the
    category-level `recent_explicit_rejection_strength` feature -- see this module's
    EXPLICIT_REJECTION_DECAY_HALF_LIFE_DAYS constant for why 30 days (RECENT_WINDOW) was
    reused rather than inventing a new half-life."""
    return math.exp(-math.log(2) / EXPLICIT_REJECTION_DECAY_HALF_LIFE_DAYS * max(0.0, age_days))


class _SessionEvent(NamedTuple):
    timestamp: datetime
    category: str
    delta: int
    label: int | None  # _session_label(): 1 positive, 0 negative, None neutral/unlabeled
    watch_percentage: float | None
    # Replay/exposure saturation weight (app.ml.replay_saturation_policy), always 1.0 for a
    # non-saturation-controlled (explicit-action) row. Applied uniformly in
    # `_session_snapshot` via `effective_decay = decay * weight` -- see that function -- so a
    # weight of 0.0 means exactly zero additional session-level influence, never a change to
    # whether the event's category still counts as "the active streak".
    weight: float = 1.0


def _dedupe(tokens: list[str] | None) -> list[str]:
    """Order-preserving dedup, applied defensively wherever a token list is consumed (not
    just at the pydantic-validation boundary in app.ml.semantic_tokens) so duplicate
    hashtags/topics never inflate affinity, even for content built directly (tests, scripts)
    without going through request validation."""
    return list(dict.fromkeys(tokens or []))


def _token_fields(content: Any) -> dict[str, list[str]]:
    """Uniform semantic-token extraction for both a Content ORM row (app.db.models.Content,
    whose hashtags/topics/entities/subgenres properties decode the *_json columns) and a
    plain object exposing the same attribute names (e.g. a request-time Candidate). `None`
    (no known content) degrades to empty lists -- neutral semantic features, never a crash.

    `title` is stored as free text (not a pre-tagged list), so its tokens are derived here via
    `extract_title_tokens` -- the exact same function used for online candidate scoring (see
    `FeatureHistory.features()` below) -- rather than read directly off the object."""
    if content is None:
        return {field: [] for field in SEMANTIC_FIELDS}
    return {
        "hashtag": _dedupe(getattr(content, "hashtags", None)),
        "topic": _dedupe(getattr(content, "topics", None)),
        "entity": _dedupe(getattr(content, "entities", None)),
        "subgenre": _dedupe(getattr(content, "subgenres", None)),
        "title": extract_title_tokens(getattr(content, "title", None)),
    }


class FeatureHistory:
    """Accumulates only events that occurred before the candidate timestamp.

    `cat["watch"]` and `cat["watch_count"]`/`cat["watch_percentage_sum"]` are deliberately
    separate (Phase A): the count/sum pair is an incrementally-maintained all-time running
    total (used for `average_category_watch_percentage`/`category_completion_rate`), while
    `cat["watch"]` remains the literal per-event list (used only for the recent-30-day-window
    filter in `.features()`). They agree exactly for any history built by `.update()` --
    `watch_count == len(watch)` and `watch_percentage_sum == sum(w for _, w, _ in watch)`
    always hold there -- so this is a pure internal refactor for the legacy row-built path,
    not a behavior change. It's what lets `from_profile()` below populate the all-time totals
    from a compact pre-aggregated snapshot while only needing the *recent* window's raw
    events (naturally small, not full lifetime history) to reproduce the recent-window
    features exactly.
    """

    def __init__(self) -> None:
        self.categories: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"watch": [], "watch_count": 0, "watch_percentage_sum": 0.0,
                     "completed": 0, "positive": 0, "negative": 0, "raw": 0.0, "last": None, "interactions": 0,
                     # "recent_events": (timestamp, delta) per interaction, using the exact same
                     # per-event `delta` .update() already computes for "raw" -- filtered to the
                     # RECENT_WINDOW at read time in .features(), mirroring how "watch" is already
                     # filtered for recent_category_watch_percentage/recent_category_completion_rate.
                     # "recent_raw_override", when not None, is a pre-aggregated recent-window raw
                     # score supplied directly by from_profile() (a UserProfile snapshot has no
                     # per-event delta to replay, only the supplying service's own pre-windowed
                     # total -- see recentRawAffinityScore in app.schemas.recommendation_schemas).
                     "recent_events": [], "recent_raw_override": None,
                     # "explicit_negative"/"last_explicit_negative": EXPLICIT_NEGATIVE_EVENT_TYPES-
                     # only counterparts to "negative"/"last" above (negative-feedback feature
                     # task) -- see this module's constants section for why these exist. Safe
                     # defaults (0/None) for history built via from_profile() from a profile that
                     # omits the corresponding optional fields, exactly like every other
                     # additive profile field this codebase already established.
                     "explicit_negative": 0, "last_explicit_negative": None}
        )
        self.creators: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"interactions": 0.0, "completed": 0.0}
        )
        # One independent affinity store per semantic field, keyed by (user_id, token) --
        # deliberately not merged into one namespace, since "MESSI" as an entity and a
        # same-named hashtag are tracked (and reported back in entity_affinity vs
        # hashtag_affinity) separately. Stays empty for history built via `from_profile()`
        # (see module docstring "Known limitation"), which is exactly what makes semantic
        # features correctly default to neutral for that path.
        self.tokens: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
            field: defaultdict(lambda: {"raw": 0.0, "positive": 0, "negative": 0, "interactions": 0,
                                         "explicit_negative": 0})
            for field in SEMANTIC_FIELDS
        }
        self.users: dict[str, float] = defaultdict(float)
        self.followed_creators: set[tuple[str, str]] = set()
        self.seen: set[tuple[str, str]] = set()
        # Replay/exposure saturation (app.ml.replay_saturation_policy): counts only
        # saturation-controlled (passive) occurrences per (user_id, content_id) -- an explicit
        # action on the same content neither advances nor is affected by this tally. Stays
        # empty for history built via `from_profile()` (a UserProfile snapshot has no
        # per-content history to replay), which is an accepted, documented limitation of that
        # path -- see `from_profile`'s own docstring.
        self.content_occurrences: dict[tuple[str, str], int] = defaultdict(int)
        # Per-user, timestamp-ordered log of recent events, bounded to roughly one
        # SESSION_WINDOW per user (see update()'s eviction below) -- backs the session_*
        # features. Kept separate from `categories` (per (user, category), unordered running
        # aggregates): session features need real cross-category event order and real
        # timestamps, not just accumulated per-category sums.
        self.session_events: dict[str, list[_SessionEvent]] = defaultdict(list)

    def _session_snapshot(self, *, user_id: str, category: str, timestamp: datetime) -> dict[str, float | int]:
        """Current-session-intent features; see module docstring for the full design. Reads
        only `self.session_events[user_id]`, which `update()`/`from_profile()` populate --
        point-in-time correct as long as callers call this before `update()` for the current
        row (see build_dataset/build_feature_rows, which already follow that order for the
        rest of `.features()`)."""
        ts = _utc(timestamp)
        log = self.session_events.get(user_id, [])
        window = [event for event in log if ts - event.timestamp <= SESSION_WINDOW]
        window = window[-SESSION_MAX_EVENTS:]

        watch_weighted_sum = 0.0
        watch_weight_total = 0.0
        raw_category_score = 0.0
        category_evidence_weight = 0.0
        category_signed_weight = 0.0
        positive_count = 0.0
        negative_count = 0.0
        for event in window:
            age_minutes = (ts - event.timestamp).total_seconds() / 60
            decay = _session_decay_weight(age_minutes)
            # Replay/exposure saturation (app.ml.replay_saturation_policy): `event.weight` is
            # 1.0 for every existing caller/scenario (every explicit-action row, and every
            # passive row that is still within its first two occurrences of this content) --
            # `effective_decay == decay` exactly in that case, so this is a pure no-op unless a
            # content has actually been passively replayed 3+ times. A weight of 0.0 (6th+
            # passive replay of the SAME content) contributes exactly zero to every
            # accumulator below, without changing whether the event's category still
            # participates in the streak walk further down.
            effective_decay = decay * event.weight
            if event.watch_percentage is not None:
                watch_weighted_sum += effective_decay * event.watch_percentage
                watch_weight_total += effective_decay
            if event.category != category:
                continue
            raw_category_score += effective_decay * event.delta
            magnitude = abs(event.delta) + (SESSION_LABEL_EVIDENCE_FLOOR if event.label is not None else 0.0)
            weight = effective_decay * magnitude
            category_evidence_weight += weight
            sign = 1 if event.label == 1 else (-1 if event.label == 0 else 0)
            category_signed_weight += weight * sign
            if event.label == 1:
                positive_count += event.weight
            elif event.label == 0:
                negative_count += event.weight

        streak = 0.0
        streak_category: str | None = None
        for event in reversed(window):
            if streak_category is None:
                streak_category, streak = event.category, event.weight
            elif event.category == streak_category:
                streak += event.weight
            else:
                break

        # Phase 2.7: session_category_streak_valence -- a signed companion to `streak` above.
        # Root cause this fixes (measured, not guessed): `streak` only counts run LENGTH, blind
        # to whether the run is positive or negative, so a 4-event run of explicit rejections
        # looks identical to a 4-event binge-watching run -- at scale, LogisticRegression learns
        # "long same-category streak correlates with positive" (true for the common,
        # engagement-driven case) and applies that to negative streaks too, which is exactly
        # backwards. This walks the SAME trailing same-category window `streak` does, but also
        # requires every event in the run to share the same session label (all positive, or all
        # negative); a neutral-labeled event or a sign flip stops the run immediately, the same
        # way a category change already stops `streak` -- so a neutral interaction can never be
        # laundered into an artificial positive or negative streak.
        streak_valence = 0.0
        valence_sign: int | None = None
        valence_run_category: str | None = None
        for event in reversed(window):
            if valence_run_category is None:
                valence_run_category = event.category
                if event.label == 1:
                    valence_sign = 1
                elif event.label == 0:
                    valence_sign = -1
                else:
                    break  # most recent session-window event is neutral -> no valence streak
                streak_valence = valence_sign * event.weight
                continue
            # valence_sign is always set here: the only way to reach this branch is a prior
            # iteration having taken the `valence_run_category is None` branch above, which
            # either sets valence_sign or breaks out of the loop entirely.
            assert valence_sign is not None
            expected_label = 1 if valence_sign == 1 else 0
            if event.category != valence_run_category or event.label != expected_label:
                break
            streak_valence += valence_sign * event.weight

        evidence_score = min(1.0, category_evidence_weight / SESSION_CONFIDENCE_SATURATION_WEIGHT)
        consistency_score = abs(category_signed_weight) / category_evidence_weight if category_evidence_weight else 0.0
        last_interaction_category_match = int(streak_category == category) if streak_category is not None else 0

        return {
            "session_category_affinity": affinity_score(raw_category_score),
            "has_session_activity": int(bool(window)),
            "session_average_watch_percentage": watch_weighted_sum / watch_weight_total if watch_weight_total else 0.0,
            "session_positive_interaction_count": math.log1p(positive_count),
            "session_negative_interaction_count": math.log1p(negative_count),
            # session_category_streak/_valence: candidate-independent, kept for diagnostics
            # (see their FEATURE_DEFINITIONS entries) but no longer selected into NUMERIC --
            # session_category_streak_valence_matched/_unmatched below is what the model
            # actually trains on.
            "session_category_streak": streak,
            "session_category_streak_valence": streak_valence,
            "session_category_streak_valence_matched": streak_valence if last_interaction_category_match else 0,
            "session_category_streak_valence_unmatched": 0 if last_interaction_category_match else streak_valence,
            "last_interaction_category_match": last_interaction_category_match,
            "session_intent_confidence": evidence_score * consistency_score,
        }

    def features(
        self,
        *,
        user_id: str,
        category: str,
        creator_id: str,
        content_id: str,
        timestamp: datetime,
        content_popularity_score: float,
        content_created_at: datetime,
        creator_followed: bool = False,
        already_seen: bool = False,
        hashtags: list[str] | None = None,
        topics: list[str] | None = None,
        entities: list[str] | None = None,
        subgenres: list[str] | None = None,
        title: str | None = None,
    ) -> dict[str, float | str]:
        ts = _utc(timestamp)
        category = category.upper()
        cat = self.categories[(user_id, category)]
        creator = self.creators[(user_id, creator_id)]
        watches = cat["watch"]
        recent = [(t, w, c, wt) for t, w, c, wt in watches if ts - t <= RECENT_WINDOW]
        # See __init__: a profile-supplied recent affinity is used verbatim (it is already
        # pre-windowed by the supplying service); a row-built history instead sums only the
        # events that actually fall inside RECENT_WINDOW of *this* candidate's timestamp,
        # using the exact same per-event `delta` .update() computed for "raw" -- one weighting
        # formula, applied to two different time windows.
        recent_raw = (
            cat["recent_raw_override"] if cat["recent_raw_override"] is not None
            else sum(delta for t, delta in cat["recent_events"] if ts - t <= RECENT_WINDOW)
        )
        # Replay/exposure saturation (app.ml.replay_saturation_policy): each recent-window
        # entry's own weight turns these two ratios into weighted averages -- 1.0 for every
        # existing scenario (every entry within its content's first two occurrences, or from
        # an explicit action), so this is byte-identical to the old plain average unless a
        # content has actually been passively replayed 3+ times within the window.
        recent_weight_total = sum(wt for _, _, _, wt in recent)
        last = cat["last"]
        content_age_hours = max(0.0, (ts - _utc(content_created_at)).total_seconds() / 3600)

        # `title` is free text; the exact same extract_title_tokens() used at ingestion and
        # dataset-construction time (_token_fields, above) tokenizes it here too, so online
        # scoring and training build identical title-token features.
        candidate_fields = {
            "hashtag": _dedupe(hashtags), "topic": _dedupe(topics), "entity": _dedupe(entities),
            "subgenre": _dedupe(subgenres), "title": extract_title_tokens(title),
        }
        matched_affinities: list[float] = []
        positive_matches = 0
        negative_matches = 0
        # Negative-feedback feature task: explicit-rejection-specific counterparts to
        # positive_matches/negative_matches above, plus `candidate_negative_semantic_match`'s
        # single bounded-scalar aggregation (see FEATURE_DEFINITIONS/module docstring).
        explicit_rejection_matches = 0
        strongest_negative_match_strength = 0.0
        field_affinities: dict[str, float] = {}
        for field, field_tokens in candidate_fields.items():
            store = self.tokens[field]
            matched_for_field: list[float] = []
            for token in field_tokens:
                entry = store.get((user_id, token))
                if entry is None or entry["interactions"] == 0:
                    continue
                affinity = affinity_score(entry["raw"])
                matched_for_field.append(affinity)
                matched_affinities.append(affinity)
                if entry["positive"] > entry["negative"]:
                    positive_matches += 1
                elif entry["negative"] > entry["positive"]:
                    negative_matches += 1
                if entry["explicit_negative"] > 0:
                    explicit_rejection_matches += 1
                    # Deviation below neutral (0.5), doubled into [0, 1] -- 0.0 for an
                    # explicitly-rejected token whose affinity has not (yet) fallen below
                    # neutral, up to 1.0 as affinity approaches 0. MAX across matched tokens,
                    # not averaged (Task 6 spec section 6/7): a single strongly-rejected
                    # subtheme must not be diluted by other, unrelated matched tokens.
                    strength = max(0.0, (0.5 - affinity) * 2.0)
                    strongest_negative_match_strength = max(strongest_negative_match_strength, strength)
            field_affinities[field] = sum(matched_for_field) / len(matched_for_field) if matched_for_field else 0.5

        session = self._session_snapshot(user_id=user_id, category=category, timestamp=ts)

        category_positive_count = math.log1p(cat["positive"])
        category_negative_count = math.log1p(cat["negative"])
        last_explicit_negative = cat["last_explicit_negative"]
        explicit_rejection_age_days = (
            (ts - last_explicit_negative).total_seconds() / 86400 if last_explicit_negative else None
        )

        return {
            "category": category,
            "category_affinity": affinity_score(cat["raw"]),
            "recent_category_affinity": affinity_score(recent_raw),
            "has_category_history": int(cat["interactions"] > 0),
            "average_category_watch_percentage": cat["watch_percentage_sum"] / cat["watch_count"] if cat["watch_count"] else 0.0,
            "recent_category_watch_percentage": (
                sum(wt * w for _, w, _, wt in recent) / recent_weight_total if recent_weight_total else 0.0
            ),
            "category_completion_rate": cat["completed"] / cat["watch_count"] if cat["watch_count"] else 0.0,
            "recent_category_completion_rate": (
                sum(wt * c for _, _, c, wt in recent) / recent_weight_total if recent_weight_total else 0.0
            ),
            "category_positive_count": category_positive_count,
            "category_negative_count": category_negative_count,
            "category_interaction_count": math.log1p(cat["interactions"]),
            "has_creator_history": int(creator["interactions"] > 0),
            "creator_interaction_count": math.log1p(creator["interactions"]),
            "creator_completion_rate": creator["completed"] / creator["interactions"] if creator["interactions"] else 0.0,
            "creator_followed": int(creator_followed or (user_id, creator_id) in self.followed_creators),
            "hashtag_affinity": field_affinities["hashtag"],
            "topic_affinity": field_affinities["topic"],
            "entity_affinity": field_affinities["entity"],
            "subgenre_affinity": field_affinities["subgenre"],
            "title_affinity": field_affinities["title"],
            "semantic_positive_match_count": positive_matches,
            "semantic_negative_match_count": negative_matches,
            "has_semantic_history": int(bool(matched_affinities)),
            "strongest_semantic_affinity": max(matched_affinities, key=lambda a: abs(a - 0.5)) if matched_affinities else 0.5,
            "average_semantic_affinity": sum(matched_affinities) / len(matched_affinities) if matched_affinities else 0.5,
            "user_total_interaction_count": math.log1p(self.users[user_id]),
            "content_popularity_score": max(0.0, min(1.0, float(content_popularity_score))),
            "content_age_hours": min(MAX_CONTENT_AGE_HOURS, content_age_hours),
            "already_seen": int(already_seen or (user_id, content_id) in self.seen),
            "hour_of_day": ts.hour,
            # 0.0 (not 365.0) for no prior category interaction at all: 365.0 is the cap for
            # genuinely-old-but-real history, and reusing it as the "no history" sentinel made
            # a total stranger's total absence of evidence indistinguishable from -- and, via
            # the model's own learned coefficient, systematically favored over -- a real user
            # with real (if long-ago) history (root-cause investigation, 2026-08-27). Matches
            # every other no-history default already used elsewhere in this same function
            # (average_category_watch_percentage, recent_category_watch_percentage,
            # category_completion_rate, recent_category_completion_rate all default to 0.0 for
            # no data, disambiguated via has_category_history) -- this was the one outlier.
            # Timestamp-hardening finding: a prior interaction's own timestamp can legitimately
            # be up to MAX_FUTURE_EVENT_SKEW_SECONDS ahead of wall-clock time (see
            # app.schemas.event_schemas), so `ts` (this scoring moment) can be earlier than
            # `last` if scoring happens before that skew has elapsed -- clamped at 0.0 (never
            # negative), matching content_age_hours' own `max(0.0, ...)` clamp above.
            "days_since_last_category_interaction": (
                min(365.0, max(0.0, (ts - last).total_seconds() / 86400)) if last else 0.0
            ),
            **session,
            # Negative-feedback feature task (not yet part of production NUMERIC/FEATURES --
            # see app.ml.negative_feedback_features): computed unconditionally so a controlled
            # experiment can select them without a second feature-computation path.
            "category_explicit_rejection_count": math.log1p(cat["explicit_negative"]),
            "recent_explicit_rejection_strength": (
                _explicit_rejection_decay_weight(explicit_rejection_age_days)
                if explicit_rejection_age_days is not None else 0.0
            ),
            "semantic_explicit_rejection_match_count": explicit_rejection_matches,
            "candidate_negative_semantic_match": strongest_negative_match_strength,
            "category_positive_minus_negative": category_positive_count - category_negative_count,
            "semantic_positive_minus_negative": positive_matches - negative_matches,
        }

    def _is_repeat_negative_subtheme(self, user_id: str, content: Any) -> bool:
        """True if ANY of `content`'s own semantic tokens already carry net-negative prior
        history for this user (interactions > 0 and negative > positive) -- i.e. this explicit-
        rejection event (CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED) targets a subtheme the
        user has already, separately, shown to dislike, not a newly-implicated one. False
        (never a repeat) when `content`
        carries no semantic metadata at all -- there is nothing to localize against, so the
        caller falls back to the full category-level penalty, matching pre-fix behavior."""
        for field, field_tokens in _token_fields(content).items():
            store = self.tokens[field]
            for token in field_tokens:
                entry = store.get((user_id, token))
                if entry is not None and entry["interactions"] > 0 and entry["negative"] > entry["positive"]:
                    return True
        return False

    def update(self, row: Any, *, content: Any = None) -> None:
        ts = _utc(row.timestamp)
        signals = interaction_signals(row)
        key = (row.user_id, row.category.upper())
        cat = self.categories[key]
        completed = _completed(row)
        # Replay/exposure saturation (app.ml.replay_saturation_policy): `weight` is 1.0 for
        # every non-saturation-controlled (explicit-action) row, and for a passive row's own
        # first two occurrences of this exact content_id -- i.e. byte-identical to before this
        # fix for every scenario except a content passively replayed 3+ times by the same
        # user. Computed ONCE per row and applied uniformly below to every accumulator this
        # row would otherwise contribute a full, unweighted increment to (category/creator/
        # semantic/session/user-total) -- content_occurrences only advances for saturation-
        # controlled rows, so an explicit action never consumes/affects a content's tally.
        weight = 1.0
        if is_saturation_controlled_interaction(row):
            content_key = (row.user_id, row.content_id)
            self.content_occurrences[content_key] += 1
            weight = repeat_influence_weight(self.content_occurrences[content_key])
        if row.watch_percentage is not None:
            cat["watch"].append((ts, float(row.watch_percentage), int(completed), weight))
            cat["watch_count"] += weight
            cat["watch_percentage_sum"] += weight * float(row.watch_percentage)
        cat["completed"] += weight * int(completed)
        positive = weight * int(_positive(row))
        negative = weight * int(_negative(row))
        cat["positive"] += positive
        cat["negative"] += negative
        cat["interactions"] += weight
        # Negative-feedback feature task: EXPLICIT_NEGATIVE_EVENT_TYPES-only counters. Point-
        # in-time safe: reads only this row's own already-happened event_type/timestamp.
        is_explicit_rejection = row.event_type in EXPLICIT_NEGATIVE_EVENT_TYPES
        if is_explicit_rejection:
            cat["explicit_negative"] += 1
            cat["last_explicit_negative"] = ts
        # Single weighting formula, applied identically to the category and the creator.
        # CONTENT_NOT_INTERESTED_CATEGORY_PENALTY (-8) is the largest single weight here,
        # deliberately: a broad-category signal absorbed into a user's existing interaction
        # history is naturally diluted by volume (forensic trace confirmed this stayed
        # appropriately gradual: 0.900 -> 0.802 SPORT affinity after ONE event among 17 SPORT
        # interactions). That full weight is still applied every time a NOT_INTERESTED event
        # implicates a genuinely new subtheme -- see CONTENT_NOT_INTERESTED_CATEGORY_REPEAT_
        # PENALTY above for why a REPEAT rejection of an already-disliked subtheme instead
        # uses the smaller, dampened weight (localization fix).
        # Was literal "CONTENT_NOT_INTERESTED"-only; widened to match EXPLICIT_NEGATIVE_
        # EVENT_TYPES (CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED) -- the two were already
        # treated identically by app.ml.feature_builder.target_for (training label) and
        # app.ml.sample_weight_policy (row weight), and this affinity/semantic-suppression
        # path is the mechanism the HARD negative/notInterested eligibility gates actually key
        # on, so a LIVE_NOT_INTERESTED rejection was silently invisible to real-time
        # suppression despite being labeled/weighted as the strongest explicit-rejection signal
        # everywhere else. `content` lookup already degrades to no-op for missing metadata
        # (see _token_fields), so this is a pure widening, not a new code path.
        is_not_interested = row.event_type in EXPLICIT_NEGATIVE_EVENT_TYPES
        base_delta = (
            int(completed) * 4 + int(signals.liked) * 3 + int(signals.shared) * 5
            + int(signals.favorited) * 5 + int(signals.commented) * 2 + int(signals.creator_followed) * 6
            + int(row.event_type == "VIDEO_REWATCHED") * 4
            - int(row.event_type == "VIDEO_SKIPPED" and (row.watch_percentage or 0) < FAST_SKIP_WATCH_PERCENTAGE_THRESHOLD) * 4
        )
        # Checked BEFORE this event's own tokens are folded in below (point-in-time: "has the
        # user already rejected this subtheme", not "including this event").
        category_penalty = 0
        if is_not_interested:
            category_penalty = (
                CONTENT_NOT_INTERESTED_CATEGORY_REPEAT_PENALTY
                if self._is_repeat_negative_subtheme(row.user_id, content)
                else CONTENT_NOT_INTERESTED_CATEGORY_PENALTY
            )
        # Explicit-rejection precedence fix (behavioral audit finding): an explicit rejection is
        # an OVERRIDE, not an offset -- base_delta (completed/liked/shared/favorited/commented/
        # creator_followed, all real fields on the SAME row a client is technically free to set
        # alongside CONTENT_NOT_INTERESTED/LIVE_NOT_INTERESTED, e.g. a high watch_percentage from
        # watching most of a video before rejecting it) must never be added back in here. Mirrors
        # `_session_event_delta`'s own `if not_interested: return -SESSION_NOT_INTERESTED_PENALTY`
        # override (never `base_delta - SESSION_NOT_INTERESTED_PENALTY`) and `target_for`'s own
        # unconditional `if row.event_type in EXPLICIT_NEGATIVE_EVENT_TYPES: return 0` precedence
        # -- this long-term category/semantic layer was the one place that still computed
        # `base_delta - penalty` instead, letting enough positive flags on a rejection event
        # dilute or even invert the category/semantic affinity contribution (confirmed: 95% watch
        # + liked + shared + CONTENT_NOT_INTERESTED raised category_affinity instead of lowering
        # it). `base_delta` is still computed above (VIDEO_REWATCHED/VIDEO_SKIPPED terms never
        # apply to a rejection row anyway, since event_type is mutually exclusive), but is simply
        # never consumed on this branch. A non-rejection row is completely unaffected:
        # category_penalty is 0 there, so `-category_penalty` down below would be wrong -- the
        # branch below reads `base_delta` on that side exactly as before.
        # Semantic tokens (hashtag/topic/entity/subgenre/title) get their OWN, smaller
        # CONTENT_NOT_INTERESTED weight instead of the full category-level penalty (forensic
        # trace root cause): the full -8 used to be applied INDEPENDENTLY to every matched
        # token, so one rejection on a content item carrying ~15 tokens produced ~15
        # simultaneous, maximally-negative semantic features from a single click -- an
        # out-of-distribution pattern the model had only ever seen from sustained rejection.
        # CONTENT_NOT_INTERESTED_SEMANTIC_PENALTY (-5, calibrated to exceed a fast skip's own
        # -4 per-token contribution -- see the constant's own comment) makes a first rejection
        # read as moderate-to-strong semantic evidence (sigmoid(-5/10) = 0.3775); the existing
        # raw-sum accumulation mechanism (unchanged) naturally makes repeated rejections of the
        # same token progressively stronger (-5, -10, -15, ...) without any new state machine.
        # Every other event type's semantic contribution is unchanged (base_delta == semantic_delta).
        # Same override precedence as `delta` above -- see this block's comment there.
        # `delta`/`semantic_delta` are only ever `base_delta` (never `-category_penalty`/
        # `-CONTENT_NOT_INTERESTED_SEMANTIC_PENALTY`) on the branch `weight` can be anything
        # other than 1.0 -- a rejection row is never saturation-controlled (see `weight`'s own
        # comment above), so multiplying by `weight` here is a genuine scaling only for the
        # base_delta branch, and a provable no-op (weight is exactly 1.0) on the rejection
        # branch -- that already-tuned penalty formula is otherwise completely untouched.
        delta = -category_penalty if is_not_interested else base_delta * weight
        semantic_delta = -CONTENT_NOT_INTERESTED_SEMANTIC_PENALTY if is_not_interested else base_delta * weight
        cat["raw"] += delta
        cat["recent_events"].append((ts, delta))
        cat["last"] = ts
        creator = self.creators[(row.user_id, row.creator_id)]
        creator["interactions"] += weight
        creator["completed"] += weight * int(completed)
        if signals.creator_followed:
            self.followed_creators.add((row.user_id, row.creator_id))
        for field, field_tokens in _token_fields(content).items():
            store = self.tokens[field]
            for token in field_tokens:
                entry = store[(row.user_id, token)]
                entry["raw"] += semantic_delta
                entry["positive"] += positive
                entry["negative"] += negative
                entry["interactions"] += weight
                if is_explicit_rejection:
                    entry["explicit_negative"] += 1
        self.seen.add((row.user_id, row.content_id))
        self.users[row.user_id] += weight

        # is_explicit_rejection (computed above for the long-term category/semantic penalties)
        # reused here: the session layer must see the same explicit-rejection signal the
        # long-term layer already does, not just watch_percentage/completed/liked/shared --
        # see _session_label/_session_event_delta docstrings for the bug this fixes.
        session_label = _session_label(
            watch_percentage=row.watch_percentage, liked=signals.liked, shared=signals.shared,
            not_interested=is_explicit_rejection,
        )
        session_delta = _session_event_delta(
            watch_percentage=row.watch_percentage, completed=completed,
            liked=signals.liked, shared=signals.shared, not_interested=is_explicit_rejection,
        )
        session_log = self.session_events[row.user_id]
        session_log.append(_SessionEvent(ts, row.category.upper(), session_delta, session_label, row.watch_percentage, weight))
        # Bound memory to roughly one SESSION_WINDOW per user: events are processed in strict
        # chronological order (build_dataset/build_feature_rows/history_from_rows sort by
        # timestamp), so an entry older than SESSION_WINDOW relative to the event that just
        # extended the log can never again fall inside any *future* event's session window.
        cutoff = ts - SESSION_WINDOW
        while session_log and session_log[0].timestamp < cutoff:
            session_log.pop(0)

    @classmethod
    def from_profile(cls, user_id: str, profile: Any) -> FeatureHistory:
        """Phase A: reconstructs the same internal accumulator state `.update()` would have
        built from raw interaction rows, from a pre-aggregated `UserProfile` snapshot
        instead (see app.schemas.recommendation_schemas.UserProfile) -- `.features()` above
        is not touched or duplicated; this only populates the state it already reads.

        `profile` is duck-typed (not imported by type) to avoid a circular import between
        this module (already imported by app.schemas.recommendation_schemas for
        MAX_CONTENT_AGE_HOURS) and the schemas module that defines UserProfile.

        Semantic features (finalization spec Decision 5): `self.tokens` is populated directly
        from `profile.semantic_affinities`, when supplied -- one pre-aggregated (field, token)
        accumulator entry per prior hashtag/topic/entity/subgenre/title-token the user has
        history with, mirroring exactly how the `categories`/`creators` loops above populate
        their own accumulators from pre-aggregated counters rather than replaying raw events.
        A profile that omits `semantic_affinities` (the default, and every pre-Decision-5
        caller) leaves `self.tokens` empty, which is what makes semantic features correctly
        default to neutral (0.5) for that request -- unchanged behavior for such callers.
        """
        history = cls()
        history.users[user_id] = profile.total_interaction_count
        for category_profile in profile.categories:
            cat = history.categories[(user_id, category_profile.category.upper())]
            cat["interactions"] = category_profile.interaction_count
            cat["positive"] = category_profile.positive_count
            cat["negative"] = category_profile.negative_count
            cat["completed"] = category_profile.completed_count
            cat["watch_count"] = category_profile.watch_count
            cat["watch_percentage_sum"] = category_profile.watch_percentage_sum
            cat["raw"] = category_profile.raw_affinity_score
            # Pre-windowed by the supplying service, exactly like raw_affinity_score above --
            # used verbatim by .features() via "recent_raw_override" rather than replayed
            # through the recent_events/RECENT_WINDOW filter (there are no per-event deltas
            # on this path to filter). Absent on an older/pre-Phase-2 caller, this defaults to
            # 0.0 (schema default), which affinity_score() maps to a neutral 0.5 -- the same
            # safe cold-start behavior every other never-supplied signal already gets here.
            cat["recent_raw_override"] = category_profile.recent_raw_affinity_score
            cat["last"] = _utc(category_profile.last_interaction_at) if category_profile.last_interaction_at else None
            # Negative-feedback feature task: additive, optional fields (see
            # app.schemas.recommendation_schemas.CategoryProfile) -- `getattr` with the same
            # safe defaults the schema itself uses, so a duck-typed profile object (tests) or
            # a pre-this-task caller that omits them degrades to "never explicitly rejected"
            # rather than crashing, exactly like recentRawAffinityScore's own precedent above.
            explicit_negative_count = getattr(category_profile, "explicit_negative_count", 0)
            last_explicit_negative_at = getattr(category_profile, "last_explicit_negative_at", None)
            cat["explicit_negative"] = explicit_negative_count
            cat["last_explicit_negative"] = _utc(last_explicit_negative_at) if last_explicit_negative_at else None
            # Only the recent (~30-day) window's raw events are ever sent -- sufficient and
            # correct for the recent-window features in `.features()`, which only ever reads
            # events within RECENT_WINDOW of the scoring timestamp from this list.
            # weight=1.0 (4th tuple element, see FeatureHistory.update()): a UserProfile
            # snapshot has no per-content history to derive a replay-saturation occurrence
            # count from -- every profile-supplied watch event is treated as full-weight,
            # unsaturated evidence, the documented limitation of this path (see module
            # docstring "Semantic features on the UserProfile snapshot path" and
            # app.ml.replay_saturation_policy's own module docstring).
            cat["watch"] = [
                (_utc(event.timestamp), event.watch_percentage, int(event.completed), 1.0)
                for event in category_profile.recent_watch_events
            ]
            # Session reconstruction: recentWatchEvents is already bounded to RECENT_WINDOW
            # (30 days) per category, well beyond SESSION_WINDOW (30 minutes) -- stored here in
            # full, exactly like cat["watch"] above, and narrowed to the true current-session
            # slice at read time by _session_snapshot()'s own SESSION_WINDOW/SESSION_MAX_EVENTS
            # filtering. Each event is tagged with its parent category_profile's category,
            # since RecentWatchEvent itself carries no category (it's implicit in which
            # CategoryProfile list it's nested under).
            for event in category_profile.recent_watch_events:
                # Negative-feedback session bug fix: `not_interested`, like explicit_negative_
                # count above, is an additive/optional RecentWatchEvent field -- `getattr` with
                # a safe False default so a duck-typed profile object (tests) or a pre-fix
                # caller degrades to the (still buggy, but unchanged) old behavior rather than
                # crashing, exactly like recentRawAffinityScore's own precedent above.
                event_not_interested = getattr(event, "not_interested", False)
                label = _session_label(
                    watch_percentage=event.watch_percentage, liked=event.liked, shared=event.shared,
                    not_interested=event_not_interested,
                )
                delta = _session_event_delta(
                    watch_percentage=event.watch_percentage, completed=event.completed,
                    liked=event.liked, shared=event.shared, not_interested=event_not_interested,
                )
                history.session_events[user_id].append(
                    _SessionEvent(_utc(event.timestamp), category_profile.category.upper(), delta, label, event.watch_percentage)
                )
        history.session_events[user_id].sort(key=lambda event: event.timestamp)
        for creator_profile in profile.creators:
            creator = history.creators[(user_id, creator_profile.creator_id)]
            creator["interactions"] = creator_profile.interaction_count
            creator["completed"] = creator_profile.completed_count
        history.followed_creators = {(user_id, creator_id) for creator_id in profile.followed_creator_ids}
        history.seen = {(user_id, content_id) for content_id in profile.seen_content_ids}
        for semantic_entry in getattr(profile, "semantic_affinities", []):
            token_entry = history.tokens[semantic_entry.field][(user_id, semantic_entry.token)]
            token_entry["raw"] = semantic_entry.raw_affinity_score
            token_entry["positive"] = semantic_entry.positive_count
            token_entry["negative"] = semantic_entry.negative_count
            token_entry["interactions"] = semantic_entry.interaction_count
            token_entry["explicit_negative"] = getattr(semantic_entry, "explicit_negative_count", 0)
        return history


def _same_timestamp_update_key(row: Any) -> tuple[str, ...]:
    """Canonical post-snapshot order for simultaneous events.

    Events in one timestamp batch never observe one another.  A deterministic update order is
    still required because ordered session state (notably the trailing category streak) is read
    by later timestamps.  Event id is the normal unique tie-breaker; the remaining fields keep
    duck-typed/test rows deterministic when no event id is present.
    """
    return tuple(str(getattr(row, field, "")) for field in (
        "user_id", "event_id", "content_id", "creator_id", "category", "event_type",
    ))


def _timestamp_batches(rows: Iterable[Any]) -> Iterable[list[Any]]:
    # Python's stable sort preserves the caller's row order inside a timestamp for output
    # compatibility. State updates are canonicalized separately after every batch snapshot.
    ordered = sorted(rows, key=lambda row: _utc(row.timestamp))
    for _, timestamp_rows in groupby(ordered, key=lambda row: _utc(row.timestamp)):
        yield list(timestamp_rows)


def history_from_rows(rows: Iterable[Any], content_by_id: dict[str, Any] | None = None) -> FeatureHistory:
    """`content_by_id`, when given, supplies each row's semantic tokens (hashtags/topics/
    entities/subgenres/title-tokens) so the resulting history's semantic-affinity state
    reflects real prior behavior, not just category/creator aggregates. Optional and defaults
    to None (no semantic history) for existing callers that only need category/creator
    features."""
    content_by_id = content_by_id or {}
    history = FeatureHistory()
    for timestamp_rows in _timestamp_batches(rows):
        for row in sorted(timestamp_rows, key=_same_timestamp_update_key):
            history.update(row, content=content_by_id.get(row.content_id))
    return history


def build_feature_rows(
    rows: Iterable[Any], content_by_id: dict[str, Any] | None = None, *, warmup_rows: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """Point-in-time FEATURES for every row in `rows`, regardless of label.

    Unlike `build_dataset()` above (which only emits a row when `target_for()` is not
    None -- i.e. only training-eligible positive/negative events), this keeps every row,
    including neutral/unlabeled ones. Drift monitoring (app.ml.drift_detector) cares about
    the feature *distribution* of recent activity as a whole, not just the subset that
    happens to carry a training label -- filtering to labeled-only rows here would silently
    bias a drift sample away from the true recent traffic shape. Reuses the exact same
    `FeatureHistory` point-in-time engine as `build_dataset()`, so this is not a second,
    parallel feature-computation path that could drift from what training/serving compute.

    `warmup_rows` (corrective pass, Finding 2): optional prior interactions used only to
    prime `FeatureHistory`'s accumulator state via `.update()` *before* any row in `rows` is
    processed -- they never contribute a feature row of their own. This exists because most
    VIDEO features are history-dependent and a bare, freshly-initialized `FeatureHistory`
    would silently reconstruct every established user as an artificial cold start purely
    because their real history happens to fall outside the observation window. Per-feature
    audit of `FEATURES` (see `app.services.drift_service.video_drift_from_recent_interactions`,
    the only caller that supplies `warmup_rows`):

    - History-dependent (need `warmup_rows` for a correct value on an established user):
      `category_affinity`, `has_category_history`, `average_category_watch_percentage`,
      `category_completion_rate`, `category_positive_count`, `category_negative_count`,
      `category_interaction_count`, `has_creator_history`, `creator_interaction_count`,
      `creator_completion_rate`, `creator_followed`, `user_total_interaction_count`,
      `already_seen`, `days_since_last_category_interaction` (all-time accumulators), plus
      `recent_category_watch_percentage`/`recent_category_completion_rate`/
      `recent_category_affinity` (only the last `RECENT_WINDOW` -- 30 days -- of history, so
      these need `warmup_rows` specifically whenever the observation window itself spans
      less than 30 days), plus `session_category_affinity`/`has_session_activity`/
      `session_average_watch_percentage`/`session_positive_interaction_count`/
      `session_negative_interaction_count`/`session_category_streak`/
      `last_interaction_category_match`/`session_intent_confidence` (only the last
      `SESSION_WINDOW` -- 30 minutes -- so in practice these only need `warmup_rows` for rows
      within the first 30 minutes of the observation window; any drift sample whose
      observation window is itself longer than 30 minutes already self-warms for every later
      row without needing warmup_rows at all).
    - History-independent (correct with or without `warmup_rows`): `content_popularity_score`
      and `content_age_hours` (read from the `Content` catalog and the row's own timestamp,
      not from interaction history at all), `hour_of_day` (the row's own timestamp), and the
      categorical `category` (the row's own value).

    Warm-up rows must be chronologically prior to every row in `rows` for the same
    point-in-time integrity `build_dataset()`/`history_from_rows()` already rely on; the
    caller (app.services.drift_service) enforces this via the observation window's own
    `(timestamp, id)` boundary (app.db.repositories.warmup_interactions_for_drift).
    """
    history = FeatureHistory()
    content_by_id = content_by_id or {}
    for timestamp_rows in _timestamp_batches(warmup_rows or []):
        for row in sorted(timestamp_rows, key=_same_timestamp_update_key):
            history.update(row, content=content_by_id.get(row.content_id))
    result: list[dict[str, Any]] = []
    for timestamp_rows in _timestamp_batches(rows):
        for row in timestamp_rows:
            content = content_by_id.get(row.content_id)
            token_fields = _token_fields(content)
            features = history.features(
                user_id=row.user_id,
                category=row.category,
                creator_id=row.creator_id,
                content_id=row.content_id,
                timestamp=row.timestamp,
                content_popularity_score=getattr(content, "popularity_score", 0.5),
                content_created_at=getattr(content, "created_at", row.timestamp),
                hashtags=token_fields["hashtag"],
                topics=token_fields["topic"],
                entities=token_fields["entity"],
                subgenres=token_fields["subgenre"],
                title=getattr(content, "title", None),
            )
            result.append(features)
        for row in sorted(timestamp_rows, key=_same_timestamp_update_key):
            history.update(row, content=content_by_id.get(row.content_id))
    # FEATURES + diagnostic-only columns: has_category_history/category_interaction_count/
    # days_since_last_category_interaction are all computed unconditionally by .features() but
    # are no longer selected into model input (see NUMERIC's own comments on each) -- still
    # carried here explicitly since drift monitoring (app.ml.drift_service) legitimately wants
    # to observe these distributions even though the model itself no longer consumes them.
    return pd.DataFrame(result, columns=[
        *FEATURES, "has_category_history", "category_interaction_count", "days_since_last_category_interaction",
    ])


def build_dataset(rows: Iterable[Any], content_by_id: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build one point-in-time labeled row per event, using only strictly-prior history.

    Rows are processed in timestamp batches. Every row at timestamp T is featurized from
    history strictly before T; only after all snapshots at T are complete is the full batch
    folded into history. Simultaneous events therefore cannot invent precedence or observe
    one another, regardless of input order.

    Context/label separation (training-dataset construction only -- see
    `app.db.models.Interaction.is_training_context_only`'s own comment for the full
    rationale): a row with `is_training_context_only` truthy still updates `history` exactly
    like any other historical event (category/recent/session/creator/semantic state), but is
    NEVER emitted as a labeled training row, regardless of what `target_for(row)` would say
    about it -- `target_for` is not even called for such a row. `getattr(row, ...)` with a
    False default means every existing caller (real `Interaction` rows -- real event ingestion
    never sets this field -- and every duck-typed test/eligibility-probe row, which typically
    lacks the attribute entirely) is completely unaffected; this is purely additive.
    """
    history = FeatureHistory()
    result: list[dict[str, Any]] = []
    content_by_id = content_by_id or {}
    for timestamp_rows in _timestamp_batches(rows):
        for row in timestamp_rows:
            is_context_only = getattr(row, "is_training_context_only", False)
            label = None if is_context_only else target_for(row)
            content = content_by_id.get(row.content_id)
            if label is not None:
                signals = interaction_signals(row)
                token_fields = _token_fields(content)
                features = history.features(
                    user_id=row.user_id,
                    category=row.category,
                    creator_id=row.creator_id,
                    content_id=row.content_id,
                    timestamp=row.timestamp,
                    content_popularity_score=getattr(content, "popularity_score", 0.5),
                    content_created_at=getattr(content, "created_at", row.timestamp),
                    hashtags=token_fields["hashtag"],
                    topics=token_fields["topic"],
                    entities=token_fields["entity"],
                    subgenres=token_fields["subgenre"],
                    title=getattr(content, "title", None),
                )
                ts = _utc(row.timestamp)
                result.append({
                    **features, "target": label, "user_id": row.user_id, "creator_id": row.creator_id,
                    "content_id": row.content_id, "candidate_group": f"{row.user_id}:{ts.date().isoformat()}",
                    "timestamp": row.timestamp,
                    # "event_"-prefixed so none of these can ever collide with a FEATURES name
                    # (`features` already has its own, differently-scoped "creator_followed" --
                    # "whether the candidate's creator was followed before this scoring moment" --
                    # a plain "creator_followed" key here would silently overwrite it, since a
                    # later key wins in a dict literal). See TRAINING_METADATA_COLUMNS below.
                    "event_type": row.event_type, "event_watch_percentage": row.watch_percentage,
                    "event_liked": signals.liked, "event_shared": signals.shared,
                    "event_favorited": signals.favorited,
                    "event_creator_followed": signals.creator_followed,
                })
        for row in sorted(timestamp_rows, key=_same_timestamp_update_key):
            history.update(row, content=content_by_id.get(row.content_id))
    return pd.DataFrame(result)
