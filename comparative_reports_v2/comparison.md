# Comparative Experiment Report v2: Same Algorithm, Different Datasets

## 1. Experimental controls

- Identical across all four experiments (enforced, not just displayed): **True**
- Algorithm: `LogisticRegression`
- Hyperparameters: `{'C': 1.0, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': 0.0, 'max_iter': 1000, 'n_jobs': None, 'penalty': 'deprecated', 'random_state': 42, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}`
- Dataset-generation seed: `20260805`  Model random_state: `42`
- Feature names/order: `['category', 'category_affinity', 'has_category_history', 'average_category_watch_percentage', 'recent_category_watch_percentage', 'category_completion_rate', 'recent_category_completion_rate', 'category_positive_count', 'category_negative_count', 'category_interaction_count', 'has_creator_history', 'creator_interaction_count', 'creator_completion_rate', 'creator_followed', 'user_total_interaction_count', 'content_popularity_score', 'content_age_hours', 'already_seen', 'hour_of_day', 'days_since_last_category_interaction']`
- Split strategy: 5-way chronological split: train/modelSelection/calibration/thresholdTuning/test are five distinct, non-overlapping partitions. (used fallback: False)
- Fixed (primary) scenario: `{'userId': 'comparative-fixed-test-user', 'limit': 10, 'candidateCategories': ['COMEDY', 'FASHION', 'FITNESS', 'FOOD', 'GAMING', 'MUSIC', 'NEWS', 'SPORT', 'TECH', 'TRAVEL'], 'candidateCount': 10}`

## 2. Dataset treatment

- **sport** (comparative-sport-v2): dominant=SPORT, target weights={'SPORT': 0.8, 'FITNESS': 0.1, 'TRAVEL': 0.1}
- **entertainment** (comparative-entertainment-v2): dominant=COMEDY, target weights={'COMEDY': 0.8, 'MUSIC': 0.1, 'NEWS': 0.1} -- MOVIES/ENTERTAINMENT is not a supported category in this repository; COMEDY is used as the closest existing supported category.
- **music** (comparative-music-v2): dominant=MUSIC, target weights={'MUSIC': 0.8, 'COMEDY': 0.1, 'FASHION': 0.1}
- **balanced** (comparative-balanced-v2): dominant=NONE, target weights={'SPORT': 0.2, 'MUSIC': 0.2, 'TECH': 0.2, 'FOOD': 0.2, 'GAMING': 0.2}

## 3. Realized dataset statistics

- **sport**: {'totalInteractions': 3044, 'labelledSamples': 900, 'neutralRowsExcluded': 2144, 'positiveSamples': 450, 'negativeSamples': 450, 'positiveRatio': 0.5, 'negativeRatio': 0.5, 'uniqueUsers': 60, 'uniqueContents': 100, 'uniqueCreators': 12, 'earliestTimestamp': '2026-05-03T14:39:36+00:00', 'latestTimestamp': '2026-07-31T23:29:29+00:00'}
  - realized category distribution: {'SPORT': 0.8022, 'FITNESS': 0.1078, 'TRAVEL': 0.09}
- **entertainment**: {'totalInteractions': 3044, 'labelledSamples': 900, 'neutralRowsExcluded': 2144, 'positiveSamples': 450, 'negativeSamples': 450, 'positiveRatio': 0.5, 'negativeRatio': 0.5, 'uniqueUsers': 60, 'uniqueContents': 100, 'uniqueCreators': 12, 'earliestTimestamp': '2026-05-03T14:39:36+00:00', 'latestTimestamp': '2026-07-31T23:29:29+00:00'}
  - realized category distribution: {'COMEDY': 0.8022, 'MUSIC': 0.1078, 'NEWS': 0.09}
- **music**: {'totalInteractions': 3044, 'labelledSamples': 900, 'neutralRowsExcluded': 2144, 'positiveSamples': 450, 'negativeSamples': 450, 'positiveRatio': 0.5, 'negativeRatio': 0.5, 'uniqueUsers': 60, 'uniqueContents': 100, 'uniqueCreators': 12, 'earliestTimestamp': '2026-05-03T14:39:36+00:00', 'latestTimestamp': '2026-07-31T23:29:29+00:00'}
  - realized category distribution: {'MUSIC': 0.8022, 'COMEDY': 0.1078, 'FASHION': 0.09}
- **balanced**: {'totalInteractions': 3014, 'labelledSamples': 900, 'neutralRowsExcluded': 2114, 'positiveSamples': 450, 'negativeSamples': 450, 'positiveRatio': 0.5, 'negativeRatio': 0.5, 'uniqueUsers': 60, 'uniqueContents': 100, 'uniqueCreators': 12, 'earliestTimestamp': '2026-05-03T14:39:36+00:00', 'latestTimestamp': '2026-07-31T23:29:29+00:00'}
  - realized category distribution: {'MUSIC': 0.222, 'TECH': 0.2203, 'GAMING': 0.1964, 'FOOD': 0.1855, 'SPORT': 0.1758}

## 4. Raw model-score evidence (primary scenario, pre-reranking)

app.experiments.fixed_scenario.PRIMARY_FIXED_CANDIDATES only -- one symmetric candidate per category, identical popularity/age/creatorFollowed=false/alreadySeen=false across every candidate. rawModelScore is the trained model's pre-reranking probability, captured via app.experiments.comparative_scoring (parity-checked against the real HTTP /recommendations response -- see postRerankingEvidence).

| Category | sport | entertainment | music | balanced | Score spread |
|---|---|---|---|---|---|
| COMEDY | 0.4831633618976502 | 0.5647862401113429 | 0.48056293866371574 | 0.4426650260076691 | 0.122121 |
| FASHION | 0.4831633618976502 | 0.4831633618975199 | 0.40045841730434867 | 0.4426650260076691 | 0.082705 |
| FITNESS | 0.4805629386636181 | 0.4831633618975199 | 0.48316336189751086 | 0.4426650260076691 | 0.040498 |
| FOOD | 0.4831633618976502 | 0.4831633618975199 | 0.48316336189751086 | 0.4449512049401054 | 0.038212 |
| GAMING | 0.4831633618976502 | 0.4831633618975199 | 0.48316336189751086 | 0.40841412407703054 | 0.074749 |
| MUSIC | 0.4831633618976502 | 0.4805629386636994 | 0.5647862401113379 | 0.4384408245245944 | 0.126345 |
| NEWS | 0.4831633618976502 | 0.4004584173043353 | 0.48316336189751086 | 0.4426650260076691 | 0.082705 |
| SPORT | 0.5647862401114734 | 0.4831633618975199 | 0.48316336189751086 | 0.4584441078765306 | 0.106342 |
| TECH | 0.4831633618976502 | 0.4831633618975199 | 0.48316336189751086 | 0.4637371504850383 | 0.019426 |
| TRAVEL | 0.40045841730426834 | 0.4831633618975199 | 0.48316336189751086 | 0.4426650260076691 | 0.082705 |

## 5. Post-reranking evidence (primary scenario, real adjusted response)

- **sport** (strategy=COLD_START): SPORT:0.564786, FOOD:0.483163, MUSIC:0.483163, TECH:0.483163, GAMING:0.483163, COMEDY:0.483163, NEWS:0.483163, FASHION:0.483163, FITNESS:0.480563, TRAVEL:0.400458
- **entertainment** (strategy=COLD_START): COMEDY:0.564786, FOOD:0.483163, SPORT:0.483163, TECH:0.483163, GAMING:0.483163, TRAVEL:0.483163, FASHION:0.483163, FITNESS:0.483163, MUSIC:0.480563, NEWS:0.400458
- **music** (strategy=COLD_START): MUSIC:0.564786, FOOD:0.483163, SPORT:0.483163, TECH:0.483163, GAMING:0.483163, TRAVEL:0.483163, NEWS:0.483163, FITNESS:0.483163, COMEDY:0.480563, FASHION:0.400458
- **balanced** (strategy=COLD_START): TECH:0.463737, SPORT:0.458444, FOOD:0.444951, TRAVEL:0.442665, COMEDY:0.442665, NEWS:0.442665, FASHION:0.442665, FITNESS:0.442665, MUSIC:0.438441, GAMING:0.408414

## Comparison table

| Experiment | Dataset | Algorithm | Dominant Category | F1 | PR-AUC | ROC-AUC | Top Recommended Categories (post-rerank) |
|---|---|---|---|---|---|---|---|
| comparative-sport-v2 | sport | LogisticRegression | SPORT | 0.705036 | 0.564812 | 0.537208 | SPORT, FOOD, MUSIC, TECH, GAMING, COMEDY, NEWS, FASHION, FITNESS, TRAVEL |
| comparative-entertainment-v2 | entertainment | LogisticRegression | COMEDY | 0.705036 | 0.564812 | 0.537208 | COMEDY, FOOD, SPORT, TECH, GAMING, TRAVEL, FASHION, FITNESS, MUSIC, NEWS |
| comparative-music-v2 | music | LogisticRegression | MUSIC | 0.705036 | 0.564812 | 0.537208 | MUSIC, FOOD, SPORT, TECH, GAMING, TRAVEL, NEWS, FITNESS, COMEDY, FASHION |
| comparative-balanced-v2 | balanced | LogisticRegression | NONE | 0.663158 | 0.722796 | 0.714533 | TECH, SPORT, FOOD, TRAVEL, COMEDY, NEWS, FASHION, FITNESS, MUSIC, GAMING |

## 6. Observations

- SPORT raw model score in sport: 0.5647862401114734 vs 0.4584441078765306 in balanced (cross-experiment delta +0.106342); vs. the average of the other 9 categories' raw scores within sport itself: 0.473685 (within-experiment delta +0.091101).
- COMEDY raw model score in entertainment: 0.5647862401113429 vs 0.4426650260076691 in balanced (cross-experiment delta +0.122121); vs. the average of the other 9 categories' raw scores within entertainment itself: 0.473685 (within-experiment delta +0.091101).
- MUSIC raw model score in music: 0.5647862401113379 vs 0.4384408245245944 in balanced (cross-experiment delta +0.126345); vs. the average of the other 9 categories' raw scores within music itself: 0.473685 (within-experiment delta +0.091101).

## 7. Evidence supporting each observation

- **comparative-sport-v2** (dominant=SPORT, supportsHypothesis=True): `{'rawScoreInOwnExperiment': 0.5647862401114734, 'rawScoreInBalanced': 0.4584441078765306, 'crossExperimentDelta': 0.106342, 'peerCategoryAverageRawScoreInOwnExperiment': 0.473685, 'withinExperimentDelta': 0.091101}`
- **comparative-entertainment-v2** (dominant=COMEDY, supportsHypothesis=True): `{'rawScoreInOwnExperiment': 0.5647862401113429, 'rawScoreInBalanced': 0.4426650260076691, 'crossExperimentDelta': 0.122121, 'peerCategoryAverageRawScoreInOwnExperiment': 0.473685, 'withinExperimentDelta': 0.091101}`
- **comparative-music-v2** (dominant=MUSIC, supportsHypothesis=True): `{'rawScoreInOwnExperiment': 0.5647862401113379, 'rawScoreInBalanced': 0.4384408245245944, 'crossExperimentDelta': 0.126345, 'peerCategoryAverageRawScoreInOwnExperiment': 0.473685, 'withinExperimentDelta': 0.091101}`

## Secondary scenario (business-reranking evidence only -- excluded from the verdict)

- **sport**: top 3 = COMEDY:FOLLOWED_CREATOR, SPORT:POPULAR_CONTENT, NEWS:EXPLORATION
- **entertainment**: top 3 = COMEDY:FOLLOWED_CREATOR, COMEDY:EXPLORATION, FASHION:EXPLORATION
- **music**: top 3 = COMEDY:FOLLOWED_CREATOR, MUSIC:EXPLORATION, MUSIC:POPULAR_CONTENT
- **balanced**: top 3 = TECH:EXPLORATION, COMEDY:EXPLORATION, SPORT:POPULAR_CONTENT

## 8. Confounding factors

- The sport/entertainment/music datasets share the identical generation seed and an isomorphic 80/10/10 category-weight shape -- they differ only in which category label is assigned the dominant/related/other roles, not in any other statistical property of the generation process.
- The entertainment dataset's dominant category (COMEDY) never appears in the balanced dataset's training categories at all, whereas sport/music's dominant categories (SPORT/MUSIC) both appear in balanced at a lower (20%) weight -- the entertainment-vs-balanced comparison is a 'never seen vs seen' contrast, not a 'seen more vs seen less' contrast like the other two.
- Deterministic post-generation class-balance downsampling (app.experiments.comparative_dataset_generation._stratified_downsample) removes some point-in-time history context for chronologically-later surviving rows -- applied identically across all four datasets via the same seeded algorithm, so it should not differentially bias one dataset over another, but it is a real methodological effect, not a null one.
- The fixed test user is cold-start (zero interaction history) in every experiment, so every primary-scenario feature except category, popularity, and age is a constant neutral value -- this isolates the category effect cleanly but says nothing about personalization for a user with real history.

## 9. Limitations

- All four datasets are entirely synthetic, generated from a weighted-category heuristic, not real user behavior -- category preference is injected directly into generation rather than emerging from organic engagement.
- No minimum-effect-size or statistical-significance threshold is applied to any raw-score delta -- see whatCannotBeConcludedFromSyntheticData in the conclusion.
- Absolute metric values, scores, and rankings describe this specific synthetic run only and must not be presented as production evidence.

## 10. Conclusion

**Verdict: YES**

The same locked LogisticRegression, trained under identical configuration on different declared category distributions, produced a measurably different raw probability for the dominant category in all 3 dominant-category experiment(s) tested, relative to the balanced-trained model.

- What the raw model learned: Raw (pre-reranking) model probability, primary symmetric scenario only: 3 of 3 dominant-category experiment(s) showed a higher raw score for their own dominant category than the balanced-trained model assigned the same category.
- What business reranking changed: See postRerankingEvidence for the real, adjusted (post-business-reranking) recommendation order per experiment -- compare against rawModelScoreEvidence to see whether business reranking (seen-penalty, diversity decay) preserved, amplified, or reduced the raw model's category ordering. Not itself part of this verdict.
- What cannot be concluded from synthetic data: Whether a model trained on skewed real (non-synthetic) data would show the same effect; whether the effect generalizes beyond this fixed, cold-start test user and fixed candidate set; whether the magnitude of any raw-score delta found here is practically significant for a production ranking (no minimum-effect-size threshold or statistical-significance test is applied -- deltas are reported as-is).
