# Comparative Experiment Report: Same Algorithm, Different Datasets

## Same-algorithm verification

- Algorithm identical across all four experiments: **True** ({'sport': 'LogisticRegression', 'entertainment': 'LogisticRegression', 'music': 'LogisticRegression', 'balanced': 'LogisticRegression'})
- Hyperparameters identical: **True**
- Dataset-generation seed identical: **True** ({'sport': 20260805, 'entertainment': 20260805, 'music': 20260805, 'balanced': 20260805})
- Model random_state identical: **True** ({'sport': 42, 'entertainment': 42, 'music': 42, 'balanced': 42})
- Fixed test user / candidate list / limit identical: **True** (All four experiments' /recommendations requests were built from the single shared app.experiments.fixed_scenario.FIXED_CANDIDATES constant -- identical by construction, not merely by comparison.)

## Comparison table

| Experiment | Dataset | Algorithm | Dominant Category | F1 | PR-AUC | ROC-AUC | Top Recommended Categories |
|---|---|---|---|---|---|---|---|
| comparative-sport-v1 | sport | LogisticRegression | SPORT | 0.828711 | 0.752932 | 0.566972 | SPORT, COMEDY, MUSIC, NEWS, FASHION, FOOD, TECH, GAMING, FITNESS |
| comparative-entertainment-v1 | entertainment | LogisticRegression | COMEDY | 0.828711 | 0.752371 | 0.566289 | COMEDY, FASHION, FITNESS, TRAVEL, FOOD, TECH, GAMING, SPORT, MUSIC |
| comparative-music-v1 | music | LogisticRegression | MUSIC | 0.828711 | 0.7524 | 0.566289 | MUSIC, COMEDY, NEWS, FITNESS, TRAVEL, FOOD, TECH, GAMING, SPORT |
| comparative-balanced-v1 | balanced | LogisticRegression | NONE | 0.6313 | 0.668243 | 0.71527 | COMEDY, TECH, FOOD, MUSIC, TRAVEL, FITNESS, FASHION, NEWS, GAMING |

## Recommendation ranking comparison / score differences per candidate

| Candidate | Sport rank/score | Entertainment rank/score | Music rank/score | Balanced rank/score | Score spread |
|---|---|---|---|---|---|
| fixed-cand-comedy-1 | 2/0.739479 | 1/0.808695 | 3/0.721354 | 1/0.613673 | 0.195022 |
| fixed-cand-comedy-2 | 10/0.68477 | 2/0.756486 | - | - | 0.071716 |
| fixed-cand-fashion-1 | 5/0.710869 | 3/0.709967 | - | 7/0.503268 | 0.207601 |
| fixed-cand-fitness-1 | 9/0.692017 | 4/0.708622 | 5/0.708431 | 6/0.507559 | 0.201063 |
| fixed-cand-food-1 | 6/0.707848 | 6/0.706826 | 7/0.706623 | 3/0.534175 | 0.173673 |
| fixed-cand-gaming-1 | 8/0.706143 | 8/0.705055 | 9/0.70484 | 10/0.490773 | 0.21537 |
| fixed-cand-music-1 | - | - | 2/0.749624 | 4/0.531488 | 0.218136 |
| fixed-cand-music-2 | 3/0.71159 | 10/0.693289 | 1/0.786688 | 9/0.492055 | 0.294633 |
| fixed-cand-news-1 | 4/0.710893 | - | 4/0.709781 | 8/0.49315 | 0.217743 |
| fixed-cand-sport-1 | 1/0.780762 | 9/0.70331 | 10/0.703082 | - | 0.07768 |
| fixed-cand-tech-1 | 7/0.706771 | 7/0.705707 | 8/0.705496 | 2/0.544302 | 0.162469 |
| fixed-cand-travel-1 | - | 5/0.707996 | 6/0.707801 | 5/0.508418 | 0.199578 |

## Category distribution of Top 10 recommendations

- **sport**: {'COMEDY': 0.2, 'SPORT': 0.1, 'MUSIC': 0.1, 'NEWS': 0.1, 'FASHION': 0.1, 'FOOD': 0.1, 'TECH': 0.1, 'GAMING': 0.1, 'FITNESS': 0.1}
- **entertainment**: {'COMEDY': 0.2, 'FASHION': 0.1, 'FITNESS': 0.1, 'TRAVEL': 0.1, 'FOOD': 0.1, 'TECH': 0.1, 'GAMING': 0.1, 'SPORT': 0.1, 'MUSIC': 0.1}
- **music**: {'MUSIC': 0.2, 'COMEDY': 0.1, 'NEWS': 0.1, 'FITNESS': 0.1, 'TRAVEL': 0.1, 'FOOD': 0.1, 'TECH': 0.1, 'GAMING': 0.1, 'SPORT': 0.1}
- **balanced**: {'MUSIC': 0.2, 'COMEDY': 0.1, 'TECH': 0.1, 'FOOD': 0.1, 'TRAVEL': 0.1, 'FITNESS': 0.1, 'FASHION': 0.1, 'NEWS': 0.1, 'GAMING': 0.1}

## Metric comparison

| Dataset | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| sport | 0.711538 | 0.709497 | 0.996078 | 0.828711 | 0.752932 | 0.566972 |
| entertainment | 0.711538 | 0.709497 | 0.996078 | 0.828711 | 0.752371 | 0.566289 |
| music | 0.711538 | 0.709497 | 0.996078 | 0.828711 | 0.7524 | 0.566289 |
| balanced | 0.625337 | 0.558685 | 0.72561 | 0.6313 | 0.668243 | 0.71527 |

## What the model learned differently

- **comparative-sport-v1** (dominant=SPORT): Candidate 'fixed-cand-sport-1' reached the Top 10 in sport (rank 1, score 0.780762) but not in balanced -- consistent with the hypothesis.
- **comparative-entertainment-v1** (dominant=COMEDY): Candidate 'fixed-cand-comedy-1' (category COMEDY) scored 0.808695 (rank 1) in entertainment vs 0.613673 (rank 1) in balanced -- a delta of +0.195022, consistent with the hypothesis that COMEDY-dominant training increases this candidate's score.
- **comparative-music-v1** (dominant=MUSIC): Candidate 'fixed-cand-music-1' (category MUSIC) scored 0.749624 (rank 2) in music vs 0.531488 (rank 4) in balanced -- a delta of +0.218136, consistent with the hypothesis that MUSIC-dominant training increases this candidate's score.

## Business interpretation

A recommendation service that genuinely learns from behavioral data should surface more of a user's likely-preferred category when that category dominates the training signal, without any change to the algorithm, code, or serving logic -- this is what operators should expect when retraining against a shifted content-consumption pattern (e.g. a seasonal spike in one content category). See recommendationLearningFindings for whether this run's data actually shows that; a synthetic-data PoC result here is evidence about pipeline behavior, not a production guarantee.

## Limitations of synthetic data

All four datasets are entirely synthetic (app.experiments.comparative_dataset_generation), generated from simple weighted-category heuristics, not real user behavior -- category preference is injected directly into the generation process rather than emerging from organic engagement patterns, which can make the learned signal cleaner (or noisier) than a real dataset of the same size would produce. The fixed test user is a cold-start user with zero interaction history in every experiment (a deliberate choice so 'same user context' is trivially true across four separate databases -- see app.experiments.fixed_scenario), so these results say nothing about how a model trained on skewed data would treat a user with a real, independent history of their own. Absolute metric values, scores, and rankings describe this specific synthetic run only and must not be presented as production evidence.

## Conclusion

**Verdict: YES**

The same locked algorithm and configuration produced meaningfully different recommendations when trained on different datasets: all 3 dominant-category experiment(s) with a comparable fixed candidate scored/ranked that candidate higher than the balanced-training run did.
