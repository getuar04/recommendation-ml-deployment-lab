"""Phase 3 PoC: offline Two-Tower retrieval, isolated from the production ranking pipeline.

Nothing in this package is imported by app.services.recommendation_service or any other
production request path -- see the package docstring in each module for what is reused from
the existing ranker (app.ml.dataset_builder.FeatureHistory, app.ml.feature_builder.target_for,
app.ml.splitting.chronological_group_split) vs. what is new here (candidate-independent user/
content vectors, the Two-Tower model itself, and offline retrieval/evaluation).
"""
