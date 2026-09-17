"""Independent recommendation-ranking benchmark (EASY/MEDIUM/HARD/ADVERSARIAL + temporal-shift
and NOT_INTERESTED-localization scenarios). See app.benchmark.runner for the entry point.

Deliberately separate from app.ml.trainer/app.ml.evaluator: a model must be trainable with no
benchmark run requested, and benchmark scenarios must never be fed into training data (see
scripts/run_ranking_benchmark.py and each scenario module's own docstring).
"""
