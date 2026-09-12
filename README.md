# nfl-score-predictor

From-scratch NFL final-score prediction system. See config.yaml for model
list and training schedule. Predictions are generated weekly (Wed 5pm CT)
via GitHub Actions and committed to /data/predictions.




## Running scripts
Scripts that import from other src/ modules must be run with -m from the
repo root, e.g.: python -m src.models.baseline
Standalone ingestion/feature scripts can still be run directly, e.g.:
python src/ingest/pull_schedules.py