"""
Shared feature list and leakage-safe imputation for all sklearn-style models.
Every model imports FEATURE_COLS from here rather than redefining it, so all
10 models train on an identical, consistent feature set.
"""
FEATURE_COLS = [
    "home_pregame_team_score", "home_pregame_opp_score", "home_pregame_off_epa_per_play",
    "home_pregame_def_epa_per_play", "home_pregame_off_success_rate",
    "home_pregame_def_success_rate_allowed", "home_rest_days", "home_prior_games_played",
    "away_pregame_team_score", "away_pregame_opp_score", "away_pregame_off_epa_per_play",
    "away_pregame_def_epa_per_play", "away_pregame_off_success_rate",
    "away_pregame_def_success_rate_allowed", "away_rest_days", "away_prior_games_played",
]