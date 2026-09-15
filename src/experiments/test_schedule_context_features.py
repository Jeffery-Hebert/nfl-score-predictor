"""
src/experiments/test_schedule_context_features.py

C1-C4: fix or add four schedule-derived pregame features, each tested in
isolation and then together, against the current production feature set.

C1  rest_days is wrong. build_rolling_features.py computes it as
    gameday.diff(), which gives 260 days across a season boundary -- 112 games
    exceed 100 "rest days". Linear puts +0.107/day on it, so a season opener
    contributes +21 points before the away-side coefficient accidentally
    cancels it. schedules.parquet already ships home_rest/away_rest: max 16,
    100% populated, correct across season boundaries. Replace, don't add.

C2  prior_games_played is a GLOBAL counter (0 -> 152 across 2019-2026), not a
    per-season one. It is a calendar trend wearing a football name; Linear puts
    +0.0458/game on it, a 6.1-point swing across the training range. Tested
    both as a per-season counter and dropped entirely.

C3  Neutral-site games (50: London, Mexico, Munich, Super Bowl) are currently
    modelled as ordinary home games. `location` marks them and is known
    pregame.

C4  Playoff games (89) are mixed into training with the regular season.
    `game_type` separates them and is known pregame.

C5  Overtime, on the TRAINING side only. schedules.overtime is 0% populated
    before kickoff, so it can never be a feature of the game being predicted --
    that would be target leakage, and OT games average 53.9 total points
    against 45.3. But a training game's OT status is entirely in the past and
    known, so it can be used to down-weight the distortion those inflated
    scores put into the fitted coefficients. Exactly the same reasoning as the
    finale-week masking already in build_rolling_features.py: clean a known
    distortion out of the historical signal, never consult it about the future.
    Tested at weight 0.0 (exclude) and 0.5 (halve).

Standalone: does not modify model_table.parquet, common.py, or any model.

Run: python -m src.experiments.test_schedule_context_features
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from src.models.common import FEATURE_COLS
from src.validate.calibration import bias, calibration_slope_intercept
from src.validate.walk_forward import walk_forward_evaluate, score_predictions

N_BOOT = 5000
SEED = 42

REST_COLS = ["home_rest_days", "away_rest_days"]
PRIOR_COLS = ["home_prior_games_played", "away_prior_games_played"]


def build_table() -> pd.DataFrame:
    """model_table plus the schedule context columns, none of which exist in it."""
    df = pd.read_parquet("data/processed/model_table.parquet")
    sched = pd.read_parquet("data/raw/schedules.parquet")[
        ["game_id", "home_rest", "away_rest", "location", "game_type", "overtime"]
    ]
    df = df.merge(sched, on="game_id", how="left")

    # C1: nflverse rest, correct across season boundaries
    df["home_rest_nfl"] = df["home_rest"].astype(float)
    df["away_rest_nfl"] = df["away_rest"].astype(float)

    # C2: per-season game counter instead of a global calendar index
    rolling = pd.read_parquet("data/processed/team_rolling_features.parquet")
    rolling = rolling.sort_values("gameday")
    rolling["prior_games_this_season"] = rolling.groupby(["team", "season"]).cumcount()
    per_season = rolling[["game_id", "team", "prior_games_this_season"]]
    df = df.merge(
        per_season.rename(
            columns={
                "team": "home_team",
                "prior_games_this_season": "home_prior_season_games",
            }
        ),
        on=["game_id", "home_team"],
        how="left",
    )
    df = df.merge(
        per_season.rename(
            columns={
                "team": "away_team",
                "prior_games_this_season": "away_prior_season_games",
            }
        ),
        on=["game_id", "away_team"],
        how="left",
    )

    # C3 / C4: both known before kickoff
    df["is_neutral_site"] = (df["location"] == "Neutral").astype(float)
    df["is_playoff"] = (df["game_type"] != "REG").astype(float)
    # C5: training-side only. Never enters any feature list.
    df["went_to_ot"] = df["overtime"].fillna(0).astype(float)
    return df


def ot_sample_weight(train: pd.DataFrame, weight: float) -> np.ndarray:
    """Down-weight historical overtime games when fitting.

    Leakage-safe: this reads the OT status of games already played, which is
    fully known at fit time. It never touches the row being predicted.
    """
    return np.where(train["went_to_ot"] == 1.0, weight, 1.0)


def make_linear(cols, ot_weight=None):
    def fit(train):
        X = train[cols]
        means = X.mean()
        Xf = X.fillna(means)
        w = None if ot_weight is None else ot_sample_weight(train, ot_weight)
        return {
            "h": LinearRegression().fit(Xf, train["home_score"], sample_weight=w),
            "a": LinearRegression().fit(Xf, train["away_score"], sample_weight=w),
            "means": means,
            "cols": cols,
        }

    def predict(m, test):
        X = test[m["cols"]].fillna(m["means"])
        return m["h"].predict(X), m["a"].predict(X)

    return fit, predict


def make_poisson(cols, ot_weight=None):
    def fit(train):
        X = train[cols]
        means = X.mean()
        Xf = X.fillna(means)
        sc = StandardScaler().fit(Xf)
        Xs = sc.transform(Xf)
        w = None if ot_weight is None else ot_sample_weight(train, ot_weight)
        return {
            "h": PoissonRegressor(max_iter=300).fit(
                Xs, train["home_score"], sample_weight=w
            ),
            "a": PoissonRegressor(max_iter=300).fit(
                Xs, train["away_score"], sample_weight=w
            ),
            "means": means,
            "scaler": sc,
            "cols": cols,
        }

    def predict(m, test):
        Xs = m["scaler"].transform(test[m["cols"]].fillna(m["means"]))
        return m["h"].predict(Xs), m["a"].predict(Xs)

    return fit, predict


def bootstrap_delta(base, cand):
    """Paired bootstrap over games, pooled home+away. Negative = candidate better."""
    m = base.merge(cand, on="game_id", suffixes=("_b", "_c"))
    eb = np.stack(
        [
            (m.home_pred_b - m.home_score_b).to_numpy(float),
            (m.away_pred_b - m.away_score_b).to_numpy(float),
        ]
    )
    ec = np.stack(
        [
            (m.home_pred_c - m.home_score_c).to_numpy(float),
            (m.away_pred_c - m.away_score_c).to_numpy(float),
        ]
    )
    rng = np.random.default_rng(SEED)
    n = eb.shape[1]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        k = rng.integers(0, n, n)
        d[i] = np.sqrt((ec[:, k] ** 2).mean()) - np.sqrt((eb[:, k] ** 2).mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def variants():
    """Each returns the feature column list for that hypothesis."""
    keep = [c for c in FEATURE_COLS]
    no_rest = [c for c in keep if c not in REST_COLS]
    no_prior = [c for c in keep if c not in PRIOR_COLS]
    combined = [c for c in keep if c not in REST_COLS + PRIOR_COLS] + [
        "home_rest_nfl",
        "away_rest_nfl",
        "is_neutral_site",
        "is_playoff",
    ]
    # (feature_cols, ot_weight). ot_weight is a TRAINING-side sample weight,
    # never a feature -- see the C5 note in the module docstring.
    return {
        "base (production)": (keep, None),
        "C1 nflverse rest": (no_rest + ["home_rest_nfl", "away_rest_nfl"], None),
        "C2a per-season counter": (
            no_prior + ["home_prior_season_games", "away_prior_season_games"],
            None,
        ),
        "C2b drop counter": (no_prior, None),
        "C3 +neutral site": (keep + ["is_neutral_site"], None),
        "C4 +playoff flag": (keep + ["is_playoff"], None),
        "C5a exclude OT (w=0)": (keep, 0.0),
        "C5b halve OT (w=0.5)": (keep, 0.5),
        "C1+C2b+C3+C4": (combined, None),
        "all + OT w=0.5": (combined, 0.5),
    }


def main():
    df = build_table()
    print(f"Table: {len(df)} games. Sanity on the new columns:")
    print(
        f"  production rest_days   max={df['home_rest_days'].max():.0f}  >100: {(df['home_rest_days']>100).sum()}"
    )
    print(
        f"  nflverse home_rest     max={df['home_rest_nfl'].max():.0f}  >100: {(df['home_rest_nfl']>100).sum()}"
    )
    print(f"  global prior_games     max={df['home_prior_games_played'].max():.0f}")
    print(f"  per-season prior games max={df['home_prior_season_games'].max():.0f}")
    print(
        f"  neutral-site games={int(df['is_neutral_site'].sum())}  playoff games={int(df['is_playoff'].sum())}\n"
    )

    for model_name, maker in [
        ("Linear Regression", make_linear),
        ("Poisson GLM", make_poisson),
    ]:
        print(f"\n{'=' * 76}\n{model_name}\n{'=' * 76}")
        results = {}
        for label, (cols, ot_w) in variants().items():
            fit, predict = maker(cols, ot_w)
            res = walk_forward_evaluate(df, fit, predict, min_train_seasons=2)
            results[label] = res
            m = score_predictions(res)
            cal = calibration_slope_intercept(
                res["home_score"].to_numpy(float), res["home_pred"].to_numpy(float)
            )[0]
            hb = bias(
                res["home_score"].to_numpy(float), res["home_pred"].to_numpy(float)
            )
            print(
                f"  {label:<24} home={m['home_rmse']:.4f} away={m['away_rmse']:.4f} "
                f"mean={(m['home_rmse']+m['away_rmse'])/2:.4f}  "
                f"h_bias={hb:+.3f}  slope={cal:.3f}"
            )

        base = results["base (production)"]
        print(
            f"\n  Paired bootstrap vs base ({N_BOOT} resamples over games, pooled home+away)"
        )
        print(f"  negative delta = variant is better\n")
        for label, res in results.items():
            if label == "base (production)":
                continue
            mean, lo, hi = bootstrap_delta(base, res)
            verdict = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
            print(
                f"    {label:<24} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}"
            )


if __name__ == "__main__":
    main()
