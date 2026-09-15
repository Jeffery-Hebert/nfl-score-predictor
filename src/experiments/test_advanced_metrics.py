"""
src/experiments/test_advanced_metrics.py

Tests four families of advanced metrics against the active benchmark, each in
isolation so the result attributes to a specific idea rather than a bundle.

  PACE        possessions and tempo. Points = efficiency x volume, and the
              production feature set contains no volume term at all.
  COMPETITIVE EPA restricted to 0.20 < wp < 0.80 -- strips prevent-defense
              garbage time out of the efficiency estimate. Uses nflfastR `wp`
              (game state), never `vegas_wp` (market).
  TURNOVER    fumbles forced (persists) separated from fumbles recovered
              (~coin flip), so a model can weight skill and discount luck.
  SPECIAL     field-goal conversion -- kicking turns drives into points and is
              absent from the feature set.

Active benchmark to beat (post-C1-C8, n=1426, mean of home/away RMSE):
  baseline 9.4415   linear 9.4110   poisson 9.3920

Standalone: writes nothing to production. Rolling features use the same
leakage-safe EWM + shift(1) + finale-masking pattern as the production
pipeline, at the tuned 17-week halflife.

Run: python -m src.experiments.test_advanced_metrics
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.preprocessing import StandardScaler

from src.features.build_rolling_features import is_finale_week, load_config
from src.models.common import FEATURE_COLS, ot_sample_weight
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

N_BOOT = 5000
SEED = 42

FAMILIES = {
    "PACE": [
        "off_plays",
        "off_drives",
        "off_plays_per_drive",
        "off_sec_per_play",
        "off_no_huddle_rate",
        "off_pass_oe",
    ],
    "COMPETITIVE": [
        "off_epa_competitive",
        "def_epa_competitive",
        "off_success_competitive",
        "def_success_competitive",
    ],
    "TURNOVER": [
        "off_fumbles",
        "off_fumble_retain_rate",
        "off_interceptions",
        "def_fumbles_forced",
        "def_interceptions",
    ],
    "SPECIAL": ["fg_pct", "fg_attempts", "fg_avg_distance"],
}
ALL_STATS = [c for cols in FAMILIES.values() for c in cols]


def add_pregame(group: pd.DataFrame, halflife_days: float, cols) -> pd.DataFrame:
    """Same leakage-safe pattern as build_rolling_features.add_pregame_rolling_features."""
    group = group.sort_values("gameday").reset_index(drop=True)
    finale = group.apply(lambda r: is_finale_week(r["season"], r["week"]), axis=1)
    for col in cols:
        masked = group[col].where(~finale)
        ewm = masked.ewm(
            halflife=pd.Timedelta(days=halflife_days),
            times=group["gameday"],
            ignore_na=True,
        ).mean()
        group[f"pregame_{col}"] = ewm.shift(1)
    return group


def build_table() -> pd.DataFrame:
    halflife_days = load_config()["training"]["recency_half_life_weeks"] * 7
    adv = pd.read_parquet("data/processed/advanced_team_stats.parquet")
    sched = pd.read_parquet("data/raw/schedules.parquet")[
        ["game_id", "season", "week", "gameday"]
    ]
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    adv = adv.merge(sched, on="game_id", how="inner")

    rolled = pd.concat(
        [
            add_pregame(g.copy(), halflife_days, ALL_STATS)
            for _, g in adv.groupby("team")
        ],
        ignore_index=True,
    )
    pre = [f"pregame_{c}" for c in ALL_STATS]

    model_table = pd.read_parquet("data/processed/model_table.parquet")
    home = rolled[["game_id", "team"] + pre].rename(
        columns={**{c: f"home_{c}" for c in pre}, "team": "home_team"}
    )
    away = rolled[["game_id", "team"] + pre].rename(
        columns={**{c: f"away_{c}" for c in pre}, "team": "away_team"}
    )
    df = model_table.merge(home, on=["game_id", "home_team"], how="left")
    return df.merge(away, on=["game_id", "away_team"], how="left")


def sided(stats):
    return [f"home_pregame_{c}" for c in stats] + [f"away_pregame_{c}" for c in stats]


def linear_fns(cols):
    def fit(train):
        X = train[cols]
        means = X.mean()
        Xf = X.fillna(means)
        w = ot_sample_weight(train)
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


def poisson_fns(cols):
    def fit(train):
        X = train[cols]
        means = X.mean()
        Xf = X.fillna(means)
        sc = StandardScaler().fit(Xf)
        Xs = sc.transform(Xf)
        w = ot_sample_weight(train)
        return {
            "h": PoissonRegressor(max_iter=500).fit(
                Xs, train["home_score"], sample_weight=w
            ),
            "a": PoissonRegressor(max_iter=500).fit(
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


def bootstrap(base, cand):
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


def main():
    df = build_table()
    cov = df[f"home_pregame_off_drives"].notna().mean()
    print(f"Table {len(df)} games; advanced-feature coverage {cov:.1%}\n")

    variants = {"base (production)": FEATURE_COLS}
    for name, stats in FAMILIES.items():
        variants[f"+{name}"] = FEATURE_COLS + sided(stats)
    variants["+PACE+COMPETITIVE"] = FEATURE_COLS + sided(
        FAMILIES["PACE"] + FAMILIES["COMPETITIVE"]
    )
    variants["+ALL"] = FEATURE_COLS + sided(ALL_STATS)
    # Competitive EPA REPLACING the blended all-situations EPA rather than added
    repl = [
        c
        for c in FEATURE_COLS
        if "pregame_off_epa_per_play" not in c and "pregame_def_epa_per_play" not in c
    ]
    variants["COMPETITIVE replaces blended"] = repl + sided(FAMILIES["COMPETITIVE"])

    for model_name, maker in [
        ("Linear Regression", linear_fns),
        ("Poisson GLM", poisson_fns),
    ]:
        print(f"\n{'=' * 78}\n{model_name}\n{'=' * 78}")
        results = {}
        for label, cols in variants.items():
            fit, predict = maker(cols)
            res = walk_forward_evaluate(df, fit, predict, min_train_seasons=2)
            results[label] = res
            m = score_predictions(res)
            print(
                f"  {label:<28} home={m['home_rmse']:.4f} away={m['away_rmse']:.4f} "
                f"mean={(m['home_rmse']+m['away_rmse'])/2:.4f}"
            )

        base = results["base (production)"]
        print(f"\n  Paired bootstrap vs base ({N_BOOT} resamples over games)\n")
        for label, res in results.items():
            if label == "base (production)":
                continue
            mean, lo, hi = bootstrap(base, res)
            v = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
            print(
                f"    {label:<28} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {v}"
            )


if __name__ == "__main__":
    main()
