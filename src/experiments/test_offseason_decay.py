"""
src/experiments/test_offseason_decay.py

Fixes early-season over-reaction: one game outweighing a whole prior season.

Symptom. Week 2 2026 predictions averaged 47.35 total points against a market
line of 45.31. The model is NOT generally biased on totals -- across the
backtest it runs +0.34 -- so this is specific to early season. 2026 Week 1
produced 49.44 points a game, far above 2025's 45.96, and the model projected
that forward off a 16-game sample.

Mechanism. Team form decays by CALENDAR DAYS at a 17-week half-life. Across a
~200-day offseason last season's games decay to 0.5^(200/119) ~= 0.31 weight,
while a Week 1 game sitting 7 days back holds 0.5^(7/119) ~= 0.96. So a single
game outweighs the entire prior season about 3-4x. The decay is treating an
offseason as though 200 days of football happened, when no games were played at
all. Rosters change over an offseason, but not that much -- and certainly not
enough to make one September game more informative than a full season.

This is consistent with the error analysis, where weeks 1-3 are the
worst-calibrated stretch of the year.

Three schemes compared against the current one:

  calendar (current)   decay on real elapsed days
  compressed           decay on real days in-season, but an offseason counts as
                       a fixed, shorter gap -- no football happened, so less
                       forgetting should happen
  games                decay on GAMES PLAYED rather than days, so the offseason
                       costs nothing on its own and one game is always worth
                       one game

Reports early-season accuracy separately, since that is where the defect lives
and where an overall average would hide it.

Run: python -m src.experiments.test_offseason_decay
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_rolling_features import STAT_COLS, is_finale_week, load_config
from src.models.common import (
    BASE_FEATURE_COLS,
    FEATURE_COLS,
    ot_sample_weight,
    recent_residual_offset,
)
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

ALPHAS = np.logspace(-2, 4, 25)
N_BOOT = 5000
SEED = 42
OFFSEASON_GAP_DAYS = 45  # what a summer is "worth" under the compressed scheme


def effective_age(group, scheme):
    """Distance back in time for the decay, per scheme, as a float Series."""
    days = group["gameday"].diff().dt.days.fillna(0.0)
    if scheme == "calendar":
        step = days
    elif scheme == "compressed":
        # An offseason is any gap long enough that no games were played.
        step = days.where(days < 120, OFFSEASON_GAP_DAYS)
    elif scheme == "games":
        # One game back = one unit, regardless of the calendar.
        step = pd.Series(np.where(days > 0, 7.0, 0.0), index=group.index)
    else:
        raise ValueError(scheme)
    return step.cumsum()


def add_rolling(group, halflife_days, scheme):
    group = group.sort_values("gameday").reset_index(drop=True)
    finale = group.apply(lambda r: is_finale_week(r["season"], r["week"]), axis=1)
    age = effective_age(group, scheme)

    for col in STAT_COLS:
        masked = group[col].where(~finale)
        # Manual EWM over the effective timeline: weight each prior game by
        # 0.5 ** (age_gap / halflife), then shift so the current game is excluded.
        vals, out = masked.to_numpy(float), np.full(len(group), np.nan)
        a = age.to_numpy(float)
        for i in range(1, len(group)):
            prior, va = a[:i], vals[:i]
            ok = ~np.isnan(va)
            if not ok.any():
                continue
            w = 0.5 ** ((a[i] - prior[ok]) / halflife_days)
            out[i] = np.average(va[ok], weights=w)
        group[f"pregame_{col}"] = out
    group["prior_games_played"] = group.groupby("season").cumcount()
    return group


def build_table(scheme, halflife_days):
    tg = pd.read_parquet("data/processed/team_game_stats.parquet")
    tg["gameday"] = pd.to_datetime(tg["gameday"])
    rolled = pd.concat(
        [add_rolling(g.copy(), halflife_days, scheme) for _, g in tg.groupby("team")],
        ignore_index=True,
    )
    home = rolled[rolled["is_home"] == 1][
        ["game_id", "team", "opponent", "is_neutral_site", "is_playoff"]
        + BASE_FEATURE_COLS
    ].rename(columns={c: f"home_{c}" for c in BASE_FEATURE_COLS})
    home = home.rename(columns={"team": "home_team", "opponent": "away_team"})
    away = (
        rolled[rolled["is_home"] == 0][["game_id", "team"] + BASE_FEATURE_COLS]
        .rename(columns={c: f"away_{c}" for c in BASE_FEATURE_COLS})
        .rename(columns={"team": "away_team"})
    )
    merged = home.merge(away, on=["game_id", "away_team"], how="inner")

    inj = pd.read_parquet("data/processed/injury_features.parquet")
    for side in ("home", "away"):
        merged = merged.merge(
            inj[["game_id", "team", "injury_impact"]].rename(
                columns={
                    "team": f"{side}_team",
                    "injury_impact": f"{side}_injury_impact",
                }
            ),
            on=["game_id", f"{side}_team"],
            how="left",
        )
    s = pd.read_parquet("data/raw/schedules.parquet")
    final = merged.merge(
        s[
            [
                "game_id",
                "season",
                "week",
                "gameday",
                "home_score",
                "away_score",
                "overtime",
            ]
        ],
        on="game_id",
        how="left",
    )
    final["went_to_ot"] = final["overtime"].fillna(0).astype(int)
    final["gameday"] = pd.to_datetime(final["gameday"])
    return final.drop(columns=["overtime"])


def make_fns(kind):
    def _ridge():
        return make_pipeline(
            StandardScaler(), RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(n_splits=5))
        )

    def fit(train):
        X = train[FEATURE_COLS]
        means = X.mean()
        Xf = X.fillna(means)
        w = ot_sample_weight(train)
        if kind == "ridge":
            h, a = _ridge(), _ridge()
            h.fit(Xf, train["home_score"], ridgecv__sample_weight=w)
            a.fit(Xf, train["away_score"], ridgecv__sample_weight=w)
            sc, ph, pa = None, h.predict(Xf), a.predict(Xf)
        else:
            sc = StandardScaler().fit(Xf)
            Xs = sc.transform(Xf)
            h = PoissonRegressor(alpha=1.0, max_iter=300).fit(
                Xs, train["home_score"], sample_weight=w
            )
            a = PoissonRegressor(alpha=1.0, max_iter=300).fit(
                Xs, train["away_score"], sample_weight=w
            )
            ph, pa = h.predict(Xs), a.predict(Xs)
        oh, oa = recent_residual_offset(train, ph, pa)
        return {"h": h, "a": a, "means": means, "scaler": sc, "oh": oh, "oa": oa}

    def predict(m, test):
        X = test[FEATURE_COLS].fillna(m["means"])
        if m["scaler"] is not None:
            X = m["scaler"].transform(X)
        return m["h"].predict(X) - m["oh"], m["a"].predict(X) - m["oa"]

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
    halflife_days = load_config()["training"]["recency_half_life_weeks"] * 7
    schemes = ["calendar", "compressed", "games"]
    tables = {s: build_table(s, halflife_days) for s in schemes}

    for name, kind in [("Linear (RidgeCV)", "ridge"), ("Poisson GLM", "poisson")]:
        print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
        print(
            f"  {'decay scheme':<22}{'all wk':>9}{'wk 1-3':>9}{'wk 4+':>9}"
            f"{'total bias':>12}{'wk1-3 total':>13}"
        )
        print("  " + "-" * 74)
        results = {}
        for s in schemes:
            fit, predict = make_fns(kind)
            res = walk_forward_evaluate(tables[s], fit, predict, min_train_seasons=2)
            results[s] = res
            m = score_predictions(res)
            allw = (m["home_rmse"] + m["away_rmse"]) / 2
            early = res[res.week <= 3]
            late = res[res.week > 3]

            def mean_rmse(d):
                return (
                    np.sqrt(((d.home_pred - d.home_score) ** 2).mean())
                    + np.sqrt(((d.away_pred - d.away_score) ** 2).mean())
                ) / 2

            tb = (
                (res.home_pred + res.away_pred) - (res.home_score + res.away_score)
            ).mean()
            etb = (
                (early.home_pred + early.away_pred)
                - (early.home_score + early.away_score)
            ).mean()
            label = s + (" (current)" if s == "calendar" else "")
            print(
                f"  {label:<22}{allw:>9.4f}{mean_rmse(early):>9.4f}{mean_rmse(late):>9.4f}"
                f"{tb:>+12.3f}{etb:>+13.3f}"
            )

        print(f"\n  Paired bootstrap vs calendar ({N_BOOT} resamples over games)\n")
        for s in schemes[1:]:
            mean, lo, hi = bootstrap(results["calendar"], results[s])
            v = "noise" if lo <= 0 <= hi else ("BETTER" if hi < 0 else "WORSE")
            print(
                f"    {s:<22} delta={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {v}"
            )


if __name__ == "__main__":
    main()
