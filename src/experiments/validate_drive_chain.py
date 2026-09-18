"""
src/experiments/validate_drive_chain.py

Every validation that can be applied to the drive chain, including several this
project has never been able to run before.

EXPERIMENTAL. Writes a prediction file and a distribution file under
data/processed/; production is untouched.

--------------------------------------------------------------------- why new

Every other model here emits a POINT prediction. `src/validate/calibration.py`
says so explicitly:

    "these models emit point predictions only. GP computes a genuine predictive
     sigma and gaussian_process.py discards it ... implied_interval_coverage()
     assumes a normal with sigma = residual RMSE -- it measures whether residuals
     are dispersed the way a naive interval would assume, NOT true predictive
     calibration."

The chain emits a real predictive DISTRIBUTION per game, so for the first time
the honest versions of these questions can be asked:

  CRPS            the proper scoring rule for a distributional forecast. Rewards
                  being both accurate and appropriately uncertain. A forecast
                  that hedges everything scores badly; so does one that is
                  confidently wrong. Reduces to MAE for a point forecast, which
                  is what makes the comparison fair.

  TRUE COVERAGE   does the model's own 50/80/95% interval contain the real score
                  50/80/95% of the time? Every other model here can only be
                  asked whether a NORMAL fitted to its residuals would.

  WIN PROBABILITY Brier score and log loss against the actual winner, plus a
                  reliability curve. A point model can only say "home by 3",
                  which is a 100%-confidence claim dressed as a number.

  P(COVER)/P(OVER) the same, against the market line. This is what a betting
                  decision actually needs, and the project has never had it --
                  which is exactly why it sits at coin-flip ATS with no notion of
                  when to abstain.

Run: python -m src.experiments.validate_drive_chain
"""

import numpy as np
import pandas as pd

import src.experiments.drive_chain as dc
from src.validate.calibration import (
    bias,
    calibration_slope_intercept,
    reliability_table,
    rmse,
)
from src.validate.walk_forward import score_predictions, walk_forward_evaluate

N_TRIALS = 4000
SEED = 42
OUT_PRED = "data/processed/drivechain_predictions.parquet"
OUT_DIST = "data/processed/drivechain_distributions.npz"


# ----------------------------------------------------------------- collection


class DistributionRecorder:
    """Runs the chain through walk-forward while keeping each game's simulated
    score distribution, which the harness contract (home_pred, away_pred) has no
    way to carry."""

    def __init__(self):
        self.game_ids = []
        self.home_draws = []
        self.away_draws = []

    def fit(self, train):
        return dc.fit_chain(train)

    def predict(self, model, test):
        h, a = dc.predict_chain(model, test)
        cond, trans, open_, league = (
            model["cond"],
            model["trans"],
            model["open"],
            model["league"],
        )
        ho = dc._marginal(test, "home", "off")
        ao = dc._marginal(test, "away", "off")
        hd = dc._marginal(test, "home", "def")
        ad = dc._marginal(test, "away", "def")
        for m in (ho, ao, hd, ad):
            bad = ~np.isfinite(m).all(axis=1) | (np.nansum(m, axis=1) <= 0)
            m[bad] = league

        for i, gid in enumerate(test["game_id"].to_numpy()):
            h_mix = (ho[i] / ho[i].sum() + ad[i] / ad[i].sum()) / 2
            a_mix = (ao[i] / ao[i].sum() + hd[i] / hd[i].sum()) / 2
            hc = dc.apply_team_quality(cond, h_mix, league)
            ac = dc.apply_team_quality(cond, a_mix, league)
            hs, as_ = dc.simulate_games(
                hc, ac, trans, open_, n_trials=N_TRIALS, seed=SEED
            )
            # Shift the sampled cloud onto the exact expectation. The mean is
            # computed analytically and carries the home-field and drift
            # corrections; the simulation supplies the SHAPE around it. Without
            # this the distribution would be centred on the raw chain mean and
            # every coverage number would inherit that offset.
            self.game_ids.append(gid)
            self.home_draws.append(hs - hs.mean() + h[i])
            self.away_draws.append(as_ - as_.mean() + a[i])
        return h, a


# -------------------------------------------------------------------- scoring


def crps_ensemble(draws: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Continuous Ranked Probability Score, per game, from an ensemble.

        CRPS = E|X - y|  -  0.5 * E|X - X'|

    The first term rewards accuracy, the second penalises being needlessly
    vague. A point forecast has no spread, so its second term is zero and CRPS
    collapses to absolute error -- which is what makes a distributional forecast
    directly comparable to a point one on the same scale. Lower is better.
    """
    n, m = draws.shape
    out = np.empty(n)
    for i in range(n):
        x = np.sort(draws[i])
        term1 = np.abs(x - actual[i]).mean()
        # E|X - X'| for a sorted sample, in O(m) rather than O(m^2)
        k = np.arange(1, m + 1)
        term2 = 2.0 * np.sum((2 * k - m - 1) * x) / (m * m)
        out[i] = term1 - 0.5 * term2
    return out


def coverage(draws: np.ndarray, actual: np.ndarray, levels=(0.50, 0.80, 0.95)) -> dict:
    """Fraction of actuals inside the model's OWN central interval."""
    out = {}
    for lv in levels:
        lo = np.percentile(draws, 50 * (1 - lv), axis=1)
        hi = np.percentile(draws, 50 * (1 + lv), axis=1)
        out[lv] = float(np.mean((actual >= lo) & (actual <= hi)))
    return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps=1e-9) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p: np.ndarray, y: np.ndarray, n_bins=5) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    return (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
        .reset_index(drop=True)
    )


def main():
    dcd = dc.extract_chain_data()
    sch = pd.read_parquet(dc.SCHEDULES)[["game_id", "gameday"]]
    sch["gameday"] = pd.to_datetime(sch["gameday"])
    dc._CHAIN_DATA = dcd.merge(sch, on="game_id", how="left")

    df = dc.load_table()
    rec = DistributionRecorder()
    res = walk_forward_evaluate(df, rec.fit, rec.predict, min_train_seasons=2)
    res.to_parquet(OUT_PRED, index=False)

    order = {g: i for i, g in enumerate(rec.game_ids)}
    idx = res["game_id"].map(order).to_numpy()
    H = np.stack([rec.home_draws[i] for i in idx])
    A = np.stack([rec.away_draws[i] for i in idx])
    np.savez_compressed(OUT_DIST, game_ids=res["game_id"].to_numpy(), home=H, away=A)

    ha = res["home_score"].to_numpy(float)
    aa = res["away_score"].to_numpy(float)
    hp = res["home_pred"].to_numpy(float)
    ap = res["away_pred"].to_numpy(float)

    print("=" * 80)
    print("DRIVE CHAIN -- FULL VALIDATION")
    print("=" * 80)
    print(f"  {len(res)} games, {N_TRIALS} simulations each\n")

    # ---- 1. point accuracy, the project's standard measures
    m = score_predictions(res)
    print("1. POINT ACCURACY (the project's usual scoreboard)")
    print(
        f"   home RMSE {m['home_rmse']:.3f}   away RMSE {m['away_rmse']:.3f}   "
        f"mean {(m['home_rmse'] + m['away_rmse']) / 2:.4f}"
    )
    print(
        f"   margin RMSE {rmse(ha - aa, hp - ap):.3f}   "
        f"total RMSE {rmse(ha + aa, hp + ap):.3f}"
    )
    print(f"   reference: drive_model_v2 9.4635 | baseline 9.458 | poisson 9.367")

    # ---- 2. bias and calibration
    slope, intercept = calibration_slope_intercept(ha, hp)
    print("\n2. BIAS AND CALIBRATION")
    print(f"   home bias {bias(ha, hp):+.3f}   away bias {bias(aa, ap):+.3f}")
    print(
        f"   calibration slope {slope:.3f} (1.0 = perfect), intercept {intercept:+.2f}"
    )
    print("   reliability by predicted-score bin:")
    print(reliability_table(ha, hp).to_string(index=False))

    # ---- 3. CRPS, which needs a distribution
    total_draws = H + A
    margin_draws = H - A
    print("\n3. CRPS -- proper scoring rule for a DISTRIBUTION (lower is better)")
    for label, draws, actual, point in (
        ("home score", H, ha, hp),
        ("total", total_draws, ha + aa, hp + ap),
        ("margin", margin_draws, ha - aa, hp - ap),
    ):
        c = crps_ensemble(draws, actual).mean()
        mae = np.abs(point - actual).mean()
        print(
            f"   {label:12s} CRPS {c:6.3f}   vs point-forecast MAE {mae:6.3f}   "
            f"{'better' if c < mae else 'worse'} by {abs(c - mae):.3f}"
        )
    print(
        "   (a point forecast's CRPS IS its absolute error, so this is like-for-like)"
    )

    # ---- 4. true interval coverage
    print("\n4. TRUE INTERVAL COVERAGE -- does its own interval contain reality?")
    for label, draws, actual in (
        ("home score", H, ha),
        ("total", total_draws, ha + aa),
        ("margin", margin_draws, ha - aa),
    ):
        cov = coverage(draws, actual)
        line = "   ".join(f"{int(k * 100)}%: {v:.1%}" for k, v in cov.items())
        print(f"   {label:12s} {line}")
    print("   (every other model here can only be asked whether a NORMAL fitted")
    print("    to its residuals would cover -- this is the real thing)")

    # ---- 5. win probability
    won = (ha > aa).astype(float)
    played = ha != aa
    p_win = (margin_draws > 0).mean(axis=1)
    print("\n5. WIN PROBABILITY (a point model cannot produce one at all)")
    print(
        f"   Brier score {brier(p_win[played], won[played]):.4f}   "
        f"(0.25 = always saying 50%, lower is better)"
    )
    print(
        f"   log loss    {log_loss(p_win[played], won[played]):.4f}   "
        f"(0.693 = always saying 50%)"
    )
    print("   reliability -- predicted vs actual win rate:")
    rel = reliability(p_win[played], won[played])
    for _, r in rel.iterrows():
        print(
            f"      predicted {r['predicted']:.1%}  actual {r['actual']:.1%}  "
            f"(n={r['n']:.0f})"
        )

    # ---- 6. market-relative probabilities
    sched = pd.read_parquet(dc.SCHEDULES)[["game_id", "spread_line", "total_line"]]
    j = res.merge(sched, on="game_id", how="left").dropna(subset=["spread_line"])
    ji = j.index.to_numpy()
    mask = np.isin(res.index.to_numpy(), ji)
    md = margin_draws[mask]
    td = total_draws[mask]
    true_margin = (j["home_score"] - j["away_score"]).to_numpy(float)
    true_total = (j["home_score"] + j["away_score"]).to_numpy(float)
    spread = j["spread_line"].to_numpy(float)
    tot_line = j["total_line"].to_numpy(float)

    p_cover = (md > spread[:, None]).mean(axis=1)
    covered = (true_margin > spread).astype(float)
    live = true_margin != spread
    p_over = (td > tot_line[:, None]).mean(axis=1)
    went_over = (true_total > tot_line).astype(float)
    live_t = true_total != tot_line

    print("\n6. AGAINST THE MARKET -- probabilities, not picks")
    print(
        f"   P(cover): Brier {brier(p_cover[live], covered[live]):.4f}   "
        f"log loss {log_loss(p_cover[live], covered[live]):.4f}"
    )
    print(
        f"   P(over) : Brier {brier(p_over[live_t], went_over[live_t]):.4f}   "
        f"log loss {log_loss(p_over[live_t], went_over[live_t]):.4f}"
    )
    print("   (0.25 / 0.693 is what a coin flip scores; beating them means the")
    print("    model knows something about the market, not just about football)")

    print("\n   P(cover) reliability -- is a stated 60% really 60%?")
    for _, r in reliability(p_cover[live], covered[live]).iterrows():
        print(
            f"      said {r['predicted']:.1%}  actual {r['actual']:.1%}  (n={r['n']:.0f})"
        )

    # ---- 7. does confidence identify better bets?
    print("\n7. DOES ITS OWN CONFIDENCE PICK BETTER BETS?")
    print("   The practical question. A distribution is only useful if the games")
    print("   it is most sure about are the ones it gets right.")
    conf = np.abs(p_cover - 0.5)
    pick_right = ((p_cover > 0.5) == (covered > 0.5))[live]
    cl = conf[live]
    for lo, hi in ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)):
        sel = (cl >= lo) & (cl < hi)
        if sel.sum() < 30:
            continue
        print(
            f"      confidence {lo:.0%}-{hi:.0%} from even:  "
            f"{pick_right[sel].mean():.1%} correct   (n={sel.sum()})"
        )
    print("      break-even at -110 is 52.4%")


if __name__ == "__main__":
    main()
