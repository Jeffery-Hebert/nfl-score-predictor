"""
src/experiments/tune_split_shrinkage.py

Derives SHRINKAGE_PLAYS in src/features/build_split_efficiency.py.

Those four constants decide how hard a team's observed pass/rush EPA rate is
pulled toward the league average, and they are the whole reason v2 of the
pass/rush split is not v1. A constant nobody can reproduce is a guess wearing a
number, and this project has already been bitten once by a hyperparameter that
sat unvalidated in config.yaml for its entire life (the 17-week half-life, see
tune_halflife.py). So the derivation lives in the repo next to the number.

--------------------------------------------------------------------- the model

Treat each team-season's observed rate as a noisy read on a true talent:

    observed_i = true_i + noise_i,     var(noise_i) = sigma^2_play / n_i

Then across teams:

    var(observed) = var(true) + E[sigma^2_play / n]

so the unobservable talent spread falls out of two things we can measure:

    var(true) = var(observed) - E[sigma^2_play / n]

and the shrinkage constant -- the play count at which a team's own number and
the league prior deserve equal weight -- is the ratio:

    k = sigma^2_play / var(true)

This is the standard James-Stein / empirical-Bayes setup. Nothing exotic; the
only judgement calls are the grain (team-season) and the minimum sample.

-------------------------------------------------------------------- leakage note

This estimates a HYPERPARAMETER from the full dataset, exactly as
tune_halflife.py and tune_ridge_alpha.py do. That is a weaker form of the
concern than a feature seeing the future -- it is four numbers describing how
noisy football statistics are in general, not anything about a specific game --
but it IS the same class of thing as choosing alpha=100 on the test set, which
this project explicitly refused. Two reasons it is acceptable here where that
was not:

  1. k describes measurement noise, a property of the sport's sampling
     process, not of the target. It would be very nearly the same number
     estimated on any few seasons of NFL play-by-play.
  2. the constants are pinned in source and rounded, not re-fit per fold, so
     no fold can tune itself.

If that ever stops feeling adequate, the fix is to estimate k inside each
training fold. It was not done that way because the estimate is stable to the
rounding already applied, and a per-fold estimate would add a moving part for
no measurable gain.

Writes nothing. Run: python -m src.experiments.tune_split_shrinkage
"""

import numpy as np
import pandas as pd

from src.features.build_split_efficiency import SHRINKAGE_PLAYS

MIN_PLAYS = 100  # a team-season below this is too thin to inform the estimate

# (label, key in SHRINKAGE_PLAYS, possession column, play_type)
TARGETS = [
    ("off pass EPA/play", "off_pass", "posteam", "pass"),
    ("off rush EPA/play", "off_rush", "posteam", "run"),
    ("def pass EPA/play allowed", "def_pass", "defteam", "pass"),
    ("def rush EPA/play allowed", "def_rush", "defteam", "run"),
]


def estimate_k(pbp: pd.DataFrame, side_col: str, play_type: str) -> dict:
    q = pbp[(pbp["play_type"] == play_type) & pbp[side_col].notna()]
    g = (
        q.groupby(["season", side_col])["epa"]
        .agg(n="size", mean="mean", var="var")
        .reset_index()
    )
    g = g[g["n"] >= MIN_PLAYS]

    var_observed = float(g["mean"].var(ddof=1))
    sigma2_play = float(g["var"].mean())
    var_noise = float((g["var"] / g["n"]).mean())
    var_true = max(var_observed - var_noise, 1e-9)

    return {
        "team_seasons": len(g),
        "sigma2_play": sigma2_play,
        "var_observed": var_observed,
        "var_noise": var_noise,
        "var_true": var_true,
        "k": sigma2_play / var_true,
    }


def main():
    pbp = pd.read_parquet(
        "data/raw/pbp.parquet",
        columns=["season", "posteam", "defteam", "play_type", "epa"],
    )
    pbp = pbp[pbp["epa"].notna()]

    print("=" * 78)
    print("SHRINKAGE CONSTANTS for build_split_efficiency.SHRINKAGE_PLAYS")
    print("=" * 78)
    print(f"  grain: team-season, minimum {MIN_PLAYS} plays")
    print(f"  seasons: {pbp['season'].min()}-{pbp['season'].max()}\n")
    print(
        f"{'split':28s}{'sigma2/play':>13}{'var_true':>11}"
        f"{'k (measured)':>14}{'k (pinned)':>12}"
    )
    print("-" * 78)

    for label, key, side_col, play_type in TARGETS:
        r = estimate_k(pbp, side_col, play_type)
        print(
            f"{label:28s}{r['sigma2_play']:>13.3f}{r['var_true']:>11.5f}"
            f"{r['k']:>14.0f}{SHRINKAGE_PLAYS[key]:>12.0f}"
        )

    print("\nWhat these mean in football terms (a team sees ~36 pass, ~26 rush")
    print("plays a game, so ~31 on average per split):\n")
    print(f"{'games of evidence':>20}{'  weight on the team''s own number':>38}")
    for games in (1, 2, 4, 8, 17, 34):
        n = games * 31
        weights = "  ".join(
            f"{label.split()[0]}/{label.split()[1]}:{n / (n + SHRINKAGE_PLAYS[key]):>5.0%}"
            for label, key, _, _ in TARGETS
        )
        print(f"{games:>17} gm  {weights}")

    print(
        "\nThe pinned values are these rounded. They are estimates from a few\n"
        "hundred team-seasons, not constants of nature -- three significant\n"
        "figures would be false precision."
    )


if __name__ == "__main__":
    main()
