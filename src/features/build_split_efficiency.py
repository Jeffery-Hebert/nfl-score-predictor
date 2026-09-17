"""
src/features/build_split_efficiency.py

Pregame pass/rush EPA efficiency, offense and defense, as four columns per team.

This is the SECOND attempt at this feature. The first one is a closed null
result in the project's findings doc, and reading why it failed is the whole
design brief for this file.

---------------------------------------------------------------- what failed

`src/experiments/test_pass_rush_split_only.py` (2026-09-14) replaced blended
`off_epa_per_play` / `def_epa_per_play` with their pass-only and rush-only
components and measured a real regression on Linear: +0.037 home RMSE, 95% CI
[+0.0058, +0.0685], excluding zero. The recorded diagnosis was that splitting a
full-game average roughly halves the effective play count per component, so
measurement noise rises faster than the added granularity helps.

That diagnosis is correct, and two further defects sat underneath it:

  1. IT WAS MEASURED WITH THE WRONG ESTIMATOR. The experiment used sklearn
     `LinearRegression` -- ordinary least squares, no regularization. The very
     next day the project diagnosed that exact estimator as unfit for p~20,
     n~1900 with correlated inputs, found accuracy degrading monotonically as
     features were added, and replaced it with RidgeCV. The pass/rush verdict
     was therefore recorded against an instrument the project subsequently
     declared broken, on a feature set that also had no injury_impact in it.

  2. NOTHING CORRECTED FOR THE NOISE IT DIAGNOSED. Having identified sample
     size as the failure mechanism, the implementation did nothing about it.
     Every game contributed equally to the rolling average regardless of
     whether the team ran the ball nine times or thirty-eight, and a
     four-game-old split average was handed to the model at face value.

------------------------------------------------------------- what is different

Three changes, each aimed at a specific defect above.

VOLUME WEIGHTING. The estimate is a recency-weighted rate PER PLAY, not a
recency-weighted mean of per-game rates:

        sum over prior games of  w_i * (EPA summed over that game's plays)
        -------------------------------------------------------------------
        sum over prior games of  w_i * (that game's play count)

with w_i = 0.5 ** (age_days / halflife). The old version is the special case
where every play count is 1. A 38-carry game now carries four times the
evidence of a 9-carry game, because it is four times the evidence.

SHRINKAGE TOWARD THE LEAGUE. A rate over few plays is mostly noise, so it is
pulled toward the league's own recency-weighted rate in proportion to how
little evidence stands behind it:

        estimate = (N_eff * team_rate + k * league_rate) / (N_eff + k)

N_eff is the recency-weighted play count actually behind the team number. k is
the empirical-Bayes constant: the number of plays at which a team's own rate
and the league prior deserve equal weight. Measured, not guessed -- see
SHRINKAGE_PLAYS below. The values are large (225 to 540 plays), which is the
quantitative form of the original diagnosis: after four games a team's own
pass-EPA number has earned about a quarter of the weight, and handing the raw
number to a model is handing it mostly noise.

The league rate is itself recency-weighted and computed from strictly prior
games, so the prior tracks the era (2020's passing environment differs from
2023's) without ever seeing the future.

THESE REPLACE BLENDED EPA. `pregame_off_epa_per_play` and
`pregame_def_epa_per_play` were dropped from BASE_FEATURE_COLS when this
shipped: they measure the same efficiency over all plays that the split
measures over pass and run separately, and the blend correlates 0.86 with the
shrunk pass split (0.58 rush; 0.78 / 0.45 on defence). Two descriptions of one
quantity is what an additive model can least afford.

That is structurally the same choice v1 made, and v1 was measurably worse for
it -- the difference is WHY it is now safe. v1 replaced a low-variance
measurement with two high-variance ones and left them raw. Here the variance is
handled where it arises, by weighting each game by its play count and shrinking
thin samples toward the league, so the granularity does not have to be paid for
in noise.

--------------------------------------------------------------- leakage position

Identical to every other rolling feature in the project, and asserted by
tests/test_split_efficiency_leakage.py:

  - a game's own plays never enter its own estimate; contributions are added to
    the accumulator only AFTER the value for that timestamp has been read;
  - games sharing a timestamp cannot see each other, which matters for the
    league prior (thirteen Sunday games must not inform each other) and is why
    the accumulation walks blocks of equal kickoff date rather than single rows;
  - finale-week results are masked out of what feeds forward, the same rule
    build_rolling_features.py applies, reusing its is_finale_week() rather than
    copying it a fourth time.

NOTE ON NAMING. build_rolling_features.py already emits
`pregame_off_pass_epa_per_play` and friends -- the v1, game-equal-weighted,
unshrunk versions. They are NOT in FEATURE_COLS, are read only by the archived
v1 experiments, and are deliberately left alone. The columns here carry a
`_shrunk` suffix so the two can never be confused in a table or a traceback.

Run: python -m src.features.build_split_efficiency
Output: data/processed/split_efficiency.parquet
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.features.build_rolling_features import is_finale_week

# Empirical-Bayes shrinkage constants, in plays: the point at which a team's own
# observed rate and the league prior carry equal weight.
#
# Measured 2026-09-17 from 2019-2026 play-by-play via the standard variance
# decomposition on team-season aggregates (>= 100 plays):
#
#     var(observed team rates) = var(true talent) + E[per-play variance / n]
#     k                        = per-play variance / var(true talent)
#
#                                 sigma^2/play   var_true    k
#     off pass EPA/play              2.548        0.01123    227
#     off rush EPA/play              1.029        0.00347    297
#     def pass EPA/play allowed      2.545        0.00599    425
#     def rush EPA/play allowed      1.033        0.00192    538
#
# Rounded below rather than carried to three figures -- they are estimates from
# 224 team-seasons, not constants of nature.
#
# Defense needs markedly more shrinkage than offense, which is not an artifact:
# defensive performance is independently measured as less persistent than
# offensive (split-half r = 0.33 against 0.53). A defense's pass-EPA-allowed is
# closer to noise than an offense's pass EPA, and this encodes that.
#
# Reproduce: python -m src.experiments.tune_split_shrinkage
SHRINKAGE_PLAYS = {
    "off_pass": 225.0,
    "off_rush": 300.0,
    "def_pass": 425.0,
    "def_rush": 540.0,
}

# key -> (per-play rate column, play-count column, output column)
SPLITS = {
    "off_pass": (
        "off_pass_epa_per_play",
        "off_pass_plays",
        "pregame_off_pass_epa_shrunk",
    ),
    "off_rush": (
        "off_rush_epa_per_play",
        "off_rush_plays",
        "pregame_off_rush_epa_shrunk",
    ),
    "def_pass": (
        "def_pass_epa_per_play_allowed",
        "def_pass_plays",
        "pregame_def_pass_epa_allowed_shrunk",
    ),
    "def_rush": (
        "def_rush_epa_per_play_allowed",
        "def_rush_plays",
        "pregame_def_rush_epa_allowed_shrunk",
    ),
}

OUTPUT_COLS = [out for _, _, out in SPLITS.values()]
# Effective play counts behind each estimate. Diagnostics and test assertions
# only -- deliberately NOT features. They would be a near-duplicate of
# prior_games_played, and this project's repeated lesson is that a redundant
# column costs more in variance than it returns.
EFF_COLS = [f"{out}_n_eff" for out in OUTPUT_COLS]


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def prior_weighted_rate(
    numerator: np.ndarray,
    denominator: np.ndarray,
    times: np.ndarray,
    halflife_days: float,
    include: np.ndarray | None = None,
):
    """Recency-weighted rate over rows STRICTLY BEFORE each row's timestamp.

    Returns (rate, n_eff), both the same length as the inputs.

      rate[i]  = sum(w * numerator)   / sum(w * denominator)  over j with
      n_eff[i] = sum(w * denominator)                          times[j] < times[i]

    where w = 0.5 ** ((times[i] - times[j]) / halflife_days).

    Rows are consumed in blocks of equal timestamp, and a block's own
    contribution is added only after its value has been read -- so simultaneous
    rows never inform each other. That is what makes this safe to run over the
    whole league, where a dozen games share a Sunday.

    `include` masks which rows may CONTRIBUTE to later estimates (finale-week
    games are excluded). Every row still receives a value.

    rate[i] is NaN where no prior evidence exists; callers decide what that
    means. n_eff is 0.0 there, never NaN, so it is always safe to compare.
    """
    n = len(numerator)
    rate = np.full(n, np.nan)
    n_eff = np.zeros(n)
    if n == 0:
        return rate, n_eff

    order = np.argsort(times, kind="stable")
    t = times[order]
    num = np.nan_to_num(numerator[order], nan=0.0)
    den = np.nan_to_num(denominator[order], nan=0.0)
    use = np.ones(n, dtype=bool) if include is None else include[order]

    # Running accumulators, always decayed to `anchor`.
    acc_num = 0.0
    acc_den = 0.0
    anchor = t[0]

    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1

        # Decay what we have to this block's timestamp, then read it.
        age_days = (t[i] - anchor) / np.timedelta64(1, "D")
        if age_days:
            decay = 0.5 ** (age_days / halflife_days)
            acc_num *= decay
            acc_den *= decay
            anchor = t[i]

        block = order[i:j]
        if acc_den > 0:
            rate[block] = acc_num / acc_den
            n_eff[block] = acc_den

        # Only now does this block become history.
        contrib = slice(i, j)
        acc_num += float(np.sum(num[contrib] * use[contrib]))
        acc_den += float(np.sum(den[contrib] * use[contrib]))

        i = j

    return rate, n_eff


def shrink(team_rate, n_eff, league_rate, k: float):
    """Pull a team's rate toward the league's in proportion to thin evidence.

    (n_eff * team + k * league) / (n_eff + k)

    With no team evidence at all (n_eff == 0) this returns the league rate
    exactly, which is the correct prior for a team nobody has observed yet --
    and is why the output has no NaNs to impute away. Where even the league
    prior is missing (the very first games in the dataset) the result is NaN and
    the models' existing training-fold mean imputation handles it.
    """
    team_rate = np.where(np.isnan(team_rate), 0.0, team_rate)
    return (n_eff * team_rate + k * league_rate) / (n_eff + k)


def build_split_efficiency(
    team_games: pd.DataFrame, halflife_days: float
) -> pd.DataFrame:
    """One row per (game_id, team) with four shrunk pregame split rates.

    Takes the WHOLE team-game table, not one team's slice. The league prior has
    to be accumulated across every team, so this cannot be a per-team loop with
    the prior computed inside it -- that would shrink each team toward itself,
    which is not shrinkage at all.
    """
    df = team_games.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)
    times = df["gameday"].to_numpy("datetime64[ns]")
    contributes = ~df.apply(
        lambda r: is_finale_week(r["season"], r["week"]), axis=1
    ).to_numpy()
    team_ids = df["team"].to_numpy()

    out = df[["game_id", "season", "week", "gameday", "team"]].copy()

    for key, (rate_col, count_col, out_col) in SPLITS.items():
        plays = np.nan_to_num(df[count_col].to_numpy(float), nan=0.0)
        # rate x count recovers the game's summed EPA, which is what a rate over
        # pooled plays has to be built from. A game with no plays of this type
        # contributes a clean zero to both sides rather than a NaN.
        epa_sum = np.nan_to_num(df[rate_col].to_numpy(float), nan=0.0) * plays

        # The prior: the same quantity over the entire league, same recency
        # weighting, same strictly-prior rule. Simultaneous games are kept from
        # informing each other by the block walk inside prior_weighted_rate.
        league_rate, _ = prior_weighted_rate(
            epa_sum, plays, times, halflife_days, include=contributes
        )

        team_rate = np.full(len(df), np.nan)
        n_eff = np.zeros(len(df))
        for team in np.unique(team_ids):
            idx = np.flatnonzero(team_ids == team)
            r, n = prior_weighted_rate(
                epa_sum[idx],
                plays[idx],
                times[idx],
                halflife_days,
                include=contributes[idx],
            )
            team_rate[idx] = r
            n_eff[idx] = n

        out[out_col] = shrink(team_rate, n_eff, league_rate, SHRINKAGE_PLAYS[key])
        out[f"{out_col}_n_eff"] = n_eff

    return out


def main():
    cfg = load_config()
    halflife_weeks = cfg["training"]["recency_half_life_weeks"]
    halflife_days = halflife_weeks * 7

    team_games = pd.read_parquet("data/processed/team_game_stats.parquet")
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])

    result = build_split_efficiency(team_games, halflife_days)

    out_path = Path("data/processed/split_efficiency.parquet")
    result.to_parquet(out_path, index=False)

    print(
        f"Built split efficiency for {result['team'].nunique()} teams, "
        f"{len(result)} team-game rows"
    )
    print(f"Halflife: {halflife_weeks} weeks ({halflife_days} days)")
    for out_col in OUTPUT_COLS:
        s = result[out_col]
        n_eff = result[f"{out_col}_n_eff"]
        print(
            f"  {out_col:38s} mean {s.mean():+.4f}  sd {s.std():.4f}  "
            f"null {s.isna().mean():.1%}  median n_eff {n_eff.median():.0f} plays"
        )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
