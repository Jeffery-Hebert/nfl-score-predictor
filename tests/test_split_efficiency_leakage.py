"""
Leakage and correctness gates for the pass/rush split efficiency features.

This builder does three things that are each easy to get quietly wrong, and all
three produce output that looks perfectly reasonable:

  1. it reads a running accumulator, so an off-by-one in WHEN a game is added
     lets a game inform its own estimate;
  2. it accumulates across the whole league for the prior, where a dozen games
     share a Sunday and must not see each other -- a plain cumulative sum would
     let the 1pm games leak into each other;
  3. it shrinks toward that prior by a play count, so a bug in the weighting
     silently changes how much the model trusts a number without changing
     anything about its shape.

Every assertion below is written against a hand-built history where the right
answer is known by hand, not against real data.

Run: pytest tests/test_split_efficiency_leakage.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_split_efficiency import (
    OUTPUT_COLS,
    SHRINKAGE_PLAYS,
    SPLITS,
    build_split_efficiency,
    prior_weighted_rate,
    shrink,
)

HALFLIFE = 119.0  # the production value, 17 weeks


def times(*days):
    return np.array([np.datetime64(d) for d in days])


# ------------------------------------------------- prior_weighted_rate itself


class TestStrictlyPrior:
    def test_first_row_has_no_estimate(self):
        rate, n_eff = prior_weighted_rate(
            np.array([10.0]), np.array([5.0]), times("2024-09-01"), HALFLIFE
        )
        assert np.isnan(rate[0]), "the first observation cannot have a prior"
        assert n_eff[0] == 0.0

    def test_a_row_never_sees_its_own_value(self):
        """THE core property. Row 2's estimate must be row 1 alone."""
        rate, n_eff = prior_weighted_rate(
            np.array([10.0, 1000.0]),
            np.array([10.0, 10.0]),
            times("2024-09-01", "2024-09-08"),
            HALFLIFE,
        )
        assert rate[1] == pytest.approx(1.0), (
            "the second row's estimate moved with its own value -- it is seeing "
            "the game being predicted"
        )

    def test_simultaneous_rows_cannot_see_each_other(self):
        """The league-prior case. Thirteen games kick off at 1pm on Sunday; none
        of them may inform another. A plain cumulative sum fails this."""
        rate, n_eff = prior_weighted_rate(
            np.array([10.0, 500.0, 500.0]),
            np.array([10.0, 10.0, 10.0]),
            times("2024-09-01", "2024-09-08", "2024-09-08"),
            HALFLIFE,
        )
        assert rate[1] == rate[2] == pytest.approx(1.0), (
            "same-day rows informed each other -- on real data this is a dozen "
            "Sunday games leaking into one another"
        )
        assert n_eff[1] == n_eff[2]

    def test_estimate_is_a_rate_over_pooled_plays_not_a_mean_of_rates(self):
        """The v1 defect, isolated.

        Game 1: 1.0 EPA over 40 plays (rate 0.025).
        Game 2: 1.0 EPA over 10 plays (rate 0.100).

        Mean of the two per-game rates is 0.0625. Pooled over all 50 plays it is
        2.0 / 50 = 0.04. The pooled number is the correct one, and it is the
        difference between weighting evidence and counting box scores.
        """
        rate, _ = prior_weighted_rate(
            np.array([1.0, 1.0, 0.0]),
            np.array([40.0, 10.0, 1.0]),
            times("2024-09-01", "2024-09-08", "2024-09-15"),
            halflife_days=1e9,  # no decay, so the arithmetic is exact
        )
        assert rate[2] == pytest.approx(2.0 / 50.0)
        assert rate[2] != pytest.approx(0.0625), "fell back to a mean of rates"

    def test_recent_games_outweigh_old_ones(self):
        rate, _ = prior_weighted_rate(
            np.array([0.0, 10.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            times("2020-01-01", "2024-09-08", "2024-09-15"),
            HALFLIFE,
        )
        # The recent 1.0-per-play game should dominate the ancient 0.0 one.
        assert rate[2] > 0.9

    def test_n_eff_reflects_volume_not_game_count(self):
        _, n_eff = prior_weighted_rate(
            np.array([0.0, 0.0]),
            np.array([40.0, 0.0]),
            times("2024-09-01", "2024-09-08"),
            halflife_days=1e9,
        )
        assert n_eff[1] == pytest.approx(40.0), "n_eff counted games, not plays"

    def test_masked_rows_receive_a_value_but_do_not_contribute(self):
        """Finale-week masking: the blowout still gets predicted, it just never
        feeds anything downstream."""
        include = np.array([True, False, True])
        rate, _ = prior_weighted_rate(
            np.array([0.0, 100.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            times("2024-09-01", "2024-09-08", "2024-09-15"),
            halflife_days=1e9,
            include=include,
        )
        assert not np.isnan(rate[1]), "a masked row must still be predictable"
        assert rate[2] == pytest.approx(0.0), (
            "the masked game leaked into the next estimate -- finale masking is "
            "not being applied to what feeds forward"
        )


# ------------------------------------------------------------------ shrinkage


class TestShrinkage:
    def test_no_evidence_returns_the_league_prior_exactly(self):
        got = shrink(np.array([np.nan]), np.array([0.0]), np.array([0.05]), k=225.0)
        assert got[0] == pytest.approx(0.05)

    def test_thin_evidence_stays_near_the_prior(self):
        """One game of rushing (~26 plays) against k=300 should barely move."""
        got = shrink(np.array([0.5]), np.array([26.0]), np.array([0.0]), k=300.0)
        assert abs(got[0]) < 0.05, "a single game moved the estimate too far"

    def test_heavy_evidence_approaches_the_team_number(self):
        got = shrink(np.array([0.5]), np.array([5000.0]), np.array([0.0]), k=225.0)
        assert got[0] > 0.47

    def test_shrinkage_is_monotone_in_evidence(self):
        n = np.array([10.0, 100.0, 1000.0, 10000.0])
        got = shrink(np.full(4, 1.0), n, np.zeros(4), k=225.0)
        assert list(got) == sorted(got), "more evidence must mean less shrinkage"

    def test_k_is_the_half_weight_point(self):
        """The definition of k: at n_eff == k the team and the prior are equal
        partners. If this drifts, SHRINKAGE_PLAYS no longer means what its
        docstring says."""
        got = shrink(np.array([1.0]), np.array([225.0]), np.array([0.0]), k=225.0)
        assert got[0] == pytest.approx(0.5)


# --------------------------------------------------------- the builder end to end


def history(rows, team="FAKE"):
    """rows: (gameday, season, week, off_pass_epa, off_pass_plays)."""
    df = pd.DataFrame(
        [
            {
                "game_id": f"g{i + 1}",
                "team": team,
                "season": s,
                "week": w,
                "gameday": pd.Timestamp(d),
                "off_pass_epa_per_play": e,
                "off_pass_plays": n,
            }
            for i, (d, s, w, e, n) in enumerate(rows)
        ]
    )
    # Every other split the builder reads, filled so it can run.
    for rate_col, count_col, _ in SPLITS.values():
        if rate_col not in df:
            df[rate_col] = 0.0
            df[count_col] = 30.0
    return df


class TestBuilder:
    def test_a_teams_own_game_never_enters_its_own_estimate(self):
        df = history(
            [
                ("2024-09-01", 2024, 1, 0.0, 40),
                ("2024-09-08", 2024, 2, 0.0, 40),
                ("2024-09-15", 2024, 3, 5.0, 40),  # an enormous outlier
            ]
        )
        out = build_split_efficiency(df, HALFLIFE)
        got = out["pregame_off_pass_epa_shrunk"].iloc[2]
        assert abs(got) < 0.2, (
            f"game 3's own +5.0 EPA/play reached its own pregame feature "
            f"(got {got:.3f}) -- this is the leak the whole file exists to stop"
        )

    def test_output_is_one_row_per_team_game_with_no_new_nulls(self):
        df = pd.concat(
            [
                history(
                    [
                        ("2024-09-01", 2024, 1, 0.1, 35),
                        ("2024-09-08", 2024, 2, 0.2, 35),
                    ],
                    team=t,
                )
                for t in ("AAA", "BBB")
            ],
            ignore_index=True,
        )
        out = build_split_efficiency(df, HALFLIFE)
        assert len(out) == 4
        assert not out.duplicated(subset=["game_id", "team"]).any()
        # Only the genuinely first-ever rows may be null.
        assert out[OUTPUT_COLS].isna().sum().max() <= 2

    def test_teams_playing_the_same_day_do_not_inform_each_other(self):
        """Two teams kick off simultaneously in week 1. Neither has any history,
        so neither may end up with an estimate -- and in particular AAA must not
        inherit BBB's enormous same-day performance through the league prior."""
        df = pd.concat(
            [
                history([("2024-09-08", 2024, 1, 0.0, 40)], team="AAA"),
                history([("2024-09-08", 2024, 1, 9.9, 40)], team="BBB"),
            ],
            ignore_index=True,
        )
        out = build_split_efficiency(df, HALFLIFE)
        week1 = out["pregame_off_pass_epa_shrunk"]
        assert week1.isna().all(), (
            f"a week-1 team with no league history got an estimate ({week1.tolist()}) "
            "-- the only place it could have come from is the other team's "
            "simultaneous game"
        )

    def test_a_cold_start_team_gets_the_league_prior_not_a_nan(self):
        """A team with no history still needs a number, and the honest one is
        'whatever the league has been doing'."""
        df = pd.concat(
            [
                history(
                    [
                        ("2024-09-01", 2024, 1, 0.30, 40),
                        ("2024-09-08", 2024, 2, 0.30, 40),
                    ],
                    team="OLD",
                ),
                history([("2024-09-15", 2024, 3, 0.0, 40)], team="NEW"),
            ],
            ignore_index=True,
        )
        out = build_split_efficiency(df, HALFLIFE)
        new = out[out["team"] == "NEW"]["pregame_off_pass_epa_shrunk"].iloc[0]
        n_eff = out[out["team"] == "NEW"]["pregame_off_pass_epa_shrunk_n_eff"].iloc[0]
        assert n_eff == 0.0, "a debut team has no evidence of its own"
        assert new == pytest.approx(
            0.30, abs=1e-9
        ), "a team with no history should inherit the league rate exactly"

    def test_volume_changes_the_answer(self):
        """Two histories with identical per-game RATES but different play counts
        must not produce the same estimate. Under v1's game-equal weighting they
        would -- that is precisely the bug.

        Asserted on game 3, which is the first row whose estimate actually sees
        both of the games that differ.
        """
        light = history(
            [
                ("2024-09-01", 2024, 1, 0.0, 40),
                ("2024-09-08", 2024, 2, 1.0, 5),  # good, but only 5 plays of it
                ("2024-09-15", 2024, 3, 0.0, 40),
            ]
        )
        heavy = history(
            [
                ("2024-09-01", 2024, 1, 0.0, 40),
                (
                    "2024-09-08",
                    2024,
                    2,
                    1.0,
                    40,
                ),  # the same rate, eight times the evidence
                ("2024-09-15", 2024, 3, 0.0, 40),
            ]
        )
        a = build_split_efficiency(light, HALFLIFE)["pregame_off_pass_epa_shrunk"].iloc[
            2
        ]
        b = build_split_efficiency(heavy, HALFLIFE)["pregame_off_pass_epa_shrunk"].iloc[
            2
        ]
        assert b > a, (
            f"the 40-play game and the 5-play game carried equal weight "
            f"({b:.4f} vs {a:.4f}) -- this is v1's game-equal averaging, back again"
        )


# ------------------------------------------------- built artifact sanity


@pytest.mark.requires_data
def test_built_features_are_sane():
    df = pd.read_parquet("data/processed/split_efficiency.parquet")
    assert not df.duplicated(subset=["game_id", "team"]).any()
    for col in OUTPUT_COLS:
        # Only the single first-ever game in the dataset may lack a prior.
        assert df[col].isna().sum() <= 2, f"{col} has unexpected nulls"
        finite = df[col].dropna()
        assert finite.abs().max() < 1.0, (
            f"{col} reaches {finite.abs().max():.2f} EPA/play -- shrinkage is "
            "not doing its job; league rates sit near 0.02"
        )


@pytest.mark.requires_data
def test_shrinkage_actually_compresses_the_spread():
    """The estimates must be tighter than the raw per-game rates they come from.
    If they are not, the shrinkage is a no-op and v1's noise problem is intact.
    """
    split = pd.read_parquet("data/processed/split_efficiency.parquet")
    raw = pd.read_parquet("data/processed/team_game_stats.parquet")
    assert (
        split["pregame_off_pass_epa_shrunk"].std() < raw["off_pass_epa_per_play"].std()
    )
    assert (
        split["pregame_def_rush_epa_allowed_shrunk"].std()
        < raw["def_rush_epa_per_play_allowed"].std()
    )


@pytest.mark.requires_data
def test_defence_is_shrunk_harder_than_offence():
    """Not cosmetic. Defensive performance is measurably less persistent
    (split-half r 0.33 against 0.53), so its estimates must be pulled toward the
    league prior harder. If these constants are ever 'tidied' to one number,
    this fails."""
    assert SHRINKAGE_PLAYS["def_pass"] > SHRINKAGE_PLAYS["off_pass"]
    assert SHRINKAGE_PLAYS["def_rush"] > SHRINKAGE_PLAYS["off_rush"]
