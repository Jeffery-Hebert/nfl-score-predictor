"""
Leakage and correctness gates for the rebuilt opponent-adjusted ratings.

This builder has a leakage shape the rolling features do not: a rating is a
LEAGUE-WIDE quantity. Team A's number depends on team B's games, which depend on
team C's, so the cutoff has to be global. A per-team cutoff would look correct,
produce sensible-looking ratings, and quietly let one team's Sunday result
inform another team's Sunday prediction through the joint solve.

The second risk is orientation. The design matrix is [offence dummies | defence
dummies | home], and reading the coefficient block back in the wrong order gives
you every team's defensive rating labelled as its offence. Ratings would still
be plausible, still be centred, still rank teams -- just backwards.

Run: pytest tests/test_adjusted_ratings_v2.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.experiments.adjusted_ratings_v2 import (
    RATING_COLS,
    build_adjusted_ratings,
    fit_unit_ratings,
)


def games(rows) -> pd.DataFrame:
    """rows: (gameday, season, week, team, opponent, is_home, pass_epa, pass_plays)."""
    recs = []
    for i, (day, season, week, team, opp, home, epa, plays) in enumerate(rows):
        recs.append(
            {
                "game_id": f"g{i}",
                "gameday": pd.Timestamp(day),
                "season": season,
                "week": week,
                "team": team,
                "opponent": opp,
                "is_home": home,
                "off_pass_epa_per_play": epa,
                "off_pass_plays": plays,
                "off_rush_epa_per_play": epa,
                "off_rush_plays": plays,
            }
        )
    return pd.DataFrame(recs)


def round_robin(n_teams=12, n_weeks=22, seed=0, strength=None):
    """A synthetic league where each team has a known true offensive strength.

    Every team plays every week, so the joint solve is identifiable, and the
    recovered ratings can be checked against the strengths that generated them.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = strength or {t: rng.normal(0, 0.15) for t in teams}
    rows = []
    day = pd.Timestamp("2024-09-08")
    for w in range(1, n_weeks + 1):
        order = list(teams)
        rng.shuffle(order)
        for a, b in zip(order[::2], order[1::2]):
            for team, opp, home in ((a, b, 1), (b, a, 0)):
                epa = strength[team] + rng.normal(0, 0.02)
                rows.append((day, 2024, w, team, opp, home, epa, 35))
        day += pd.Timedelta(days=7)
    return games(rows), strength


class TestLeakage:
    def test_a_weeks_ratings_use_only_games_before_that_week(self):
        """The core property, checked by making the final week extreme: if it
        leaked, the ratings AT that week would move."""
        base, strength = round_robin(seed=1)
        spiked = base.copy()
        last = spiked["week"] == spiked["week"].max()
        spiked.loc[last, "off_pass_epa_per_play"] = 5.0  # absurd

        a = build_adjusted_ratings(base).set_index(["week", "team"])
        b = build_adjusted_ratings(spiked).set_index(["week", "team"])
        common = a.index.intersection(b.index)
        for col in RATING_COLS:
            np.testing.assert_allclose(
                a.loc[common, col].to_numpy(),
                b.loc[common, col].to_numpy(),
                atol=1e-9,
                err_msg=f"{col} moved when only the FINAL week's results changed",
            )

    def test_the_cutoff_is_league_wide_not_per_team(self):
        """A rating depends on every team's games through the joint solve, so a
        per-team cutoff would let one team's result reach another's rating for
        the same week."""
        base, _ = round_robin(seed=2)
        # Blow up ONE team's most recent game only.
        spiked = base.copy()
        last_week = spiked["week"].max()
        target = (spiked["week"] == last_week) & (spiked["team"] == "T0")
        spiked.loc[target, "off_pass_epa_per_play"] = 9.0

        a = build_adjusted_ratings(base).set_index(["week", "team"])
        b = build_adjusted_ratings(spiked).set_index(["week", "team"])
        common = a.index.intersection(b.index)
        np.testing.assert_allclose(
            a.loc[common, "adj_def_pass_allowed"].to_numpy(),
            b.loc[common, "adj_def_pass_allowed"].to_numpy(),
            atol=1e-9,
            err_msg="one team's latest game reached another team's rating",
        )

    def test_no_ratings_before_there_is_enough_evidence(self):
        small, _ = round_robin(n_teams=4, n_weeks=3)
        out = build_adjusted_ratings(small)
        assert out.empty or out["week"].min() > 1, (
            "ratings were produced from almost no history, where 64 team effects "
            "cannot be identified"
        )


class TestRecoversTruth:
    def test_a_stronger_offense_gets_a_higher_rating(self):
        """The joint solve has to actually work, not merely run."""
        strength = {
            "T0": 0.40,
            "T1": 0.20,
            "T2": 0.0,
            "T3": -0.20,
            "T4": -0.40,
            "T5": 0.0,
        }
        df, _ = round_robin(n_teams=6, n_weeks=40, seed=3, strength=strength)
        out = build_adjusted_ratings(df)
        final = out[out["week"] == out["week"].max()].set_index("team")
        assert final.loc["T0", "adj_off_pass"] > final.loc["T2", "adj_off_pass"]
        assert final.loc["T2", "adj_off_pass"] > final.loc["T4", "adj_off_pass"]

    def test_ratings_correlate_with_the_strengths_that_generated_them(self):
        strength = {
            f"T{i}": v for i, v in enumerate([0.4, 0.25, 0.1, -0.1, -0.25, -0.4])
        }
        df, _ = round_robin(n_teams=6, n_weeks=40, seed=4, strength=strength)
        out = build_adjusted_ratings(df)
        final = out[out["week"] == out["week"].max()].set_index("team")
        r = np.corrcoef(
            [strength[t] for t in final.index], final["adj_off_pass"].to_numpy()
        )[0, 1]
        assert r > 0.9, f"recovered ratings correlate only {r:.2f} with the truth"


class TestPlayWeighting:
    def test_a_high_volume_game_carries_more_weight_than_a_low_volume_one(self):
        """The v2 correction. Under v1's unweighted fit these two would give the
        same rating."""
        teams = ["A", "B"]
        common = [
            ("2024-09-08", 2024, 1, "A", "B", 1, 0.0, 35),
            ("2024-09-08", 2024, 1, "B", "A", 0, 0.0, 35),
        ]
        few = games(
            common
            + [
                ("2024-09-15", 2024, 2, "A", "B", 1, 1.0, 3),
                ("2024-09-15", 2024, 2, "B", "A", 0, 0.0, 35),
            ]
        )
        many = games(
            common
            + [
                ("2024-09-15", 2024, 2, "A", "B", 1, 1.0, 60),
                ("2024-09-15", 2024, 2, "B", "A", 0, 0.0, 35),
            ]
        )
        cutoff = pd.Timestamp("2024-09-22")
        a, _ = fit_unit_ratings(
            few, teams, "off_pass_epa_per_play", "off_pass_plays", cutoff, min_rows=1
        )
        b, _ = fit_unit_ratings(
            many, teams, "off_pass_epa_per_play", "off_pass_plays", cutoff, min_rows=1
        )
        assert b["A"] > a["A"], (
            "a 60-play performance did not outweigh a 3-play one -- the fit is "
            "still counting games rather than plays"
        )


class TestOrientation:
    def test_offense_and_defense_blocks_are_not_swapped(self):
        """Give one team a great offence and check that shows up in its OFFENCE
        rating, not in its defensive one."""
        strength = {"T0": 0.6, "T1": 0.0, "T2": 0.0, "T3": 0.0}
        df, _ = round_robin(n_teams=4, n_weeks=60, seed=5, strength=strength)
        out = build_adjusted_ratings(df)
        final = out[out["week"] == out["week"].max()].set_index("team")
        others = [t for t in final.index if t != "T0"]
        assert (
            final.loc["T0", "adj_off_pass"] > final.loc[others, "adj_off_pass"].max()
        ), (
            "the team with the best offence does not have the best offensive "
            "rating -- the coefficient blocks are likely swapped"
        )

    def test_facing_a_strong_offense_shows_up_as_defense_allowed(self):
        """A defence that keeps meeting T0 should not be penalised for it -- that
        is the entire purpose of adjusting for opponent."""
        strength = {"T0": 0.6, "T1": 0.0, "T2": 0.0, "T3": 0.0}
        df, _ = round_robin(n_teams=4, n_weeks=60, seed=6, strength=strength)
        out = build_adjusted_ratings(df)
        final = out[out["week"] == out["week"].max()].set_index("team")
        spread = final["adj_def_pass_allowed"].std()
        assert spread < 0.25, (
            f"defensive ratings vary by {spread:.3f} when every defence is "
            "identical by construction -- opponent strength is leaking into them"
        )


@pytest.mark.requires_data
class TestAgainstRealData:
    @pytest.fixture(scope="class")
    def built(self):
        return build_adjusted_ratings()

    def test_every_team_has_a_rating_at_every_cutoff(self, built):
        counts = built.groupby(["season", "week"])["team"].nunique()
        assert counts.min() == 32, (
            "some cutoffs are missing teams; ratings would not be comparable "
            "across weeks"
        )

    def test_ratings_are_centred_and_bounded(self, built):
        for c in RATING_COLS:
            v = built[c]
            assert abs(v.mean()) < 0.02, f"{c} is not centred (mean {v.mean():+.4f})"
            assert v.abs().max() < 1.0, f"{c} reaches {v.abs().max():.2f} EPA/play"

    def test_pass_and_rush_ratings_are_genuinely_different(self, built):
        """The headline v2 change. If these correlate near 1.0 the split has
        bought nothing and v1's single blended rating was adequate."""
        r = built["adj_off_pass"].corr(built["adj_off_rush"])
        assert abs(r) < 0.8, (
            f"pass and rush ratings correlate {r:+.2f} -- separating them added "
            "no information"
        )
