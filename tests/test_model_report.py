"""
The evaluation arithmetic behind src/validate/model_report.py, on hand-built
data with known answers. An evaluation that is silently wrong is worse than
none: it is what decides which models forecast real games.

Run: pytest tests/test_model_report.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.validate import model_report as mr


class TestKickoffSlot:
    @pytest.mark.parametrize(
        "day, time, slot",
        [
            ("2026-09-27", "13:00", "Sun early (1pm)"),
            ("2026-09-27", "16:25", "Sun late (4pm)"),
            ("2026-09-27", "20:20", "Sun night"),
            ("2026-09-28", "20:15", "Mon night"),
            ("2026-09-24", "20:15", "Thu night"),
            ("2026-11-26", "16:30", "Thu day (Thanksgiving)"),
            ("2026-12-19", "16:30", "Saturday"),
            ("2026-10-04", "09:30", "international/morning"),
            ("2026-12-25", "13:00", "other weekday"),  # a Friday Christmas game
        ],
    )
    def test_windows(self, day, time, slot):
        assert mr.kickoff_slot(pd.Timestamp(day).dayofweek, time) == slot

    def test_missing_time_is_unknown_not_a_guess(self):
        assert mr.kickoff_slot(6, None) == "unknown"
        assert mr.kickoff_slot(6, float("nan")) == "unknown"


class TestBlockBootstrap:
    def test_rmse_delta_point_estimate_is_exact(self):
        e_a = np.array([1.0, -1.0, 2.0, -2.0])
        e_b = np.array([2.0, -2.0, 2.0, -2.0])
        bb = mr.BlockBootstrap(np.array([1, 1, 2, 2]), n_boot=200)
        d, lo, hi, p = bb.rmse_delta(e_a, e_b)
        assert d == pytest.approx(np.sqrt(2.5) - 2.0)
        assert lo <= d <= hi

    def test_resampling_is_by_week_not_by_game(self):
        """With every game in one week identical, a block bootstrap has ONE
        block: every resample is the same, so the interval collapses to a point.
        A per-game bootstrap would still vary -- which is exactly the
        overconfidence the block version exists to avoid in the other direction
        (games in a week are not independent draws)."""
        x = np.array([1.0, 5.0, 9.0])
        lo, hi = mr.BlockBootstrap(np.array([7, 7, 7]), n_boot=500).mean_ci(x)
        assert lo == pytest.approx(hi) == pytest.approx(5.0)

    def test_a_clear_bias_gets_an_interval_excluding_zero(self):
        rng = np.random.default_rng(0)
        blocks = np.repeat(np.arange(60), 16)
        x = 1.5 + rng.normal(0, 9, len(blocks))
        lo, hi = mr.BlockBootstrap(blocks, n_boot=2000).mean_ci(x)
        assert lo > 0

    def test_intervals_are_reproducible(self):
        blocks = np.repeat(np.arange(20), 4)
        x = np.random.default_rng(1).normal(size=80)
        a = mr.BlockBootstrap(blocks, n_boot=300).mean_ci(x)
        b = mr.BlockBootstrap(blocks, n_boot=300).mean_ci(x)
        assert a == b


def long_frame():
    """Two models, four games, two weeks. Model 'hi' always predicts 3 points
    too high on both sides; model 'exact' is perfect."""
    rows = []
    games = [
        ("g1", 2025, 1, "AAA", "BBB", 24, 20),
        ("g2", 2025, 1, "CCC", "AAA", 17, 27),
        ("g3", 2025, 2, "BBB", "CCC", 30, 10),
        ("g4", 2025, 2, "AAA", "CCC", 21, 21),
    ]
    for model, off in (("hi", 3.0), ("exact", 0.0)):
        for gid, season, week, home, away, hs, as_ in games:
            rows.append(
                {
                    "model": model,
                    "game_id": gid,
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "home_score": hs,
                    "away_score": as_,
                    "home_pred": hs + off,
                    "away_pred": as_ + off,
                }
            )
    df = pd.DataFrame(rows)
    df["home_err"] = df["home_pred"] - df["home_score"]
    df["away_err"] = df["away_pred"] - df["away_score"]
    df["total_err"] = df["home_err"] + df["away_err"]
    df["margin_err"] = (df["home_pred"] - df["away_pred"]) - (
        df["home_score"] - df["away_score"]
    )
    df["block"] = df["season"] * 100 + df["week"]
    return df


class TestMissDistribution:
    def test_direction_and_size_of_a_known_bias(self):
        m = mr.miss_distribution(long_frame(), n_boot=200).set_index(
            ["model", "target"]
        )
        hi_home = m.loc[("hi", "home")]
        assert hi_home["mean_err"] == pytest.approx(3.0)
        assert hi_home["median_err"] == pytest.approx(3.0)
        assert hi_home["worst_over"] == pytest.approx(3.0)
        assert hi_home["pct_over"] == pytest.approx(1.0)
        assert hi_home["consistent_direction"] == "OVER-predicts"
        # an equal miss on both sides is +6 on the total and 0 on the margin
        assert m.loc[("hi", "total"), "mean_err"] == pytest.approx(6.0)
        assert m.loc[("hi", "margin"), "mean_err"] == pytest.approx(0.0)
        assert m.loc[("exact", "home"), "rmse"] == pytest.approx(0.0)


class TestTeamPerspective:
    def test_each_game_appears_once_per_team_with_the_right_orientation(self):
        tl = mr.team_long(long_frame())
        hi = tl[tl["model"] == "hi"]
        assert len(hi) == 8  # 4 games x 2 teams
        # AAA was home in g1 and g4 and away in g2
        aaa = hi[hi["team"] == "AAA"].set_index("game_id")
        assert list(aaa.loc[["g1", "g2", "g4"], "venue"]) == ["home", "away", "home"]

    def test_margin_error_flips_sign_for_the_away_team(self):
        df = long_frame()
        df.loc[0, "home_pred"] += 4  # model over-rates the HOME team of g1 by 4
        df["home_err"] = df["home_pred"] - df["home_score"]
        df["margin_err"] = (df["home_pred"] - df["away_pred"]) - (
            df["home_score"] - df["away_score"]
        )
        tl = mr.team_long(df).set_index(["model", "game_id", "team"])
        assert tl.loc[("hi", "g1", "AAA"), "margin_err"] == pytest.approx(4.0)
        assert tl.loc[("hi", "g1", "BBB"), "margin_err"] == pytest.approx(-4.0)
        assert tl.loc[("hi", "g1", "AAA"), "scored_err"] == pytest.approx(7.0)
        assert tl.loc[("hi", "g1", "BBB"), "allowed_err"] == pytest.approx(7.0)


class TestCommonGames:
    def test_stacking_does_not_shrink_everyone_elses_games(self):
        df = pd.DataFrame(
            {
                "model": ["a", "a", "b", "b", "stacking"],
                "game_id": ["g1", "g2", "g1", "g2", "g2"],
            }
        )
        assert mr.common_games(df) == {"g1", "g2"}
