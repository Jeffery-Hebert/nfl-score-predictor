"""
Correctness gates for starting field position.

The extraction has one trap that produces a plausible and entirely WRONG answer:
take the first play of each drive and you pick up KICKOFFS at the 35, which
piles half of all drives into a single bucket and flattens the expected-points
curve. I hit that trap on the first attempt and the resulting curve looked
perfectly reasonable -- 4.2 EP at the top, 1.4 at the bottom, monotone. It was
wrong. Nothing about the output said so.

So the extraction is checked two ways: against nflverse's own
`drive_start_yard_line` string on real data, and against hand-built drives where
the answer is arithmetic.

The other thing pinned here is the set of facts that decide whether this feature
is worth having, kept explicitly separate because conflating them is what I did
the first time:

    1. does field position matter per drive          -> yes, hugely
    2. do teams differ from each other               -> yes, ~7.7 yards
    3. is it knowable in advance                     -> barely, r ~ 0.11

Run: pytest tests/test_field_position.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.experiments.field_position import (
    FP_COLS,
    build_pregame_field_position,
    expected_points_curve,
    extract_drive_starts,
    team_game_field_position,
    within_team_slope,
)


def plays(rows) -> pd.DataFrame:
    """rows: (play_id, drive, posteam, defteam, down, yardline_100, result).

    `down` is None for kickoffs and punts -- which is exactly the signal the
    extraction uses, so fixtures must include some.
    """
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "play_id": pid,
                "drive": dr,
                "posteam": pos,
                "defteam": dff,
                "down": dn,
                "yardline_100": yl,
                "fixed_drive_result": res,
            }
            for pid, dr, pos, dff, dn, yl, res in rows
        ]
    )


class TestExtraction:
    def test_kickoffs_are_not_mistaken_for_the_drive_start(self):
        """THE trap. A kickoff sits at the 35 and has no down; the drive really
        starts where the offence first snaps it."""
        df = plays(
            [
                (1, 1.0, "AAA", "BBB", None, 35.0, "Touchdown"),  # kickoff
                (2, 1.0, "AAA", "BBB", 1.0, 75.0, "Touchdown"),  # real start, own 25
                (3, 1.0, "AAA", "BBB", 2.0, 68.0, "Touchdown"),
            ]
        )
        got = extract_drive_starts(df)
        assert len(got) == 1
        assert got.iloc[0]["start_100"] == 75.0, (
            f"drive start came out {got.iloc[0]['start_100']} -- the kickoff at "
            "the 35 was taken as the start"
        )

    def test_punts_are_not_mistaken_for_the_drive_start(self):
        df = plays(
            [
                (1, 2.0, "AAA", "BBB", None, 40.0, "Punt"),  # punt, no down
                (2, 2.0, "AAA", "BBB", 1.0, 80.0, "Punt"),
            ]
        )
        assert extract_drive_starts(df).iloc[0]["start_100"] == 80.0

    def test_the_first_scrimmage_play_wins_not_the_lowest_yardline(self):
        """A drive that advances must be recorded at where it BEGAN."""
        df = plays(
            [
                (1, 1.0, "AAA", "BBB", 1.0, 75.0, "Touchdown"),
                (2, 1.0, "AAA", "BBB", 1.0, 40.0, "Touchdown"),
                (3, 1.0, "AAA", "BBB", 1.0, 5.0, "Touchdown"),
            ]
        )
        assert extract_drive_starts(df).iloc[0]["start_100"] == 75.0

    def test_plays_are_ordered_by_play_id_not_frame_order(self):
        """Shuffled input must still yield the first play chronologically."""
        df = plays(
            [
                (3, 1.0, "AAA", "BBB", 1.0, 5.0, "Touchdown"),
                (1, 1.0, "AAA", "BBB", 1.0, 75.0, "Touchdown"),
                (2, 1.0, "AAA", "BBB", 1.0, 40.0, "Touchdown"),
            ]
        )
        assert extract_drive_starts(df).iloc[0]["start_100"] == 75.0

    def test_each_drive_is_one_row_with_its_own_possessor(self):
        df = plays(
            [
                (1, 1.0, "AAA", "BBB", 1.0, 75.0, "Punt"),
                (2, 2.0, "BBB", "AAA", 1.0, 60.0, "Touchdown"),
            ]
        )
        got = extract_drive_starts(df).sort_values("drive")
        assert list(got["posteam"]) == ["AAA", "BBB"]
        assert list(got["start_100"]) == [75.0, 60.0]

    def test_points_are_attached_from_the_drive_result(self):
        df = plays(
            [
                (1, 1.0, "AAA", "BBB", 1.0, 75.0, "Touchdown"),
                (2, 2.0, "AAA", "BBB", 1.0, 75.0, "Field goal"),
                (3, 3.0, "AAA", "BBB", 1.0, 75.0, "Punt"),
            ]
        )
        got = extract_drive_starts(df).sort_values("drive")
        assert list(got["points"]) == [6.95, 3.0, 0.0]


class TestUnits:
    def test_lower_yardline_100_means_better_field_position(self):
        """yardline_100 is yards to the OPPONENT end zone. Getting this backwards
        would invert every conclusion while leaving the numbers plausible."""
        df = plays(
            [
                (1, 1.0, "AAA", "BBB", 1.0, 10.0, "Touchdown"),  # opponent 10
                (2, 2.0, "AAA", "BBB", 1.0, 90.0, "Punt"),  # own 10
            ]
        )
        got = extract_drive_starts(df).sort_values("drive")
        good, bad = got.iloc[0], got.iloc[1]
        assert good["start_100"] < bad["start_100"]
        assert good["points"] > bad["points"]


class TestTeamGame:
    def test_own_and_allowed_are_the_two_sides_of_one_coin(self):
        df = plays(
            [
                (1, 1.0, "AAA", "BBB", 1.0, 80.0, "Punt"),
                (2, 2.0, "BBB", "AAA", 1.0, 40.0, "Touchdown"),
            ]
        )
        tg = extract_drive_starts(df).pipe(team_game_field_position).set_index("team")
        assert tg.loc["AAA", "own_fp"] == 80.0
        assert (
            tg.loc["AAA", "fp_allowed"] == 40.0
        ), "AAA's defence let BBB start at the 40; that is AAA's fp_allowed"
        assert tg.loc["BBB", "own_fp"] == 40.0
        assert tg.loc["BBB", "fp_allowed"] == 80.0


class TestPregameLeakage:
    @staticmethod
    def _history(starts, team="AAA"):
        rows, day = [], pd.Timestamp("2024-09-08")
        for i, st in enumerate(starts):
            rows.append(
                {
                    "game_id": f"g{i}",
                    "posteam": team,
                    "defteam": "ZZZ",
                    "drive": 1.0,
                    "start_100": float(st),
                    "result": "Punt",
                    "points": 0.0,
                }
            )
        return pd.DataFrame(rows)

    def test_a_games_own_field_position_never_reaches_its_own_feature(
        self, tmp_path, monkeypatch
    ):
        """THE leakage property."""
        import src.experiments.field_position as fpmod

        drives = self._history([75, 75, 5])  # the last game is an extreme outlier
        sched = pd.DataFrame(
            {
                "game_id": ["g0", "g1", "g2"],
                "season": [2024, 2024, 2024],
                "gameday": pd.to_datetime(["2024-09-08", "2024-09-15", "2024-09-22"]),
            }
        )
        path = tmp_path / "s.parquet"
        sched.to_parquet(path, index=False)
        monkeypatch.setattr(fpmod, "SCHEDULES", str(path))

        # team_game_field_position emits a row for the OPPONENT of every game
        # too (that is its fp_allowed side), so select the team under test.
        out = fpmod.build_pregame_field_position(drives)
        out = out[out["team"] == "AAA"].set_index("game_id")
        assert out.loc["g2", "pregame_own_fp"] == pytest.approx(
            75.0
        ), "game 3's own start of 5 reached its own pregame feature"

    def test_the_first_game_has_no_pregame_value(self, tmp_path, monkeypatch):
        import src.experiments.field_position as fpmod

        drives = self._history([75])
        sched = pd.DataFrame(
            {
                "game_id": ["g0"],
                "season": [2024],
                "gameday": pd.to_datetime(["2024-09-08"]),
            }
        )
        path = tmp_path / "s.parquet"
        sched.to_parquet(path, index=False)
        monkeypatch.setattr(fpmod, "SCHEDULES", str(path))
        out = fpmod.build_pregame_field_position(drives)
        assert out[out["team"] == "AAA"]["pregame_own_fp"].isna().all()


# ------------------------------------------------- the three separate questions


@pytest.mark.requires_data
class TestTheThreeQuestions:
    """These are pinned because conflating them is the mistake that got this
    feature wrongly screened out the first time."""

    @pytest.fixture(scope="class")
    def drives(self):
        return extract_drive_starts()

    def test_extraction_agrees_with_nflverse_on_real_data(self, drives):
        """The authoritative check: nflverse ships its own drive start as a
        'TEAM YD' string. Decode it and the two must agree."""
        pbp = pd.read_parquet(
            PBP_COLS := "data/raw/pbp.parquet",
            columns=[
                "game_id",
                "play_id",
                "drive",
                "posteam",
                "down",
                "drive_start_yard_line",
            ],
        )
        scrim = pbp[pbp["down"].notna() & pbp["posteam"].notna()].sort_values("play_id")
        nfl = (
            scrim.groupby(["game_id", "posteam", "drive"])["drive_start_yard_line"]
            .first()
            .reset_index()
        )
        m = drives.merge(nfl, on=["game_id", "posteam", "drive"], how="inner").dropna(
            subset=["drive_start_yard_line"]
        )

        def decode(row):
            parts = row["drive_start_yard_line"].split()
            if len(parts) == 1:  # midfield ships as a bare "50"
                return 50.0
            side, yd = parts
            yd = int(yd)
            return yd if side != row["posteam"] else 100 - yd

        m["nfl_100"] = m.apply(decode, axis=1)
        agree = np.isclose(m["start_100"], m["nfl_100"]).mean()
        assert agree > 0.99, (
            f"only {agree:.1%} agreement with nflverse's own drive start -- the "
            "extraction is picking up the wrong play"
        )

    def test_q1_field_position_matters_enormously_per_drive(self, drives):
        """Not in dispute and worth pinning: a steep, monotone curve."""
        curve = expected_points_curve(drives)
        assert curve[
            "ep"
        ].is_monotonic_decreasing, (
            f"EP curve is not monotone in yards-to-goal:\n{curve}"
        )
        spread = curve["ep"].max() - curve["ep"].min()
        # Deciles, not fixed-width buckets. The 3.23 figure quoted elsewhere
        # comes from fixed-width buckets whose top one (inside the opponent 20)
        # holds only 730 of 42,917 drives; across DECILES of the actual
        # distribution the spread is ~1.96, which is the more representative
        # number for typical variation. Both are true; they answer different
        # questions and the decile one is the honest headline.
        assert spread > 1.8, f"EP spread is only {spread:.2f} points per drive"

    def test_q1b_the_effect_survives_removing_team_quality(self, drives):
        """The across-team slope is confounded -- good teams have good field
        position AND good offences. De-meaned by team-season, the causal slope
        must still be substantial."""
        sched = pd.read_parquet("data/raw/schedules.parquet")[["game_id", "season"]]
        slope = within_team_slope(drives, sched)
        assert slope < -0.02, (
            f"within-team slope is only {slope:+.4f} points/drive/yard; the "
            "field-position effect does not survive removing team quality"
        )

    def test_q2_teams_differ_meaningfully(self, drives):
        sched = pd.read_parquet("data/raw/schedules.parquet")[["game_id", "season"]]
        d = drives.merge(sched, on="game_id")
        ts = d.groupby(["season", "posteam"]).agg(
            fp=("start_100", "mean"), n=("points", "size")
        )
        ts = ts[ts["n"] >= 150]
        spread = ts["fp"].max() - ts["fp"].min()
        assert spread > 5.0, (
            f"best-to-worst team-season field position differs by only "
            f"{spread:.1f} yards"
        )

    def test_q3_it_barely_persists_which_is_why_it_may_not_help(self):
        """The decisive one, and the only one of the three that bears on a
        FORECAST. If this ever rises materially, the feature deserves another
        look -- so it is pinned as a fact rather than left as a memory."""
        fp = build_pregame_field_position().dropna(subset=FP_COLS)
        tg = team_game_field_position(extract_drive_starts())
        j = fp.merge(tg, on=["game_id", "team"], how="inner").dropna(
            subset=["pregame_own_fp", "own_fp"]
        )
        r = j["pregame_own_fp"].corr(j["own_fp"])
        assert r < 0.30, (
            f"prior field position now predicts future field position at "
            f"r={r:+.3f}. It was 0.108, which is why this feature was screened "
            "out. If it has risen, re-run the model test."
        )
        assert r > 0.0, "prior FP should still carry SOME signal"
