"""
Which plays the per-team efficiency stats are measured on.

The defect this pins (fixed 2026-10-01). build_team_game_stats.py computed
off_success_rate / def_success_rate_allowed -- and the EPA per play and play
counts beside them -- over every row with a play_type: kickoffs, punts, extra
points, field goals, kneel-downs, spikes and penalty-nullified snaps included.
23% of the plays behind two of the model's success-rate features were not
offensive or defensive snaps, and extra points (94% success) and field goals
(85%) turned the rate partly into a count of scoring.

All synthetic, no data/: one game in which every kind of play appears, with
values chosen so that ANY non-snap leaking into a rate changes the answer.

Run: pytest tests/test_team_game_stats_snaps.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_team_game_stats import (
    SCRIMMAGE_PLAY_TYPES,
    build_defense_stats,
    build_offense_stats,
    scrimmage_snaps,
)

G = "2026_04_AWY_HOM"

# (posteam, defteam, play_type, pass, rush, epa, success, cpoe)
# HOM's offence: five snaps that count, then every kind of play that must not.
COUNTED_HOM = [
    ("HOM", "AWY", "pass", 1, 0, 1.0, 1, 5.0),  # completion
    ("HOM", "AWY", "pass", 1, 0, -0.5, 0, -20.0),  # incompletion
    ("HOM", "AWY", "pass", 1, 0, -1.5, 0, np.nan),  # sack: a pass, no cpoe
    ("HOM", "AWY", "run", 0, 1, 0.3, 1, np.nan),  # designed run
    ("HOM", "AWY", "run", 1, 0, 0.8, 1, np.nan),  # scramble: a run here
]
NOT_COUNTED_HOM = [
    ("HOM", "AWY", "extra_point", 0, 0, 0.02, 1, np.nan),
    ("HOM", "AWY", "field_goal", 0, 0, 1.2, 1, np.nan),
    ("HOM", "AWY", "kickoff", 0, 0, 0.3, 1, np.nan),  # HOM receiving
    ("HOM", "AWY", "punt", 0, 0, 0.4, 1, np.nan),
    ("HOM", "AWY", "qb_kneel", 0, 0, -0.4, 0, np.nan),
    ("HOM", "AWY", "qb_spike", 0, 0, -0.1, 0, np.nan),
    ("HOM", "AWY", "no_play", 0, 0, -0.4, 0, np.nan),  # false start
    ("HOM", "AWY", "no_play", 1, 0, -0.9, 0, np.nan),  # holding wipes a pass
    ("HOM", "AWY", "no_play", 0, 1, 0.5, 1, np.nan),  # def. holding on a run
]
# AWY's offence, so HOM's defence has something to be measured on.
COUNTED_AWY = [
    ("AWY", "HOM", "pass", 1, 0, 0.6, 1, 10.0),
    ("AWY", "HOM", "run", 0, 1, -0.2, 0, np.nan),
    ("AWY", "HOM", "run", 0, 1, -0.6, 0, np.nan),
]
NOT_COUNTED_AWY = [
    ("AWY", "HOM", "extra_point", 0, 0, 0.02, 1, np.nan),
    ("AWY", "HOM", "kickoff", 0, 0, -0.3, 0, np.nan),
]
# Rows that are not plays at all (timeouts, end of quarter): no play_type.
NON_PLAYS = [
    ("HOM", "AWY", None, 0, 0, np.nan, np.nan, np.nan),
    (None, None, None, 0, 0, np.nan, np.nan, np.nan),
]


def pbp(rows) -> pd.DataFrame:
    cols = ["posteam", "defteam", "play_type", "pass", "rush", "epa", "success"]
    df = pd.DataFrame([r[:7] for r in rows], columns=cols)
    df["cpoe"] = [r[7] for r in rows]
    df.insert(0, "game_id", G)
    return df


@pytest.fixture
def game():
    return pbp(
        COUNTED_HOM + NOT_COUNTED_HOM + COUNTED_AWY + NOT_COUNTED_AWY + NON_PLAYS
    )


def row(stats: pd.DataFrame, team: str) -> pd.Series:
    (r,) = stats[stats["team"] == team].to_dict("records")
    return pd.Series(r)


# ----------------------------------------------------------------- the rule --


def test_only_pass_and_run_snaps_are_scrimmage_snaps(game):
    got = scrimmage_snaps(game, "posteam")
    assert set(got["play_type"]) == set(SCRIMMAGE_PLAY_TYPES) == {"pass", "run"}
    assert len(got) == len(COUNTED_HOM) + len(COUNTED_AWY)


def test_offensive_success_rate_counts_only_the_teams_snaps(game):
    off = row(build_offense_stats(game), "HOM")
    assert off["off_plays"] == 5
    assert off["off_success_rate"] == pytest.approx(3 / 5)  # 1, 0, 0, 1, 1
    assert off["off_epa_per_play"] == pytest.approx((1.0 - 0.5 - 1.5 + 0.3 + 0.8) / 5)


def test_defensive_success_rate_counts_only_the_opponents_snaps(game):
    deff = row(build_defense_stats(game), "HOM")
    assert deff["def_plays"] == 3
    assert deff["def_success_rate_allowed"] == pytest.approx(1 / 3)
    assert deff["def_epa_per_play"] == pytest.approx((0.6 - 0.2 - 0.6) / 3)


def test_the_old_all_plays_definition_would_have_given_a_different_answer(game):
    """Guards against a vacuous fixture: the leak must be detectable here."""
    plays = game[game["play_type"].notna() & (game["posteam"] == "HOM")]
    new = row(build_offense_stats(game), "HOM")
    # 3 of 5 snaps succeed, plus 5 of the 9 non-snaps (XP, FG, kickoff, punt,
    # the defensive-holding no_play) -- 8 of 14.
    assert plays["success"].mean() == pytest.approx(8 / 14)
    assert plays["success"].mean() != pytest.approx(new["off_success_rate"])
    assert plays["epa"].mean() != pytest.approx(new["off_epa_per_play"])
    assert len(plays) != new["off_plays"]


@pytest.mark.parametrize(
    "play_type",
    ["extra_point", "field_goal", "kickoff", "punt", "qb_kneel", "qb_spike", "no_play"],
)
def test_no_non_snap_play_type_can_move_any_rate(game, play_type):
    """Rewrite every play of one excluded type to an extreme outcome: no
    offensive or defensive rate, and no count, may change."""
    before_off, before_def = build_offense_stats(game), build_defense_stats(game)
    rigged = game.copy()
    sel = rigged["play_type"] == play_type
    assert sel.any(), "fixture lost a play type"
    rigged.loc[sel, ["epa", "success"]] = [99.0, 1]
    pd.testing.assert_frame_equal(build_offense_stats(rigged), before_off)
    pd.testing.assert_frame_equal(build_defense_stats(rigged), before_def)


def test_penalty_nullified_dropbacks_and_runs_are_left_out(game):
    """nflfastR's convention (pass == 1 | rush == 1) would count them. It was an
    arm of the experiment and was no better; leaving them out is also how the
    pass/rush split has always counted. Pinned so the choice is explicit."""
    nullified = game[
        (game["play_type"] == "no_play") & ((game["pass"] == 1) | (game["rush"] == 1))
    ]
    assert len(nullified) == 2
    assert not scrimmage_snaps(game, "posteam").index.isin(nullified.index).any()


# --------------------------------------------- consistency, and what is kept --


def test_offence_and_defence_are_two_views_of_the_same_snaps(game):
    off, deff = build_offense_stats(game), build_defense_stats(game)
    for attacker, defender in (("HOM", "AWY"), ("AWY", "HOM")):
        o, d = row(off, attacker), row(deff, defender)
        assert o["off_plays"] == d["def_plays"]
        assert o["off_success_rate"] == pytest.approx(d["def_success_rate_allowed"])
        assert o["off_epa_per_play"] == pytest.approx(d["def_epa_per_play"])


def test_the_pass_rush_split_is_unchanged(game):
    """The split always used play_type pass / run; the fix must not move it."""
    off = row(build_offense_stats(game), "HOM")
    assert off["off_pass_plays"] == 3 and off["off_rush_plays"] == 2
    assert off["off_pass_epa_per_play"] == pytest.approx((1.0 - 0.5 - 1.5) / 3)
    assert off["off_rush_epa_per_play"] == pytest.approx((0.3 + 0.8) / 2)
    assert off["off_cpoe"] == pytest.approx((5.0 - 20.0) / 2)  # sack has no cpoe
    deff = row(build_defense_stats(game), "HOM")
    assert deff["def_pass_plays"] == 1 and deff["def_rush_plays"] == 2
    assert deff["def_rush_epa_per_play_allowed"] == pytest.approx(-0.4)


def test_a_team_with_no_snaps_gets_no_rate_rather_than_a_special_teams_one():
    """A team-game whose only plays are special teams must come out MISSING --
    imputed downstream like any absent stat -- not scored on its kickoffs."""
    only_st = pbp(COUNTED_AWY + NOT_COUNTED_HOM[:4])
    assert "HOM" not in set(build_offense_stats(only_st)["team"])
    assert "AWY" in set(build_offense_stats(only_st)["team"])


def test_non_plays_never_count(game):
    assert scrimmage_snaps(game, "posteam")["play_type"].notna().all()
    assert scrimmage_snaps(game, "defteam")["defteam"].notna().all()
