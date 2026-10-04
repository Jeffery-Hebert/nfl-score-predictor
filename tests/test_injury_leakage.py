"""
Leakage gates for the injury availability features.

This builder shipped two bugs on the first attempt, both of which produced
plausible-looking output, so it gets hand-built fixtures with known answers:

  1. The leakage filter dropped rows whose date_modified was NaT along with
     rows genuinely published after kickoff. nflverse stopped populating that
     timestamp in 2025, so it silently discarded two entire seasons.
  2. Snap share was joined on (game_id, gsis_id) -- but a player ruled Out has
     no snap row for that game, so the join failed for exactly the players the
     feature exists to measure. qb_out came out identically zero across 289
     candidate rows and nothing flagged it.

Run: pytest tests/test_injury_leakage.py -v
"""

import pandas as pd
import pytest

from src.features.build_injury_features import (
    team_totals,
    QB_STARTER_SNAP_THRESHOLD,
    attach_prior_share,
    latest_report_per_player,
    snap_history,
    unmeasurable,
)


def _history(rows):
    """rows: list of (gsis_id, kickoff_str, role_share)."""
    return (
        pd.DataFrame(
            [
                {"gsis_id": g, "kickoff": pd.Timestamp(k, tz="UTC"), "share_to_date": s}
                for g, k, s in rows
            ]
        )
        .sort_values("kickoff")
        .reset_index(drop=True)
    )


def _injuries(rows):
    """rows: list of (gsis_id, kickoff_str)."""
    return pd.DataFrame(
        [{"gsis_id": g, "kickoff": pd.Timestamp(k, tz="UTC")} for g, k in rows]
    )


def test_as_of_join_uses_only_games_before_kickoff():
    """The core property. A player's share must come from earlier games, never
    from the game being predicted."""
    hist = _history(
        [
            ("P1", "2024-09-08 17:00", 0.40),
            ("P1", "2024-09-15 17:00", 0.60),
            ("P1", "2024-09-22 17:00", 0.90),  # the game being predicted
        ]
    )
    inj = _injuries([("P1", "2024-09-22 17:00")])
    got = attach_prior_share(inj, hist)
    assert got.loc[0, "prior_share"] == 0.60, (
        "as-of join picked up the current game's own snap share -- that is "
        "the game we are predicting"
    )


def test_player_with_no_prior_games_gets_no_share():
    hist = _history([("P1", "2024-09-15 17:00", 0.80)])
    inj = _injuries([("P2", "2024-09-22 17:00")])
    got = attach_prior_share(inj, hist)
    assert pd.isna(got.loc[0, "prior_share"]), "unknown player must not inherit a share"


def test_first_ever_game_has_no_prior_share():
    hist = _history([("P1", "2024-09-22 17:00", 0.70)])
    inj = _injuries([("P1", "2024-09-22 17:00")])
    got = attach_prior_share(inj, hist)
    assert pd.isna(got.loc[0, "prior_share"])


def test_an_injured_player_still_gets_a_share_though_he_did_not_play():
    """The bug that made the feature useless. A player ruled Out has no snap
    row for that game; his share must still be recoverable from history."""
    hist = _history(
        [
            ("QB1", "2024-09-08 17:00", 1.00),
            ("QB1", "2024-09-15 17:00", 1.00),
            # no row for 2024-09-22 -- he was inactive
        ]
    )
    inj = _injuries([("QB1", "2024-09-22 17:00")])
    got = attach_prior_share(inj, hist)
    assert got.loc[0, "prior_share"] == 1.00, (
        "an inactive player's prior role was lost -- this is the join bug that "
        "made qb_out identically zero"
    )
    assert got.loc[0, "prior_share"] >= QB_STARTER_SNAP_THRESHOLD


def test_snap_history_never_includes_the_current_game_for_the_next_lookup():
    """share_to_date includes the row's own game by construction; the as-of
    join is what excludes it, via allow_exact_matches=False."""
    snaps = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "pfr_player_id": ["X", "X"],
            "offense_pct": [0.0, 1.0],
            "defense_pct": [0.0, 0.0],
        }
    )
    sched = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "kickoff": pd.to_datetime(
                ["2024-09-08 17:00", "2024-09-15 17:00"], utc=True
            ),
        }
    )
    players = pd.DataFrame({"gsis_id": ["GX"], "pfr_id": ["X"]})
    hist = snap_history(snaps, sched, players)
    assert list(hist["share_to_date"]) == [0.0, 0.5]

    inj = _injuries([("GX", "2024-09-15 17:00")])
    got = attach_prior_share(inj, hist)
    assert got.loc[0, "prior_share"] == 0.0, "must see only g1, not g2"


# ------------------------------------------------- built artifact sanity


@pytest.mark.requires_data
def test_built_features_are_sane():
    df = pd.read_parquet("data/processed/injury_features.parquet")
    assert (df["injury_impact"].dropna() >= 0).all(), "impact must be non-negative"
    assert df["qb_out"].dropna().isin([0.0, 1.0]).all()
    assert df.duplicated(subset=["game_id", "team"]).sum() == 0
    # The 2025+ seasons have no date_modified; if they had been dropped as a
    # side effect of the leakage filter, their impact would be identically 0.
    recent = df[df["season"] >= 2025]
    assert recent["injury_impact"].sum() > 0, (
        "2025+ rows carry no injury signal -- the NaT-timestamp rows were "
        "probably dropped again"
    )


@pytest.mark.requires_data
def test_missing_means_unmeasurable_and_nothing_else():
    """injury_impact is left missing only where the data cannot measure it:
    played games, every team's first game in the data, and team-weeks with no
    report. A code-mapping slip (a relocated team's snaps not recognised)
    would blank whole seasons and fail the coverage floor."""
    df = pd.read_parquet("data/processed/injury_features.parquet")
    sched = pd.read_parquet("data/raw/schedules.parquet")[["game_id", "home_score"]]
    df = df.merge(sched, on="game_id", how="left")
    played = df["home_score"].notna()
    missing = df["injury_impact"].isna()
    assert (missing == df["qb_out"].isna()).all(), "qb_out and impact disagree"
    assert not (missing & ~played).any(), (
        "a game not yet played is missing injury_impact -- before its report is "
        "out it must stay 0 (predict_week holds it back until the report is in)"
    )
    first = df[(df["season"] == df["season"].min()) & (df["week"] == 1) & played]
    assert (
        len(first) and first["injury_impact"].isna().all()
    ), "the first week of the data has no snap history: impact cannot be known"
    later = df[played & (df["season"] > df["season"].min())]
    coverage = 1 - later.groupby("season")["injury_impact"].apply(
        lambda s: s.isna().mean()
    )
    assert (coverage >= 0.95).all(), f"seasons mostly unmeasured:\n{coverage}"


def _team_games(rows):
    """rows: (game_id, team, kickoff_str, played)."""
    return pd.DataFrame(
        [
            {"game_id": g, "team": t, "kickoff": pd.Timestamp(k, tz="UTC"), "played": p}
            for g, t, k, p in rows
        ]
    )


def _snaps(rows):
    """rows: (team, kickoff_str) -- one per player-game."""
    return pd.DataFrame(
        [{"team": t, "kickoff": pd.Timestamp(k, tz="UTC")} for t, k in rows]
    )


REPORTED = pd.DataFrame({"game_id": ["g2", "g3"], "team": ["AAA", "AAA"]})


def test_a_played_game_with_no_report_is_unknown_not_healthy():
    tg = _team_games(
        [
            ("g2", "AAA", "2024-09-15 17:00", True),  # reported
            ("g4", "AAA", "2024-09-29 17:00", True),  # no report in the data
        ]
    )
    snaps = _snaps([("AAA", "2024-09-08 17:00")])
    assert list(unmeasurable(tg, REPORTED, snaps)) == [False, True]


def test_a_game_not_yet_played_keeps_zero_before_its_report():
    """Tuesday of every week: the coming games have no report rows yet. That
    is the normal cycle, not missing data -- predict_week waits for the final
    report -- so the builder must not blank them (the upcoming-week checks in
    test_prediction_freshness.py would stop the daily pipeline)."""
    tg = _team_games([("g9", "AAA", "2024-12-01 18:00", False)])
    snaps = _snaps([("AAA", "2024-09-08 17:00")])
    assert list(unmeasurable(tg, REPORTED, snaps)) == [False]


def test_a_teams_first_game_in_the_snap_history_is_unknown():
    tg = _team_games(
        [
            ("g2", "AAA", "2024-09-15 17:00", True),  # its first snap game
            ("g3", "AAA", "2024-09-22 17:00", True),  # one game of history
        ]
    )
    snaps = _snaps([("AAA", "2024-09-15 17:00"), ("AAA", "2024-09-22 17:00")])
    assert list(unmeasurable(tg, REPORTED, snaps)) == [True, False]


def test_a_relocated_teams_snap_history_is_recognised():
    """Snap counts carry the code of the season (OAK in 2019); the team-games
    carry today's (LV). Without normalising, the Raiders would have no history
    at all and their whole first season would be blanked."""
    reported = pd.DataFrame({"game_id": ["g2"], "team": ["LV"]})
    tg = _team_games([("g2", "LV", "2019-09-15 20:00", True)])
    snaps = _snaps([("OAK", "2019-09-09 20:00")])
    assert list(unmeasurable(tg, reported, snaps)) == [False]


def test_a_player_listed_twice_in_a_week_counts_once_at_his_latest_status():
    """2024 week 15: Cade Stover was listed Questionable at 03:34 and Out at
    14:17. Summing both rows counted him 1.5 times; the game-day Out stands."""
    rows = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "team": ["HOU", "HOU", "HOU"],
            "gsis_id": ["stover", "stover", "other"],
            "report_status": ["Out", "Questionable", "Out"],
            "date_modified": pd.to_datetime(
                ["2024-12-15 14:17", "2024-12-15 03:34", "2024-12-13 20:00"], utc=True
            ),
        }
    )
    for seed in range(5):  # whatever order the file arrives in
        got = latest_report_per_player(rows.sample(frac=1, random_state=seed))
        assert len(got) == 2
        assert got.set_index("gsis_id").at["stover", "report_status"] == "Out"


def test_team_totals_do_not_depend_on_row_order():
    """A pull that only reorders the injury file must leave every total
    bitwise identical, or the content fingerprints report history as changed.
    These eleven impacts (one team's report) sum to 2.66503 in one order and
    2.6650300000000002 in another under an unordered groupby -- found by
    search, since pandas' compensated summation hides it on small groups."""
    impacts = [0.220185, 0.13908, 0.1101, 0.071595, 0.224955, 0.04252,
               0.09303, 0.9951, 0.332115, 0.0575, 0.37885]  # fmt: skip
    rows = pd.DataFrame(
        {
            "game_id": ["g1"] * 11 + ["g2"],
            "team": ["AAA"] * 11 + ["BBB"],
            "gsis_id": [f"p{i}" for i in range(12)],
            "impact": impacts + [0.5],
            "is_qb_out": [False] * 10 + [True, False],
        }
    )
    reference = team_totals(rows)
    for seed in range(50):
        shuffled = rows.sample(frac=1, random_state=seed)
        pd.testing.assert_frame_equal(
            team_totals(shuffled), reference, check_exact=True
        )
    assert reference["qb_out"].tolist() == [1.0, 0.0]
