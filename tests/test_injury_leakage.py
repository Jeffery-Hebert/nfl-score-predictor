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
    QB_STARTER_SNAP_THRESHOLD,
    attach_prior_share,
    snap_history,
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
    assert (df["injury_impact"] >= 0).all(), "impact must be non-negative"
    assert df["qb_out"].isin([0.0, 1.0]).all()
    assert df.duplicated(subset=["game_id", "team"]).sum() == 0
    # The 2025+ seasons have no date_modified; if they had been dropped as a
    # side effect of the leakage filter, their impact would be identically 0.
    recent = df[df["season"] >= 2025]
    assert recent["injury_impact"].sum() > 0, (
        "2025+ rows carry no injury signal -- the NaT-timestamp rows were "
        "probably dropped again"
    )
