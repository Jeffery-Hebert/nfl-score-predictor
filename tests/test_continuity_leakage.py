"""
Leakage and correctness gates for the roster-continuity features.

Snap counts are POST-game data, which makes this builder the same shape of risk
as the injury features: safe only because it reads them exclusively from games
that have already finished. Two traps specifically:

  1. asking "who is on this roster" by looking at the CURRENT game's snaps. That
     both leaks and is unservable -- on Wednesday nobody knows who will take
     snaps on Sunday. It is the precise defect that shelved the QB-identity
     feature, which worked historically and had no pre-game source.

  2. a measure that looks fine because it is bounded and smooth while being
     computed from the wrong games entirely. Continuity numbers all live in
     [0,1] and hover around 0.7, so a broken one is indistinguishable from a
     working one by inspection.

Every fixture below is hand-built with a known answer.

Run: pytest tests/test_continuity_leakage.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.experiments.continuity import (
    CONTINUITY_COLS,
    _weighted_jaccard,
    build_continuity,
)


def snaps(rows) -> pd.DataFrame:
    """rows: (gameday, season, team, [player ids]) -> a snap frame.

    Every listed player takes a full share, so line-up overlap is the only
    thing varying.
    """
    recs = []
    for i, (day, season, team, players) in enumerate(rows):
        for p in players:
            recs.append(
                {
                    "game_id": f"g{i}",
                    "season": season,
                    "team": team,
                    "pfr_player_id": p,
                    "role_share": 1.0,
                    "gameday": pd.Timestamp(day),
                }
            )
    return pd.DataFrame(recs)


class TestStrictlyPrior:
    def test_first_ever_game_has_no_continuity(self):
        s = snaps([("2024-09-08", 2024, "AAA", ["p1", "p2"])])
        out = build_continuity(s).iloc[0]
        for c in CONTINUITY_COLS:
            assert pd.isna(out[c]), f"{c} was fabricated for a team with no history"

    def test_this_games_lineup_never_sets_its_own_tenure(self):
        """THE property. Two histories identical except for who plays in the
        final game must produce the same tenure for that game."""
        base = [
            ("2024-09-08", 2024, "AAA", ["vet1", "vet2"]),
            ("2024-09-15", 2024, "AAA", ["vet1", "vet2"]),
        ]
        a = build_continuity(
            snaps(base + [("2024-09-22", 2024, "AAA", ["vet1", "vet2"])])
        )
        b = build_continuity(
            snaps(base + [("2024-09-22", 2024, "AAA", ["rookie1", "rookie2"])])
        )
        ta = a[a.game_id == "g2"]["snap_weighted_tenure"].iloc[0]
        tb = b[b.game_id == "g2"]["snap_weighted_tenure"].iloc[0]
        assert ta == pytest.approx(tb), (
            f"tenure moved with the current game's line-up ({ta} vs {tb}) -- it "
            "is reading who actually played, which is unknowable before kickoff"
        )

    def test_a_later_game_cannot_change_an_earlier_row(self):
        early = snaps(
            [
                ("2024-09-08", 2024, "AAA", ["p1", "p2"]),
                ("2024-09-15", 2024, "AAA", ["p1", "p2"]),
                ("2024-09-22", 2024, "AAA", ["p1", "p2"]),
            ]
        )
        late = snaps(
            [
                ("2024-09-08", 2024, "AAA", ["p1", "p2"]),
                ("2024-09-15", 2024, "AAA", ["p1", "p2"]),
                ("2024-09-22", 2024, "AAA", ["p1", "p2"]),
                ("2024-09-29", 2024, "AAA", ["x1", "x2", "x3"]),  # total turnover
            ]
        )
        a = build_continuity(early).set_index("game_id")
        b = build_continuity(late).set_index("game_id")
        for g in ("g0", "g1", "g2"):
            for c in CONTINUITY_COLS:
                va, vb = a.loc[g, c], b.loc[g, c]
                assert (pd.isna(va) and pd.isna(vb)) or va == pytest.approx(
                    vb
                ), f"{c} for {g} changed when a LATER game was added"


class TestCarryover:
    def test_a_fully_returning_squad_scores_one(self):
        s = snaps(
            [
                ("2023-09-10", 2023, "AAA", ["p1", "p2"]),
                ("2024-09-08", 2024, "AAA", ["p1", "p2"]),
                ("2024-09-15", 2024, "AAA", ["p1", "p2"]),
            ]
        )
        got = build_continuity(s).set_index("game_id").loc["g2", "roster_carryover"]
        assert got == pytest.approx(1.0)

    def test_a_completely_new_squad_scores_zero(self):
        s = snaps(
            [
                ("2023-09-10", 2023, "AAA", ["old1", "old2"]),
                ("2024-09-08", 2024, "AAA", ["new1", "new2"]),
                ("2024-09-15", 2024, "AAA", ["new1", "new2"]),
            ]
        )
        got = build_continuity(s).set_index("game_id").loc["g2", "roster_carryover"]
        assert got == pytest.approx(0.0)

    def test_a_half_new_squad_scores_a_half(self):
        s = snaps(
            [
                ("2023-09-10", 2023, "AAA", ["p1", "p2"]),
                ("2024-09-08", 2024, "AAA", ["p1", "new1"]),
                ("2024-09-15", 2024, "AAA", ["p1", "new1"]),
            ]
        )
        got = build_continuity(s).set_index("game_id").loc["g2", "roster_carryover"]
        assert got == pytest.approx(0.5)

    def test_season_one_has_no_carryover_to_measure(self):
        s = snaps(
            [
                ("2024-09-08", 2024, "AAA", ["p1"]),
                ("2024-09-15", 2024, "AAA", ["p1"]),
            ]
        )
        got = build_continuity(s).set_index("game_id").loc["g1", "roster_carryover"]
        assert pd.isna(got), "carryover was invented with no prior season on file"


class TestStability:
    def test_an_unchanged_lineup_is_perfectly_stable(self):
        s = snaps(
            [
                ("2024-09-08", 2024, "AAA", ["p1", "p2"]),
                ("2024-09-15", 2024, "AAA", ["p1", "p2"]),
                ("2024-09-22", 2024, "AAA", ["p1", "p2"]),
            ]
        )
        got = build_continuity(s).set_index("game_id").loc["g2", "lineup_stability"]
        assert got == pytest.approx(1.0)

    def test_a_fully_churned_lineup_is_unstable(self):
        s = snaps(
            [
                ("2024-09-08", 2024, "AAA", ["a1", "a2"]),
                ("2024-09-15", 2024, "AAA", ["b1", "b2"]),
                ("2024-09-22", 2024, "AAA", ["c1", "c2"]),
            ]
        )
        got = build_continuity(s).set_index("game_id").loc["g2", "lineup_stability"]
        assert got == pytest.approx(0.0)

    def test_needs_two_prior_games_to_have_a_transition(self):
        s = snaps(
            [
                ("2024-09-08", 2024, "AAA", ["p1"]),
                ("2024-09-15", 2024, "AAA", ["p1"]),
            ]
        )
        out = build_continuity(s).set_index("game_id")
        assert pd.isna(out.loc["g1", "lineup_stability"]), (
            "stability was computed from a single prior game, which has no "
            "transition in it"
        )


class TestWeightedJaccard:
    def test_identical_lineups_overlap_completely(self):
        a = {"p1": 1.0, "p2": 0.5}
        assert _weighted_jaccard(a, dict(a)) == pytest.approx(1.0)

    def test_disjoint_lineups_do_not_overlap(self):
        assert _weighted_jaccard({"a": 1.0}, {"b": 1.0}) == pytest.approx(0.0)

    def test_swapping_a_starter_costs_more_than_swapping_a_backup(self):
        """The reason for weighting at all. A plain set overlap would score
        these two changes identically."""
        base = {"starter": 1.0, "backup": 0.1}
        lost_starter = {"replacement": 1.0, "backup": 0.1}
        lost_backup = {"starter": 1.0, "replacement": 0.1}
        assert _weighted_jaccard(base, lost_starter) < _weighted_jaccard(
            base, lost_backup
        )

    def test_is_symmetric(self):
        a, b = {"p1": 1.0, "p2": 0.3}, {"p1": 0.6, "p3": 0.4}
        assert _weighted_jaccard(a, b) == pytest.approx(_weighted_jaccard(b, a))

    def test_an_empty_lineup_has_no_defined_overlap(self):
        assert np.isnan(_weighted_jaccard({}, {"p1": 1.0}))


class TestTeamsAreIndependent:
    def test_one_teams_history_never_reaches_another(self):
        s = snaps(
            [
                ("2024-09-08", 2024, "AAA", ["a1", "a2"]),
                ("2024-09-08", 2024, "BBB", ["b1", "b2"]),
                ("2024-09-15", 2024, "AAA", ["a1", "a2"]),
                ("2024-09-15", 2024, "BBB", ["b1", "b2"]),
                ("2024-09-22", 2024, "AAA", ["a1", "a2"]),
            ]
        )
        out = build_continuity(s)
        row = out[(out.team == "AAA") & (out.game_id == "g4")].iloc[0]
        assert row["lineup_stability"] == pytest.approx(
            1.0
        ), "another team's roster churn reached this team's stability"


@pytest.mark.requires_data
class TestAgainstRealData:
    @pytest.fixture(scope="class")
    @classmethod
    def built(cls):
        return build_continuity()

    def test_measures_are_in_their_defined_ranges(self, built):
        for c in ("roster_carryover", "lineup_stability"):
            v = built[c].dropna()
            assert v.between(0.0, 1.0).all(), f"{c} escaped [0,1]"
        assert (built["snap_weighted_tenure"].dropna() >= 0).all()

    def test_carryover_is_football_plausible(self, built):
        """NFL rosters turn over roughly a quarter to a third a year. A mean
        near 1.0 would mean the cross-season join is matching everything; near
        0.0 would mean it is matching nothing -- both are the same bug wearing
        different clothes."""
        m = built["roster_carryover"].mean()
        assert 0.5 < m < 0.85, f"mean carryover {m:.3f} is not football-plausible"

    def test_week_one_has_no_within_season_measures(self, built):
        """Nothing has happened yet this season, so stability is undefined."""
        sched = pd.read_parquet("data/raw/schedules.parquet")[["game_id", "week"]]
        j = built.merge(sched, on="game_id", how="left")
        wk1 = j[j["week"] == 1]
        assert wk1["lineup_stability"].isna().all(), (
            "week 1 has a within-season stability value, so it is reading games "
            "from another season or from the future"
        )

    def test_the_three_measures_are_not_duplicates_of_each_other(self, built):
        """If they correlate near 1.0 they are one feature with three names, and
        this project has repeatedly found redundant columns cost more than they
        return."""
        c = built[CONTINUITY_COLS].corr().abs()
        offdiag = c.to_numpy()[~np.eye(len(CONTINUITY_COLS), dtype=bool)]
        assert offdiag.max() < 0.9, f"continuity measures are collinear:\n{c}"


class TestHistoryUpdateUsesTheCurrentGame:
    """Regression gates for the 2026-09-24 fix.

    The carryover loop reused the variable holding the current game's line-up,
    so from the second season onward the history was updated with the PREVIOUS
    game's players: each game entered one game late, week 1 was counted twice
    and each season's final game never counted. Every earlier test passed,
    because none pinned a value in a second season with a changing line-up.
    """

    def test_tenure_counts_every_prior_appearance_in_a_later_season(self):
        s = snaps(
            [
                ("2023-09-10", 2023, "AAA", ["a"]),
                ("2024-09-08", 2024, "AAA", ["a"]),
                ("2024-09-15", 2024, "AAA", ["b"]),
                ("2024-09-22", 2024, "AAA", ["b"]),
                ("2024-09-29", 2024, "AAA", ["b"]),
            ]
        )
        got = build_continuity(s).set_index("game_id").loc["g4", "snap_weighted_tenure"]
        assert got == pytest.approx(2.0), (
            f"b played g2 and g3 before g4, so tenure is 2 -- got {got}; the "
            "history is being updated with the wrong game's line-up"
        )

    def test_a_seasons_final_game_counts_toward_next_seasons_carryover(self):
        s = snaps(
            [
                ("2023-09-10", 2023, "AAA", ["a"]),
                ("2024-09-08", 2024, "AAA", ["a"]),
                ("2024-09-15", 2024, "AAA", ["a"]),
                ("2024-09-22", 2024, "AAA", ["b"]),  # b's only 2024 game: the last
                ("2025-09-07", 2025, "AAA", ["b"]),
                ("2025-09-14", 2025, "AAA", ["b"]),
            ]
        )
        got = build_continuity(s).set_index("game_id").loc["g5", "roster_carryover"]
        assert got == pytest.approx(
            1.0
        ), f"b played for this team in 2024, so 2025's carryover is 1.0 -- got {got}"
