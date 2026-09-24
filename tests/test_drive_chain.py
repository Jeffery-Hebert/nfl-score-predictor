"""
Correctness gates for the Markov-chain drive model.

The chain's whole claim is that field position is ENDOGENOUS -- drive N's
outcome sets drive N+1's starting position -- rather than a team trait to be
forecast. That claim is only worth anything if three things hold, and all three
are invisible in the output (every version produces scores in the low twenties):

  1. the transition actually happens: a turnover must hand the OPPONENT better
     field position, and that must show up in the opponent's points;
  2. the expectation is exact, not sampled, so chaining does not compound
     sampling noise across 22 possessions;
  3. the probability tables stay proper distributions after team modulation.

Run: pytest tests/test_drive_chain.py -v
"""

import numpy as np
import pandas as pd
import pytest

import src.experiments.drive_chain as dc

CATS = dc.OUTCOMES
N = len(CATS)


def dist(**kw) -> np.ndarray:
    """An outcome distribution; unnamed categories share the remainder."""
    v = np.zeros(N)
    for k, x in kw.items():
        v[CATS.index(k)] = x
    left = 1.0 - v.sum()
    blanks = [i for i, c in enumerate(CATS) if c not in kw]
    if blanks and left > 0:
        v[blanks] = left / len(blanks)
    return v


def uniform_cond() -> np.ndarray:
    """A conditional table with no field-position effect: every bin identical.

    Useful as a control -- under it the chain must reduce to the independent
    -drive model, so any difference is attributable to field position.
    """
    return np.tile(dist(Touchdown=0.23, **{"Field goal": 0.15}), (dc.N_BINS, 1))


def flat_transition() -> np.ndarray:
    """Every outcome leads to the same starting bin -- no feedback at all."""
    t = np.zeros((N, dc.N_BINS))
    t[:, 4] = 1.0  # everyone starts own 34-20
    return t


def opening_at(bin_idx: int) -> np.ndarray:
    o = np.zeros(dc.N_BINS)
    o[bin_idx] = 1.0
    return o


class TestBinning:
    def test_lower_yards_to_goal_is_a_better_bin(self):
        """yardline_100 is distance to the OPPONENT end zone, so bin 0 is the
        opponent's red zone and the last bin is your own."""
        assert dc._bin_of(np.array([10.0]))[0] == 0
        assert dc._bin_of(np.array([95.0]))[0] == dc.N_BINS - 1

    def test_binning_is_monotone(self):
        got = dc._bin_of(np.array([5.0, 30.0, 45.0, 60.0, 75.0, 95.0]))
        assert list(got) == sorted(got)

    def test_out_of_range_values_are_clipped_not_dropped(self):
        assert dc._bin_of(np.array([-5.0]))[0] == 0
        assert dc._bin_of(np.array([150.0]))[0] == dc.N_BINS - 1


class TestExpectationIsExact:
    """Claim 2. The answer to 'isn't this circular?' -- a sampled chain would
    compound noise over 22 possessions; a propagated distribution does not."""

    def test_repeated_calls_give_identical_answers(self):
        c, t, o = uniform_cond(), flat_transition(), opening_at(4)
        runs = [dc.expected_points(c, c, t, o) for _ in range(5)]
        assert len({r[0] for r in runs}) == 1, "the expectation is being sampled"

    def test_matches_a_hand_computed_two_drive_chain(self):
        """Two possessions, one each, no field-position effect: the answer is
        just the outcome distribution dotted with the points vector."""
        c, t, o = uniform_cond(), flat_transition(), opening_at(4)
        h, a = dc.expected_points(c, c, t, o, drives_per_team=1)
        per_drive = float(c[4] @ dc.OFFENSE_POINTS)
        conceded = float(c[4] @ dc.DEFENSE_POINTS)
        assert h == pytest.approx(per_drive + conceded)
        assert a == pytest.approx(per_drive + conceded)

    def test_scales_with_the_number_of_possessions(self):
        c, t, o = uniform_cond(), flat_transition(), opening_at(4)
        one = dc.expected_points(c, c, t, o, drives_per_team=1)[0]
        five = dc.expected_points(c, c, t, o, drives_per_team=5)[0]
        assert five == pytest.approx(5 * one, rel=1e-9)

    def test_reduces_to_independent_drives_when_there_is_no_feedback(self):
        """With a flat conditional AND a flat transition the chain must be
        exactly the bag-of-drives model. If it is not, the extra structure is
        doing something unintended."""
        c, t, o = uniform_cond(), flat_transition(), opening_at(4)
        h, _ = dc.expected_points(c, c, t, o, drives_per_team=11)
        expected = 11 * float(c[4] @ dc.OFFENSE_POINTS) + 11 * float(
            c[4] @ dc.DEFENSE_POINTS
        )
        assert h == pytest.approx(expected)


class TestFeedback:
    """Claim 1, and the reason the chain exists at all."""

    @staticmethod
    def _transition_where_turnovers_give_good_position():
        t = np.zeros((N, dc.N_BINS))
        t[:, 4] = 1.0  # everything else -> own 34-20
        t[CATS.index("Turnover")] = 0.0
        t[CATS.index("Turnover"), 1] = 1.0  # a turnover -> opponent 21-35
        return t

    def test_a_turnover_hands_the_opponent_better_field_position(self):
        """v2 scores a turnover as zero and stops. The chain must also move the
        opponent's starting state, and that must raise the opponent's points."""
        cond = np.zeros((dc.N_BINS, N))
        # Scoring chance depends on the bin, so better position must pay.
        cond[1] = dist(Touchdown=0.50, **{"Field goal": 0.25})
        cond[4] = dist(Touchdown=0.20, **{"Field goal": 0.15})
        for b in range(dc.N_BINS):
            if cond[b].sum() == 0:
                cond[b] = cond[4]

        trans = self._transition_where_turnovers_give_good_position()
        clean = dist(Punt=0.60, Touchdown=0.20, **{"Field goal": 0.15})
        sloppy = clean.copy()
        sloppy[CATS.index("Punt")] -= 0.30
        sloppy[CATS.index("Turnover")] += 0.30

        home_clean = np.tile(clean, (dc.N_BINS, 1))
        home_sloppy = np.tile(sloppy, (dc.N_BINS, 1))
        _, away_vs_clean = dc.expected_points(home_clean, cond, trans, opening_at(4))
        _, away_vs_sloppy = dc.expected_points(home_sloppy, cond, trans, opening_at(4))

        assert away_vs_sloppy > away_vs_clean, (
            f"the opponent scored {away_vs_sloppy:.2f} against a turnover-prone "
            f"offence and {away_vs_clean:.2f} against a clean one -- the field "
            "position handed over by turnovers is not reaching their points"
        )

    def test_better_starting_position_produces_more_points(self):
        cond = np.zeros((dc.N_BINS, N))
        cond[0] = dist(Touchdown=0.60)
        cond[dc.N_BINS - 1] = dist(Touchdown=0.10)
        for b in range(1, dc.N_BINS - 1):
            cond[b] = dist(Touchdown=0.30)
        t = flat_transition()
        good = dc.expected_points(cond, cond, t, opening_at(0), drives_per_team=1)[0]
        bad = dc.expected_points(
            cond, cond, t, opening_at(dc.N_BINS - 1), drives_per_team=1
        )[0]
        assert good > bad, "starting in the red zone did not beat starting pinned back"


class TestTeamQuality:
    """Claim 3. The modulation must leave proper distributions behind."""

    def test_rows_remain_probability_distributions(self):
        cond = dc.outcome_given_bin(
            pd.DataFrame(
                {
                    "bin": [0, 1, 2, 3, 4, 5] * 20,
                    "result": (["Touchdown", "Punt", "Field goal"] * 40)[:120],
                }
            )
        )
        league = np.full(N, 1.0 / N)
        team = dist(Touchdown=0.40, Punt=0.30)
        out = dc.apply_team_quality(cond, team, league)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-9)
        assert (out >= 0).all()

    def test_a_league_average_team_is_left_unchanged(self):
        cond = uniform_cond()
        league = cond[0].copy()
        out = dc.apply_team_quality(cond, league, league)
        # 1e-6 rather than 1e-9: the epsilon guard in apply_team_quality makes
        # the per-element ratio differ by ~1e-8, which survives renormalisation.
        # Still four orders of magnitude tighter than anything that matters.
        np.testing.assert_allclose(out, cond, rtol=1e-6)

    def test_a_better_offense_scores_more(self):
        cond = uniform_cond()
        league = cond[0].copy()
        good = league.copy()
        good[CATS.index("Touchdown")] *= 2.0
        good = good / good.sum()
        t, o = flat_transition(), opening_at(4)
        strong = dc.expected_points(
            dc.apply_team_quality(cond, good, league), cond, t, o, drives_per_team=1
        )[0]
        average = dc.expected_points(cond, cond, t, o, drives_per_team=1)[0]
        assert strong > average

    def test_field_position_structure_survives_the_tilt(self):
        """Tilting by team quality must not flatten the bin differences -- that
        structure is measured on 43k drives and is not team-specific."""
        cond = np.zeros((dc.N_BINS, N))
        cond[0] = dist(Touchdown=0.60)
        for b in range(1, dc.N_BINS):
            cond[b] = dist(Touchdown=0.20)
        league = np.full(N, 1.0 / N)
        out = dc.apply_team_quality(cond, dist(Touchdown=0.30), league)
        td = CATS.index("Touchdown")
        assert out[0, td] > out[1, td], "the bin structure was flattened by the tilt"


@pytest.mark.requires_data
class TestAgainstRealData:
    @pytest.fixture(scope="class")
    @classmethod
    def chain(cls):
        d = dc.extract_chain_data()
        return d, dc.outcome_given_bin(d), dc.bin_given_outcome(d)

    def test_scoring_chance_falls_monotonically_with_distance(self, chain):
        _, cond, _ = chain
        td = cond[:, CATS.index("Touchdown")]
        assert list(td) == sorted(
            td, reverse=True
        ), f"touchdown rate is not monotone in field position: {td.round(3)}"

    def test_turnovers_really_do_hand_over_better_position(self, chain):
        """The measured fact the whole model rests on."""
        d, _, trans = chain
        centre = np.arange(dc.N_BINS)
        exp_bin = trans @ centre
        punt = exp_bin[CATS.index("Punt")]
        turnover = exp_bin[CATS.index("Turnover")]
        assert turnover < punt, (
            "a turnover does not lead to a better expected starting bin than a "
            "punt; the transition table is wrong or inverted"
        )

    def test_the_tables_are_well_determined(self, chain):
        d, cond, trans = chain
        counts = d["bin"].value_counts()
        assert counts.min() > 300, (
            f"smallest field-position bin has only {counts.min()} drives across "
            f"{N} outcomes"
        )
        np.testing.assert_allclose(cond.sum(axis=1), 1.0, rtol=1e-6)
        np.testing.assert_allclose(trans.sum(axis=1), 1.0, rtol=1e-6)

    def test_the_simulated_distribution_matches_reality(self, chain):
        """What the chain is FOR. A regression gives a point estimate; this must
        produce a score distribution with the right spread AND the right
        correlation between the two teams."""
        d, cond, trans = chain
        league = np.array(
            d["result"].value_counts(normalize=True).reindex(CATS).fillna(0.0).tolist()
        )
        league = (league + 1e-9) / (league + 1e-9).sum()
        c = dc.apply_team_quality(cond, league, league)
        h, a = dc.simulate_games(c, c, trans, dc.opening_distribution(d), n_trials=2000)

        act = pd.read_parquet("data/processed/model_table.parquet").dropna(
            subset=["home_score"]
        )
        assert 8.0 < h.std() < 12.0, (
            f"simulated team-score sd is {h.std():.2f}; the real one is "
            f"{act['home_score'].std():.2f}"
        )
        sim_total = (h + a).std()
        act_total = (act["home_score"] + act["away_score"]).std()
        assert (
            abs(sim_total - act_total) < 2.5
        ), f"simulated total sd {sim_total:.2f} against actual {act_total:.2f}"
        assert abs(np.corrcoef(h, a)[0, 1]) < 0.25, (
            "the two teams' simulated scores are far more correlated than the "
            "real ones"
        )


class TestPossessionRules:
    """Fixed 2026-09-24. The chain used to alternate possession after EVERY
    outcome and always gave the home team the opening kickoff. After a returned
    touchdown the team that threw it receives the next kickoff, and who receives
    first is a coin toss."""

    @staticmethod
    def _all(outcome: str) -> np.ndarray:
        row = np.zeros(N)
        row[CATS.index(outcome)] = 1.0
        return np.tile(row, (dc.N_BINS, 1))

    def test_a_pick_six_gives_the_ball_back_to_the_team_that_threw_it(self):
        """Home throws a pick-six on every drive; away punts on every drive.
        Home receives after its own pick-six AND after each away punt, so from
        the second drive on it has the ball every time: 0.5 + 21 home drives,
        each worth 6.95 to the away team. The old alternating chain gave home
        exactly 11 drives (76.45 points to away)."""
        t, o = flat_transition(), opening_at(4)
        _, away = dc.expected_points(
            self._all("Opp touchdown"), self._all("Punt"), t, o
        )
        assert away == pytest.approx(21.5 * 6.95), (
            f"away scored {away:.2f}; possession is still alternating after a "
            "returned touchdown"
        )

    def test_without_pick_sixes_each_team_gets_exactly_its_drives(self):
        """With no returned touchdowns possession strictly alternates, so over
        22 drives each team has 11, whoever wins the coin toss."""
        home = self._all("Touchdown")
        away = self._all("Punt")
        h, a = dc.expected_points(home, away, flat_transition(), opening_at(4))
        assert h == pytest.approx(11 * 6.95)
        assert a == pytest.approx(0.0)

    def test_the_opening_kickoff_is_a_coin_toss(self):
        """Identical teams must score identically -- the first version gave the
        home team the first possession of every game, a structural home edge
        unrelated to home-field advantage."""
        cond = uniform_cond()
        h, a = dc.expected_points(cond, cond, flat_transition(), opening_at(4), 3)
        assert h == pytest.approx(a)

    def test_the_sampler_agrees_with_the_exact_expectation(self):
        """simulate_games and expected_points implement the same rules; their
        means must agree within Monte Carlo error when pick-sixes are common."""
        home = np.tile(dist(Touchdown=0.3, **{"Opp touchdown": 0.15}), (dc.N_BINS, 1))
        away = np.tile(dist(Touchdown=0.2, **{"Field goal": 0.2}), (dc.N_BINS, 1))
        t, o = flat_transition(), opening_at(4)
        eh, ea = dc.expected_points(home, away, t, o)
        sh, sa = dc.simulate_games(home, away, t, o, n_trials=40000, seed=1)
        assert sh.mean() == pytest.approx(eh, abs=0.25)
        assert sa.mean() == pytest.approx(ea, abs=0.25)

    def test_transition_rows_come_from_the_right_owner(self):
        """The pick-six row is estimated from drives where the SAME team had the
        ball next; every other row from drives where it changed hands."""
        d = pd.DataFrame(
            {
                "result": ["Opp touchdown", "Opp touchdown", "Punt", "Punt"],
                "possession_changed": [False, True, True, False],
                "next_bin": [4, 0, 2, 5],
            }
        )
        trans = dc.bin_given_outcome(d)
        assert trans[CATS.index("Opp touchdown")].argmax() == 4
        assert trans[CATS.index("Punt")].argmax() == 2
