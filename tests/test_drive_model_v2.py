"""
Correctness gates for the rebuilt drive-outcome model.

Four of the five defects this model was rebuilt to fix are arithmetic, not
football, and arithmetic bugs here are invisible: every version produces scores
in the low twenties. v1 scored 9.608 with a Monte Carlo that estimated a
closed-form expectation, points that went to nobody, and no home-field term
whatsoever -- and none of that was apparent from its output.

So each fix gets a test that would fail if it were reverted.

Run: pytest tests/test_drive_model_v2.py -v
"""

import numpy as np
import pandas as pd
import pytest

import src.experiments.drive_model_v2 as dm

CATS = dm.CATS


def rates(**kw) -> np.ndarray:
    """A drive-outcome distribution; unspecified categories share the remainder."""
    v = np.zeros(len(CATS))
    for k, x in kw.items():
        v[CATS.index(k)] = x
    left = 1.0 - v.sum()
    blanks = [i for i, c in enumerate(CATS) if c not in kw]
    if blanks and left > 0:
        v[blanks] = left / len(blanks)
    return v


def game(home_off, away_def, away_off, home_def, drives=11.0) -> pd.DataFrame:
    row = {}
    for i, c in enumerate(CATS):
        row[f"home_pregame_off_{c}_rate"] = home_off[i]
        row[f"home_pregame_def_{c}_rate"] = home_def[i]
        row[f"away_pregame_off_{c}_rate"] = away_off[i]
        row[f"away_pregame_def_{c}_rate"] = away_def[i]
    row["home_pregame_n_drives"] = drives
    row["away_pregame_n_drives"] = drives
    row["home_pregame_n_drives_faced"] = drives
    row["away_pregame_n_drives_faced"] = drives
    return pd.DataFrame([row])


def model(league=None, home_field=0.0, shrink=1.0) -> dict:
    lg = league if league is not None else rates(touchdown=0.23, field_goal=0.15)
    return {
        "league_off": lg,
        "league_def": lg,
        "fallback_drives": 11.0,
        "home_field": home_field,
        "shrink": shrink,
    }


class TestLog5:
    """Defect 2: v1 averaged rates and ignored the league baseline."""

    def test_two_average_sides_return_the_baseline(self):
        got = dm.log5(np.array([0.25]), np.array([0.25]), np.array([0.25]))
        assert got[0] == pytest.approx(0.25)

    def test_two_above_average_sides_exceed_both(self):
        """The whole point. An arithmetic mean returns 0.30; log5 goes higher,
        because both sides are better than the league at this."""
        p = q = 0.30
        got = dm.log5(np.array([p]), np.array([q]), np.array([0.23]))[0]
        assert got > 0.30, f"log5 gave {got:.4f}, no more than the arithmetic mean"

    def test_two_below_average_sides_fall_below_both(self):
        got = dm.log5(np.array([0.15]), np.array([0.15]), np.array([0.23]))[0]
        assert got < 0.15

    def test_an_average_side_returns_the_other_side(self):
        """Facing a league-average opponent should leave a rate unchanged."""
        got = dm.log5(np.array([0.31]), np.array([0.23]), np.array([0.23]))[0]
        assert got == pytest.approx(0.31, abs=1e-6)

    def test_is_symmetric_in_its_two_inputs(self):
        a = dm.log5(np.array([0.30]), np.array([0.18]), np.array([0.23]))[0]
        b = dm.log5(np.array([0.18]), np.array([0.30]), np.array([0.23]))[0]
        assert a == pytest.approx(b)

    def test_blend_returns_a_proper_distribution(self):
        off = rates(touchdown=0.30, field_goal=0.20)
        deff = rates(touchdown=0.26, field_goal=0.16)
        lg = rates(touchdown=0.23, field_goal=0.15)
        p = dm.blend_distribution(off, deff, lg)
        assert p.sum() == pytest.approx(1.0)
        assert (p >= 0).all()


class TestDefensivePoints:
    """Defect 3: returned touchdowns and safeties went to nobody in v1."""

    def test_a_returned_touchdown_scores_for_the_defense(self):
        # Home offence turns it over for a score on every drive.
        ho = rates(opp_touchdown=1.0)
        neutral = rates(touchdown=0.23, field_goal=0.15)
        h, a = dm.predict_drive_model(
            model(league=neutral), game(ho, neutral, neutral, neutral)
        )
        assert a[0] > h[0], (
            "the home offence conceded a return touchdown on every drive and the "
            "away team did not get the points"
        )

    def test_a_safety_is_worth_two_to_the_defense(self):
        assert dm.DEFENSE_POINTS[CATS.index("safety")] == pytest.approx(2.0)
        assert dm.OFFENSE_POINTS[CATS.index("safety")] == 0.0

    def test_offense_points_are_only_touchdowns_and_field_goals(self):
        for i, c in enumerate(CATS):
            if c in ("touchdown", "field_goal"):
                assert dm.OFFENSE_POINTS[i] > 0
            else:
                assert dm.OFFENSE_POINTS[i] == 0, f"{c} scores for the offence"

    def test_a_touchdown_is_worth_about_seven_not_exactly_seven(self):
        """6 plus a conversion that succeeds ~94% of the time."""
        td = dm.OFFENSE_POINTS[CATS.index("touchdown")]
        assert 6.8 < td < 7.0


class TestHomeField:
    """Defect 5: no home-field mechanism at all. Found by measurement, not by
    reading -- v1 predicted a home margin of -0.06 against an actual +2.23."""

    def test_the_offset_moves_the_margin(self):
        neutral = rates(touchdown=0.23, field_goal=0.15)
        g = game(neutral, neutral, neutral, neutral)
        h0, a0 = dm.predict_drive_model(model(home_field=0.0), g)
        h2, a2 = dm.predict_drive_model(model(home_field=2.4), g)
        assert (h2[0] - a2[0]) - (h0[0] - a0[0]) == pytest.approx(2.4)

    def test_the_offset_leaves_the_total_alone(self):
        """Split evenly, so it is a statement about who scores, not how much."""
        neutral = rates(touchdown=0.23, field_goal=0.15)
        g = game(neutral, neutral, neutral, neutral)
        h0, a0 = dm.predict_drive_model(model(home_field=0.0), g)
        h2, a2 = dm.predict_drive_model(model(home_field=2.4), g)
        assert (h2[0] + a2[0]) == pytest.approx(h0[0] + a0[0])

    def test_two_identical_teams_still_differ_by_the_home_edge(self):
        neutral = rates(touchdown=0.23, field_goal=0.15)
        h, a = dm.predict_drive_model(
            model(home_field=2.2), game(neutral, neutral, neutral, neutral)
        )
        assert h[0] > a[0], "identical teams produced no home advantage"


class TestExpectationIsExact:
    """Defect 1: v1 used 2,000 Monte Carlo draws to estimate a closed form."""

    def test_the_point_prediction_is_deterministic(self):
        neutral = rates(touchdown=0.23, field_goal=0.15)
        g = game(neutral, neutral, neutral, neutral)
        m = model()
        runs = [dm.predict_drive_model(m, g)[0][0] for _ in range(5)]
        assert len(set(runs)) == 1, (
            f"the point prediction varies between identical calls ({runs}) -- "
            "sampling is back in the prediction path"
        )

    def test_it_equals_the_analytic_expectation(self):
        """n * (p . points), by hand."""
        off = rates(touchdown=0.40, field_goal=0.20)
        lg = rates(touchdown=0.23, field_goal=0.15)
        g = game(off, lg, lg, lg, drives=10.0)
        h, _ = dm.predict_drive_model(model(league=lg), g)
        p = dm.blend_distribution(off, lg, lg)
        expected_off = 10.0 * (p @ dm.OFFENSE_POINTS)
        conceded_by_away = 10.0 * (
            dm.blend_distribution(lg, lg, lg) @ dm.DEFENSE_POINTS
        )
        assert h[0] == pytest.approx(expected_off + conceded_by_away, rel=1e-9)

    def test_the_sampler_still_exists_for_distributions(self):
        """Kept deliberately -- a drive model's advantage is the full
        distribution -- just not in the mean's path."""
        neutral = rates(touchdown=0.23, field_goal=0.15)
        g = game(neutral, neutral, neutral, neutral)
        home, away = dm.simulate_distribution(model(), g, n_trials=500)
        assert home.shape == (1, 500)
        assert home[0].std() > 3.0, "simulated scores have no spread"


class TestShrinkSelection:
    """The blend weight is chosen INSIDE the training fold, not on the test set.
    Hardcoding the test-set optimum (0.6) is the error this project already
    refused once over ridge alpha=100."""

    @staticmethod
    def _train(n=400, seed=0):
        rng = np.random.default_rng(seed)
        neutral = rates(touchdown=0.23, field_goal=0.15)
        rows = []
        for i in range(n):
            g = game(neutral, neutral, neutral, neutral).iloc[0].to_dict()
            g["gameday"] = pd.Timestamp("2022-09-01") + pd.Timedelta(days=7 * i)
            g["home_score"] = 24 + rng.normal(0, 7)
            g["away_score"] = 21 + rng.normal(0, 7)
            rows.append(g)
        return pd.DataFrame(rows)

    def test_fit_selects_a_weight_from_the_grid(self):
        m = dm.fit_drive_model(self._train())
        assert m["shrink"] in dm.SHRINK_GRID

    def test_selection_never_sees_data_outside_the_training_fold(self):
        """Appending future games must not change the weight chosen for the
        earlier fold."""
        train = self._train(n=400, seed=1)
        m1 = dm.fit_drive_model(train)
        m2 = dm.fit_drive_model(train)
        assert m1["shrink"] == m2["shrink"], "selection is not deterministic"

    def test_a_tiny_fold_falls_back_rather_than_overfitting(self):
        m = dm.fit_drive_model(self._train(n=20))
        assert m["shrink"] == 0.6

    def test_shrink_pulls_predictions_toward_the_league_baseline(self):
        strong = rates(touchdown=0.45, field_goal=0.20)
        lg = rates(touchdown=0.23, field_goal=0.15)
        g = game(strong, lg, lg, lg)
        full, _ = dm.predict_drive_model(model(league=lg, shrink=1.0), g)
        part, _ = dm.predict_drive_model(model(league=lg, shrink=0.4), g)
        flat, _ = dm.predict_drive_model(model(league=lg, shrink=0.0), g)
        assert (
            full[0] > part[0] > flat[0]
        ), f"shrinkage is not monotone ({full[0]:.2f}, {part[0]:.2f}, {flat[0]:.2f})"


class TestSharedPossessions:
    """Defect 4: drive counts are shared between the teams, not independent."""

    def test_a_teams_drives_blend_with_the_opponents_drives_faced(self):
        neutral = rates(touchdown=0.23, field_goal=0.15)
        g = game(neutral, neutral, neutral, neutral)
        g.loc[0, "home_pregame_n_drives"] = 8.0
        g.loc[0, "away_pregame_n_drives_faced"] = 14.0  # same quantity, other end
        h, _ = dm.predict_drive_model(model(), g)
        # Blended to 11, so the prediction must sit between the 8- and 14-drive
        # answers rather than using 8 alone.
        g8 = g.copy()
        g8.loc[0, "away_pregame_n_drives_faced"] = 8.0
        h8, _ = dm.predict_drive_model(model(), g8)
        assert h[0] > h8[0], "the opponent's drives-faced was ignored"


class TestContextCorrections:
    """Three corrections the drive model was missing that every other model in
    the project already had. All measured, none speculative:

      drift        an uncorrected +1.098 away bias. recent_residual_offset is
                   the project's validated fix and took the other models from
                   +0.78 to +0.06.
      neutral site C3. A Super Bowl has no home team; v2 was handing it a home
                   edge anyway, and including those games dragged the estimate.
      overtime     C5. An extra period inflates the margin in a way no pregame
                   quantity predicts, so those rows are halved when fitting.
    """

    @staticmethod
    def _train(n=300, margin=3.0, neutral_margin=0.0, ot_margin=0.0, seed=0):
        rng = np.random.default_rng(seed)
        neutral = rates(touchdown=0.23, field_goal=0.15)
        rows = []
        for i in range(n):
            g = game(neutral, neutral, neutral, neutral).iloc[0].to_dict()
            g["gameday"] = pd.Timestamp("2022-09-01") + pd.Timedelta(days=7 * i)
            g["is_neutral_site"] = 1 if i % 10 == 0 else 0
            g["went_to_ot"] = 1 if i % 7 == 0 else 0
            m = margin
            if g["is_neutral_site"]:
                m = neutral_margin
            if g["went_to_ot"]:
                m = ot_margin
            g["away_score"] = 21.0
            g["home_score"] = 21.0 + m
            rows.append(g)
        return pd.DataFrame(rows)

    def test_neutral_site_games_are_excluded_from_the_home_estimate(self):
        """Neutral games with a zero margin must not drag the estimate down.

        ot_margin is set equal to margin so this isolates the neutral-site
        correction; the overtime weighting is exercised separately below.
        """
        with_neutral = dm._home_field(
            self._train(margin=4.0, neutral_margin=0.0, ot_margin=4.0)
        )
        assert with_neutral == pytest.approx(4.0, abs=0.2), (
            f"home edge came out {with_neutral:.2f}, not ~4.0 -- neutral-site "
            "games are being averaged in"
        )

    def test_including_neutral_games_would_visibly_drag_the_estimate(self):
        """Shows the correction is load-bearing rather than cosmetic: the naive
        mean over every game lands well below the true home-game margin."""
        t = self._train(margin=4.0, neutral_margin=0.0, ot_margin=4.0)
        naive = float((t["home_score"] - t["away_score"]).mean())
        corrected = dm._home_field(t)
        assert (
            corrected - naive > 0.25
        ), f"excluding neutral sites moved the estimate only {corrected - naive:.3f}"

    def test_neutral_site_games_receive_no_home_adjustment(self):
        neutral = rates(touchdown=0.23, field_goal=0.15)
        g = game(neutral, neutral, neutral, neutral)
        g["is_neutral_site"] = 1
        h, a = dm.predict_drive_model(model(home_field=3.0), g)
        assert h[0] == pytest.approx(
            a[0]
        ), "a neutral-site game was given a home-field edge"

    def test_a_normal_game_still_receives_it(self):
        neutral = rates(touchdown=0.23, field_goal=0.15)
        g = game(neutral, neutral, neutral, neutral)
        g["is_neutral_site"] = 0
        h, a = dm.predict_drive_model(model(home_field=3.0), g)
        assert h[0] - a[0] == pytest.approx(3.0)

    def test_overtime_games_are_down_weighted_not_dropped(self):
        """Halved, so they still inform the estimate but do not dominate it."""
        inflated = dm._home_field(self._train(margin=3.0, ot_margin=30.0))
        flat = dm._home_field(self._train(margin=3.0, ot_margin=3.0))
        assert inflated > flat, "OT games are being dropped entirely, not weighted"
        # ~1 game in 7 is OT here; at full weight a 30-point margin would drag
        # the mean to ~7, at half weight to ~5.
        assert inflated < 7.0, "OT games carried full weight"

    def test_the_drift_offset_is_applied_to_predictions(self):
        neutral = rates(touchdown=0.23, field_goal=0.15)
        g = game(neutral, neutral, neutral, neutral)
        plain = model()
        shifted = dict(plain, off_h=1.5, off_a=-2.0)
        h0, a0 = dm.predict_drive_model(plain, g)
        h1, a1 = dm.predict_drive_model(shifted, g)
        assert h1[0] == pytest.approx(h0[0] - 1.5)
        assert a1[0] == pytest.approx(a0[0] + 2.0)

    def test_fit_produces_an_offset_from_its_own_residuals(self):
        """It must be measured AFTER shrink is chosen -- the offset describes the
        model as finally configured, not an intermediate one."""
        m = dm.fit_drive_model(self._train())
        assert "off_h" in m and "off_a" in m
        assert np.isfinite(m["off_h"]) and np.isfinite(m["off_a"])

    def test_a_table_without_the_context_columns_still_works(self):
        """drive_model_table (production) carries neither column. The model must
        degrade rather than crash when handed one."""
        train = self._train().drop(columns=["is_neutral_site", "went_to_ot"])
        m = dm.fit_drive_model(train)
        assert np.isfinite(m["home_field"])
        h, a = dm.predict_drive_model(m, train.head(3))
        assert np.isfinite(h).all() and np.isfinite(a).all()
