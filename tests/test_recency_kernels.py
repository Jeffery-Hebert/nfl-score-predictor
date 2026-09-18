"""
Unit tests for the generalised recency weighting in src/experiments/recency.py.

The point of these is NOT "does it run". It is that every later comparison in
this line of work is measured against a control arm, and the control arm is only
meaningful if it genuinely reproduces production. If prior_weighted_mean with an
Exponential kernel and anchor="previous" differs from what
build_rolling_features.py actually computes, then every "two-timescale beats
production" number is really "my reimplementation differs from pandas", which is
worthless and would look identical on a scoreboard.

So the first test here compares against pandas' own ewm on random data, and the
rest pin the individual properties -- strict priority, finale masking, the shape
claims the two-timescale kernel is supposed to have -- each on a fixture where
the right answer is known by hand rather than by running the code.

Run: pytest tests/test_recency_kernels.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.experiments.recency import (
    Exponential,
    prior_weighted_rate,
    RecencyPlan,
    TwoTimescale,
    prior_weighted_mean,
    production_plan,
)

H = 17 * 7.0


def weekly(n, start="2024-09-01"):
    return pd.to_datetime(
        [pd.Timestamp(start) + pd.Timedelta(days=7 * i) for i in range(n)]
    )


# ---------------------------------------------------- equivalence to production


class TestReproducesProduction:
    """The control arm has to be the real thing, not an approximation of it."""

    @staticmethod
    def _pandas_reference(values, times, halflife_days, include=None):
        """Exactly what build_rolling_features.add_pregame_rolling_features does."""
        s = pd.Series(values, dtype=float)
        if include is not None:
            s = s.where(pd.Series(include))
        # ignore_na=False is the clean definition: a weighted mean of the
        # observed values by real elapsed time. Production passes True, which is
        # a different normalisation -- see test_documents_the_production_masking
        # convention below, and the module docstring.
        ewm = s.ewm(
            halflife=pd.Timedelta(days=halflife_days), times=times, ignore_na=False
        ).mean()
        return ewm.shift(1).to_numpy()

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_pandas_ewm_plus_shift_on_random_data(self, seed):
        rng = np.random.default_rng(seed)
        n = 40
        values = rng.normal(22, 7, n)
        times = weekly(n)

        mine = prior_weighted_mean(
            values, times.to_numpy(), Exponential(H), anchor="previous"
        )
        theirs = self._pandas_reference(values, times, H)

        both = np.isfinite(mine) & np.isfinite(theirs)
        assert both.sum() >= n - 2, "almost every row should be comparable"
        np.testing.assert_allclose(mine[both], theirs[both], rtol=1e-9, atol=1e-9)

    def test_matches_pandas_with_irregular_spacing(self):
        """Byes and offseasons are where a time-based scheme earns its keep, and
        where an index-based reimplementation would silently diverge."""
        times = pd.to_datetime(
            [
                "2023-09-10",
                "2023-09-17",
                "2023-10-01",  # bye
                "2023-12-31",
                "2024-09-08",
                "2024-09-15",  # offseason
            ]
        )
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        mine = prior_weighted_mean(
            values, times.to_numpy(), Exponential(H), anchor="previous"
        )
        theirs = self._pandas_reference(values, times, H)
        both = np.isfinite(mine) & np.isfinite(theirs)
        np.testing.assert_allclose(mine[both], theirs[both], rtol=1e-9, atol=1e-9)

    def test_matches_pandas_with_masked_rows(self):
        """Finale masking is `.where(~mask)` plus ignore_na in production."""
        n = 20
        rng = np.random.default_rng(7)
        values = rng.normal(22, 7, n)
        times = weekly(n)
        include = np.ones(n, dtype=bool)
        include[[5, 11, 17]] = False

        mine = prior_weighted_mean(
            values, times.to_numpy(), Exponential(H), include=include, anchor="previous"
        )
        theirs = self._pandas_reference(values, times, H, include=include)
        both = np.isfinite(mine) & np.isfinite(theirs)
        np.testing.assert_allclose(mine[both], theirs[both], rtol=1e-9, atol=1e-9)

    def test_documents_the_production_masking_convention(self):
        """Production passes ignore_na=True, which is NOT a weighted mean of the
        observed values -- it renormalises differently around NaN rows.

        Pinned rather than matched. On real data the gap is ~0.012 against a
        team_score sd of 9.9, which is why the experiments use production's own
        function as their control arm instead of this one. If this test ever
        starts passing, pandas changed its semantics and the control-arm
        reasoning needs revisiting.
        """
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        times = weekly(5)
        include = np.array([True, False, True, True, True])
        s_masked = pd.Series(vals).where(pd.Series(include))
        clean = (
            s_masked.ewm(halflife=pd.Timedelta(days=H), times=times, ignore_na=False)
            .mean()
            .shift(1)
            .to_numpy()
        )
        production = (
            s_masked.ewm(halflife=pd.Timedelta(days=H), times=times, ignore_na=True)
            .mean()
            .shift(1)
            .to_numpy()
        )
        assert clean[3] != pytest.approx(production[3]), (
            "the two ignore_na conventions agree -- the control-arm argument in "
            "the module docstring no longer holds"
        )
        # ours is the clean one
        mine = prior_weighted_mean(
            vals, times.to_numpy(), Exponential(H), include=include, anchor="previous"
        )
        assert mine[3] == pytest.approx(clean[3])

    def test_production_plan_is_the_production_scheme(self):
        p = production_plan()
        assert p.anchor == "previous", "the control must anchor the way production does"
        assert p.per_stat is None
        assert p.kernel_for("anything").halflife_days == H


# --------------------------------------------------------------- leakage rules


class TestStrictlyPrior:
    def test_first_row_has_no_estimate(self):
        got = prior_weighted_mean(
            np.array([10.0]), weekly(1).to_numpy(), Exponential(H)
        )
        assert np.isnan(got[0])

    def test_a_row_never_sees_its_own_value(self):
        """THE property. Row 2's estimate must be row 1 alone, whatever row 2 is."""
        for own in (0.0, 1e6, -500.0):
            got = prior_weighted_mean(
                np.array([10.0, own]), weekly(2).to_numpy(), Exponential(H)
            )
            assert got[1] == pytest.approx(
                10.0
            ), f"row 2's own value {own} reached its own estimate"

    def test_a_row_never_sees_the_future(self):
        got = prior_weighted_mean(
            np.array([10.0, 20.0, 1e6]), weekly(3).to_numpy(), Exponential(H)
        )
        assert got[1] == pytest.approx(10.0)
        assert 10.0 <= got[2] <= 20.0, "a later observation leaked backwards"

    def test_masked_rows_are_predicted_but_never_contribute(self):
        include = np.array([True, False, True])
        got = prior_weighted_mean(
            np.array([10.0, 999.0, 0.0]),
            weekly(3).to_numpy(),
            Exponential(H),
            include=include,
        )
        assert np.isfinite(got[1]), "a masked row must still get a value"
        assert got[2] == pytest.approx(10.0), "the masked 999 leaked forward"

    def test_nan_observations_are_skipped_not_zeroed(self):
        """A missing stat must be absent from the average, not counted as 0."""
        got = prior_weighted_mean(
            np.array([10.0, np.nan, 10.0, 0.0]), weekly(4).to_numpy(), Exponential(H)
        )
        assert got[2] == pytest.approx(10.0), "NaN was treated as a zero observation"


# ------------------------------------------------------------------- anchoring


class TestAnchoring:
    """Anchoring is a no-op for a single exponential and real for a mixture.

    This surprised me, so it is pinned. w_i = 0.5^((A-t_i)/H) factors into
    0.5^(A/H) * 0.5^(-t_i/H); the first term is constant across observations and
    cancels in a normalised mean. Production's .shift(1) is therefore harmless,
    which is the opposite of what it looks like.
    """

    TIMES = pd.to_datetime(["2024-09-01", "2024-09-08", "2025-09-07"]).to_numpy()
    VALUES = np.array([30.0, 10.0, 0.0])

    def test_anchor_is_a_no_op_for_a_single_exponential(self):
        cur = prior_weighted_mean(
            self.VALUES, self.TIMES, Exponential(H), anchor="current"
        )
        prev = prior_weighted_mean(
            self.VALUES, self.TIMES, Exponential(H), anchor="previous"
        )
        np.testing.assert_allclose(cur[1:], prev[1:], rtol=1e-12)

    def test_anchor_changes_the_answer_for_a_mixture(self):
        """Two components pick up two different constants, so nothing cancels."""
        k = TwoTimescale(halflife_fast_days=21, halflife_slow_days=H, weight_fast=0.5)
        cur = prior_weighted_mean(self.VALUES, self.TIMES, k, anchor="current")
        prev = prior_weighted_mean(self.VALUES, self.TIMES, k, anchor="previous")
        assert cur[2] != pytest.approx(prev[2]), (
            "the anchor made no difference to a mixture -- either the mixture "
            "collapsed to one component or the anchor is not being applied"
        )

    def test_a_mixture_anchored_at_a_long_gap_forgets_recent_form(self):
        """Seen from a year later, last season's last two games are both ancient
        and the fast component has died on both, so they weight nearly evenly.
        Seen from the day after game 2, the fast component still favours it."""
        k = TwoTimescale(halflife_fast_days=21, halflife_slow_days=H, weight_fast=0.5)
        cur = prior_weighted_mean(self.VALUES, self.TIMES, k, anchor="current")
        prev = prior_weighted_mean(self.VALUES, self.TIMES, k, anchor="previous")
        even = self.VALUES[:2].mean()
        assert abs(cur[2] - even) < abs(prev[2] - even), (
            f"the current-date anchor should even out across a long layoff "
            f"(cur={cur[2]:.3f}, prev={prev[2]:.3f}, even={even:.3f})"
        )

    def test_identical_when_history_is_flat(self):
        """A constant history returns the constant under any weighting. Guards
        against a test that passes by arithmetic coincidence."""
        values = np.full(6, 21.0)
        times = weekly(6).to_numpy()
        for k in (Exponential(H), TwoTimescale(21, H, 0.5)):
            cur = prior_weighted_mean(values, times, k, anchor="current")
            np.testing.assert_allclose(cur[1:], 21.0)


# ------------------------------------------------------------ kernel behaviour


class TestExponentialKernel:
    def test_halves_at_the_halflife(self):
        k = Exponential(100.0)
        assert k(np.array([0.0]))[0] == pytest.approx(1.0)
        assert k(np.array([100.0]))[0] == pytest.approx(0.5)
        assert k(np.array([200.0]))[0] == pytest.approx(0.25)

    def test_recent_games_outweigh_old_ones(self):
        values = np.array([0.0, 100.0, 0.0])
        times = pd.to_datetime(["2020-01-01", "2024-09-08", "2024-09-15"]).to_numpy()
        got = prior_weighted_mean(values, times, Exponential(H))
        assert got[2] > 90.0, "a four-year-old game is still dominating"


class TestTwoTimescaleKernel:
    def test_is_steeper_than_a_single_exponential_near_the_present(self):
        """The whole reason this family exists. At short ages the mixture must
        fall away faster than a single exponential with the same long tail."""
        slow = Exponential(H)
        mix = TwoTimescale(halflife_fast_days=21, halflife_slow_days=H, weight_fast=0.5)
        near = np.array([14.0])
        assert (
            mix(near)[0] < slow(near)[0]
        ), "the mixture is not steeper than the single exponential it contains"

    def test_retains_a_long_tail(self):
        """...and must NOT collapse to the fast component, or it is just a short
        half-life, which has already been tested and lost."""
        fast_only = Exponential(21)
        mix = TwoTimescale(halflife_fast_days=21, halflife_slow_days=H, weight_fast=0.5)
        far = np.array([300.0])
        assert mix(far)[0] > fast_only(far)[0] * 5, (
            "the mixture decayed like its fast component -- the slow tail is "
            "doing nothing"
        )

    def test_starts_at_one_like_any_weighting(self):
        mix = TwoTimescale(21, H, 0.5)
        assert mix(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_reduces_to_the_fast_component_at_weight_one(self):
        mix = TwoTimescale(21, H, 1.0)
        exp = Exponential(21)
        ages = np.array([0.0, 30.0, 200.0])
        np.testing.assert_allclose(mix(ages), exp(ages))

    def test_reduces_to_the_slow_component_at_weight_zero(self):
        mix = TwoTimescale(21, H, 0.0)
        exp = Exponential(H)
        ages = np.array([0.0, 30.0, 200.0])
        np.testing.assert_allclose(mix(ages), exp(ages))

    def test_rejects_a_fast_component_that_is_not_fast(self):
        with pytest.raises(ValueError, match="must be shorter"):
            TwoTimescale(
                halflife_fast_days=200, halflife_slow_days=100, weight_fast=0.5
            )

    def test_rejects_a_weight_outside_the_unit_interval(self):
        with pytest.raises(ValueError, match="weight_fast"):
            TwoTimescale(21, H, 1.5)


# --------------------------------------------------------------- the plan type


class TestRecencyPlan:
    def test_per_stat_overrides_the_default(self):
        plan = RecencyPlan(
            default=Exponential(H), per_stat={"off_success_rate": Exponential(52 * 7)}
        )
        assert plan.kernel_for("off_success_rate").halflife_days == 52 * 7
        assert plan.kernel_for("anything_else").halflife_days == H

    def test_an_empty_plan_is_the_production_scheme(self):
        plan = RecencyPlan(default=Exponential(H))
        assert plan.kernel_for("off_epa_per_play").halflife_days == H

    def test_describe_is_informative_enough_to_read_in_a_results_table(self):
        plan = RecencyPlan(
            default=Exponential(H), per_stat={"a": Exponential(7)}, anchor="current"
        )
        d = plan.describe()
        assert "17w" in d and "current" in d and "per-stat" in d


class TestSimultaneousObservations:
    """Rows sharing a timestamp must not inform each other.

    This function is used two ways: per team, where a team plays once a day and
    the question never arises, and across the WHOLE LEAGUE for the split
    features' prior, where a dozen games share a Sunday. A mask built on array
    position rather than timestamp passes every per-team test and leaks at
    league scale -- the earlier-SORTED of two simultaneous games feeding the
    later-sorted one, a leak whose existence depends on nothing but sort order.
    """

    SUNDAY = pd.to_datetime(
        ["2024-09-01", "2024-09-08", "2024-09-08", "2024-09-08"]
    ).to_numpy()

    def test_simultaneous_rows_get_identical_estimates(self):
        values = np.array([10.0, 500.0, 500.0, 500.0])
        got = prior_weighted_mean(values, self.SUNDAY, Exponential(H))
        assert got[1] == got[2] == got[3] == pytest.approx(10.0), (
            f"same-day rows informed each other (got {got[1:]}) -- on real data "
            "this is a dozen Sunday games leaking into one another"
        )

    def test_simultaneous_rows_are_visible_to_a_later_day(self):
        """...but they must still count for everything that comes after."""
        values = np.array([0.0, 10.0, 10.0, 10.0])
        times = pd.to_datetime(
            ["2024-09-01", "2024-09-08", "2024-09-08", "2024-09-15"]
        ).to_numpy()
        got = prior_weighted_mean(values, times, Exponential(H))
        assert got[3] > 5.0, "the Sunday games were excluded from the following week"

    def test_rate_version_also_excludes_simultaneous_rows(self):
        num = np.array([0.0, 100.0, 100.0])
        den = np.array([10.0, 10.0, 10.0])
        rate, n_eff = prior_weighted_rate(num, den, self.SUNDAY[:3], Exponential(H))
        assert (
            rate[1] == rate[2] == pytest.approx(0.0)
        ), "a simultaneous row's 100 EPA reached the other's rate"
        assert n_eff[1] == pytest.approx(n_eff[2]), "the two saw different histories"
        # n_eff is the RECENCY-WEIGHTED denominator, not a raw count: the only
        # prior game is 10 plays a week old, so 10 * 0.5^(7/119) = 9.60. What
        # matters is that it is one game's worth and not two -- ~19.6 would mean
        # the simultaneous row had been counted.
        assert n_eff[1] == pytest.approx(10 * 0.5 ** (7 / 119), rel=1e-6)
        assert n_eff[1] < 11.0, "a second, simultaneous game entered the sample"
