"""
Unit tests for the offense-vs-defense interaction terms.

The failure this file is really guarding against is ORIENTATION. Pair a home
passing attack against the HOME pass defence instead of the away one and you get
a column that is finite, well-scaled, correlated with plenty of things, and
completely meaningless. Nothing downstream would notice: the model would fit it,
the RMSE would move slightly, and the result would be reported as "interactions
don't help".

So these tests assert the pairing directly, against hand-built rows where the
right answer is arithmetic rather than a matter of reading the column name.

Run: pytest tests/test_matchup_terms.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.experiments.matchup import (
    INJURY_COLS,
    INJURY_TERMS,
    MATCHUP_COLS,
    MATCHUP_TERMS,
    add_matchup_terms,
)

SPLIT = [
    "pregame_off_pass_epa_shrunk",
    "pregame_off_rush_epa_shrunk",
    "pregame_def_pass_epa_allowed_shrunk",
    "pregame_def_rush_epa_allowed_shrunk",
]


def frame(**overrides) -> pd.DataFrame:
    """One game with every split column present and distinct, so a mis-pairing
    cannot hide behind two equal values."""
    row = {}
    for i, side in enumerate(("home", "away")):
        for j, c in enumerate(SPLIT):
            row[f"{side}_{c}"] = 0.1 * (i + 1) + 0.01 * (j + 1)
        row[f"{side}_injury_impact"] = 0.5 * (i + 1)
    row.update(overrides)
    return pd.DataFrame([row])


class TestOrientation:
    """A team's offence must meet the OPPONENT's defence."""

    def test_home_pass_matchup_pairs_home_offense_with_away_defense(self):
        df = frame(
            home_pregame_off_pass_epa_shrunk=0.20,
            away_pregame_def_pass_epa_allowed_shrunk=0.30,
            home_pregame_def_pass_epa_allowed_shrunk=999.0,  # must NOT be used
        )
        got = add_matchup_terms(df)["home_pass_matchup"].iloc[0]
        assert got == pytest.approx(0.06), (
            f"expected 0.20 x 0.30 = 0.06, got {got} -- the home passing attack "
            "was paired against the wrong defence"
        )

    def test_away_pass_matchup_pairs_away_offense_with_home_defense(self):
        df = frame(
            away_pregame_off_pass_epa_shrunk=0.20,
            home_pregame_def_pass_epa_allowed_shrunk=0.30,
            away_pregame_def_pass_epa_allowed_shrunk=999.0,
        )
        got = add_matchup_terms(df)["away_pass_matchup"].iloc[0]
        assert got == pytest.approx(0.06)

    def test_rush_terms_use_rush_columns_not_pass_ones(self):
        df = frame(
            home_pregame_off_rush_epa_shrunk=0.40,
            away_pregame_def_rush_epa_allowed_shrunk=0.50,
            home_pregame_off_pass_epa_shrunk=999.0,
            away_pregame_def_pass_epa_allowed_shrunk=999.0,
        )
        got = add_matchup_terms(df)["home_rush_matchup"].iloc[0]
        assert got == pytest.approx(0.20), "a rush matchup picked up a pass column"

    def test_every_term_uses_one_home_and_one_away_input(self):
        """Structural guard: an interaction between two columns on the SAME side
        is not a matchup, it is a self-product."""
        for name, off_col, def_col in MATCHUP_TERMS:
            sides = {off_col.split("_")[0], def_col.split("_")[0]}
            assert sides == {"home", "away"}, (
                f"{name} multiplies two {sides.pop()}-side columns -- that is not "
                "a matchup"
            )

    def test_every_term_pairs_an_offense_with_a_defense(self):
        for name, off_col, def_col in MATCHUP_TERMS:
            assert "_off_" in off_col, f"{name}'s first input is not an offence"
            assert "_def_" in def_col, f"{name}'s second input is not a defence"

    def test_pass_terms_pair_pass_with_pass_and_rush_with_rush(self):
        """A passing attack meets a pass defence. Crossing them would be a
        different and unmotivated quantity."""
        for name, off_col, def_col in MATCHUP_TERMS:
            unit = "pass" if "pass" in name else "rush"
            assert (
                unit in off_col and unit in def_col
            ), f"{name} crosses play types: {off_col} x {def_col}"


class TestArithmetic:
    def test_a_good_offense_against_a_bad_defense_is_the_largest_product(self):
        """The football claim, stated as a test. Both inputs are 'higher is more
        scoring' -- offensive EPA generated, and defensive EPA allowed -- so the
        product is largest exactly where the mismatch is."""
        good_vs_bad = frame(
            home_pregame_off_pass_epa_shrunk=0.25,
            away_pregame_def_pass_epa_allowed_shrunk=0.25,
        )
        bad_vs_good = frame(
            home_pregame_off_pass_epa_shrunk=-0.25,
            away_pregame_def_pass_epa_allowed_shrunk=-0.25,
        )
        mixed = frame(
            home_pregame_off_pass_epa_shrunk=0.25,
            away_pregame_def_pass_epa_allowed_shrunk=-0.25,
        )
        g = add_matchup_terms(good_vs_bad)["home_pass_matchup"].iloc[0]
        m = add_matchup_terms(mixed)["home_pass_matchup"].iloc[0]
        assert g > m, "a mismatch did not produce a larger interaction than a wash"
        # Two negatives also multiply positive -- which is correct and worth
        # pinning, because it is the one place the sign convention surprises.
        b = add_matchup_terms(bad_vs_good)["home_pass_matchup"].iloc[0]
        assert b > 0, (
            "bad offence against good defence should give a positive product; the "
            "model learns the sign of its coefficient, not this column"
        )

    def test_term_is_exactly_the_product_of_its_inputs(self):
        rng = np.random.default_rng(0)
        df = frame()
        for _ in range(5):
            df.loc[0, "home_pregame_off_pass_epa_shrunk"] = rng.normal()
            df.loc[0, "away_pregame_def_pass_epa_allowed_shrunk"] = rng.normal()
            out = add_matchup_terms(df)
            assert out["home_pass_matchup"].iloc[0] == pytest.approx(
                df["home_pregame_off_pass_epa_shrunk"].iloc[0]
                * df["away_pregame_def_pass_epa_allowed_shrunk"].iloc[0]
            )


class TestContract:
    def test_adds_exactly_four_columns_and_mutates_nothing(self):
        df = frame()
        before = df.copy()
        out = add_matchup_terms(df)
        assert set(out.columns) - set(before.columns) == set(MATCHUP_COLS)
        pd.testing.assert_frame_equal(df, before), "the input frame was mutated"

    def test_a_missing_input_raises_rather_than_producing_nan(self):
        """A NaN column would be imputed to the training mean by every model
        here, turning it into a silent constant that contributes nothing while
        still appearing in the feature list."""
        df = frame().drop(columns=["away_pregame_def_pass_epa_allowed_shrunk"])
        with pytest.raises(KeyError, match="home_pass_matchup"):
            add_matchup_terms(df)

    def test_injury_terms_are_a_separate_opt_in_arm(self):
        """They must not ride along on the matchup arm's result -- this project
        has been burned by bundled features (qb_out diluting injury_impact)."""
        assert not set(INJURY_COLS) & set(MATCHUP_COLS)
        out = add_matchup_terms(frame())
        assert not set(INJURY_COLS) & set(out.columns)
        out2 = add_matchup_terms(frame(), terms=INJURY_TERMS)
        assert set(INJURY_COLS) <= set(out2.columns)

    def test_works_on_many_rows_at_once(self):
        df = pd.concat([frame() for _ in range(50)], ignore_index=True)
        out = add_matchup_terms(df)
        assert len(out) == 50
        assert out[MATCHUP_COLS].notna().all().all()


@pytest.mark.requires_data
class TestAgainstRealTable:
    def test_terms_are_finite_and_bounded_on_the_real_model_table(self):
        """The inputs are shrunk toward the league mean, so the products cannot
        blow up the way raw-EPA interactions would. If this starts failing, the
        shrinkage stopped working."""
        df = pd.read_parquet("data/processed/model_table.parquet")
        out = add_matchup_terms(df)
        for c in MATCHUP_COLS:
            v = out[c].dropna()
            assert np.isfinite(v).all(), f"{c} contains non-finite values"
            assert v.abs().max() < 0.25, (
                f"{c} reaches {v.abs().max():.3f}; shrunk EPA rates sit near "
                "+/-0.15 so a product should stay well under 0.25"
            )

    def test_terms_are_not_collinear_with_their_own_inputs(self):
        """If the product correlates ~1.0 with a main effect it is not adding an
        interaction, it is adding a duplicate -- which this project has
        repeatedly found to be worse than nothing."""
        df = add_matchup_terms(pd.read_parquet("data/processed/model_table.parquet"))
        for name, off_col, def_col in MATCHUP_TERMS:
            for parent in (off_col, def_col):
                r = df[name].corr(df[parent])
                assert abs(r) < 0.9, (
                    f"{name} correlates {r:+.2f} with {parent} -- it is a "
                    "restatement of a feature already in the model"
                )
