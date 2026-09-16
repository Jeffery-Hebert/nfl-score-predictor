"""
Grading gates for the prediction ledger.

The report page claims two things that are easy to get quietly wrong, and both
are wrong in the direction of flattering the model:

  1. the sign of an error. "Prediction minus reality" has to hold for every
     column, or a model that consistently overshoots reads as if it undershoots.
  2. what counts as a correct pick. Getting the winner right is about the sign
     of the MARGIN, not about which predicted score is larger in isolation.

There is a third claim, subtler and more important than either: a graded week
must be graded against the prediction as it was written, never a re-derived
one. The parquet on disk is the record; build_report only reads it. The test
for that is structural -- grade() takes the stored predictions and the final
score and has no path back to a model.

Run: pytest tests/test_report_grading.py -v
"""

import pandas as pd
import pytest

from src.predict.build_report import grade, week_payload


def preds(home, away):
    return {"home": home, "away": away}


class TestGradeOneGame:
    def test_error_is_prediction_minus_reality(self):
        # predicted 30-20, actual 24-17: both sides overshot
        g = grade({"m": preds(30.0, 20.0)}, {"home": 24.0, "away": 17.0})["m"]
        assert g["err_home"] == 6.0
        assert g["err_away"] == 3.0
        assert g["abs_total"] == 9.0

    def test_undershoot_is_negative(self):
        g = grade({"m": preds(17.0, 14.0)}, {"home": 31.0, "away": 20.0})["m"]
        assert g["err_home"] == -14.0
        assert g["err_away"] == -6.0
        assert g["abs_total"] == 20.0

    def test_abs_total_never_cancels(self):
        """One side too high and the other too low is still 10 points of error,
        not zero. Signed errors cancelling here would be the single most
        flattering bug available."""
        g = grade({"m": preds(30.0, 10.0)}, {"home": 25.0, "away": 15.0})["m"]
        assert g["err_home"] == 5.0 and g["err_away"] == -5.0
        assert g["abs_total"] == 10.0

    def test_margin_error_is_signed(self):
        # predicted home by 10, home actually won by 3 -> 7 too confident
        g = grade({"m": preds(27.0, 17.0)}, {"home": 20.0, "away": 17.0})["m"]
        assert g["margin_err"] == 7.0

    @pytest.mark.parametrize(
        "ph,pa,ah,aa,ok",
        [
            (27.0, 17.0, 30, 10, True),  # home picked, home won
            (27.0, 17.0, 10, 30, False),  # home picked, away won
            (17.0, 27.0, 10, 30, True),  # away picked, away won
            (17.0, 27.0, 30, 10, False),  # away picked, home won
            (24.5, 24.4, 21, 20, True),  # a tenth of a point still picks a side
            (20.0, 20.0, 21, 20, False),  # dead-heat prediction picks nobody
        ],
    )
    def test_winner_is_the_sign_of_the_margin(self, ph, pa, ah, aa, ok):
        g = grade({"m": preds(ph, pa)}, {"home": ah, "away": aa})["m"]
        assert g["winner_correct"] is ok

    def test_a_blown_out_but_correct_pick_still_counts(self):
        """Scoring the winner and scoring the points are separate questions.
        A model can be 20 points off and still have called the game."""
        g = grade({"m": preds(21.0, 17.0)}, {"home": 45.0, "away": 3.0})["m"]
        assert g["winner_correct"] is True
        assert g["abs_total"] == 38.0

    def test_every_model_is_graded_independently(self):
        out = grade(
            {"a": preds(30.0, 20.0), "b": preds(20.0, 30.0)},
            {"home": 28.0, "away": 21.0},
        )
        assert out["a"]["winner_correct"] is True
        assert out["b"]["winner_correct"] is False
        assert set(out) == {"a", "b"}


def fake_week(tmp_path, generated_at, scores=True):
    rows = []
    for i, (h, a) in enumerate([("KC", "DEN"), ("SF", "SEA")]):
        rows.append(
            {
                "game_id": f"2025_01_{a}_{h}",
                "season": 2025,
                "week": 1,
                "gameday": pd.Timestamp("2025-09-07"),
                "home_team": h,
                "away_team": a,
                "combined_home": 27.0,
                "combined_away": 20.0,
                "linear_home": 26.0,
                "linear_away": 21.0,
                "spread_line": 3.0,
                "total_line": 45.0,
                "market_home": 24.0,
                "market_away": 21.0,
                "generated_at": generated_at,
                "trained_through": "2025-02-09",
                "n_training_games": 1800,
            }
        )
    path = tmp_path / "2025_wk01.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    actuals = pd.DataFrame(
        [
            {"game_id": "2025_01_DEN_KC", "home_score": 24.0, "away_score": 17.0},
            {"game_id": "2025_01_SEA_SF", "home_score": 13.0, "away_score": 30.0},
        ]
    )
    return path, (actuals if scores else actuals.iloc[:0])


class TestWeekPayload:
    def test_ungraded_week_has_no_grades(self, tmp_path):
        path, _ = fake_week(tmp_path, "2025-09-05T12:00:00+00:00")
        w = week_payload(
            path, pd.DataFrame(columns=["game_id", "home_score", "away_score"])
        )
        assert w["n_graded"] == 0
        assert w["summary"] == {}
        assert all("grade" not in g for g in w["games"])

    def test_graded_week_summarizes_every_model(self, tmp_path):
        path, actuals = fake_week(tmp_path, "2025-09-05T12:00:00+00:00")
        w = week_payload(path, actuals)
        assert w["n_graded"] == 2
        # combined picked home in both; right once, wrong once
        assert w["summary"]["combined"]["winners"] == 1
        assert w["summary"]["combined"]["n"] == 2
        # per-score MAE: KC (3,3) + SF (14,10) = 30 points over 4 scores
        assert w["summary"]["combined"]["mae"] == 7.5
        assert "gp" not in w["summary"]  # absent from the file, absent from the page

    def test_prediction_columns_survive_grading_unchanged(self, tmp_path):
        """Grading must never rewrite the forecast. The stored number is the
        record of what was claimed before kickoff."""
        path, actuals = fake_week(tmp_path, "2025-09-05T12:00:00+00:00")
        before = pd.read_parquet(path)["combined_home"].tolist()
        week_payload(path, actuals)
        after = pd.read_parquet(path)["combined_home"].tolist()
        assert before == after

    def test_prediction_before_kickoff_is_not_flagged(self, tmp_path):
        path, actuals = fake_week(tmp_path, "2025-09-05T12:00:00+00:00")
        assert week_payload(path, actuals)["backfilled"] is False

    def test_prediction_after_kickoff_is_flagged(self, tmp_path):
        """A week predicted after the fact is still out of sample, but nothing
        stopped it from being regenerated until it looked good. The page has to
        say so."""
        path, actuals = fake_week(tmp_path, "2026-09-15T12:00:00+00:00")
        assert week_payload(path, actuals)["backfilled"] is True
