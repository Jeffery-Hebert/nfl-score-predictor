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

# week_payload resolves kickoff times to order the slate. These fixtures use
# synthetic game_ids that no schedule can resolve, so an empty index is the
# honest input -- it also exercises the fallback path.
NO_KICKOFFS = pd.Series(dtype="datetime64[ns, UTC]")


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
                # The backfill flag is now per game and compares generated_at to
                # THIS game's kickoff, so the fixture has to carry one.
                "kickoff": pd.Timestamp("2025-09-07T17:00:00", tz="UTC"),
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
            path,
            pd.DataFrame(columns=["game_id", "home_score", "away_score"]),
            NO_KICKOFFS,
        )
        assert w["n_graded"] == 0
        assert w["summary"] == {}
        assert all("grade" not in g for g in w["games"])

    def test_graded_week_summarizes_every_model(self, tmp_path):
        path, actuals = fake_week(tmp_path, "2025-09-05T12:00:00+00:00")
        w = week_payload(path, actuals, NO_KICKOFFS)
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
        week_payload(path, actuals, NO_KICKOFFS)
        after = pd.read_parquet(path)["combined_home"].tolist()
        assert before == after

    def test_prediction_before_kickoff_is_not_flagged(self, tmp_path):
        path, actuals = fake_week(tmp_path, "2025-09-05T12:00:00+00:00")
        assert week_payload(path, actuals, NO_KICKOFFS)["backfilled"] is False

    def test_prediction_after_kickoff_is_flagged(self, tmp_path):
        """A week predicted after the fact is still out of sample, but nothing
        stopped it from being regenerated until it looked good. The page has to
        say so."""
        path, actuals = fake_week(tmp_path, "2026-09-15T12:00:00+00:00")
        assert week_payload(path, actuals, NO_KICKOFFS)["backfilled"] is True


class TestSlateOrdering:
    """Games must be listed in the order they kick off.

    Row order in a week's parquet is whatever predict_week produced -- not
    chronological, and not even stable across runs now that a week is written
    three times as its slates come up. Week 2's Thursday night game sat fourth
    on the page. gameday cannot fix it on its own either: a dozen games share a
    Sunday and kick off across seven hours.
    """

    @staticmethod
    def _week(tmp_path, rows, with_kickoff=True):
        recs = []
        for i, (gid, kick) in enumerate(rows):
            # Distinct team codes per row so uniqueness assertions test the
            # code rather than the fixture's naming.
            r = {
                "game_id": gid,
                "season": 2026,
                "week": 2,
                "gameday": pd.Timestamp(kick).tz_localize(None).normalize(),
                "home_team": f"H{i:02d}",
                "away_team": f"A{i:02d}",
                "combined_home": 24.0,
                "combined_away": 20.0,
                "generated_at": "2026-09-17T12:00:00+00:00",
                "trained_through": "2026-09-14",
                "n_training_games": 1976,
            }
            if with_kickoff:
                r["kickoff"] = pd.Timestamp(kick, tz="UTC")
            recs.append(r)
        path = tmp_path / "2026_wk02.parquet"
        pd.DataFrame(recs).to_parquet(path, index=False)
        return path

    # deliberately shuffled, with the Thursday game buried in the middle
    ROWS = [
        ("SUN_LATE", "2026-09-20T20:25:00"),
        ("SUN_EARLY", "2026-09-20T17:00:00"),
        ("THU_NIGHT", "2026-09-18T00:15:00"),
        ("MON_NIGHT", "2026-09-22T00:15:00"),
        ("SUN_NIGHT", "2026-09-21T00:20:00"),
    ]
    EXPECTED = ["THU_NIGHT", "SUN_EARLY", "SUN_LATE", "SUN_NIGHT", "MON_NIGHT"]

    def test_games_are_ordered_by_kickoff(self, tmp_path):
        path = self._week(tmp_path, self.ROWS)
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), NO_KICKOFFS)
        order = [g["kickoff_ts"] for g in w["games"]]
        assert order == sorted(order), f"not chronological: {order}"

    def test_same_day_games_are_ordered_by_time_not_just_date(self, tmp_path):
        """The case gameday alone cannot solve."""
        path = self._week(
            tmp_path,
            [
                ("LATE_AAA", "2026-09-20T20:25:00"),
                ("EARLY_BBB", "2026-09-20T17:00:00"),
            ],
        )
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), NO_KICKOFFS)
        assert w["games"][0]["kickoff_ts"] < w["games"][1]["kickoff_ts"]

    def test_a_week_without_stored_kickoffs_falls_back_to_the_schedule(self, tmp_path):
        """Weeks predicted before the kickoff column existed still have to sort."""
        path = self._week(tmp_path, self.ROWS, with_kickoff=False)
        schedule = pd.Series({gid: pd.Timestamp(k, tz="UTC") for gid, k in self.ROWS})
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), schedule)
        order = [g["kickoff_ts"] for g in w["games"]]
        assert all(o is not None for o in order), "schedule fallback produced no times"
        assert order == sorted(order)

    def test_unresolvable_kickoffs_do_not_crash_and_sort_last(self, tmp_path):
        """Neither source knows these games. The page must still render."""
        path = self._week(tmp_path, self.ROWS, with_kickoff=False)
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), NO_KICKOFFS)
        assert len(w["games"]) == len(self.ROWS), "games were dropped"
        assert all(g["kickoff_ts"] is None for g in w["games"])

    def test_no_game_is_lost_or_duplicated_by_sorting(self, tmp_path):
        path = self._week(tmp_path, self.ROWS)
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), NO_KICKOFFS)
        assert len(w["games"]) == len(self.ROWS)
        assert len({g["away"] for g in w["games"]}) == len(self.ROWS)


class TestPerGameBackfillFlag:
    """Whether a forecast was pre-registered is a per-GAME question.

    It used to be decided for the whole week off the week's first kickoff, so a
    Thursday night game that had already been played condemned the fifteen
    untouched Sunday forecasts alongside it. That matters now that a week is
    written in three passes -- Thursday, Sunday, Monday -- and is routinely part
    pre-registered and part not.
    """

    @staticmethod
    def _week(tmp_path, rows):
        """rows: (game_id, kickoff, generated_at)."""
        recs = [
            {
                "game_id": gid,
                "season": 2026,
                "week": 2,
                "gameday": pd.Timestamp(kick).tz_localize(None).normalize(),
                "kickoff": pd.Timestamp(kick, tz="UTC"),
                "home_team": f"H{i:02d}",
                "away_team": f"A{i:02d}",
                "combined_home": 24.0,
                "combined_away": 20.0,
                "generated_at": gen,
                "trained_through": "2026-09-14",
                "n_training_games": 1976,
            }
            for i, (gid, kick, gen) in enumerate(rows)
        ]
        path = tmp_path / "2026_wk02.parquet"
        pd.DataFrame(recs).to_parquet(path, index=False)
        return path

    def test_a_forecast_written_before_its_own_kickoff_is_not_flagged(self, tmp_path):
        path = self._week(
            tmp_path,
            [("THU", "2026-09-18T00:15:00", "2026-09-17T18:00:00+00:00")],
        )
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), NO_KICKOFFS)
        assert w["games"][0]["backfilled"] is False
        assert w["n_backfilled"] == 0

    def test_a_forecast_written_after_its_own_kickoff_is_flagged(self, tmp_path):
        path = self._week(
            tmp_path,
            [("THU", "2026-09-18T00:15:00", "2026-09-19T12:00:00+00:00")],
        )
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), NO_KICKOFFS)
        assert w["games"][0]["backfilled"] is True
        assert w["n_backfilled"] == 1

    def test_a_played_thursday_game_does_not_condemn_the_sunday_slate(self, tmp_path):
        """THE regression this replaces. Under the old week-level rule all three
        of these counted as backfilled because the Thursday game had kicked off
        before the Sunday forecasts were written."""
        path = self._week(
            tmp_path,
            [
                # written late, after its own kickoff -- genuinely backfilled
                ("THU", "2026-09-18T00:15:00", "2026-09-20T11:00:00+00:00"),
                # written Sunday morning, hours before their own kickoffs
                ("SUN1", "2026-09-20T17:00:00", "2026-09-20T11:00:00+00:00"),
                ("SUN2", "2026-09-20T20:25:00", "2026-09-20T11:00:00+00:00"),
            ],
        )
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), NO_KICKOFFS)
        flags = {g["away"]: g["backfilled"] for g in w["games"]}
        assert sum(flags.values()) == 1, (
            f"expected exactly the Thursday game flagged, got {flags} -- the "
            "week-level rule is back"
        )
        assert w["n_backfilled"] == 1
        assert w["backfilled"] is True, "the week still carries a partial flag"

    def test_the_running_record_counts_backfilled_games_not_weeks(self, tmp_path):
        """n_backfilled_graded is what the page discounts. Only graded games
        can be in a record, so an ungraded late forecast must not inflate it."""
        path = self._week(
            tmp_path,
            [
                ("PLAYED", "2026-09-18T00:15:00", "2026-09-20T11:00:00+00:00"),
                ("UNPLAYED", "2026-09-22T00:15:00", "2026-09-23T11:00:00+00:00"),
            ],
        )
        actuals = pd.DataFrame(
            [{"game_id": "PLAYED", "home_score": 24.0, "away_score": 20.0}]
        )
        w = week_payload(path, actuals, NO_KICKOFFS)
        assert w["n_backfilled"] == 2, "both forecasts postdate their kickoffs"
        assert w["n_backfilled_graded"] == 1, "only one of them has a result"

    def test_a_missing_kickoff_is_not_assumed_backfilled(self, tmp_path):
        """Absence of evidence is not evidence of lateness -- a week with no
        resolvable kickoff must not be smeared as backfilled."""
        recs = [
            {
                "game_id": "X",
                "season": 2026,
                "week": 2,
                "gameday": pd.Timestamp("2026-09-20"),
                "home_team": "AAA",
                "away_team": "BBB",
                "combined_home": 24.0,
                "combined_away": 20.0,
                "generated_at": "2026-09-25T12:00:00+00:00",
                "trained_through": "2026-09-14",
                "n_training_games": 1976,
            }
        ]
        path = tmp_path / "2026_wk02.parquet"
        pd.DataFrame(recs).to_parquet(path, index=False)
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), NO_KICKOFFS)
        assert w["games"][0]["backfilled"] is False
        assert w["n_backfilled"] == 0


class TestModelsComeFromTheRecords:
    """The page used to hard-code combined/linear/poisson/gp, so changing the
    live models in config.yaml would have silently dropped the new ones from the
    ledger. It now shows every <name>_home/<name>_away pair a week carries."""

    def _week(self, tmp_path, extra=None):
        row = {
            "game_id": "2026_03_AAA_BBB",
            "season": 2026,
            "week": 3,
            "gameday": pd.Timestamp("2026-09-27"),
            "kickoff": pd.Timestamp("2026-09-27T17:00:00", tz="UTC"),
            "home_team": "BBB",
            "away_team": "AAA",
            "combined_home": 24.0,
            "combined_away": 20.0,
            "catboost_home": 23.0,
            "catboost_away": 21.0,
            "baseline_home": 22.0,
            "baseline_away": 21.0,
            "market_home": 23.5,
            "market_away": 20.5,
            "generated_at": "2026-09-27T11:00:00+00:00",
            "trained_through": "2026-09-21",
            "n_training_games": 1992,
        }
        row.update(extra or {})
        path = tmp_path / "2026_wk03.parquet"
        pd.DataFrame([row]).to_parquet(path, index=False)
        return path

    def test_any_model_in_the_file_is_shown(self, tmp_path):
        w = week_payload(
            self._week(tmp_path), pd.DataFrame(columns=["game_id"]), NO_KICKOFFS
        )
        assert w["models"] == ["combined", "catboost"]
        assert set(w["games"][0]["models"]) == {"combined", "catboost"}

    def test_market_and_baseline_are_not_models_on_the_page(self, tmp_path):
        w = week_payload(
            self._week(tmp_path), pd.DataFrame(columns=["game_id"]), NO_KICKOFFS
        )
        assert "market" not in w["models"] and "baseline" not in w["models"]

    def test_a_pre_injury_report_forecast_is_flagged(self, tmp_path):
        path = self._week(tmp_path, {"injury_report_final": False})
        w = week_payload(path, pd.DataFrame(columns=["game_id"]), NO_KICKOFFS)
        assert w["games"][0]["provisional"] is True
        assert w["n_provisional"] == 1

    def test_a_week_written_before_the_flag_existed_is_not_smeared(self, tmp_path):
        w = week_payload(
            self._week(tmp_path), pd.DataFrame(columns=["game_id"]), NO_KICKOFFS
        )
        assert w["games"][0]["provisional"] is False


# ------------------------------------------------ the page and the records --

# 2026-09-27: GitHub's pipeline recorded Sunday's forecasts on their FINAL
# injury reports, a `git pull` brought them in, and the ledger page -- then an
# untracked local file nothing had rebuilt -- still tagged them
# "pre-injury-report". Nothing was miscomputed; the page was showing records
# that no longer existed. These pin the guarantees added after it: the page is
# a pure function of its inputs, carries a fingerprint of every record, is
# verified against them game by game before it is written, and the page in the
# repository must match the records in the repository.


def ledger_dir(tmp_path, final=(False, False)):
    """A predictions directory with one week of two games."""
    d = tmp_path / "predictions"
    d.mkdir()
    rows = []
    for (away, home), is_final in zip([("DEN", "KC"), ("SEA", "SF")], final):
        rows.append(
            {
                "game_id": f"2026_03_{away}_{home}",
                "season": 2026,
                "week": 3,
                "gameday": pd.Timestamp("2026-09-27"),
                "kickoff": pd.Timestamp("2026-09-27T17:00:00", tz="UTC"),
                "home_team": home,
                "away_team": away,
                "combined_home": 24.6,
                "combined_away": 21.0,
                "linear_home": 24.9,
                "linear_away": 20.7,
                "injury_report_final": is_final,
                "generated_at": "2026-09-24T22:26:26+00:00",
                "trained_through": "2026-09-21",
                "n_training_games": 1992,
            }
        )
    pd.DataFrame(rows).to_parquet(d / "2026_wk03.parquet", index=False)
    return d


def render(d):
    from src.predict import build_report

    return build_report.render(d, d / "no-schedule.parquet")


def rewrite(d, **changes):
    """Change the record after the page was built, as a later forecast does."""
    path = d / "2026_wk03.parquet"
    df = pd.read_parquet(path)
    for col, value in changes.items():
        df[col] = value
    df.to_parquet(path, index=False)


class TestTheLedgerMatchesTheRecords:
    def test_a_fresh_page_verifies(self, tmp_path):
        from src.predict.build_report import verify

        d = ledger_dir(tmp_path)
        render(d)
        assert verify(pred_dir=d) == []

    def test_the_page_is_a_pure_function_of_its_inputs(self, tmp_path):
        # No build time or other clock in it: the same records give the same
        # bytes, so the tracked page changes only when they do.
        d = ledger_dir(tmp_path)
        render(d)
        first = (d / "index.html").read_bytes()
        render(d)
        assert (d / "index.html").read_bytes() == first

    def test_the_2026_09_27_case_is_caught(self, tmp_path):
        from src.predict.build_report import verify

        d = ledger_dir(tmp_path, final=(False, False))
        render(d)
        rewrite(d, injury_report_final=True, generated_at="2026-09-26T17:23:47+00:00")
        problems = verify(pred_dir=d)
        assert "2026_wk03.parquet has changed since the page was built" in problems
        assert any(
            "DEN @ KC: page says provisional, the record says final" in p
            for p in problems
        )

    def test_a_forecast_the_page_misstates_is_caught(self, tmp_path):
        from src.predict.build_report import verify

        d = ledger_dir(tmp_path)
        render(d)
        page = d / "index.html"
        page.write_text(page.read_text().replace('"home":24.6', '"home":27.0', 1))
        assert any("combined is" in p for p in verify(pred_dir=d))

    def test_a_page_that_would_disagree_is_never_written(self, tmp_path, monkeypatch):
        from src.predict import build_report

        d = ledger_dir(tmp_path)
        render(d)
        good = (d / "index.html").read_bytes()
        real = build_report.week_payload

        def wrong_status(*a, **k):
            w = real(*a, **k)
            w["games"][0]["provisional"] = not w["games"][0]["provisional"]
            return w

        monkeypatch.setattr(build_report, "week_payload", wrong_status)
        with pytest.raises(RuntimeError, match="would not match the records"):
            render(d)
        assert (d / "index.html").read_bytes() == good, "the old page must stay"
        assert not (d / "index.html.tmp").exists()

    def test_check_reports_a_stale_page_and_fails(self, tmp_path, monkeypatch, capsys):
        from src.predict import build_report

        d = ledger_dir(tmp_path)
        render(d)
        rewrite(d, injury_report_final=True)
        monkeypatch.setattr(build_report, "PRED_DIR", d)
        monkeypatch.setattr(build_report, "OUT", d / "index.html")
        monkeypatch.setattr("sys.argv", ["build_report", "--check"])
        assert build_report.main() == 1
        assert "STALE" in capsys.readouterr().out


@pytest.mark.parametrize(
    "value, provisional",
    [
        (True, False),
        (False, True),
        (None, False),  # a week written before the column existed
        (float("nan"), False),
        (pd.NA, False),
        ("False", True),  # bool("False") is True -- read, never truth-tested
        ("True", False),
        (0, True),
        (1, False),
    ],
)
def test_one_reading_of_the_injury_report_flag(value, provisional):
    import numpy as np

    from src.predict.injury_readiness import is_provisional

    assert is_provisional(value) is provisional
    if isinstance(value, bool):
        assert is_provisional(np.bool_(value)) is provisional


@pytest.mark.ledger_sync
def test_the_committed_ledger_matches_the_committed_records():
    """CI runs this on every push: a commit that changes a record without the
    page (or the page without the record) fails here, before anyone opens it.
    Fix: python -m src.predict.build_report, and commit data/predictions/."""
    from src.predict.build_report import verify

    assert verify() == []


# ---------------------------------------------------------- market picks --

# The ledger's Market / Model line / Spread pick / Total pick columns. nflverse's
# spread_line is the HOME team's expected margin (positive = home favoured), so
# the home team's betting line is -spread_line: every case below pins that down.


def picks(home, away, spread, total, home_team="HOME", away_team="AWAY"):
    from src.predict.build_report import market_picks

    return market_picks(home, away, spread, total, home_team, away_team)


class TestMarketPicks:
    def test_thursday_night_of_week_3(self):
        # Combined ATL 21.0 - GB 24.6 against GB -5.5, O/U 43.5.
        assert picks(24.6, 21.0, 5.5, 43.5, "GB", "ATL") == {
            "market": {"favorite": "GB", "by": 5.5, "total": 43.5},
            "model": {"favorite": "GB", "by": 3.6, "total": 45.6},
            "spread": {"team": "ATL", "line": 5.5, "edge": 1.9},
            "total": {"side": "Over", "line": 43.5, "edge": 2.1},
        }

    def test_a_home_favourite_the_model_likes_more(self):
        # KC by 7 against KC -3: KC covers, at KC's line -3.
        p = picks(27.0, 20.0, 3.0, 45.0, "KC", "DEN")
        assert p["spread"] == {"team": "KC", "line": -3.0, "edge": 4.0}
        assert p["total"] == {"side": "Over", "line": 45.0, "edge": 2.0}

    def test_an_away_favourite(self):
        # CAR favoured by 2.5 at CLE; the model has CAR by 2.9: CAR -2.5.
        p = picks(20.7, 23.6, -2.5, 42.5, "CLE", "CAR")
        assert p["market"]["favorite"] == "CAR" and p["model"]["favorite"] == "CAR"
        assert p["spread"] == {"team": "CAR", "line": -2.5, "edge": 0.4}

    def test_a_home_underdog(self):
        # CIN favoured by 3.5 at PIT; the model has CIN by only 1.9: PIT +3.5.
        p = picks(20.1, 22.0, -3.5, 42.5, "PIT", "CIN")
        assert p["spread"] == {"team": "PIT", "line": 3.5, "edge": 1.6}
        assert p["total"] == {"side": "Under", "line": 42.5, "edge": 0.4}

    def test_the_model_favouring_the_other_team(self):
        # Market GB -1.5, model ATL by 2.0: ATL +1.5, 3.5 points apart.
        p = picks(21.0, 23.0, 1.5, 44.0, "GB", "ATL")
        assert p["model"] == {"favorite": "ATL", "by": 2.0, "total": 44.0}
        assert p["spread"] == {"team": "ATL", "line": 1.5, "edge": 3.5}
        assert p["total"] is None  # 44.0 on a 44.0 line

    def test_a_pickem_line(self):
        p = picks(24.0, 21.0, 0.0, 45.0, "KC", "DEN")
        assert p["market"]["favorite"] is None
        assert p["spread"] == {"team": "KC", "line": 0.0, "edge": 3.0}

    def test_no_pick_when_the_model_sits_on_the_line(self):
        import math

        p = picks(24.0, 21.0, 3.0, 45.0)
        assert p["spread"] is None and p["total"] is None
        assert math.copysign(1, p["model"]["by"]) == 1  # never "-0.0"

    @pytest.mark.parametrize("spread, total", [(float("nan"), 43.5), (5.5, None)])
    def test_no_line_means_no_picks(self, spread, total):
        assert picks(24.6, 21.0, spread, total) is None

    def test_float_noise_never_reaches_the_page(self):
        # 3.6 - 5.5 is -1.9000000000000004 in floating point.
        assert repr(picks(24.6, 21.0, 5.5, 43.5)["spread"]["edge"]) == "1.9"

    def test_it_agrees_with_exact_decimal_arithmetic(self):
        from decimal import Decimal
        import random

        rng = random.Random(0)
        for _ in range(500):
            home = Decimal(rng.randint(50, 400)) / 10  # forecasts: tenths
            away = Decimal(rng.randint(50, 400)) / 10
            spread = Decimal(rng.randint(-40, 40)) / 2  # lines: half points
            total = Decimal(rng.randint(60, 120)) / 2
            p = picks(float(home), float(away), float(spread), float(total), "H", "A")
            gap, over_by = (home - away) - spread, (home + away) - total
            if gap == 0:
                assert p["spread"] is None
            else:
                team = "H" if gap > 0 else "A"
                # the pick's line, seen from its own team, is what it must cover
                line = -spread if team == "H" else spread
                assert p["spread"] == {
                    "team": team,
                    "line": float(line),
                    "edge": float(abs(gap)),
                }
                # and the model's margin for that team does cover it
                model_margin = (home - away) if team == "H" else (away - home)
                assert model_margin + line > 0
            if over_by == 0:
                assert p["total"] is None
            else:
                assert p["total"]["side"] == ("Over" if over_by > 0 else "Under")
                assert Decimal(str(p["total"]["edge"])) == abs(over_by)


class TestPickResults:
    @staticmethod
    def result(p, home, away, home_team="HOME"):
        from src.predict.build_report import grade_picks

        return grade_picks(p, home_team, home, away)

    def test_thursday_night_of_week_3_won_both(self):
        # ATL +5.5 and Over 43.5; final ATL 35 - GB 14 (49 points).
        p = picks(24.6, 21.0, 5.5, 43.5, "GB", "ATL")
        assert self.result(p, 14.0, 35.0, "GB") == {"spread": "win", "total": "win"}

    @pytest.mark.parametrize(
        "home, away, spread_result, total_result",
        [
            (24.0, 20.0, "loss", "loss"),  # GB by 4: covers 5.5? no -> ATL wins
            (30.0, 20.0, "loss", "win"),  # GB by 10: ATL +5.5 loses; 50 points
            (20.0, 21.0, "win", "loss"),  # ATL wins outright; 41 points
        ],
    )
    def test_each_outcome(self, home, away, spread_result, total_result):
        p = picks(24.6, 21.0, 5.5, 43.5, "HOME", "AWAY")  # AWAY +5.5, Over
        got = self.result(p, home, away)
        want_spread = "win" if (away - home) + 5.5 > 0 else "loss"
        assert got["spread"] == want_spread
        assert got["total"] == ("win" if home + away > 43.5 else "loss")

    def test_landing_on_the_line_is_a_push(self):
        p = picks(27.0, 20.0, 3.0, 45.0)  # HOME -3, Over 45
        assert self.result(p, 24.0, 21.0) == {"spread": "push", "total": "push"}

    def test_no_pick_has_no_result(self):
        p = picks(24.0, 21.0, 3.0, 45.0)
        assert self.result(p, 30.0, 10.0) == {"spread": None, "total": None}


class TestPicksOnThePage:
    def test_every_game_carries_its_picks_and_graded_games_their_results(
        self, tmp_path
    ):
        from src.predict.build_report import grade_picks, market_picks

        path, actuals = fake_week(tmp_path, "2025-09-06T12:00:00+00:00")
        w = week_payload(path, actuals, NO_KICKOFFS)
        for g in w["games"]:
            for m, p in g["models"].items():
                want = market_picks(
                    p["home"], p["away"], 3.0, 45.0, g["home"], g["away"]
                )
                assert g["picks"][m] == want
                assert g["pick_results"][m] == grade_picks(
                    want, g["home"], g["actual"]["home"], g["actual"]["away"]
                )
        # combined 20-27 against home -3 / 45: home -3 and Over 45.
        # DEN @ KC ended 17-24 (KC by 7, 41 points): won, lost.
        # SEA @ SF ended 30-13 (SEA by 17, 43 points): lost, lost.
        assert [g["pick_results"]["combined"] for g in w["games"]] == [
            {"spread": "win", "total": "loss"},
            {"spread": "loss", "total": "loss"},
        ]
        assert w["summary"]["combined"]["picks"] == {
            "spread": {"win": 1, "loss": 1, "push": 0},
            "total": {"win": 0, "loss": 2, "push": 0},
        }

    def test_a_game_without_a_line_has_no_picks(self, tmp_path):
        path, actuals = fake_week(tmp_path, "2025-09-06T12:00:00+00:00")
        df = pd.read_parquet(path)
        df["spread_line"] = float("nan")
        df.to_parquet(path, index=False)
        w = week_payload(path, actuals, NO_KICKOFFS)
        assert all(g["picks"] is None for g in w["games"])

    def test_the_verifier_recomputes_the_picks_from_the_record(self, tmp_path):
        from src.predict.build_report import verify

        d = ledger_dir(tmp_path)
        df = pd.read_parquet(d / "2026_wk03.parquet")
        df["spread_line"], df["total_line"] = 5.5, 43.5
        df["market_home"], df["market_away"] = 24.5, 19.0  # as predict_week stores
        df.to_parquet(d / "2026_wk03.parquet", index=False)
        render(d)
        assert verify(pred_dir=d) == []
        page = d / "index.html"
        page.write_text(page.read_text().replace('"team":"DEN"', '"team":"KC"', 1))
        assert any("picks on the page" in p for p in verify(pred_dir=d))


# --------------------------------------------------------- implied score --


class TestImpliedScore:
    """The market's line read as a score, beside the forecasts: an over/under
    of 43.5 with GB favoured by 5.5 implies ATL 19.0 - GB 24.5."""

    def test_it_is_the_line_as_a_score(self, tmp_path):
        path, actuals = fake_week(tmp_path, "2025-09-06T12:00:00+00:00")
        w = week_payload(path, actuals, NO_KICKOFFS)
        for g in w["games"]:
            m = g["market"]
            assert m == {"spread": 3.0, "total": 45.0, "away": 21.0, "home": 24.0}
            assert m["home"] == (m["total"] + m["spread"]) / 2
            assert m["away"] == (m["total"] - m["spread"]) / 2

    def test_every_stored_implied_score_matches_its_line(self):
        """Every committed record -- 48 games through week 3 -- stores the
        implied score its own lines give, to the rounding (quarter points
        round to a tenth)."""
        from src.predict.build_report import records

        files = records()
        assert files, "no committed records"
        for f in files:
            df = pd.read_parquet(f).dropna(subset=["spread_line", "total_line"])
            home = (df["total_line"] + df["spread_line"]) / 2
            away = (df["total_line"] - df["spread_line"]) / 2
            assert ((df["market_home"] - home).abs() <= 0.05 + 1e-9).all(), f.name
            assert ((df["market_away"] - away).abs() <= 0.05 + 1e-9).all(), f.name

    def test_the_verifier_catches_a_wrong_implied_score(self, tmp_path):
        from src.predict.build_report import verify

        d = ledger_dir(tmp_path)
        df = pd.read_parquet(d / "2026_wk03.parquet")
        df["spread_line"], df["total_line"] = 5.5, 43.5
        df["market_home"], df["market_away"] = 24.5, 19.0
        df.to_parquet(d / "2026_wk03.parquet", index=False)
        render(d)
        assert verify(pred_dir=d) == []
        page = d / "index.html"
        page.write_text(page.read_text().replace('"away":19.0', '"away":21.0', 1))
        assert any("market is" in p for p in verify(pred_dir=d))


class TestMarketCellLayout:
    """2026-10-01: each market cell is a value with a qualifier under it ("PIT
    −1.1" over "42.5 pts", "CLE +2.5" over "edge 1.4"). The qualifier span's
    block rule was scoped to the matchup cell, so in the five market cells the
    two ran together inline and "PIT −1.1" + "42.5 pts" read "−1.142.5"."""

    @staticmethod
    def css():
        import re

        from src.predict.build_report import TEMPLATE

        html = TEMPLATE.read_text()
        return re.sub(r"/\*.*?\*/", "", html.split("</style>")[0], flags=re.S)

    def test_the_qualifier_sits_on_its_own_line(self):
        import re

        rule = re.search(r"td\.mkt \.dt\{([^}]*)\}", self.css())
        assert rule, "no rule for the qualifier line in market cells"
        assert "display:block" in rule.group(1)

    def test_a_market_value_never_wraps_inside_itself(self):
        import re

        rule = re.search(r"td\.mkt\{([^}]*)\}", self.css())
        assert rule and "white-space:nowrap" in rule.group(1)

    def test_the_qualifier_is_a_separate_element_from_the_value(self):
        """The layout depends on drawBody putting the qualifier in its own
        .dt span after the value's text node."""
        from src.predict.build_report import TEMPLATE

        js = TEMPLATE.read_text()
        assert "td.append(document.createTextNode(text));" in js
        assert 's.className = "dt"; s.append(...sub);' in js
