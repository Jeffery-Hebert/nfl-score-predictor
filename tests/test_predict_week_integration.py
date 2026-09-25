"""
predict_week end to end, on a small synthetic season -- no data/ needed.

The unit tests pin each helper; this pins the WIRING of main(): the config
decides the models, injury readiness decides which games are forecast, the
composite is the mean of its members' unrounded forecasts, every row carries
its provenance, and nothing is written when nothing is ready. A mistake in how
those pieces are joined would pass every unit test and still publish a wrong
forecast.

Run: pytest tests/test_predict_week_integration.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.models.common import FEATURE_COLS
from src.predict import injury_readiness, predict_week

TEAMS = [f"T{i:02d}" for i in range(8)]
LIVE = {
    "models": ["poisson", "linear"],
    "composite": {
        "name": "combined",
        "members": ["poisson", "linear"],
        "method": "mean",
    },
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Twenty played weeks ending last week, and one upcoming week whose games
    kick off three days from now -- dated relative to TODAY, because main()
    decides what is pending from the real clock."""
    rng = np.random.default_rng(0)
    today = pd.Timestamp.now().normalize()
    first = today - pd.Timedelta(days=7 * 20 + 4)
    rows, sched = [], []
    for w in range(21):
        day = first + pd.Timedelta(days=7 * w)
        played = w < 20
        if not played:
            day = today + pd.Timedelta(days=3)
        order = list(TEAMS)
        rng.shuffle(order)
        for g, (h, a) in enumerate(zip(order[::2], order[1::2])):
            gid = f"2026_{w + 1:02d}_{a}_{h}"
            feats = {c: rng.normal() for c in FEATURE_COLS}
            feats.update(
                {
                    "is_neutral_site": 0,
                    "is_playoff": 0,
                    "home_pregame_team_score": 22 + rng.normal(0, 3),
                    "away_pregame_team_score": 21 + rng.normal(0, 3),
                    "home_pregame_opp_score": 21 + rng.normal(0, 3),
                    "away_pregame_opp_score": 22 + rng.normal(0, 3),
                }
            )
            hs = float(rng.poisson(23)) if played else np.nan
            as_ = float(rng.poisson(21)) if played else np.nan
            rows.append(
                {
                    **feats,
                    "game_id": gid,
                    "season": 2026,
                    "week": w + 1,
                    "gameday": day.date().isoformat(),
                    "home_team": h,
                    "away_team": a,
                    "home_score": hs,
                    "away_score": as_,
                    "went_to_ot": 0,
                }
            )
            sched.append(
                {
                    "game_id": gid,
                    "season": 2026,
                    "week": w + 1,
                    "gameday": day.date().isoformat(),
                    "gametime": "13:00",
                    "home_team": h,
                    "away_team": a,
                    "home_score": hs,
                    "away_score": as_,
                    "spread_line": 3.0,
                    "total_line": 44.5,
                }
            )
    upcoming_week = 21
    inj = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": upcoming_week,
                "team": t,
                "gsis_id": f"{t}1",
                "report_status": "Out",
            }
            for t in TEAMS
        ]
    )
    paths = {
        "MODEL_TABLE": tmp_path / "model_table.parquet",
        "SCHEDULES": tmp_path / "schedules.parquet",
        "INJURIES": tmp_path / "injuries.parquet",
    }
    pd.DataFrame(rows).to_parquet(paths["MODEL_TABLE"], index=False)
    pd.DataFrame(sched).to_parquet(paths["SCHEDULES"], index=False)
    inj.to_parquet(paths["INJURIES"], index=False)
    for name, p in paths.items():
        monkeypatch.setattr(predict_week, name, p)
    monkeypatch.setattr(predict_week, "OUT_DIR", tmp_path / "predictions")
    monkeypatch.setattr(predict_week, "live_settings", lambda: LIVE)
    out_file = tmp_path / "predictions" / f"2026_wk{upcoming_week:02d}.parquet"
    return {"today": today, "out": out_file, "monkeypatch": monkeypatch}


def pulled(world, when):
    world["monkeypatch"].setattr(injury_readiness, "injuries_pulled_at", lambda: when)


def test_nothing_is_written_while_every_final_report_is_pending(world, capsys):
    pulled(world, world["today"].tz_localize("UTC") - pd.Timedelta(days=2))
    assert predict_week.main(["--season", "2026", "--week", "21"]) == 0
    assert not world["out"].exists(), "a forecast was written on an incomplete report"
    assert "held back" in capsys.readouterr().out


def test_a_settled_week_is_forecast_with_full_provenance(world):
    pulled(world, pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2, hours=23))
    assert predict_week.main(["--season", "2026", "--week", "21"]) == 0
    out = pd.read_parquet(world["out"])
    assert len(out) == 4
    for col in (
        "baseline_home",
        "poisson_home",
        "linear_home",
        "combined_home",
        "combined_margin",
        "combined_total",
        "market_home",
        "kickoff",
        "generated_at",
        "trained_through",
        "n_training_games",
        "models",
        "composite_members",
        "code_version",
        "injury_report_final",
    ):
        assert col in out.columns, col
    assert out["injury_report_final"].all()
    assert set(out["models"]) == {"baseline,poisson,linear"}
    assert set(out["composite_members"]) == {"poisson,linear"}
    assert (out["n_training_games"] == 80).all(), "trained on every played game"
    # Each member column is rounded on its own and the composite is rounded
    # from the UNROUNDED mean, so the two differ by at most two half-steps.
    approx = (out["poisson_home"] + out["linear_home"]) / 2
    assert (out["combined_home"] - approx).abs().max() <= 0.1 + 1e-9
    assert out["kickoff"].is_monotonic_increasing


def test_the_composite_is_the_mean_of_the_unrounded_members(world):
    # With rounding pushed out of the way the bound above becomes an identity.
    world["monkeypatch"].setattr(predict_week, "DP", 9)
    pulled(world, pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2, hours=23))
    predict_week.main(["--season", "2026", "--week", "21"])
    out = pd.read_parquet(world["out"])
    for side in ("home", "away"):
        mean = (out[f"poisson_{side}"] + out[f"linear_{side}"]) / 2
        assert np.allclose(out[f"combined_{side}"], mean, atol=1e-8), side
    assert np.allclose(
        out["combined_margin"], out["combined_home"] - out["combined_away"], atol=1e-8
    )


def test_the_override_forecasts_early_and_says_so(world):
    pulled(world, world["today"].tz_localize("UTC") - pd.Timedelta(days=2))
    assert (
        predict_week.main(
            ["--season", "2026", "--week", "21", "--allow-unsettled-injuries"]
        )
        == 0
    )
    out = pd.read_parquet(world["out"])
    assert len(out) == 4
    assert not out["injury_report_final"].any(), "early forecasts must be flagged"


def test_a_missing_injury_file_stops_with_the_fix(world):
    predict_week.INJURIES.unlink()
    with pytest.raises(SystemExit) as e:
        predict_week.main(["--season", "2026", "--week", "21"])
    assert "pull_all" in str(e.value)


def test_skipping_a_member_shrinks_the_composite_and_records_it(world):
    pulled(world, pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2, hours=23))
    predict_week.main(["--season", "2026", "--week", "21", "--skip", "linear"])
    out = pd.read_parquet(world["out"])
    assert "linear_home" not in out.columns
    assert set(out["composite_members"]) == {"poisson"}
    assert (out["combined_home"] == out["poisson_home"]).all()


# ------------------------------------------- re-runs, and which week to run --

FINAL = pd.Timedelta(days=2, hours=23)  # pulled after every final report was due


def forecast_cols(df):
    return [c for c in df.columns if c.startswith("combined_")]


def test_a_rerun_that_reproduces_the_forecasts_leaves_the_record_alone(world):
    # The scheduled pipeline runs daily; each identical re-run must not rewrite
    # the record, move its generated_at later, or produce a commit.
    pulled(world, pd.Timestamp.now(tz="UTC") + FINAL)
    predict_week.main(["--season", "2026", "--week", "21"])
    first = world["out"].read_bytes()
    predict_week.main(["--season", "2026", "--week", "21"])
    assert world["out"].read_bytes() == first


def test_a_rerun_whose_forecasts_moved_rewrites_the_record(world):
    pulled(world, pd.Timestamp.now(tz="UTC") + FINAL)
    predict_week.main(["--season", "2026", "--week", "21"])
    before = pd.read_parquet(world["out"]).sort_values("game_id")
    table = pd.read_parquet(predict_week.MODEL_TABLE)
    table.loc[table["week"] == 21, FEATURE_COLS] += 5.0  # new information
    table.to_parquet(predict_week.MODEL_TABLE, index=False)
    predict_week.main(["--season", "2026", "--week", "21"])
    after = pd.read_parquet(world["out"]).sort_values("game_id")
    assert (after[forecast_cols(after)] != before[forecast_cols(before)]).any().any()
    assert (after["generated_at"].values > before["generated_at"].values).all()


def test_by_default_the_week_is_the_next_one_with_a_game_to_kick_off(world):
    table = predict_week.load_table()
    # Week 20 kicked off eleven days ago. Say its results never arrived: the
    # default must still move on to the week that has games left to forecast.
    table.loc[table["week"] == 20, ["home_score", "away_score"]] = np.nan
    assert predict_week.pick_week(table, None, None) == (2026, 21)


def test_with_no_game_left_to_kick_off_there_is_nothing_to_do(world, capsys):
    table = pd.read_parquet(predict_week.MODEL_TABLE)
    table[table["week"] < 21].to_parquet(predict_week.MODEL_TABLE, index=False)
    assert predict_week.main([]) == 0, "the offseason is not an error"
    assert not world["out"].exists()
    assert "Nothing to forecast" in capsys.readouterr().out


def test_half_a_week_address_is_refused(world):
    with pytest.raises(SystemExit, match="both"):
        predict_week.main(["--week", "21"])


class TestUnchanged:
    @pytest.fixture
    def on_disk(self, tmp_path):
        rec = pd.DataFrame(
            {
                "game_id": ["a", "b"],
                "combined_home": [24.6, 20.7],
                "injury_report_final": [True, True],
                "kickoff": pd.to_datetime(["2026-09-25 00:15", "2026-09-27 17:00"], utc=True),
                "generated_at": ["2026-09-24T13:40:00+00:00"] * 2,
                "code_version": ["abc"] * 2,
                "spread_line": [5.5, -1.5],
                "total_line": [43.5, 42.5],
                "market_home": [24.5, 20.5],
                "market_away": [19.0, 22.0],
            }
        )  # fmt: skip
        path = tmp_path / "rec.parquet"
        rec.to_parquet(path, index=False)
        return path, rec

    def test_run_metadata_and_market_moves_are_not_changes(self, on_disk):
        path, rec = on_disk
        again = rec.iloc[::-1].copy()  # and row order is not a change either
        again["generated_at"] = "2026-09-26T13:41:00+00:00"
        again["code_version"] = "def"
        again["spread_line"] += 1.0
        again["market_home"] += 0.5
        again["kickoff"] = again["kickoff"].astype("datetime64[ns, UTC]")
        assert predict_week.unchanged(path, again)

    @pytest.mark.parametrize(
        "column, value",
        [("combined_home", 24.7), ("injury_report_final", False)],
    )
    def test_a_different_forecast_or_status_is(self, on_disk, column, value):
        path, rec = on_disk
        again = rec.copy()
        again.loc[0, column] = value
        assert not predict_week.unchanged(path, again)

    def test_a_new_column_or_no_file_is(self, on_disk, tmp_path):
        path, rec = on_disk
        assert not predict_week.unchanged(path, rec.assign(extra=1))
        assert not predict_week.unchanged(tmp_path / "missing.parquet", rec)
