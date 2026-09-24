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
