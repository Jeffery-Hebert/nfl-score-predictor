"""
--season on the benchmark tools: model_scoreboard and error_analysis score one
season's games and nothing else. (model_report's filter runs on real backtests
in the weekly re-benchmark; it shares the same one-line filter.)

Run: pytest tests/test_season_filters.py -v
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def preds(tmp_path):
    """A saved backtest spanning two seasons: 2025 perfect, 2026 off by 7."""
    rng = np.random.default_rng(0)
    rows = []
    for season, miss in ((2025, 0.0), (2026, 7.0)):
        for i in range(12):
            h, a = float(rng.integers(10, 35)), float(rng.integers(10, 35))
            rows.append(
                {
                    "game_id": f"{season}_{i:02d}",
                    "season": season,
                    "week": 1 + i // 4,
                    "home_score": h,
                    "away_score": a,
                    "home_pred": h + miss + rng.normal(0, 1e-3),
                    "away_pred": a + miss + rng.normal(0, 1e-3),
                }
            )
    d = tmp_path / "processed"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "linear_predictions.parquet", index=False)
    return d


def test_the_scoreboard_scores_only_that_season(preds, monkeypatch, capsys):
    from src.validate import model_scoreboard as sb

    monkeypatch.setattr(
        sb, "discover", lambda: {"linear": preds / "linear_predictions.parquet"}
    )
    monkeypatch.setattr(
        "src.validate.backtest_io.stale_inputs", lambda name: []
    )  # synthetic file: no sidecar to check
    assert sb.main(["--season", "2026"]) == 0
    out = capsys.readouterr().out
    assert "Season 2026 only." in out
    line = next(l for l in out.splitlines() if l.startswith("linear"))
    n, home = line.split()[1:3]
    assert n == "12" and float(home) == pytest.approx(7.0, abs=0.01)


def test_error_analysis_analyzes_only_that_season(preds, capsys):
    from src.validate.error_analysis import analyze

    analyze(str(preds / "linear_predictions.parquet"), "linear", season=2026)
    out = capsys.readouterr().out
    assert "linear -- 2026 season, 12 games" in out
    assert "Home bias: +7.00" in out


def test_a_season_with_no_games_says_so(preds, capsys):
    from src.validate.error_analysis import analyze

    analyze(str(preds / "linear_predictions.parquet"), "linear", season=2019)
    assert "no games" in capsys.readouterr().out
