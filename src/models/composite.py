"""
The composite ("combined") forecast: an equal-weight average of the live member
models named in config.yaml (live.composite.members).

Why it needs its own backtest. The ledger's headline model is this average, but
until now it had never been scored: the only ensemble in the walk-forward was
the ridge stack in stacking.py, which is not what predict_week publishes. This
module scores exactly the published rule.

Why equal weights rather than fitted ones. The members' errors correlate at
~0.99 (they read the same features), so any fitted weighting mostly fits noise
-- the A5 finding in the project log is that the ridge stack does not beat its
best member. An average of near-identical forecasts cannot be much better than
its best member, but it cannot be much worse either, and it has no parameter
that can overfit. src/validate/model_report.py reports how in-fold weighting
alternatives compare, so this choice stays checked rather than assumed.

Leakage: every member prediction was made by a model that saw only games before
that week, so their average is out of sample too. No fitting happens here.

Run: python -m src.models.composite       (after the members' backtests)
"""

import pandas as pd

from src.config import live_settings
from src.validate.backtest_io import predictions_path, save_predictions
from src.validate.walk_forward import score_predictions

KEYS = ["game_id", "season", "week", "home_score", "away_score"]


def combine(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average member predictions over the games EVERY member predicted.

    Inner join, not outer: a game some member skipped would otherwise be
    averaged over fewer models than the ones on either side of it, and the
    composite would silently mean different things on different rows.

    Joined on game_id alone, then the members must AGREE on its season, week and
    score. Joining on the score too would quietly drop a game that one member
    scored against an older table -- the composite's game set would then depend
    on which member happened to be stale. That is an error, not a filter.
    """
    if not frames:
        raise ValueError("no member predictions to combine")
    merged, first = None, None
    for name, df in frames.items():
        part = df[KEYS + ["home_pred", "away_pred"]].rename(
            columns={"home_pred": f"{name}_home", "away_pred": f"{name}_away"}
        )
        if merged is None:
            merged, first = part, name
            continue
        merged = merged.merge(
            part,
            on="game_id",
            how="inner",
            validate="one_to_one",
            suffixes=("", "__other"),
        )
        for key in KEYS[1:]:
            ours, theirs = merged[key], merged[f"{key}__other"]
            differ = (ours != theirs) & ~(ours.isna() & theirs.isna())
            if differ.any():
                raise ValueError(
                    f"members {first!r} and {name!r} disagree on {key} for "
                    f"{int(differ.sum())} game(s), e.g. "
                    f"{merged.loc[differ, 'game_id'].iloc[0]} -- one of them was "
                    "backtested against an older table; re-run it"
                )
        merged = merged.drop(columns=[f"{k}__other" for k in KEYS[1:]])
    names = list(frames)
    out = merged[KEYS].copy()
    out["home_pred"] = merged[[f"{n}_home" for n in names]].mean(axis=1)
    out["away_pred"] = merged[[f"{n}_away" for n in names]].mean(axis=1)
    return out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def main():
    comp = live_settings()["composite"]
    members = comp["members"]
    paths = [predictions_path(m) for m in members]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"ERROR: member backtests missing: {missing}. Run them first, e.g.\n"
            "  python -m src.models.run_all --only " + " ".join(members)
        )
    frames = {m: pd.read_parquet(p) for m, p in zip(members, paths)}
    results = combine(frames)
    print(f"Composite of {members}: {len(results)} games common to every member")
    for k, v in score_predictions(results).items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    save_predictions(results, "composite", inputs=paths)


if __name__ == "__main__":
    main()
