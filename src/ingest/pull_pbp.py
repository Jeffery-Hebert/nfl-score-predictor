"""
Pulls play-by-play data (2019-current) from nflverse via nflreadpy.
Source of EPA and drive-outcome features used across nearly every model
in the ensemble. This is a larger pull than schedules -- expect it to
take longer and produce a bigger file.

Run: python src/ingest/pull_pbp.py
Output: data/raw/pbp.parquet
"""

import nflreadpy as nfl
from pathlib import Path

SEASON_START = 2019
SEASON_END = 2026


def main():
    seasons = list(range(SEASON_START, SEASON_END + 1))
    df = nfl.load_pbp(seasons=seasons).to_pandas()

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pbp.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Pulled {len(df)} plays across seasons {seasons[0]}-{seasons[-1]}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
