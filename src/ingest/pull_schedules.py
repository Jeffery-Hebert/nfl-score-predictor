"""
Pulls NFL schedule/results data (2019-current) from nflverse via nflreadpy.
This is the source of truth for the prediction target: final home/away scores.

Run: python src/ingest/pull_schedules.py
Output: data/raw/schedules.parquet
"""

import nflreadpy as nfl
from pathlib import Path

SEASON_START = 2019
SEASON_END = 2026  # current season; extend each year


def main():
    seasons = list(range(SEASON_START, SEASON_END + 1))
    df = nfl.load_schedules(seasons=seasons).to_pandas()

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "schedules.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Pulled {len(df)} games across seasons {seasons[0]}-{seasons[-1]}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
