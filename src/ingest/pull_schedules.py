"""
Pulls NFL schedule/results data (2019-current) from nflverse via nflreadpy.
This is the source of truth for the prediction target: final home/away scores.

Run: python -m src.ingest.pull_schedules   (or all three: python -m src.ingest.pull_all)
Output: data/raw/schedules.parquet
"""

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a file: python src/ingest/<this>.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nflreadpy as nfl  # noqa: E402

from src.ingest.manifest import record_pull  # noqa: E402
from src.ingest.seasons import seasons as seasons_to_pull  # noqa: E402


def main():
    seasons = seasons_to_pull()
    df = nfl.load_schedules(seasons=seasons).to_pandas()

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "schedules.parquet"
    df.to_parquet(out_path, index=False)
    record_pull("schedules", out_path, df, seasons)

    print(f"Pulled {len(df)} games across seasons {seasons[0]}-{seasons[-1]}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
