"""
Pulls official NFL injury reports, weekly snap counts, and the player-id
crosswalk needed to join them.

Injury reports are the largest genuine information gap in this project: the
model has never had any visibility into who is actually playing. Vegas prices
this; the model was blind to it.

Leakage position. Injury reports carry a `date_modified` timestamp, and 99.94%
of rows publish before kickoff with a median lead of 49 hours -- the Friday
report before a Sunday game. The remaining 0.06% do not, so the feature builder
filters on the timestamp rather than trusting the week key.

Snap counts are post-game, which is fine: they are used only to measure a
player's importance from games ALREADY PLAYED, never from the game being
predicted. A star out matters; a fourth-stringer out does not, and snap share
is how you tell them apart.

The two sources use different player ids -- injuries key on gsis_id, snap
counts on pfr_player_id -- so load_players supplies the crosswalk.

Run: python -m src.ingest.pull_injuries   (or all three: python -m src.ingest.pull_all)

TIMING: injury statuses are only final shortly before each game (see
src/predict/injury_readiness.py). Pull again right before forecasting.
Output: data/raw/injuries.parquet
        data/raw/snap_counts.parquet
        data/raw/player_ids.parquet
"""

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a file: python src/ingest/<this>.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nflreadpy as nfl  # noqa: E402

from src.ingest.manifest import record_pull  # noqa: E402
from src.ingest.seasons import load_seasons  # noqa: E402
from src.ingest.seasons import seasons as seasons_to_pull  # noqa: E402


def main():
    seasons = seasons_to_pull()
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    injuries = load_seasons(nfl.load_injuries, seasons).to_pandas()
    injuries.to_parquet(out_dir / "injuries.parquet", index=False)
    record_pull("injuries", out_dir / "injuries.parquet", injuries, seasons)
    print(f"Pulled {len(injuries)} injury-report rows, {seasons[0]}-{seasons[-1]}")

    snaps = load_seasons(nfl.load_snap_counts, seasons).to_pandas()
    snaps.to_parquet(out_dir / "snap_counts.parquet", index=False)
    record_pull("snap_counts", out_dir / "snap_counts.parquet", snaps, seasons)
    print(f"Pulled {len(snaps)} snap-count rows")

    players = nfl.load_players().to_pandas()
    keep = [
        c
        for c in ["gsis_id", "pfr_id", "display_name", "position"]
        if c in players.columns
    ]
    players[keep].to_parquet(out_dir / "player_ids.parquet", index=False)
    record_pull("player_ids", out_dir / "player_ids.parquet", players[keep], seasons)
    print(f"Pulled {len(players)} player-id crosswalk rows ({keep})")

    print("Saved to data/raw/{injuries,snap_counts,player_ids}.parquet")


if __name__ == "__main__":
    main()
