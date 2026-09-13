"""
Builds fixed-length sequences of each team's prior games (raw per-game
stats, chronologically ordered) for the RNN model. Zero-padded if a team
has fewer than SEQ_LEN prior games (early-season/early-history cold start).

Leakage rule: game N's sequence contains only games 1..N-1 for that team.

Run: python src/features/build_sequences.py
Output: data/processed/team_sequences.npz
"""
import pandas as pd
import numpy as np
from pathlib import Path

SEQ_LEN = 8
SEQ_FEATURES = ["team_score", "opp_score", "off_epa_per_play",
                "def_epa_per_play", "off_success_rate", "def_success_rate_allowed"]

def build_team_sequences(df: pd.DataFrame) -> dict:
    df = df.sort_values("gameday").reset_index(drop=True)
    sequences = {}
    for team, group in df.groupby("team"):
        group = group.sort_values("gameday").reset_index(drop=True)
        feats = group[SEQ_FEATURES].fillna(0).values
        for i, row in group.iterrows():
            game_id = row["game_id"]
            history = feats[max(0, i - SEQ_LEN):i]  # strictly prior games only
            n_real = len(history)
            padded = np.zeros((SEQ_LEN, len(SEQ_FEATURES)))
            if n_real > 0:
                padded[-n_real:] = history
            mask = np.zeros(SEQ_LEN)
            if n_real > 0:
                mask[-n_real:] = 1
            sequences[(game_id, team)] = (padded, mask)
    return sequences

def main():
    df = pd.read_parquet("data/processed/team_game_stats.parquet")
    df["gameday"] = pd.to_datetime(df["gameday"])
    sequences = build_team_sequences(df)

    keys = list(sequences.keys())
    game_ids = np.array([k[0] for k in keys])
    teams = np.array([k[1] for k in keys])
    padded = np.stack([sequences[k][0] for k in keys])
    masks = np.stack([sequences[k][1] for k in keys])

    out_path = Path("data/processed/team_sequences.npz")
    np.savez(out_path, game_ids=game_ids, teams=teams, padded=padded, masks=masks)
    print(f"Built sequences for {len(keys)} team-game entries")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()