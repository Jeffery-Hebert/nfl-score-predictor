"""
RNN (LSTM): shared LSTM encoder processes each team's sequence of prior
games (from build_sequences.py), concatenates home+away representations,
predicts both scores jointly. Uses season-level walk-forward -- same
reasoning as Bayesian/GP models: refitting a neural net 145 times (weekly)
is unnecessary compute cost for a model already expected to struggle with
this little data (same lesson as the MLP result).

Run: python -m src.models.rnn_lstm
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from src.validate.walk_forward import walk_forward_evaluate_by_season, score_predictions

SEQ_PATH = "data/processed/team_sequences.npz"
N_SEQ_FEATURES = 6  # matches SEQ_FEATURES in build_sequences.py

def load_sequence_lookup():
    data = np.load(SEQ_PATH, allow_pickle=True)
    lookup = {}
    for i, (gid, team) in enumerate(zip(data["game_ids"], data["teams"])):
        lookup[(gid, team)] = data["padded"][i]
    return lookup

SEQ_LOOKUP = load_sequence_lookup()

class TeamEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]

class ScorePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super().__init__()
        self.encoder = TeamEncoder(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, home_seq, away_seq):
        home_repr = self.encoder(home_seq)
        away_repr = self.encoder(away_seq)
        return self.head(torch.cat([home_repr, away_repr], dim=1))

def _gather_sequences(df: pd.DataFrame):
    zero_seq = np.zeros((8, N_SEQ_FEATURES))
    home_seqs = [SEQ_LOOKUP.get((r["game_id"], r["home_team"]), zero_seq) for _, r in df.iterrows()]
    away_seqs = [SEQ_LOOKUP.get((r["game_id"], r["away_team"]), zero_seq) for _, r in df.iterrows()]
    return np.stack(home_seqs), np.stack(away_seqs)

def fit_rnn(train: pd.DataFrame) -> dict:
    torch.manual_seed(42)
    home_seq, away_seq = _gather_sequences(train)

    flat = np.concatenate([home_seq.reshape(-1, N_SEQ_FEATURES), away_seq.reshape(-1, N_SEQ_FEATURES)])
    scaler = StandardScaler().fit(flat)
    home_scaled = scaler.transform(home_seq.reshape(-1, N_SEQ_FEATURES)).reshape(home_seq.shape)
    away_scaled = scaler.transform(away_seq.reshape(-1, N_SEQ_FEATURES)).reshape(away_seq.shape)

    X_home = torch.tensor(home_scaled, dtype=torch.float32)
    X_away = torch.tensor(away_scaled, dtype=torch.float32)
    y = torch.tensor(train[["home_score", "away_score"]].values, dtype=torch.float32)

    model = ScorePredictor(input_dim=N_SEQ_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X_home, X_away)
        loss = loss_fn(pred, y)
        assert torch.isfinite(loss), f"Non-finite loss at epoch {epoch} -- stopping, do not trust this model"
        loss.backward()
        optimizer.step()

    return {"model": model, "scaler": scaler}

def predict_rnn(model: dict, test: pd.DataFrame):
    home_seq, away_seq = _gather_sequences(test)
    scaler = model["scaler"]
    home_scaled = scaler.transform(home_seq.reshape(-1, N_SEQ_FEATURES)).reshape(home_seq.shape)
    away_scaled = scaler.transform(away_seq.reshape(-1, N_SEQ_FEATURES)).reshape(away_seq.shape)

    X_home = torch.tensor(home_scaled, dtype=torch.float32)
    X_away = torch.tensor(away_scaled, dtype=torch.float32)

    model["model"].eval()
    with torch.no_grad():
        pred = model["model"](X_home, X_away).numpy()
    return pred[:, 0], pred[:, 1]

def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    results = walk_forward_evaluate_by_season(df, fit_rnn, predict_rnn, min_train_seasons=2)
    metrics = score_predictions(results)
    print("RNN (LSTM) walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    results.to_parquet("data/processed/rnn_predictions.parquet", index=False)

if __name__ == "__main__":
    main()