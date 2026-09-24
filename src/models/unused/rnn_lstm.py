"""
RNN (LSTM): shared LSTM encoder processes each team's sequence of prior
games (from build_sequences.py), concatenates home+away representations,
predicts both scores jointly. Uses season-level walk-forward -- same
reasoning as Bayesian/GP models: refitting a neural net 145 times (weekly)
is unnecessary compute cost for a model already expected to struggle with
this little data (same lesson as the MLP result).

----------------------------------------------------- fixed 2026-09-24

The first version was degenerate: its predictions had a standard deviation of
0.5 points (home) against ~3 for every other model -- it had learned the mean
score and nothing else. It regressed RAW scores (~22) with 100 full-batch Adam
steps at lr 0.01 from a random initialisation, so nearly all of its training
budget went into moving the output from 0 to 22, and the loss surface gave the
sequence encoder almost no gradient to learn from.

Changes:
  - the target is standardised per fold (per side), so training starts at the
    mean and every step is spent on the signal;
  - up to 400 epochs with early stopping on the most recent 15% of the
    training fold (chronological, never random), restoring the best weights;
  - historical overtime games weighted 0.5 in the loss (C5), and the
    recent-residual drift offset every other model carries.
  - the sequence lookup is loaded on first use, not at import, so importing
    this module (the registry, CI's import check) needs neither data nor a fit.

Run: python -m src.models.unused.rnn_lstm      (needs torch; see requirements.txt)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.common import chronological, ot_sample_weight, recent_residual_offset
from src.validate.backtest_io import save_predictions
from src.validate.walk_forward import score_predictions, walk_forward_evaluate_by_season

MODEL_TABLE = "data/processed/model_table.parquet"
SEQ_PATH = "data/processed/team_sequences.npz"
N_SEQ_FEATURES = 6  # matches SEQ_FEATURES in build_sequences.py
SEQ_LEN = 8
MAX_EPOCHS = 400
PATIENCE = 30
VALIDATION_FRACTION = 0.15
SEED = 42

_SEQ_LOOKUP = None


def sequence_lookup() -> dict:
    """(game_id, team) -> padded prior-game sequence, loaded once on demand."""
    global _SEQ_LOOKUP
    if _SEQ_LOOKUP is None:
        data = np.load(SEQ_PATH, allow_pickle=True)
        _SEQ_LOOKUP = {
            (gid, team): data["padded"][i]
            for i, (gid, team) in enumerate(zip(data["game_ids"], data["teams"]))
        }
    return _SEQ_LOOKUP


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
    lookup = sequence_lookup()
    zero_seq = np.zeros((SEQ_LEN, N_SEQ_FEATURES))
    home = [lookup.get(k, zero_seq) for k in zip(df["game_id"], df["home_team"])]
    away = [lookup.get(k, zero_seq) for k in zip(df["game_id"], df["away_team"])]
    return np.stack(home), np.stack(away)


def _to_tensors(scaler, home_seq, away_seq):
    def scale(seq):
        flat = scaler.transform(seq.reshape(-1, N_SEQ_FEATURES))
        return torch.tensor(flat.reshape(seq.shape), dtype=torch.float32)

    return scale(home_seq), scale(away_seq)


def _weighted_mse(pred, y, w):
    return (w[:, None] * (pred - y) ** 2).sum() / (w.sum() * y.shape[1])


def fit_rnn(train: pd.DataFrame) -> dict:
    torch.manual_seed(SEED)
    train = chronological(train)
    home_seq, away_seq = _gather_sequences(train)

    flat = np.concatenate(
        [home_seq.reshape(-1, N_SEQ_FEATURES), away_seq.reshape(-1, N_SEQ_FEATURES)]
    )
    scaler = StandardScaler().fit(flat)
    X_home, X_away = _to_tensors(scaler, home_seq, away_seq)

    y_raw = train[["home_score", "away_score"]].to_numpy(float)
    y_mean, y_sd = y_raw.mean(axis=0), y_raw.std(axis=0)
    y = torch.tensor((y_raw - y_mean) / y_sd, dtype=torch.float32)
    w_np = ot_sample_weight(train)
    w = torch.tensor(
        np.ones(len(train)) if w_np is None else np.asarray(w_np, float),
        dtype=torch.float32,
    )

    # Early stopping on the most RECENT slice of the fold: chronological, so
    # the stopping rule is itself a forecast of later games.
    cut = int(len(train) * (1 - VALIDATION_FRACTION))
    if len(train) - cut < 20:  # too small to stop on: train on everything
        cut = len(train)
    tr, va = slice(0, cut), slice(cut if cut < len(train) else 0, len(train))

    model = ScorePredictor(input_dim=N_SEQ_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    best_loss, best_state, since_best = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = _weighted_mse(model(X_home[tr], X_away[tr]), y[tr], w[tr])
        assert torch.isfinite(
            loss
        ), f"Non-finite loss at epoch {epoch} -- stopping, do not trust this model"
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val = float(_weighted_mse(model(X_home[va], X_away[va]), y[va], w[va]))
        if val < best_loss - 1e-6:
            best_loss, since_best = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break
    if best_state is not None:  # None only if validation never produced a number
        model.load_state_dict(best_state)

    fitted = {"model": model, "scaler": scaler, "y_mean": y_mean, "y_sd": y_sd}
    fitted["off_h"], fitted["off_a"] = 0.0, 0.0
    h, a = predict_rnn(fitted, train)
    fitted["off_h"], fitted["off_a"] = recent_residual_offset(train, h, a)
    fitted["epochs"] = epoch + 1
    return fitted


def predict_rnn(model: dict, test: pd.DataFrame):
    home_seq, away_seq = _gather_sequences(test)
    X_home, X_away = _to_tensors(model["scaler"], home_seq, away_seq)
    model["model"].eval()
    with torch.no_grad():
        pred = model["model"](X_home, X_away).numpy()
    pred = pred * model["y_sd"] + model["y_mean"]
    return pred[:, 0] - model["off_h"], pred[:, 1] - model["off_a"]


def main():
    df = pd.read_parquet(MODEL_TABLE)
    results = walk_forward_evaluate_by_season(
        df, fit_rnn, predict_rnn, min_train_seasons=2
    )
    metrics = score_predictions(results)
    print("RNN (LSTM) walk-forward results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    save_predictions(results, "rnn", inputs=[MODEL_TABLE, SEQ_PATH])


if __name__ == "__main__":
    main()
