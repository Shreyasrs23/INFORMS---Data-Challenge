# demo_seq2seq.py
# Simple shared Seq2Seq per-county forecaster (outage + ALL weather features).
# Trains one LSTM encoder over sliding windows from all counties, then predicts
# each county's next H hours using only the last SEQ_LEN hours of its own history.
# Outputs per-county CSVs for 24h and 48h horizons.

import os
from tqdm import tqdm
import numpy as np
import time
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- config ----------------
SPLIT_DIR = "./../../Data Given for Challenge/data/"
OUT_DIR   = "./../../results/"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN    = 8
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DHIDDEN    = 64
NLAYERS    = 1

# ---------------- utils ----------------
def open_ds(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path)

def get_all_features(ds):
    if "feature" not in ds.dims:
        raise RuntimeError("Dataset missing 'feature' dim for weather variables.")
    return list(map(str, ds.feature.values))

def zfit(arr):
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd

def zapply(arr, mu, sd):
    return (arr - mu) / sd

def build_windows(X_loc, y_loc, seq_len, horizon):
    """
    Build (X,y) windows for one location.
    X_loc: (T, D), y_loc: (T,)
    Returns Xw: (N, seq_len, D), Yw: (N, horizon)
    """
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return np.empty((0, seq_len, X_loc.shape[1]), dtype=float), np.empty((0, horizon), dtype=float)
    Xw, Yw = [], []
    for i in range(N):
        Xw.append(X_loc[i:i+seq_len])
        Yw.append(y_loc[i+seq_len:i+seq_len+horizon])
    return np.asarray(Xw, dtype=np.float32), np.asarray(Yw, dtype=np.float32)

# ---------------- model ----------------
class SimpleSeq2Seq(nn.Module):
    def __init__(self, din, dh=64, nl=1, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(din, dh, nl, batch_first=True)
        self.head = nn.Linear(dh, horizon)
    def forward(self, x):
        # x: (B, T, D)
        _, (h, _) = self.lstm(x)
        hlast = h[-1]               # (B, dh)
        return self.head(hlast)     # (B, H)

def train_model(X, Y, din, horizon):
    if len(X) == 0:
        return None
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(Y, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleSeq2Seq(din=din, dh=DHIDDEN, nl=NLAYERS, horizon=horizon).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for xb, yb in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dl.dataset)
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s")

    return model

@torch.no_grad()
def infer(model, Xin):
    model.eval()
    out = model(torch.tensor(Xin, dtype=torch.float32).to(DEVICE))
    return out.cpu().numpy()

# ---------------- pipeline helpers ----------------
def prepare_training(ds_train, horizon):
    """
    Build global scalers and training windows from ALL counties.
    Inputs per time step are [y_scaled, all weather features (scaled)] for that county.
    """
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, L)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, L, F)
    T, L, F = w.shape

    # Global scalers over all (time, county)
    y_mu, y_sd = zfit(y.reshape(-1, 1))
    w_mu, w_sd = zfit(w.reshape(-1, F))

    # Apply scaling
    y_sc = zapply(y.reshape(-1,1), y_mu, y_sd).reshape(T, L)
    w_sc = zapply(w.reshape(-1, F), w_mu, w_sd).reshape(T, L, F)

    Din = 1 + F
    X_list, Y_list = [], []

    for li in range(L):
        y_loc = y_sc[:, li]          # (T,)
        w_loc = w_sc[:, li, :]       # (T, F)
        X_loc = np.concatenate([y_loc.reshape(-1,1), w_loc], axis=1)  # (T, 1+F)
        Xw, Yw = build_windows(X_loc, y_loc, SEQ_LEN, horizon)
        if len(Xw):
            X_list.append(Xw)
            Y_list.append(Yw)

    X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, SEQ_LEN, Din), dtype=float)
    Y = np.concatenate(Y_list, axis=0) if Y_list else np.empty((0, horizon), dtype=float)
    scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
    return X, Y, Din, scalers

def predict_per_county(ds_train, model, scalers, horizon, ts_future):
    """
    For each county, form the last SEQ_LEN inputs from train history only, then
    predict the next 'horizon' values. If insufficient history or model is None,
    output zeros for that county.
    """
    locs = list(map(str, ds_train.location.values))
    y = ds_train.out.transpose("timestamp", "location").values.astype(float)                 # (T, L)
    w = ds_train.weather.transpose("timestamp", "location", "feature").values.astype(float) # (T, L, F)
    T, L, F = w.shape
    Din = 1 + F

    # Scale with train scalers
    y_sc = zapply(y.reshape(-1,1), scalers["y_mu"], scalers["y_sd"]).reshape(T, L)
    w_sc = zapply(w.reshape(-1,F), scalers["w_mu"], scalers["w_sd"]).reshape(T, L, F)

    rows = []
    for li, county in enumerate(locs):
        # Build last SEQ_LEN inputs for this county
        if T < SEQ_LEN or model is None:
            pred_sc = np.zeros(horizon, dtype=float)
        else:
            y_loc = y_sc[:, li]
            w_loc = w_sc[:, li, :]
            X_loc = np.concatenate([y_loc.reshape(-1,1), w_loc], axis=1)  # (T, Din)
            Xin = X_loc[-SEQ_LEN:].reshape(1, SEQ_LEN, Din)
            pred_sc = infer(model, Xin)[0]  # (H,)

        # Invert scaling for y
        pred = (pred_sc * scalers["y_sd"].flatten()[0]) + scalers["y_mu"].flatten()[0]

        # >>> non-negative clamp (and NaN guard)
        pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)

        rows.append(pd.DataFrame({"timestamp": ts_future, "location": county, "pred": pred.astype(float)}))

    return pd.concat(rows, ignore_index=True)

# ---------------- main ----------------
def main():
    # Load train + test
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    ds_test24 = open_ds(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ds_test48 = open_ds(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))

    # Ensure features exist (and implicitly use all of them)
    _ = get_all_features(ds_train)

    # Horizons and timestamps from tests (length only, no weather leakage)
    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    # Train two shared models (one per horizon), using all counties' windows
    X24, Y24, Din, scalers24 = prepare_training(ds_train, horizon=H24)
    mdl24 = train_model(X24, Y24, Din, H24)

    X48, Y48, _, scalers48 = prepare_training(ds_train, horizon=H48)
    mdl48 = train_model(X48, Y48, Din, H48)

    # Predict per county
    df24 = predict_per_county(ds_train, mdl24, scalers24, H24, ts24) if mdl24 is not None else None
    df48 = predict_per_county(ds_train, mdl48, scalers48, H48, ts48) if mdl48 is not None else None

    # Fallback to zeros if no windows (extremely short train)
    if df24 is None:
        locs = list(map(str, ds_train.location.values))
        df24 = pd.DataFrame({"timestamp": np.repeat(ts24, len(locs)),
                             "location": np.tile(locs, len(ts24)),
                             "pred": 0.0})
    if df48 is None:
        locs = list(map(str, ds_train.location.values))
        df48 = pd.DataFrame({"timestamp": np.repeat(ts48, len(locs)),
                             "location": np.tile(locs, len(ts48)),
                             "pred": 0.0})

    # Save 
    out24 = os.path.join(OUT_DIR, "seq2seq_pred_24h.csv")
    out48 = os.path.join(OUT_DIR, "seq2seq_pred_48h.csv")
    df24.to_csv(out24, index=False)
    df48.to_csv(out48, index=False)

    print("Saved predictions:")
    print(f"  {out24}")
    print(f"  {out48}")

    # Clean up
    ds_train.close(); ds_test24.close(); ds_test48.close()

if __name__ == "__main__":
    main()
