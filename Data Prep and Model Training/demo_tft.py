import os
import numpy as np
import pandas as pd
import xarray as xr
import torch

# from pytorch_lightning import Trainer
from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE

# ---------------- config ----------------
SPLIT_DIR = "./../../Data Given for Challenge/data/"
OUT_DIR   = "./../../results/"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ_LEN    = 72
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DHIDDEN    = 64

# ---------------- utils ----------------
def open_ds(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path)

def ds_to_dataframe(ds):
    """Convert NetCDF to flat dataframe for pytorch-forecasting."""
    df_out = []
    ts = pd.to_datetime(ds.timestamp.values)
    y = ds.out.transpose("timestamp", "location").values.astype(float)
    w = ds.weather.transpose("timestamp", "location", "feature").values.astype(float)
    features = list(map(str, ds.feature.values))
    T, L, F = w.shape

    for li in range(L):
        df_loc = pd.DataFrame({
            "time_idx": np.arange(T),
            "timestamp": ts,
            "out": y[:, li],
            "county": str(ds.location.values[li])
        })
        weather_df = pd.DataFrame(w[:, li, :], columns=[f"w_{fname}" for fname in features])
        df_loc = pd.concat([df_loc, weather_df], axis=1)
        df_out.append(df_loc)

    return pd.concat(df_out, ignore_index=True)

# ---------------- training ----------------
def train_tft(df, horizon, max_encoder_length):
    real_cols = [c for c in df.columns if c not in ["time_idx", "timestamp", "county"]]

    ts_dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="out",
        group_ids=["county"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=horizon,
        time_varying_known_reals=[],
        time_varying_unknown_reals=real_cols,
        target_normalizer=None,
    )

    train_ds = TimeSeriesDataSet.from_dataset(ts_dataset, df, predict=False, stop_randomization=True)
    val_ds   = TimeSeriesDataSet.from_dataset(ts_dataset, df, predict=True, stop_randomization=True)

    train_loader = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader   = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        ts_dataset,
        hidden_size=DHIDDEN,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=DHIDDEN,
        loss=RMSE(),
        learning_rate=LR,
    )

    trainer = Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=10,
    )

    # trainer.fit(tft)
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return tft, ts_dataset

# ---------------- prediction ----------------
def predict_tft(model, ts_dataset, horizon, ts_future):
    # get predictions aligned with dataset rows
    preds, x = model.predict(ts_dataset, mode="prediction", return_x=True)
    preds = preds.cpu().numpy()

    # build dataframe
    df_preds = pd.DataFrame({
        "time_idx": x["time_idx"],
        "location": x["county"],
        "pred": np.clip(preds, 0.0, None)
    })

    # map time_idx to actual timestamps
    ts_map = {i: ts for i, ts in enumerate(ts_future)}
    df_preds["timestamp"] = df_preds["time_idx"].map(ts_map)

    # reorder columns like seq2seq output
    return df_preds[["timestamp", "location", "pred"]].reset_index(drop=True)

# ---------------- main ----------------
def main():
    ds_train = open_ds(os.path.join(SPLIT_DIR, "train.nc"))
    ds_test24 = open_ds(os.path.join(SPLIT_DIR, "test_24h_demo.nc"))
    ds_test48 = open_ds(os.path.join(SPLIT_DIR, "test_48h_demo.nc"))

    df_train = ds_to_dataframe(ds_train)

    ts24 = pd.to_datetime(ds_test24.timestamp.values)
    ts48 = pd.to_datetime(ds_test48.timestamp.values)
    H24, H48 = len(ts24), len(ts48)

    model24, ds24 = train_tft(df_train, horizon=H24, max_encoder_length=SEQ_LEN)
    model48, ds48 = train_tft(df_train, horizon=H48, max_encoder_length=SEQ_LEN)

    df24 = predict_tft(model24, ds24, H24, ts24)
    df48 = predict_tft(model48, ds48, H48, ts48)

    out24 = os.path.join(OUT_DIR, "tft_pred_24h.csv")
    out48 = os.path.join(OUT_DIR, "tft_pred_48h.csv")
    df24.to_csv(out24, index=False)
    df48.to_csv(out48, index=False)

    print("Saved predictions:")
    print(f"  {out24}")
    print(f"  {out48}")

if __name__ == "__main__":
    main()