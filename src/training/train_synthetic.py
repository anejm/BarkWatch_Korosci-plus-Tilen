"""
train_synthetic.py
------------------
Single-stage LightGBM regressor per horizon (h1..h12) for synthetic data.

No classifier stage — synthetic bark beetle counts are continuous and always
non-zero, so the two-stage sparsity-handling approach used for real posek
data does not apply here.

For each horizon h_i:
  Features   = base features (sanitised + derived)  +  predictions h1..h(i-1)
  Model      = LGBMRegressor (Huber loss)
  Prediction = max(reg_output, 0)

Inputs:
  data/synthetic/splits/train_synthetic.csv
  data/synthetic/splits/val_synthetic.csv
  data/synthetic/synthetic_target.csv

Output:
  models/lgb_models_synthetic.pkl
    dict: horizon -> {"reg": LGBMRegressor, "feature_cols": [col, ...]}
"""

import sys
import time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "synthetic" / "splits"
TARGET_PATH = ROOT / "data" / "synthetic" / "synthetic_target.csv"
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH  = MODELS_DIR / "lgb_models_synthetic.pkl"

sys.path.insert(0, str(ROOT))
from models.model import add_derived_features, sanitize_columns

# ---------------------------------------------------------------------------
# Column config
# ---------------------------------------------------------------------------
POSSIBLE_KEYS = ["ggo", "odsek", "odsek_id", "leto_mesec"]

DROP_COLS = [
    "datum", "leto", "target", "log1p_target",
    "sosedi_target_sum", "sosedi_target_mean",
    "sosedi_target_std", "sosedi_target_median",
    "sosedi_log1p_target_mean",
]
TARGET_COLS = [f"h{h}" for h in range(1, 13)]

# ---------------------------------------------------------------------------
# Regressor hyperparameters
# ---------------------------------------------------------------------------
REG_PARAMS = dict(
    objective="huber",
    alpha=0.9,
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_merge() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train_synthetic.csv"
    val_path   = DATA_DIR / "val_synthetic.csv"
    for p in (train_path, val_path, TARGET_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print(f"Loading X from {train_path.name} / {val_path.name} …")
    train_x = pd.read_csv(train_path, low_memory=False)
    val_x   = pd.read_csv(val_path,   low_memory=False)

    print(f"Loading targets from {TARGET_PATH.name} …")
    targets = pd.read_csv(TARGET_PATH, low_memory=False)

    keys = sorted(set(train_x.columns) & set(targets.columns) & set(POSSIBLE_KEYS))
    if not keys:
        raise ValueError("No common join keys between features and target.csv")
    print(f"  Join keys: {keys}")

    target_cols_present = [c for c in TARGET_COLS if c in targets.columns]
    if len(target_cols_present) != 12:
        raise ValueError(f"synthetic_target.csv missing horizon cols; found: {target_cols_present}")

    for df in (train_x, val_x):
        overlap = [c for c in TARGET_COLS if c in df.columns]
        if overlap:
            print(f"  WARNING: feature CSV contains horizon cols {overlap} — dropping")
            train_x = train_x.drop(columns=[c for c in overlap if c in train_x.columns])
            val_x   = val_x.drop(  columns=[c for c in overlap if c in val_x.columns])
            break

    train = train_x.merge(targets[keys + TARGET_COLS], on=keys, how="inner")
    val   = val_x.merge(  targets[keys + TARGET_COLS], on=keys, how="inner")
    print(f"  After merge — train: {len(train):,}  |  val: {len(val):,}")
    return train, val


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_cols   = [c for c in POSSIBLE_KEYS if c in df.columns]
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in set(id_cols + drop_cols + TARGET_COLS)]

    if drop_cols:
        print(f"    Dropping leaky columns: {drop_cols}")

    id_df = df[id_cols].reset_index(drop=True)
    X     = df[feat_cols].reset_index(drop=True)
    y     = df[TARGET_COLS].reset_index(drop=True)

    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"    Filling NaN in {len(nan_cols)} columns with 0")
        X = X.fillna(0)

    X = sanitize_columns(X)
    X = add_derived_features(X)
    return id_df, X, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> None:
    train_df, val_df = load_and_merge()

    print("\nPreparing features …")
    _, X_tr_base, y_train = prepare_xy(train_df)
    _, X_v_base,  y_val   = prepare_xy(val_df)

    print(f"  Train : {len(X_tr_base):,} rows × {X_tr_base.shape[1]} features")
    print(f"  Val   : {len(X_v_base):,} rows")

    models: dict[str, dict] = {}
    X_tr = X_tr_base.copy()
    X_v  = X_v_base.copy()
    t_total = time.time()

    for i, col in enumerate(TARGET_COLS):
        t0 = time.time()
        print(f"\n[{i+1:2d}/12]  {col}  ({X_tr.shape[1]} features)")

        y_tr = y_train[col].values
        y_v  = y_val[col].values

        reg = lgb.LGBMRegressor(**REG_PARAMS)
        reg.fit(
            X_tr, y_tr,
            eval_set=[(X_v, y_v)],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        v_preds = np.maximum(reg.predict(X_v), 0.0)
        mae  = mean_absolute_error(y_v, v_preds)
        bias = float(np.mean(v_preds - y_v))
        best = getattr(reg, "best_iteration_", "n/a")
        print(f"  best={best}  val MAE={mae:.4f}  bias={bias:.4f}  {time.time()-t0:.1f}s")

        models[col] = {
            "reg":          reg,
            "feature_cols": list(X_tr.columns),
        }

        if i < 11:
            tr_preds = np.maximum(reg.predict(X_tr), 0.0)
            pred_col = f"pred_{col}"
            tr_noise_scale = float(np.std(tr_preds)) * 0.15
            rng = np.random.default_rng(seed=42 + i)
            X_tr = X_tr.copy()
            X_v  = X_v.copy()
            X_tr[pred_col] = tr_preds + rng.normal(0, tr_noise_scale, size=len(tr_preds))
            X_v[pred_col]  = v_preds

    print(f"\nTotal training time: {(time.time() - t_total) / 60:.1f} min")

    # ── Final val evaluation ──────────────────────────────────────────────────
    print("\nReconstructing val predictions for final metrics …")
    X_v_eval  = X_v_base.copy()
    val_preds = np.zeros((len(X_v_eval), 12))

    for i, col in enumerate(TARGET_COLS):
        m         = models[col]
        X_aligned = X_v_eval[m["feature_cols"]]
        preds_i   = np.maximum(m["reg"].predict(X_aligned), 0.0)
        val_preds[:, i] = preds_i
        if i < 11:
            X_v_eval = X_v_eval.copy()
            X_v_eval[f"pred_{col}"] = preds_i

    y_val_arr = y_val.values
    mae  = mean_absolute_error(y_val_arr, val_preds)
    rmse = root_mean_squared_error(y_val_arr, val_preds)
    bias = float(np.mean(val_preds - y_val_arr))

    print(f"\nVal metrics (all 12 horizons, log1p space):")
    print(f"  Overall  — MAE: {mae:.4f}   RMSE: {rmse:.4f}   Bias: {bias:.4f}")

    print(f"\n  {'':4s}  {'MAE':>10s}  {'RMSE':>10s}  {'Bias':>10s}")
    for i, col in enumerate(TARGET_COLS):
        yt = y_val_arr[:, i]
        yp = val_preds[:, i]
        print(
            f"  {col:4s}  "
            f"{mean_absolute_error(yt, yp):10.4f}  "
            f"{root_mean_squared_error(yt, yp):10.4f}  "
            f"{float(np.mean(yp - yt)):10.4f}"
        )

    joblib.dump(models, MODEL_PATH)
    print(f"\nModels saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
