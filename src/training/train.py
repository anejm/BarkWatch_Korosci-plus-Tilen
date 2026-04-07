"""
train.py
--------
Two-stage model per horizon (h1..h12):
  Stage 1 – XGBClassifier  → predicts whether the value is zero or non-zero
  Stage 2 – XGBRegressor   → predicts the magnitude, trained only on non-zero rows

Final prediction = classifier_output * regressor_output
  (if the classifier says 0, the prediction is 0 regardless of the regressor)

This approach handles the heavily zero-inflated distribution (~92 % zeros) much
better than a plain regressor, which tends to under-predict non-zero events.

Inputs:
  data/processed/splits/train.csv
  data/processed/splits/val.csv

Output:
  models/xgb_models.pkl   — dict mapping horizon → {"clf": XGBClassifier,
                                                      "reg": XGBRegressor}
"""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "xgb_models.pkl"

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
INDEX_COLS  = ["odsek", "leto_mesec"]
DROP_COLS   = ["datum"]
TARGET_COLS = [f"h{h}" for h in range(1, 13)]

# ---------------------------------------------------------------------------
# Shared XGBoost parameters
# ---------------------------------------------------------------------------
COMMON = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
)

CLF_PARAMS = dict(
    **COMMON,
    eval_metric="logloss",
    # scale_pos_weight set dynamically per horizon (ratio of negatives to positives)
)

REG_PARAMS = dict(
    **COMMON,
    eval_metric="rmse",
    # Tweedie loss handles heavy-tailed, non-negative distributions well
    objective="reg:tweedie",
    tweedie_variance_power=1.5,   # 1 → Poisson-like, 2 → Gamma-like
)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("Loading train / val …")
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df   = pd.read_csv(SPLITS_DIR / "val.csv")


def split_xy(df: pd.DataFrame):
    df = df.drop(columns=DROP_COLS + INDEX_COLS, errors="ignore")
    df = df.dropna()
    X = df.drop(columns=TARGET_COLS)
    y = df[TARGET_COLS]
    return X, y


X_train, y_train = split_xy(train_df)
X_val,   y_val   = split_xy(val_df)

print(f"  Train : {X_train.shape[0]:,} rows  |  {X_train.shape[1]} features")
print(f"  Val   : {X_val.shape[0]:,} rows")

# ---------------------------------------------------------------------------
# 2. Train two-stage model per horizon
# ---------------------------------------------------------------------------
models: dict[str, dict] = {}
val_preds = np.zeros((len(y_val), len(TARGET_COLS)))

t_total = time.time()

for i, col in enumerate(TARGET_COLS):
    t0 = time.time()
    print(f"\n[{i+1:2d}/12]  {col}")

    y_tr = y_train[col].values
    y_v  = y_val[col].values

    # Binary labels: 1 = non-zero event, 0 = no event
    y_tr_bin = (y_tr > 0).astype(int)
    y_v_bin  = (y_v  > 0).astype(int)

    n_neg = (y_tr_bin == 0).sum()
    n_pos = (y_tr_bin == 1).sum()
    spw   = n_neg / max(n_pos, 1)          # scale_pos_weight

    # ── Stage 1: classifier ──────────────────────────────────────────────
    print(f"  Stage 1 clf  (pos={n_pos:,}, neg={n_neg:,}, spw={spw:.1f}) …",
          end=" ", flush=True)

    clf = XGBClassifier(**CLF_PARAMS, scale_pos_weight=spw)
    clf.fit(
        X_train, y_tr_bin,
        eval_set=[(X_val, y_v_bin)],
        verbose=False,
    )
    clf_pred_bin = clf.predict(X_val)          # hard 0/1
    clf_pred_prob = clf.predict_proba(X_val)[:, 1]

    f1  = f1_score(y_v_bin, clf_pred_bin, zero_division=0)
    pre = precision_score(y_v_bin, clf_pred_bin, zero_division=0)
    rec = recall_score(y_v_bin, clf_pred_bin, zero_division=0)
    print(f"best={clf.best_iteration}  F1={f1:.3f}  P={pre:.3f}  R={rec:.3f}")

    # ── Stage 2: regressor on non-zero train rows only ───────────────────
    nz_mask_tr = y_tr > 0
    X_tr_nz    = X_train[nz_mask_tr]
    y_tr_nz    = y_tr[nz_mask_tr]

    nz_mask_v  = y_v > 0
    X_v_nz     = X_val[nz_mask_v]
    y_v_nz     = y_v[nz_mask_v]

    print(f"  Stage 2 reg  (train non-zero rows: {nz_mask_tr.sum():,}) …",
          end=" ", flush=True)

    reg = XGBRegressor(**REG_PARAMS)
    reg.fit(
        X_tr_nz, y_tr_nz,
        eval_set=[(X_v_nz, y_v_nz)],
        verbose=False,
    )
    reg_pred_nz = reg.predict(X_val)           # magnitude for every row

    # ── Combine: zero out rows the classifier marks as 0 ─────────────────
    combined = clf_pred_bin * np.maximum(reg_pred_nz, 0)
    val_preds[:, i] = combined

    # Non-zero MAE (only rows where y_true > 0)
    if nz_mask_v.sum() > 0:
        mae_nz = mean_absolute_error(y_v[nz_mask_v], combined[nz_mask_v])
    else:
        mae_nz = float("nan")

    elapsed = time.time() - t0
    print(f"best={reg.best_iteration}  non-zero MAE={mae_nz:.2f}  | {elapsed:.1f}s")

    models[col] = {"clf": clf, "reg": reg}

print(f"\nTotal training time: {(time.time() - t_total) / 60:.1f} min")

# ---------------------------------------------------------------------------
# 3. Evaluate on val
# ---------------------------------------------------------------------------
y_val_arr = y_val.values

mae_all  = mean_absolute_error(y_val_arr, val_preds)
rmse_all = root_mean_squared_error(y_val_arr, val_preds)

# Non-zero only (where any horizon is > 0)
nz_mask_any = (y_val_arr > 0)
if nz_mask_any.sum() > 0:
    mae_nz_all  = mean_absolute_error(y_val_arr[nz_mask_any], val_preds[nz_mask_any])
    rmse_nz_all = root_mean_squared_error(y_val_arr[nz_mask_any], val_preds[nz_mask_any])
else:
    mae_nz_all = rmse_nz_all = float("nan")

print(f"\nVal metrics (all 12 horizons averaged):")
print(f"  Overall  — MAE: {mae_all:.4f}   RMSE: {rmse_all:.4f}")
print(f"  Non-zero — MAE: {mae_nz_all:.4f}   RMSE: {rmse_nz_all:.4f}")

print("\nPer-horizon metrics:")
print(f"  {'':4s}  {'MAE-all':>10s}  {'MAE-nz':>10s}  {'RMSE':>10s}")
for i, col in enumerate(TARGET_COLS):
    y_true_col = y_val_arr[:, i]
    y_pred_col = val_preds[:, i]
    mae_col  = mean_absolute_error(y_true_col, y_pred_col)
    rmse_col = root_mean_squared_error(y_true_col, y_pred_col)
    nz = y_true_col > 0
    mae_nz_col = mean_absolute_error(y_true_col[nz], y_pred_col[nz]) if nz.sum() else float("nan")
    print(f"  {col:4s}  {mae_col:10.4f}  {mae_nz_col:10.4f}  {rmse_col:10.4f}")

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
joblib.dump(models, MODEL_PATH)
print(f"\nModels saved → {MODEL_PATH}")
