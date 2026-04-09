"""
train.py
--------
Sequential two-stage model per horizon (h1..h12).

For each horizon h_i:
  Features   = base features  +  predictions from models h1..h(i-1)
  Stage 1    = XGBClassifier  → is h_i zero or non-zero?
  Stage 2    = XGBRegressor   → magnitude, trained only on non-zero rows
  Prediction = clf_output * max(reg_output, 0)

Inputs:
  data/train.csv  (or data/train_short.csv with --short)
  data/val.csv    (or data/val_short.csv   with --short)
  data/processed/target.csv  — columns: ggo, odsek, leto_mesec, h1..h12

Output:
  models/xgb_models.pkl
    dict: horizon -> {"clf": XGBClassifier, "reg": XGBRegressor,
                      "feature_cols": [col, ...]}
"""

import argparse
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    mean_absolute_error, root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "processed" / "splits"
MODELS_DIR  = ROOT / "models"
TARGET_PATH = ROOT / "data" / "processed" / "target.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "xgb_models.pkl"

# ---------------------------------------------------------------------------
# Column config
# ---------------------------------------------------------------------------
POSSIBLE_KEYS = ["ggo", "odsek", "odsek_id", "leto_mesec"]
DROP_COLS     = ["datum", "leto", "target", "log1p_target"]
TARGET_COLS   = [f"h{h}" for h in range(1, 13)]

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
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

CLF_PARAMS = dict(**COMMON, eval_metric="logloss")
REG_PARAMS = dict(
    **COMMON,
    eval_metric="rmse",
    objective="reg:tweedie",
    tweedie_variance_power=1.5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BarkWatch horizon models")
    p.add_argument("--short", action="store_true",
                   help="Use *_short.csv splits instead of full splits")
    return p.parse_args()


def _join_keys(df_x: pd.DataFrame, df_target: pd.DataFrame) -> list[str]:
    """Return columns present in both X and target that can serve as join keys."""
    target_keys = set(df_target.columns) & set(POSSIBLE_KEYS)
    x_keys      = set(df_x.columns) & set(POSSIBLE_KEYS)
    common      = sorted(target_keys & x_keys)
    if not common:
        raise ValueError(
            f"No common join keys between X {list(df_x.columns[:8])} "
            f"and target {list(df_target.columns)}"
        )
    return common


def load_and_merge(short: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/val X files and target.csv, merge on common keys.
    Returns (train_df, val_df) each containing features + h1..h12 columns.
    """
    suffix = "_short" if short else ""
    train_path = DATA_DIR / f"train{suffix}.csv"
    val_path   = DATA_DIR / f"val{suffix}.csv"

    for p in (train_path, val_path, TARGET_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print(f"Loading X from {train_path.name} / {val_path.name} …")
    train_x = pd.read_csv(train_path, low_memory=False)
    val_x   = pd.read_csv(val_path,   low_memory=False)

    print(f"Loading targets from {TARGET_PATH.name} …")
    targets = pd.read_csv(TARGET_PATH, low_memory=False)

    keys = _join_keys(train_x, targets)
    print(f"  Join keys: {keys}")

    target_cols_present = [c for c in TARGET_COLS if c in targets.columns]
    if len(target_cols_present) != 12:
        raise ValueError(f"target.csv is missing horizon columns; found: {target_cols_present}")

    train = train_x.merge(targets[keys + TARGET_COLS], on=keys, how="inner")
    val   = val_x.merge(  targets[keys + TARGET_COLS], on=keys, how="inner")

    print(f"  After merge — train: {len(train):,} rows  |  val: {len(val):,} rows")
    return train, val


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a merged dataframe into (id_df, X_base, y).
    Drops rows with NaN features.
    """
    id_cols   = [c for c in POSSIBLE_KEYS if c in df.columns]
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in id_cols + drop_cols + TARGET_COLS]

    id_df = df[id_cols].reset_index(drop=True)
    X     = df[feat_cols].reset_index(drop=True)
    y     = df[TARGET_COLS].reset_index(drop=True)

    valid   = X.notna().all(axis=1)
    dropped = (~valid).sum()
    if dropped:
        print(f"    Dropped {dropped:,} rows with NaN features")

    return (
        id_df[valid].reset_index(drop=True),
        X[valid].reset_index(drop=True),
        y[valid].reset_index(drop=True),
    )


def _two_stage_predict(clf, reg, X: pd.DataFrame) -> np.ndarray:
    clf_bin  = clf.predict(X)
    reg_pred = np.maximum(reg.predict(X), 0)
    return clf_bin * reg_pred


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(short: bool) -> None:
    train_df, val_df = load_and_merge(short)

    print("\nPreparing features …")
    _, X_tr_base, y_train = prepare_xy(train_df)
    _, X_v_base,  y_val   = prepare_xy(val_df)

    print(f"  Train : {len(X_tr_base):,} rows × {X_tr_base.shape[1]} features")
    print(f"  Val   : {len(X_v_base):,} rows")

    models: dict[str, dict] = {}

    # Running augmented feature matrices (grow by 1 column per horizon)
    X_tr = X_tr_base.copy()
    X_v  = X_v_base.copy()

    t_total = time.time()

    for i, col in enumerate(TARGET_COLS):
        t0 = time.time()
        print(f"\n[{i+1:2d}/12]  {col}  ({X_tr.shape[1]} features)")

        y_tr = y_train[col].values
        y_v  = y_val[col].values

        # Binary labels for classifier
        y_tr_bin = (y_tr > 0).astype(int)
        y_v_bin  = (y_v  > 0).astype(int)

        n_neg = int((y_tr_bin == 0).sum())
        n_pos = int((y_tr_bin == 1).sum())
        spw   = n_neg / max(n_pos, 1)

        # ── Stage 1: classifier ──────────────────────────────────────────────
        print(f"  clf  pos={n_pos:,}  neg={n_neg:,}  spw={spw:.1f} …", end=" ", flush=True)

        clf = XGBClassifier(**CLF_PARAMS, scale_pos_weight=spw)
        clf.fit(X_tr, y_tr_bin, eval_set=[(X_v, y_v_bin)], verbose=False)

        clf_pred_bin = clf.predict(X_v)
        f1  = f1_score(y_v_bin, clf_pred_bin, zero_division=0)
        pre = precision_score(y_v_bin, clf_pred_bin, zero_division=0)
        rec = recall_score(y_v_bin, clf_pred_bin, zero_division=0)
        print(f"best={clf.best_iteration}  F1={f1:.3f}  P={pre:.3f}  R={rec:.3f}")

        # ── Stage 2: regressor on non-zero rows ──────────────────────────────
        nz_tr = y_tr > 0
        nz_v  = y_v  > 0
        print(f"  reg  train non-zero: {nz_tr.sum():,} …", end=" ", flush=True)

        reg = XGBRegressor(**REG_PARAMS)
        if nz_v.sum() > 0:
            reg.fit(
                X_tr[nz_tr], y_tr[nz_tr],
                eval_set=[(X_v[nz_v], y_v[nz_v])],
                verbose=False,
            )
        else:
            # No non-zero val rows: train without early stopping
            reg_params_no_es = {k: v for k, v in REG_PARAMS.items()
                                if k != "early_stopping_rounds"}
            reg = XGBRegressor(**reg_params_no_es)
            reg.fit(X_tr[nz_tr], y_tr[nz_tr], verbose=False)

        combined_v = _two_stage_predict(clf, reg, X_v)
        mae_nz = (mean_absolute_error(y_v[nz_v], combined_v[nz_v])
                  if nz_v.sum() > 0 else float("nan"))

        best_it = getattr(reg, "best_iteration", "n/a")
        print(f"best={best_it}  non-zero MAE={mae_nz:.4f}  {time.time()-t0:.1f}s")

        models[col] = {
            "clf":          clf,
            "reg":          reg,
            "feature_cols": list(X_tr.columns),
        }

        # ── Augment features for next horizon ────────────────────────────────
        # Predictions on train set (using current augmented X) become a feature
        # for subsequent horizon models — mirrors inference behaviour.
        if i < 11:
            tr_preds = _two_stage_predict(clf, reg, X_tr)
            v_preds  = _two_stage_predict(clf, reg, X_v)
            pred_col = f"pred_{col}"
            X_tr = X_tr.copy()
            X_v  = X_v.copy()
            X_tr[pred_col] = tr_preds
            X_v[pred_col]  = v_preds

    print(f"\nTotal training time: {(time.time() - t_total) / 60:.1f} min")

    # ── Final val evaluation across all horizons ─────────────────────────────
    print("\nReconstructing val predictions for final metrics …")
    X_v_eval = X_v_base.copy()
    val_preds = np.zeros((len(X_v_eval), 12))

    for i, col in enumerate(TARGET_COLS):
        m = models[col]
        X_aligned = X_v_eval[m["feature_cols"]]
        preds_i = _two_stage_predict(m["clf"], m["reg"], X_aligned)
        val_preds[:, i] = preds_i
        if i < 11:
            X_v_eval = X_v_eval.copy()
            X_v_eval[f"pred_{col}"] = preds_i

    y_val_arr = y_val.values
    mae  = mean_absolute_error(y_val_arr, val_preds)
    rmse = root_mean_squared_error(y_val_arr, val_preds)
    nz   = y_val_arr > 0

    print(f"\nVal metrics (all 12 horizons):")
    print(f"  Overall  — MAE: {mae:.4f}   RMSE: {rmse:.4f}")
    if nz.sum():
        print(f"  Non-zero — MAE: {mean_absolute_error(y_val_arr[nz], val_preds[nz]):.4f}"
              f"   RMSE: {root_mean_squared_error(y_val_arr[nz], val_preds[nz]):.4f}")

    print(f"\n  {'':4s}  {'MAE-all':>10s}  {'MAE-nz':>10s}  {'RMSE':>10s}")
    for i, col in enumerate(TARGET_COLS):
        yt = y_val_arr[:, i]
        yp = val_preds[:, i]
        nz_col = yt > 0
        mae_col  = mean_absolute_error(yt, yp)
        rmse_col = root_mean_squared_error(yt, yp)
        mae_nz   = mean_absolute_error(yt[nz_col], yp[nz_col]) if nz_col.sum() else float("nan")
        print(f"  {col:4s}  {mae_col:10.4f}  {mae_nz:10.4f}  {rmse_col:10.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    joblib.dump(models, MODEL_PATH)
    print(f"\nModels saved → {MODEL_PATH}")


if __name__ == "__main__":
    args = parse_args()
    train(short=args.short)
