"""
train.py
--------
Sequential two-stage model per horizon (h1..h12).

For each horizon h_i:
  Features   = base features  +  predictions from models h1..h(i-1)
  Stage 1    = XGBClassifier  → is h_i zero or non-zero?
               Threshold is tuned on val set to match actual positive rate
               (eliminates the systematic positive bias from scale_pos_weight).
  Stage 2    = XGBRegressor   → magnitude, trained only on non-zero rows
  Prediction = (prob >= threshold) * max(reg_output, 0)

Inputs:
  data/processed/splits/train.csv  (or train_short.csv with --short)
  data/processed/splits/val.csv    (or val_short.csv   with --short)
  data/processed/target.csv        — columns: ggo, odsek, leto_mesec, h1..h12
                                     (targets are in log1p space)

Output:
  models/xgb_models.pkl
    dict: horizon -> {"clf": XGBClassifier, "reg": XGBRegressor,
                      "threshold": float, "feature_cols": [col, ...]}
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

# Columns to drop from features:
#   - datum / leto_mesec string / leto: date identifiers or temporal trend
#     (leto encourages extrapolation beyond training years)
#   - target / log1p_target: current-month harvest leaks into lag_0 position;
#     lag_1 already captures the previous month, which is safer for inference
#   - sosedi_target_* / sosedi_log1p_target_mean: contemporaneous neighbour
#     harvest — not available at prediction time
DROP_COLS = [
    "datum", "leto", "target", "log1p_target",
    "sosedi_target_sum", "sosedi_target_mean",
    "sosedi_target_std", "sosedi_target_median",
    "sosedi_log1p_target_mean",
]
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

# No scale_pos_weight — we compensate via threshold tuning instead.
# scale_pos_weight with spw=n_neg/n_pos (often 10-20x) caused the classifier
# to predict "nonzero" for nearly all rows → systematic positive bias ≈ 6.7
# in log1p space. Threshold tuning gives a cleaner operating point on the
# precision-recall curve without the global positive shift.
CLF_PARAMS = dict(**COMMON, eval_metric="logloss")

# reg:squarederror is appropriate because targets are in log1p space
# (approximately Gaussian). reg:tweedie is designed for raw count data with
# a specific variance structure; applying it to log1p-transformed values
# double-transforms the data.
REG_PARAMS = dict(
    **COMMON,
    eval_metric="rmse",
    objective="reg:squarederror",
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

    # Drop target horizon columns from X if they already exist (avoids _x/_y suffixes)
    for df in (train_x, val_x):
        overlap = [c for c in TARGET_COLS if c in df.columns]
        if overlap:
            print(f"  WARNING: feature CSV already contains horizon cols {overlap} — dropping before merge")
            train_x = train_x.drop(columns=[c for c in overlap if c in train_x.columns])
            val_x   = val_x.drop(  columns=[c for c in overlap if c in val_x.columns])
            break

    train = train_x.merge(targets[keys + TARGET_COLS], on=keys, how="inner")
    val   = val_x.merge(  targets[keys + TARGET_COLS], on=keys, how="inner")

    print(f"  After merge — train: {len(train):,} rows  |  val: {len(val):,} rows")
    return train, val


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a merged dataframe into (id_df, X_base, y).
    NaN features are filled with 0 (rolling/lag features have NaN at series start).
    """
    id_cols   = [c for c in POSSIBLE_KEYS if c in df.columns]
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in set(id_cols + drop_cols + TARGET_COLS)]

    # Report any dropped leaky columns
    if drop_cols:
        print(f"    Dropping columns: {drop_cols}")

    id_df = df[id_cols].reset_index(drop=True)
    X     = df[feat_cols].reset_index(drop=True)
    y     = df[TARGET_COLS].reset_index(drop=True)

    # Fill NaN with 0 instead of dropping rows.
    # Rolling and lag features are NaN at the start of each odsek time series;
    # dropping entire rows wastes data and biases toward odseki with long histories.
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"    Filling NaN in {len(nan_cols)} columns with 0 (lag/rolling start-of-series)")
        X = X.fillna(0)

    return id_df, X, y


def find_threshold(clf, X_val: pd.DataFrame, y_val_bin: np.ndarray) -> float:
    """
    Find the classifier probability threshold that makes the predicted positive
    rate match the actual positive rate on the validation set.

    This cancels out the effect of scale_pos_weight (or any classifier bias)
    on the final positive/negative split, eliminating systematic bias in the
    two-stage prediction.

    Returns the best threshold in [0.01, 0.99].
    """
    probs = clf.predict_proba(X_val)[:, 1]
    actual_pos_rate = float(y_val_bin.mean())

    best_thresh = 0.5
    best_gap    = float("inf")

    for thresh in np.linspace(0.01, 0.99, 99):
        pred_pos_rate = float((probs >= thresh).mean())
        gap = abs(pred_pos_rate - actual_pos_rate)
        if gap < best_gap:
            best_gap    = gap
            best_thresh = float(thresh)

    return best_thresh


def _two_stage_predict(clf, reg, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    clf_prob = clf.predict_proba(X)[:, 1]
    clf_bin  = (clf_prob >= threshold).astype(int)
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

        # Binary labels for classifier (nonzero in log1p space)
        y_tr_bin = (y_tr > 0).astype(int)
        y_v_bin  = (y_v  > 0).astype(int)

        n_neg = int((y_tr_bin == 0).sum())
        n_pos = int((y_tr_bin == 1).sum())
        print(f"  clf  pos={n_pos:,}  neg={n_neg:,}  "
              f"actual_pos_rate={y_v_bin.mean():.3f} …", end=" ", flush=True)

        # ── Stage 1: classifier ──────────────────────────────────────────────
        clf = XGBClassifier(**CLF_PARAMS)
        clf.fit(X_tr, y_tr_bin, eval_set=[(X_v, y_v_bin)], verbose=False)

        # Tune threshold to match actual positive rate (eliminates systematic bias)
        threshold = find_threshold(clf, X_v, y_v_bin)

        clf_pred_bin = (clf.predict_proba(X_v)[:, 1] >= threshold).astype(int)
        f1  = f1_score(y_v_bin, clf_pred_bin, zero_division=0)
        pre = precision_score(y_v_bin, clf_pred_bin, zero_division=0)
        rec = recall_score(y_v_bin, clf_pred_bin, zero_division=0)
        print(f"best={clf.best_iteration}  thresh={threshold:.2f}  "
              f"F1={f1:.3f}  P={pre:.3f}  R={rec:.3f}")

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
            reg_params_no_es = {k: v for k, v in REG_PARAMS.items()
                                if k != "early_stopping_rounds"}
            reg = XGBRegressor(**reg_params_no_es)
            reg.fit(X_tr[nz_tr], y_tr[nz_tr], verbose=False)

        combined_v = _two_stage_predict(clf, reg, X_v, threshold)
        mae_nz = (mean_absolute_error(y_v[nz_v], combined_v[nz_v])
                  if nz_v.sum() > 0 else float("nan"))

        # Bias check: positive bias means model over-predicts zeros
        bias = float(np.mean(combined_v - y_v))
        best_it = getattr(reg, "best_iteration", "n/a")
        print(f"best={best_it}  non-zero MAE={mae_nz:.4f}  bias={bias:.4f}  "
              f"{time.time()-t0:.1f}s")

        models[col] = {
            "clf":          clf,
            "reg":          reg,
            "threshold":    threshold,
            "feature_cols": list(X_tr.columns),
        }

        # ── Augment features for next horizon ────────────────────────────────
        # NOTE: using the model's own training-set predictions as features
        # for the next horizon creates a training-time advantage — the model
        # has already seen training rows so pred_h{i} ≈ actual h{i} on train.
        # At test time pred_h{i} is noisier, causing covariate shift.
        # Proper fix = out-of-fold predictions (complex); for now we add
        # small Gaussian noise to training pred_h{i} to partially simulate
        # the test-time prediction error and reduce over-reliance.
        if i < 11:
            tr_preds = _two_stage_predict(clf, reg, X_tr, threshold)
            v_preds  = _two_stage_predict(clf, reg, X_v,  threshold)
            pred_col = f"pred_{col}"
            X_tr = X_tr.copy()
            X_v  = X_v.copy()
            # Add noise proportional to training prediction std to partially
            # break the train-set optimism (val set stays clean)
            tr_noise_scale = float(np.std(tr_preds)) * 0.1
            rng = np.random.default_rng(seed=42 + i)
            X_tr[pred_col] = tr_preds + rng.normal(0, tr_noise_scale, size=len(tr_preds))
            X_v[pred_col]  = v_preds

    print(f"\nTotal training time: {(time.time() - t_total) / 60:.1f} min")

    # ── Final val evaluation across all horizons ─────────────────────────────
    print("\nReconstructing val predictions for final metrics …")
    X_v_eval = X_v_base.copy()
    val_preds = np.zeros((len(X_v_eval), 12))

    for i, col in enumerate(TARGET_COLS):
        m = models[col]
        X_aligned = X_v_eval[m["feature_cols"]]
        preds_i = _two_stage_predict(m["clf"], m["reg"], X_aligned, m["threshold"])
        val_preds[:, i] = preds_i
        if i < 11:
            X_v_eval = X_v_eval.copy()
            X_v_eval[f"pred_{col}"] = preds_i

    y_val_arr = y_val.values
    mae  = mean_absolute_error(y_val_arr, val_preds)
    rmse = root_mean_squared_error(y_val_arr, val_preds)
    nz   = y_val_arr > 0
    bias_overall = float(np.mean(val_preds - y_val_arr))

    print(f"\nVal metrics (all 12 horizons, log1p space):")
    print(f"  Overall  — MAE: {mae:.4f}   RMSE: {rmse:.4f}   Bias: {bias_overall:.4f}")
    if nz.sum():
        print(f"  Non-zero — MAE: {mean_absolute_error(y_val_arr[nz], val_preds[nz]):.4f}"
              f"   RMSE: {root_mean_squared_error(y_val_arr[nz], val_preds[nz]):.4f}")

    print(f"\n  {'':4s}  {'MAE-all':>10s}  {'MAE-nz':>10s}  {'RMSE':>10s}  {'Bias':>10s}")
    for i, col in enumerate(TARGET_COLS):
        yt = y_val_arr[:, i]
        yp = val_preds[:, i]
        nz_col = yt > 0
        mae_col  = mean_absolute_error(yt, yp)
        rmse_col = root_mean_squared_error(yt, yp)
        mae_nz   = mean_absolute_error(yt[nz_col], yp[nz_col]) if nz_col.sum() else float("nan")
        bias_col = float(np.mean(yp - yt))
        print(f"  {col:4s}  {mae_col:10.4f}  {mae_nz:10.4f}  {rmse_col:10.4f}  {bias_col:10.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    joblib.dump(models, MODEL_PATH)
    print(f"\nModels saved → {MODEL_PATH}")


if __name__ == "__main__":
    args = parse_args()
    train(short=args.short)
