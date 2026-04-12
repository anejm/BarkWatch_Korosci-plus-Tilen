"""
train.py
--------
Sequential two-stage LightGBM model per horizon (h1..h12).

For each horizon h_i:
  Features   = base features (sanitised + derived)  +  predictions h1..h(i-1)
  Stage 1    = LGBMClassifier  → is h_i zero or non-zero?
               Threshold tuned on val set via F_beta maximisation.
               beta=0.7 for h8-h12 (favours precision, reduces positive-bias
               that otherwise compounds through the sequential chain).
  Stage 2    = LGBMRegressor   → magnitude (Huber loss, non-zero rows only)
  Prediction = (prob >= threshold) * max(reg_output, 0)

Improvements over XGBoost baseline:
  - LightGBM + Huber loss: robust to large-harvest outliers that caused
    RMSE >> MAE (21.7 vs 5.3 m³) in the original model.
  - Derived features: trend/ratio/flag signals added via add_derived_features().
  - Column sanitisation: spaces in names stripped (LightGBM requirement).
  - Precision-mode threshold for late horizons: reduces positive bias at h8-h12.

Inputs:
  data/processed/splits/train.csv
  data/processed/splits/val.csv
  data/processed/target.csv   — h1..h12 in log1p space

Output:
  models/lgb_models.pkl
    dict: horizon -> {"clf": LGBMClassifier, "reg": LGBMRegressor,
                      "threshold": float, "feature_cols": [col, ...]}

For synthetic data training, use train_synthetic.py instead.
"""

import argparse
import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    mean_absolute_error, root_mean_squared_error,
)

# ---------------------------------------------------------------------------
# Paths & model import
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "processed" / "splits"
MODELS_DIR  = ROOT / "models"
TARGET_PATH = ROOT / "data" / "processed" / "target.csv"

SYNTHETIC_DATA_DIR    = ROOT / "data" / "synthetic" / "splits"
SYNTHETIC_TARGET_PATH = ROOT / "data" / "synthetic" / "synthetic_target.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH           = MODELS_DIR / "lgb_models.pkl"
MODEL_PATH_SYNTHETIC = MODELS_DIR / "lgb_models_synthetic.pkl"

# Allow importing from project root
sys.path.insert(0, str(ROOT))
from models.model import TwoStageHorizonModel, add_derived_features, sanitize_columns

# ---------------------------------------------------------------------------
# Column config
# ---------------------------------------------------------------------------
POSSIBLE_KEYS = ["ggo", "odsek", "odsek_id", "leto_mesec"]

DROP_COLS = [
    "datum", "leto",
]
TARGET_COLS = [f"h{h}" for h in range(1, 13)]

# Horizons where positive bias compounds — use precision-favouring threshold
PRECISION_MODE_HORIZONS = {f"h{h}" for h in range(8, 13)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BarkWatch LightGBM horizon models")
    p.add_argument("--short", action="store_true",
                   help="Use *_short.csv splits instead of full splits")
    p.add_argument("--synthetic", action="store_true",
                   help="Train on synthetic data; saves to lgb_models_synthetic.pkl")
    return p.parse_args()


def _join_keys(df_x: pd.DataFrame, df_target: pd.DataFrame) -> list[str]:
    target_keys = set(df_target.columns) & set(POSSIBLE_KEYS)
    x_keys      = set(df_x.columns)     & set(POSSIBLE_KEYS)
    common      = sorted(target_keys & x_keys)
    if not common:
        raise ValueError(
            f"No common join keys between X {list(df_x.columns[:8])} "
            f"and target {list(df_target.columns)}"
        )
    return common


def load_and_merge(short: bool, synthetic: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if synthetic:
        train_path  = SYNTHETIC_DATA_DIR / "train_synthetic.csv"
        val_path    = SYNTHETIC_DATA_DIR / "val_synthetic.csv"
        target_path = SYNTHETIC_TARGET_PATH
    else:
        suffix      = "_short" if short else ""
        train_path  = DATA_DIR / f"train{suffix}.csv"
        val_path    = DATA_DIR / f"val{suffix}.csv"
        target_path = TARGET_PATH

    for p in (train_path, val_path, target_path):
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
        raise ValueError(f"target.csv missing horizon cols; found: {target_cols_present}")

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
    """Split merged df into (id_df, X_base, y).

    Processing pipeline:
      1. Drop leaky/temporal columns.
      2. Fill NaN with 0 (lag/rolling features NaN at series start).
      3. Sanitise column names (strip spaces → LightGBM requirement).
      4. Add derived features (trends, ratios, flags).
    """
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


def _two_stage_predict(model: TwoStageHorizonModel, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(short: bool, synthetic: bool = False) -> None:
    train_df, val_df = load_and_merge(short, synthetic=synthetic)

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

        y_tr_bin = (y_tr > 0).astype(int)
        y_v_bin  = (y_v  > 0).astype(int)

        n_neg = int((y_tr_bin == 0).sum())
        n_pos = int((y_tr_bin == 1).sum())
        print(f"  clf  pos={n_pos:,}  neg={n_neg:,}  "
              f"val_pos_rate={y_v_bin.mean():.3f} …", end=" ", flush=True)

        # scale_pos_weight: 0.75-power ratio — stronger than sqrt (~5x) but
        # less aggressive than full ratio (~25x), balancing recall and precision.
        spw = (n_neg / max(n_pos, 1)) ** 0.75

        # For late horizons, use precision-favouring threshold (beta=0.7)
        # to suppress the positive bias that compounds through sequential chain.
        precision_mode = col in PRECISION_MODE_HORIZONS

        model = TwoStageHorizonModel(
            scale_pos_weight=spw,
            precision_mode=precision_mode,
        )
        model.fit(X_tr, y_tr, X_v, y_v)

        threshold = model.threshold
        clf_pred_bin = (model.clf.predict_proba(X_v)[:, 1] >= threshold).astype(int)
        f1  = f1_score(y_v_bin, clf_pred_bin, zero_division=0)
        pre = precision_score(y_v_bin, clf_pred_bin, zero_division=0)
        rec = recall_score(y_v_bin, clf_pred_bin, zero_division=0)

        clf_best = getattr(model.clf, "best_iteration_", "n/a")
        print(f"clf_best={clf_best}  thresh={threshold:.2f}  "
              f"F1={f1:.3f}  P={pre:.3f}  R={rec:.3f}")

        nz_v = y_v > 0
        combined_v = model.predict(X_v)
        mae_nz = (mean_absolute_error(y_v[nz_v], combined_v[nz_v])
                  if nz_v.sum() > 0 else float("nan"))
        bias = float(np.mean(combined_v - y_v))

        reg_best = getattr(model.reg, "best_iteration_", "n/a")
        print(f"  reg  best={reg_best}  non-zero MAE={mae_nz:.4f}  "
              f"bias={bias:.4f}  {time.time()-t0:.1f}s")

        # Store in dict (backward-compatible with testing.py which directly
        # accesses clf, reg, threshold, feature_cols)
        models[col] = {
            "clf":          model.clf,
            "reg":          model.reg,
            "threshold":    model.threshold,
            "feature_cols": list(model.feature_cols),
        }

        # ── Augment features for next horizon ────────────────────────────────
        # Add predictions from this horizon as a feature for the next.
        # On training set: add Gaussian noise (σ = 15% of pred std) to
        # simulate the test-time prediction error and reduce over-fitting.
        # On validation set: use clean predictions.
        if i < 11:
            tr_preds = model.predict(X_tr)
            v_preds  = model.predict(X_v)
            pred_col = f"pred_{col}"
            X_tr = X_tr.copy()
            X_v  = X_v.copy()
            tr_noise_scale = float(np.std(tr_preds)) * 0.15   # 15% vs 10%
            rng = np.random.default_rng(seed=42 + i)
            X_tr[pred_col] = tr_preds + rng.normal(0, tr_noise_scale, size=len(tr_preds))
            X_v[pred_col]  = v_preds

    print(f"\nTotal training time: {(time.time() - t_total) / 60:.1f} min")

    # ── Final val evaluation ──────────────────────────────────────────────────
    print("\nReconstructing val predictions for final metrics …")
    X_v_eval  = X_v_base.copy()
    val_preds = np.zeros((len(X_v_eval), 12))

    for i, col in enumerate(TARGET_COLS):
        m = models[col]
        X_aligned    = X_v_eval[m["feature_cols"]]
        preds_i      = _lgb_predict(m, X_aligned)
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
        nz_col   = yt > 0
        mae_col  = mean_absolute_error(yt, yp)
        rmse_col = root_mean_squared_error(yt, yp)
        mae_nz   = mean_absolute_error(yt[nz_col], yp[nz_col]) if nz_col.sum() else float("nan")
        bias_col = float(np.mean(yp - yt))
        print(f"  {col:4s}  {mae_col:10.4f}  {mae_nz:10.4f}  {rmse_col:10.4f}  {bias_col:10.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = MODEL_PATH_SYNTHETIC if synthetic else MODEL_PATH
    joblib.dump(models, save_path)
    print(f"\nModels saved → {save_path}")


def _lgb_predict(m: dict, X: pd.DataFrame) -> np.ndarray:
    """Unified predict using stored clf/reg/threshold (testing.py compatible)."""
    clf_prob = m["clf"].predict_proba(X)[:, 1]
    clf_bin  = (clf_prob >= m["threshold"]).astype(int)
    reg_pred = np.maximum(m["reg"].predict(X), 0.0)
    return clf_bin * reg_pred


if __name__ == "__main__":
    args = parse_args()
    train(short=args.short, synthetic=args.synthetic)
