"""
model.py
--------
LightGBM-based two-stage horizon model for BarkWatch.

Stage 1 — LGBMClassifier  (binary: is harvest > 0?)
Stage 2 — LGBMRegressor   (Huber loss, trained on non-zero rows only)

Final prediction = (P(non-zero) >= threshold) × max(reg_output, 0)

Key improvements over XGBoost baseline:
  - Huber loss (alpha=0.9) for regressor: robust to extreme harvest outliers
    that cause RMSE >> MAE in the original model (RMSE 21.7 vs MAE 5.3).
  - LightGBM leaf-wise tree growth (num_leaves=63): more expressive than
    max_depth=4/6 level-wise trees, faster training.
  - Derived features: short/med/long-term trends, relative-level ratios,
    neighbour-vs-self ratio, activity flags, seasonal interactions.
  - Column sanitisation: spaces → underscores (required by LightGBM).
  - Precision-favouring threshold (F_beta, beta=0.7) for later horizons
    (h8-h12) to suppress the positive bias that compounds through the
    sequential augmentation chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import fbeta_score
from typing import Optional


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def sanitize_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Strip trailing spaces and replace internal spaces with underscores.

    LightGBM raises errors for feature names that contain spaces.
    The raw CSV has names like 'relief_1 ' and 'sestoji_sklep_ _sum'.
    """
    return X.rename(columns=lambda c: c.strip().replace(" ", "_"))


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features derived from existing columns.

    All additions are conditional on their source columns being present, so
    this function is safe to call on any feature DataFrame (train, val, test,
    future predictions).  Returns a new DataFrame; does not mutate input.

    New features
    ────────────
    Trend signals
      trend_short     : lag_1 - lag_3          (1-vs-3 month delta)
      trend_med       : lag_3 - lag_6          (3-vs-6 month delta)
      trend_long      : lag_6 - lag_12         (6-vs-12 month delta)
      trend_annual    : lag_12 - lag_24        (year-over-year delta)
      acceleration    : Δ(lag_1-lag_3) - Δ(lag_3-lag_6) (2nd derivative)

    Relative-level ratios (log1p space)
      lag1_vs_annual  : lag_1 / (rolling_mean_12 + ε)   (is this month above avg?)
      lag1_z          : lag_1 / (rolling_std_12  + ε)   (standardised harvest)
      vol_ratio       : rolling_std_3 / (rolling_std_12 + ε)  (volatility regime)

    Neighbour signals
      neighbor_vs_self  : sosedi_lag_1_sum  / (lag_1 + ε)
      neighbor_trend    : sosedi_lag_1_mean / (sosedi_rolling_mean_12_mean + ε)

    Activity flags  (binary floats)
      has_lag1          : lag_1 > 0
      has_recent_harvest: rolling_mean_3 > 0
      neighbor_active   : sosedi_lag_1_sum > 0

    Seasonal interactions
      season_sin_x_lag1 : mesec_sin × lag_1
      season_cos_x_lag1 : mesec_cos × lag_1

    Rolling max proxy  (from available lag checkpoints)
      max_recent_12m    : max(lag_1, lag_3, lag_6, lag_12)
      max_recent_24m    : max(lag_1, lag_3, lag_6, lag_12, lag_24)
      any_harvest_12m   : binary — any lag in 12m window > 0
      harvest_freq_12m  : fraction of 12m lag checkpoints > 0
      total_recent_12m  : sum of 12m lag checkpoints

    Year-over-year / historical ratios
      yoy_ratio         : lag_12 / (lag_24 + ε)

    Neighbor peak activity
      neighbor_max_recent  : max(sosedi_lag_1_sum, _lag_3_sum, _lag_6_sum)
      neighbor_any_recent  : binary — any neighbor active in past 6m

    Weather stress
      drought_stress    : avg_temp / (annual_precip / 100)  — bark beetle risk proxy

    Cross signals
      neighbor_self_activity : neighbor_max_recent × max_recent_12m
    """
    X = X.copy()
    cols = set(X.columns)

    def _div(a: pd.Series, b: pd.Series, eps: float = 0.01) -> pd.Series:
        return a / (b.abs() + eps)

    # ── Trend signals ────────────────────────────────────────────────────────
    if {"lag_1", "lag_3"} <= cols:
        X["trend_short"] = X["lag_1"] - X["lag_3"]
    if {"lag_3", "lag_6"} <= cols:
        X["trend_med"] = X["lag_3"] - X["lag_6"]
    if {"lag_6", "lag_12"} <= cols:
        X["trend_long"] = X["lag_6"] - X["lag_12"]
    if {"lag_12", "lag_24"} <= cols:
        X["trend_annual"] = X["lag_12"] - X["lag_24"]
    if {"lag_1", "lag_3", "lag_6"} <= cols:
        X["acceleration"] = (X["lag_1"] - X["lag_3"]) - (X["lag_3"] - X["lag_6"])

    # ── Relative-level ratios ─────────────────────────────────────────────────
    if {"lag_1", "rolling_mean_12"} <= cols:
        X["lag1_vs_annual"] = _div(X["lag_1"], X["rolling_mean_12"])
    if {"lag_1", "rolling_std_12"} <= cols:
        X["lag1_z"] = _div(X["lag_1"], X["rolling_std_12"])
    if {"rolling_std_3", "rolling_std_12"} <= cols:
        X["vol_ratio"] = _div(X["rolling_std_3"], X["rolling_std_12"])

    # ── Neighbour signals ────────────────────────────────────────────────────
    if {"sosedi_lag_1_sum", "lag_1"} <= cols:
        X["neighbor_vs_self"] = _div(X["sosedi_lag_1_sum"], X["lag_1"])
    if {"sosedi_lag_1_mean", "sosedi_rolling_mean_12_mean"} <= cols:
        X["neighbor_trend"] = _div(
            X["sosedi_lag_1_mean"], X["sosedi_rolling_mean_12_mean"]
        )

    # ── Activity flags ────────────────────────────────────────────────────────
    if "lag_1" in cols:
        X["has_lag1"] = (X["lag_1"] > 0).astype(np.float32)
    if "rolling_mean_3" in cols:
        X["has_recent_harvest"] = (X["rolling_mean_3"] > 0).astype(np.float32)
    if "sosedi_lag_1_sum" in cols:
        X["neighbor_active"] = (X["sosedi_lag_1_sum"] > 0).astype(np.float32)

    # ── Seasonal interactions ─────────────────────────────────────────────────
    if {"mesec_sin", "lag_1"} <= cols:
        X["season_sin_x_lag1"] = X["mesec_sin"] * X["lag_1"]
    if {"mesec_cos", "lag_1"} <= cols:
        X["season_cos_x_lag1"] = X["mesec_cos"] * X["lag_1"]

    # ── Rolling max proxy (from available lag checkpoints) ───────────────────
    # True rolling max unavailable; max over lag checkpoints captures whether
    # there was ever a large harvest event in the window.
    _lag_12 = [c for c in ["lag_1", "lag_3", "lag_6", "lag_12"] if c in cols]
    if len(_lag_12) >= 2:
        X["max_recent_12m"] = X[_lag_12].max(axis=1)
        X["any_harvest_12m"] = (X[_lag_12].max(axis=1) > 0).astype(np.float32)
        X["harvest_freq_12m"] = (X[_lag_12] > 0).sum(axis=1) / len(_lag_12)
        X["total_recent_12m"] = X[_lag_12].sum(axis=1)

    _lag_24 = [c for c in ["lag_1", "lag_3", "lag_6", "lag_12", "lag_24"] if c in cols]
    if len(_lag_24) >= 3:
        X["max_recent_24m"] = X[_lag_24].max(axis=1)

    # ── Year-over-year ratio ──────────────────────────────────────────────────
    if {"lag_12", "lag_24"} <= cols:
        X["yoy_ratio"] = _div(X["lag_12"], X["lag_24"])

    # ── Neighbor peak activity ────────────────────────────────────────────────
    _sosedi_lags = [c for c in ["sosedi_lag_1_sum", "sosedi_lag_3_sum", "sosedi_lag_6_sum"] if c in cols]
    if len(_sosedi_lags) >= 2:
        X["neighbor_max_recent"] = X[_sosedi_lags].max(axis=1)
        X["neighbor_any_recent"] = (X[_sosedi_lags].max(axis=1) > 0).astype(np.float32)

    # ── Weather drought stress: high temp + low rain → bark beetle risk ───────
    if {"leto_povp_T_avg", "leto_padavine_skupaj_mm"} <= cols:
        X["drought_stress"] = X["leto_povp_T_avg"] / (X["leto_padavine_skupaj_mm"].clip(lower=1) / 100)

    # ── Neighbor × self activity interaction ─────────────────────────────────
    if {"max_recent_12m", "neighbor_max_recent"} <= set(X.columns):
        X["neighbor_self_activity"] = X["neighbor_max_recent"] * X["max_recent_12m"]

    return X


# ---------------------------------------------------------------------------
# Two-stage per-horizon model
# ---------------------------------------------------------------------------

# Default hyperparameters
_CLF_DEFAULTS: dict = dict(
    n_estimators=2000,
    learning_rate=0.03,      # lower LR → more trees, better convergence on minority class
    num_leaves=127,          # more expressive than 63
    max_depth=-1,            # leaf-wise; num_leaves controls complexity
    min_child_samples=20,    # allow fitting smaller minority-class groups (was 30)
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.7,    # was 0.6
    reg_lambda=1.0,          # less regularization to capture minority class signal (was 2.0)
    metric='auc',            # AUC for early stopping — logloss converges at iter 2 on imbalanced data
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

_REG_DEFAULTS: dict = dict(
    objective="huber",
    alpha=0.9,               # Huber α: top-10% outlier residuals get L1 treatment
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=127,          # was 63
    max_depth=-1,
    min_child_samples=20,    # was 30
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


class TwoStageHorizonModel:
    """LightGBM two-stage model for a single forecast horizon.

    Stage 1 (classifier): binary — is harvest > 0?
    Stage 2 (regressor) : magnitude in log1p space using Huber loss,
                          trained on non-zero rows only.

    Threshold for the classifier is calibrated on a held-out validation set
    by maximising F_beta.  For horizons ≥ 8 (where positive bias compounds
    through the sequential chain), beta=0.7 gives slightly more weight to
    precision, suppressing false positives that drive the bias up.
    """

    def __init__(
        self,
        scale_pos_weight: float = 1.0,
        precision_mode: bool = False,   # unused — kept for API compat
        clf_overrides: Optional[dict] = None,
        reg_overrides: Optional[dict] = None,
    ):
        self.precision_mode = precision_mode
        # beta=2.0 strongly favours recall: we want to predict non-zero whenever
        # there is any reasonable signal, accepting more false positives.
        self._fbeta = 2.0

        clf_params = {**_CLF_DEFAULTS, "scale_pos_weight": scale_pos_weight}
        if clf_overrides:
            clf_params.update(clf_overrides)
        self.clf = lgb.LGBMClassifier(**clf_params)

        reg_params = {**_REG_DEFAULTS}
        if reg_overrides:
            reg_params.update(reg_overrides)
        self.reg = lgb.LGBMRegressor(**reg_params)

        self.threshold: float = 0.5
        self.feature_cols: list[str] = []

    # -----------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        early_stopping_rounds: int = 100,
    ) -> None:
        self.feature_cols = list(X_train.columns)

        y_tr_bin = (y_train > 0).astype(int)
        y_v_bin  = (y_val  > 0).astype(int)

        # ── Stage 1: classifier ──────────────────────────────────────────────
        _cbs = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        self.clf.fit(
            X_train, y_tr_bin,
            eval_set=[(X_val, y_v_bin)],
            callbacks=_cbs,
        )
        self.threshold = self._find_threshold(X_val, y_v_bin)

        # ── Stage 2: regressor (non-zero rows only) ──────────────────────────
        nz_tr = y_train > 0
        nz_v  = y_val   > 0

        _cbs_reg = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        if nz_tr.sum() > 0:
            if nz_v.sum() > 0:
                self.reg.fit(
                    X_train[nz_tr], y_train[nz_tr],
                    eval_set=[(X_val[nz_v], y_val[nz_v])],
                    callbacks=_cbs_reg,
                )
            else:
                self.reg.set_params(n_estimators=self.reg.n_estimators)
                self.reg.fit(X_train[nz_tr], y_train[nz_tr])

    # -----------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return combined two-stage prediction (log1p space).

        Soft-blend: samples above the threshold get full reg_pred; samples
        below threshold are scaled by clf_prob/threshold rather than hard-zeroed.
        This ensures borderline-positive samples still produce a non-zero
        prediction proportional to the classifier's confidence.
        """
        clf_prob = self.clf.predict_proba(X)[:, 1]
        reg_pred = np.maximum(self.reg.predict(X), 0.0)
        # weight=1 above threshold, weight=prob/threshold below it (still >0)
        weight = np.where(clf_prob >= self.threshold, 1.0, clf_prob / self.threshold)
        return weight * reg_pred

    # -----------------------------------------------------------------------
    def _find_threshold(
        self, X_val: pd.DataFrame, y_val_bin: np.ndarray
    ) -> float:
        """Grid-search probability threshold to maximise F_beta on val set."""
        probs = self.clf.predict_proba(X_val)[:, 1]
        best_thresh, best_score = 0.5, -1.0
        for thresh in np.linspace(0.01, 0.99, 99):
            pred_bin = (probs >= thresh).astype(int)
            score = fbeta_score(
                y_val_bin, pred_bin, beta=self._fbeta, zero_division=0
            )
            if score > best_score:
                best_score, best_thresh = score, float(thresh)
        return best_thresh
