"""
model.py — LightGBM two-stage horizon model for BarkWatch.

Stage 1 — LGBMClassifier  (binary: is harvest > 0?)
Stage 2 — LGBMRegressor   (Huber loss, trained on non-zero rows only)

Final prediction = (P(non-zero) >= threshold) × max(reg_output, 0)
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
    """Strip spaces and replace internal spaces with underscores (LightGBM requirement)."""
    return X.rename(columns=lambda c: c.strip().replace(" ", "_"))


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Safe to call on any split — skips missing source cols."""
    X = X.copy()
    cols = set(X.columns)

    def _div(a: pd.Series, b: pd.Series, eps: float = 0.01) -> pd.Series:
        return a / (b.abs() + eps)

    # Trend signals
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

    # Relative-level ratios
    if {"lag_1", "rolling_mean_12"} <= cols:
        X["lag1_vs_annual"] = _div(X["lag_1"], X["rolling_mean_12"])
    if {"lag_1", "rolling_std_12"} <= cols:
        X["lag1_z"] = _div(X["lag_1"], X["rolling_std_12"])
    if {"rolling_std_3", "rolling_std_12"} <= cols:
        X["vol_ratio"] = _div(X["rolling_std_3"], X["rolling_std_12"])

    # Neighbour signals
    if {"sosedi_lag_1_sum", "lag_1"} <= cols:
        X["neighbor_vs_self"] = _div(X["sosedi_lag_1_sum"], X["lag_1"])
    if {"sosedi_lag_1_mean", "sosedi_rolling_mean_12_mean"} <= cols:
        X["neighbor_trend"] = _div(X["sosedi_lag_1_mean"], X["sosedi_rolling_mean_12_mean"])

    # Activity flags
    if "lag_1" in cols:
        X["has_lag1"] = (X["lag_1"] > 0).astype(np.float32)
    if "rolling_mean_3" in cols:
        X["has_recent_harvest"] = (X["rolling_mean_3"] > 0).astype(np.float32)
    if "sosedi_lag_1_sum" in cols:
        X["neighbor_active"] = (X["sosedi_lag_1_sum"] > 0).astype(np.float32)

    # Seasonal interactions
    if {"mesec_sin", "lag_1"} <= cols:
        X["season_sin_x_lag1"] = X["mesec_sin"] * X["lag_1"]
    if {"mesec_cos", "lag_1"} <= cols:
        X["season_cos_x_lag1"] = X["mesec_cos"] * X["lag_1"]

    # Rolling max proxy from lag checkpoints
    _lag_12 = [c for c in ["lag_1", "lag_3", "lag_6", "lag_12"] if c in cols]
    if len(_lag_12) >= 2:
        X["max_recent_12m"]   = X[_lag_12].max(axis=1)
        X["any_harvest_12m"]  = (X[_lag_12].max(axis=1) > 0).astype(np.float32)
        X["harvest_freq_12m"] = (X[_lag_12] > 0).sum(axis=1) / len(_lag_12)
        X["total_recent_12m"] = X[_lag_12].sum(axis=1)

    _lag_24 = [c for c in ["lag_1", "lag_3", "lag_6", "lag_12", "lag_24"] if c in cols]
    if len(_lag_24) >= 3:
        X["max_recent_24m"] = X[_lag_24].max(axis=1)

    # Year-over-year ratio
    if {"lag_12", "lag_24"} <= cols:
        X["yoy_ratio"] = _div(X["lag_12"], X["lag_24"])

    # Neighbor peak activity
    _sosedi_lags = [c for c in ["sosedi_lag_1_sum", "sosedi_lag_3_sum", "sosedi_lag_6_sum"] if c in cols]
    if len(_sosedi_lags) >= 2:
        X["neighbor_max_recent"] = X[_sosedi_lags].max(axis=1)
        X["neighbor_any_recent"] = (X[_sosedi_lags].max(axis=1) > 0).astype(np.float32)

    # Weather drought stress proxy
    if {"leto_povp_T_avg", "leto_padavine_skupaj_mm"} <= cols:
        X["drought_stress"] = X["leto_povp_T_avg"] / (X["leto_padavine_skupaj_mm"].clip(lower=1) / 100)

    # Neighbor × self activity interaction
    if {"max_recent_12m", "neighbor_max_recent"} <= set(X.columns):
        X["neighbor_self_activity"] = X["neighbor_max_recent"] * X["max_recent_12m"]

    return X


# ---------------------------------------------------------------------------
# Two-stage per-horizon model
# ---------------------------------------------------------------------------

_CLF_DEFAULTS: dict = dict(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    metric="auc",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

_REG_DEFAULTS: dict = dict(
    objective="huber",
    alpha=0.9,
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
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


class TwoStageHorizonModel:
    """LightGBM two-stage model for a single forecast horizon.

    Stage 1 (classifier): binary — is harvest > 0?
    Stage 2 (regressor) : magnitude in log1p space, trained on non-zero rows only.
    Prediction          : (clf_prob >= threshold) * max(reg_pred, 0)
    """

    def __init__(
        self,
        scale_pos_weight: float = 1.0,
        clf_overrides: Optional[dict] = None,
        reg_overrides: Optional[dict] = None,
    ):
        # beta=2.0 strongly favours recall — we'd rather predict non-zero than miss an event
        self._fbeta = 1.5

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

        _cbs = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=-1)]
        self.clf.fit(X_train, y_tr_bin, eval_set=[(X_val, y_v_bin)], callbacks=_cbs)
        self.threshold = self._find_threshold(X_val, y_v_bin)

        nz_tr = y_train > 0
        nz_v  = y_val   > 0
        if nz_tr.sum() > 0:
            _cbs_reg = [lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=-1)]
            if nz_v.sum() > 0:
                self.reg.fit(X_train[nz_tr], y_train[nz_tr], eval_set=[(X_val[nz_v], y_val[nz_v])], callbacks=_cbs_reg)
            else:
                self.reg.fit(X_train[nz_tr], y_train[nz_tr])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        clf_prob = self.clf.predict_proba(X)[:, 1]
        reg_pred = np.maximum(self.reg.predict(X), 0.0)
        return np.where(clf_prob >= self.threshold, reg_pred, 0.0)

    def _find_threshold(self, X_val: pd.DataFrame, y_val_bin: np.ndarray) -> float:
        """Grid-search threshold to maximise F_beta on val set."""
        probs = self.clf.predict_proba(X_val)[:, 1]
        best_thresh, best_score = 0.5, -1.0
        for thresh in np.linspace(0.01, 0.99, 99):
            score = fbeta_score(y_val_bin, (probs >= thresh).astype(int), beta=self._fbeta, zero_division=0)
            if score > best_score:
                best_score, best_thresh = score, float(thresh)
        return best_thresh
