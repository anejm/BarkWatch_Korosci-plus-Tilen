"""Train one-stage horizon regressors with zero-inflation tweaks.

Modeling strategy:
1. Train one RandomForestRegressor per horizon (h1..h12).
2. Upweight non-zero training rows to fight heavy zero inflation.
3. Tune a per-horizon cutoff on validation predictions:
   predictions below this cutoff are set to zero.

This remains one-stage because each horizon uses a single regressor model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
	f1_score,
	mean_absolute_error,
	precision_score,
	recall_score,
	roc_auc_score,
	root_mean_squared_error,
)
from sklearn.pipeline import Pipeline


RANDOM_SEED = 42
INDEX_COLS = ["ggo", "odsek", "leto_mesec"]
TARGET_COLS = [f"h{i}" for i in range(1, 13)]
LEAKY_COLS = {"datum", "leto", "target", "log1p_target"}


def parse_args() -> argparse.Namespace:
	root = Path(__file__).resolve().parents[2]
	parser = argparse.ArgumentParser(
		description="Train one-stage zero-inflated multi-horizon regressors"
	)
	parser.add_argument(
		"--train-path",
		type=Path,
		default=root / "data" / "processed" / "splits" / "train.csv",
		help="Path to training features CSV",
	)
	parser.add_argument(
		"--val-path",
		type=Path,
		default=root / "data" / "processed" / "splits" / "val.csv",
		help="Path to validation features CSV",
	)
	parser.add_argument(
		"--target-path",
		type=Path,
		default=root / "data" / "processed" / "target.csv",
		help="Path to target CSV with h1..h12",
	)
	parser.add_argument(
		"--output-model",
		type=Path,
		default=root / "models" / "model.pkl",
		help="Output model artifact path",
	)
	parser.add_argument(
		"--n-estimators",
		type=int,
		default=500,
		help="Number of trees per horizon regressor",
	)
	parser.add_argument(
		"--max-depth",
		type=int,
		default=20,
		help="Maximum tree depth in each regressor",
	)
	parser.add_argument(
		"--min-samples-leaf",
		type=int,
		default=2,
		help="Minimum samples in each tree leaf",
	)
	parser.add_argument(
		"--positive-weight-scale",
		type=float,
		default=0.5,
		help=(
			"Non-zero sample weight boost exponent. "
			"Boost = (n_zero / n_nonzero)^scale"
		),
	)
	parser.add_argument(
		"--threshold-grid-size",
		type=int,
		default=61,
		help="Number of validation cutoffs to test per horizon",
	)
	parser.add_argument(
		"--threshold-max-quantile",
		type=float,
		default=0.995,
		help="Upper quantile of validation predictions used in cutoff search",
	)
	parser.add_argument(
		"--min-threshold",
		type=float,
		default=0.0,
		help="Lower bound for zero cutoff search",
	)
	return parser.parse_args()


def setup_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%H:%M:%S",
	)


def load_csv(path: Path, name: str) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Missing {name}: {path}")
	return pd.read_csv(path, low_memory=False)


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(f"{name} missing required columns: {missing}")


def normalize_index_columns(df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
	out = df.copy()
	for col in index_cols:
		out[col] = out[col].astype(str).str.strip()
	return out


def build_target_frame(target_df: pd.DataFrame) -> pd.DataFrame:
	require_columns(target_df, INDEX_COLS + TARGET_COLS, "target CSV")
	target_norm = normalize_index_columns(target_df, INDEX_COLS)
	subset = target_norm[INDEX_COLS + TARGET_COLS].copy()
	subset[TARGET_COLS] = (
		subset[TARGET_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
	)
	subset[TARGET_COLS] = subset[TARGET_COLS].clip(lower=0.0)
	return subset


def merge_features_with_target(
	feature_df: pd.DataFrame,
	target_df: pd.DataFrame,
	split_name: str,
) -> pd.DataFrame:
	require_columns(feature_df, INDEX_COLS, f"{split_name} features")
	feat_norm = normalize_index_columns(feature_df, INDEX_COLS)

	leakage_cols = [col for col in TARGET_COLS if col in feat_norm.columns]
	if leakage_cols:
		logging.warning(
			"%s split contains horizon targets %s; dropping before merge.",
			split_name,
			leakage_cols,
		)
		feat_norm = feat_norm.drop(columns=leakage_cols)

	merged = feat_norm.merge(target_df[INDEX_COLS + TARGET_COLS], on=INDEX_COLS, how="inner")
	if merged.empty:
		raise ValueError(f"{split_name} merge produced 0 rows. Check index consistency.")

	logging.info("%s rows: source=%d merged=%d", split_name, len(feat_norm), len(merged))
	return merged


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
	drop_cols = set(INDEX_COLS) | set(TARGET_COLS) | LEAKY_COLS
	candidate = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
	numeric = candidate.select_dtypes(include=[np.number]).copy()
	if numeric.shape[1] == 0:
		raise ValueError("No numeric feature columns left after preprocessing.")
	return numeric


def to_regression_target(series: pd.Series) -> np.ndarray:
	values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
	return np.clip(values, 0.0, None)


def build_sample_weights(y: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
	non_zero_mask = y > 0
	n_non_zero = int(non_zero_mask.sum())
	n_zero = int((~non_zero_mask).sum())

	if n_non_zero == 0 or n_zero == 0:
		return np.ones_like(y, dtype=float), 1.0

	boost = max(1.0, (n_zero / max(n_non_zero, 1)) ** scale)
	weights = np.ones_like(y, dtype=float)
	weights[non_zero_mask] = boost
	return weights, float(boost)


def build_regressor(args: argparse.Namespace, seed: int) -> Pipeline:
	model = RandomForestRegressor(
		n_estimators=args.n_estimators,
		max_depth=args.max_depth,
		min_samples_leaf=args.min_samples_leaf,
		random_state=seed,
		n_jobs=-1,
	)
	return Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("model", model),
		]
	)


def tune_zero_threshold(
	y_true: np.ndarray,
	y_pred_raw: np.ndarray,
	grid_size: int,
	min_threshold: float,
	max_quantile: float,
) -> float:
	y_true_bin = (y_true > 0).astype(np.int8)

	if len(y_pred_raw) == 0:
		return float(min_threshold)

	upper = float(np.quantile(y_pred_raw, np.clip(max_quantile, 0.5, 1.0)))
	upper = max(upper, min_threshold)

	if upper == min_threshold:
		candidates = np.array([float(min_threshold)], dtype=float)
	else:
		candidates = np.linspace(float(min_threshold), upper, max(3, grid_size))

	best_threshold = float(min_threshold)
	best_f1 = -1.0
	best_precision = -1.0

	for threshold in candidates:
		y_pred_bin = (y_pred_raw >= threshold).astype(np.int8)
		f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
		precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)

		if f1 > best_f1 or (np.isclose(f1, best_f1) and precision > best_precision):
			best_f1 = float(f1)
			best_precision = float(precision)
			best_threshold = float(threshold)

	return best_threshold


def apply_zero_cutoff(y_pred_raw: np.ndarray, cutoff: float) -> np.ndarray:
	y_pred = np.clip(y_pred_raw, 0.0, None)
	return np.where(y_pred >= cutoff, y_pred, 0.0)


def evaluate_horizon(
	y_true: np.ndarray,
	y_pred_raw: np.ndarray,
	cutoff: float,
) -> dict[str, float]:
	y_pred = apply_zero_cutoff(y_pred_raw, cutoff)
	y_true_bin = (y_true > 0).astype(np.int8)
	y_pred_bin = (y_pred > 0).astype(np.int8)

	metrics: dict[str, float] = {
		"f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
		"precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
		"recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"rmse": float(root_mean_squared_error(y_true, y_pred)),
		"true_non_zero_rate": float(np.mean(y_true_bin)),
		"pred_non_zero_rate": float(np.mean(y_pred_bin)),
	}

	if np.unique(y_true_bin).size < 2:
		metrics["roc_auc"] = float("nan")
	else:
		metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_pred_raw))

	mask = y_true > 0
	if mask.any():
		metrics["non_zero_mae"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
		metrics["non_zero_rmse"] = float(root_mean_squared_error(y_true[mask], y_pred[mask]))
	else:
		metrics["non_zero_mae"] = float("nan")
		metrics["non_zero_rmse"] = float("nan")

	return metrics


def train_models(args: argparse.Namespace) -> None:
	train_raw = load_csv(args.train_path, "train split")
	val_raw = load_csv(args.val_path, "validation split")
	target_raw = load_csv(args.target_path, "target matrix")

	target_df = build_target_frame(target_raw)
	train_df = merge_features_with_target(train_raw, target_df, split_name="train")
	val_df = merge_features_with_target(val_raw, target_df, split_name="val")

	x_train = build_feature_matrix(train_df)
	x_val = build_feature_matrix(val_df)

	x_train = x_train.reindex(sorted(x_train.columns), axis=1)
	x_val = x_val.reindex(columns=x_train.columns)

	model_bundle: dict[str, Any] = {
		"model_type": "one_stage_zero_inflated_random_forest_regression",
		"zero_inflation_tweaks": {
			"positive_weight_scale": float(args.positive_weight_scale),
			"cutoff_tuning": "validation F1 over threshold grid",
		},
		"index_cols": INDEX_COLS,
		"target_cols": TARGET_COLS,
		"feature_columns": list(x_train.columns),
		"leaky_columns": sorted(LEAKY_COLS),
		"random_seed": RANDOM_SEED,
		"models": {},
	}

	metric_rows: list[dict[str, float | str]] = []

	logging.info("Training with %d rows and %d numeric features", len(x_train), x_train.shape[1])

	for i, horizon in enumerate(TARGET_COLS, start=1):
		y_train = to_regression_target(train_df[horizon])
		y_val = to_regression_target(val_df[horizon])

		unique_train = np.unique(y_train)
		if unique_train.size == 1:
			constant_value = float(unique_train[0])
			y_pred_raw_val = np.full(len(y_val), constant_value, dtype=float)
			cutoff = tune_zero_threshold(
				y_true=y_val,
				y_pred_raw=y_pred_raw_val,
				grid_size=args.threshold_grid_size,
				min_threshold=args.min_threshold,
				max_quantile=args.threshold_max_quantile,
			)
			artifact = {
				"estimator": None,
				"constant_prediction": constant_value,
				"zero_cutoff": cutoff,
				"positive_weight": 1.0,
			}
		else:
			sample_weight, positive_weight = build_sample_weights(y_train, args.positive_weight_scale)
			estimator = build_regressor(args, seed=RANDOM_SEED + i)
			estimator.fit(x_train, y_train, model__sample_weight=sample_weight)

			y_pred_raw_val = np.clip(estimator.predict(x_val), 0.0, None)
			cutoff = tune_zero_threshold(
				y_true=y_val,
				y_pred_raw=y_pred_raw_val,
				grid_size=args.threshold_grid_size,
				min_threshold=args.min_threshold,
				max_quantile=args.threshold_max_quantile,
			)
			artifact = {
				"estimator": estimator,
				"constant_prediction": None,
				"zero_cutoff": cutoff,
				"positive_weight": positive_weight,
			}

		model_bundle["models"][horizon] = artifact

		metrics = evaluate_horizon(y_val, y_pred_raw_val, cutoff)
		metric_rows.append({"horizon": horizon, "cutoff": cutoff, **metrics})

		logging.info(
			"[%02d/12] %s | cutoff=%.5f F1=%.4f P=%.4f R=%.4f AUC=%s MAE=%.4f RMSE=%.4f",
			i,
			horizon,
			cutoff,
			metrics["f1"],
			metrics["precision"],
			metrics["recall"],
			"nan" if np.isnan(metrics["roc_auc"]) else f"{metrics['roc_auc']:.4f}",
			metrics["mae"],
			metrics["rmse"],
		)

	metrics_df = pd.DataFrame(metric_rows)
	logging.info("Validation metrics by horizon:\n%s", metrics_df.round(4).to_string(index=False))

	macro = metrics_df[["f1", "precision", "recall", "roc_auc", "mae", "rmse"]].mean(
		numeric_only=True
	)
	logging.info(
		"Validation macro averages | F1=%.4f P=%.4f R=%.4f AUC=%.4f MAE=%.4f RMSE=%.4f",
		macro["f1"],
		macro["precision"],
		macro["recall"],
		macro["roc_auc"],
		macro["mae"],
		macro["rmse"],
	)

	args.output_model.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model_bundle, args.output_model)
	logging.info("Saved trained artifact to %s", args.output_model)


def main() -> None:
	setup_logging()
	args = parse_args()
	train_models(args)


if __name__ == "__main__":
	main()
