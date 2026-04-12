"""Run one-stage zero-inflated regression inference using models/model.pkl."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
	f1_score,
	mean_absolute_error,
	precision_score,
	recall_score,
	roc_auc_score,
	root_mean_squared_error,
)
from sklearn.pipeline import Pipeline


INDEX_COLS = ["ggo", "odsek", "leto_mesec"]
TARGET_COLS = [f"h{i}" for i in range(1, 13)]
LEAKY_COLS = {"datum", "leto", "target", "log1p_target"}


def parse_args() -> argparse.Namespace:
	root = Path(__file__).resolve().parents[2]
	parser = argparse.ArgumentParser(description="Run one-stage zero-inflated test inference")
	parser.add_argument(
		"--test-path",
		type=Path,
		default=root / "data" / "processed" / "splits" / "test.csv",
		help="Path to test features CSV",
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=root / "models" / "model.pkl",
		help="Path to trained model artifact",
	)
	parser.add_argument(
		"--output-path",
		type=Path,
		default=root / "predictions" / "test_predictions.csv",
		help="Path for prediction CSV output",
	)
	parser.add_argument(
		"--target-path",
		type=Path,
		default=root / "data" / "processed" / "target.csv",
		help="Optional path for evaluation",
	)
	parser.add_argument(
		"--no-eval",
		action="store_true",
		help="Skip optional evaluation even if target file exists",
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


def build_feature_matrix(df: pd.DataFrame, index_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
	drop_cols = set(index_cols) | set(target_cols) | LEAKY_COLS
	candidate = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
	numeric = candidate.select_dtypes(include=[np.number]).copy()
	if numeric.shape[1] == 0:
		raise ValueError("No numeric feature columns found in test split.")
	return numeric


def apply_zero_cutoff(y_pred_raw: np.ndarray, cutoff: float) -> np.ndarray:
	y_pred = np.clip(y_pred_raw, 0.0, None)
	return np.where(y_pred >= cutoff, y_pred, 0.0)


def run_inference(
	args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
	if not args.model_path.exists():
		raise FileNotFoundError(f"Model artifact not found: {args.model_path}")

	bundle: dict[str, Any] = joblib.load(args.model_path)
	index_cols = bundle.get("index_cols", INDEX_COLS)
	target_cols = bundle.get("target_cols", TARGET_COLS)
	feature_columns = bundle.get("feature_columns", [])
	models = bundle.get("models", {})

	if not feature_columns:
		raise ValueError("Loaded artifact has no feature_columns metadata.")

	test_raw = load_csv(args.test_path, "test split")
	require_columns(test_raw, index_cols, "test split")
	test_df = normalize_index_columns(test_raw, index_cols)

	x_test_raw = build_feature_matrix(test_df, index_cols, target_cols)
	x_test = x_test_raw.reindex(columns=feature_columns)

	thresholded_predictions: dict[str, np.ndarray] = {}
	raw_predictions: dict[str, np.ndarray] = {}

	for horizon in target_cols:
		if horizon not in models:
			raise ValueError(f"Model for horizon {horizon} not found in artifact.")

		model_info = models[horizon]
		cutoff = float(model_info.get("zero_cutoff", 0.0))
		constant_prediction = model_info.get("constant_prediction")

		if constant_prediction is not None:
			y_raw = np.full(len(x_test), float(constant_prediction), dtype=float)
		else:
			estimator: Pipeline = model_info["estimator"]
			y_raw = np.clip(estimator.predict(x_test), 0.0, None)

		y_pred = apply_zero_cutoff(y_raw, cutoff)
		raw_predictions[horizon] = y_raw
		thresholded_predictions[horizon] = y_pred

	prediction_df = pd.concat(
		[
			test_df[index_cols].reset_index(drop=True),
			pd.DataFrame(thresholded_predictions),
		],
		axis=1,
	)
	raw_df = pd.concat(
		[
			test_df[index_cols].reset_index(drop=True),
			pd.DataFrame({f"{k}_raw": v for k, v in raw_predictions.items()}),
		],
		axis=1,
	)

	args.output_path.parent.mkdir(parents=True, exist_ok=True)
	prediction_df.to_csv(args.output_path, index=False)
	logging.info("Saved predictions to %s", args.output_path)
	return prediction_df, raw_df, index_cols, target_cols


def maybe_evaluate(
	prediction_df: pd.DataFrame,
	raw_df: pd.DataFrame,
	target_path: Path,
	index_cols: list[str],
	target_cols: list[str],
) -> None:
	if not target_path.exists():
		logging.info("Skipping evaluation: target file not found at %s", target_path)
		return

	target_df = load_csv(target_path, "target matrix")
	require_columns(target_df, index_cols + target_cols, "target matrix")
	target_df = normalize_index_columns(target_df, index_cols)

	merged = prediction_df.merge(raw_df, on=index_cols, how="inner").merge(
		target_df[index_cols + target_cols], on=index_cols, how="inner", suffixes=("_pred", "_true")
	)
	if merged.empty:
		logging.info("Skipping evaluation: no overlapping rows between predictions and target.")
		return

	metric_rows: list[dict[str, float | str]] = []

	for horizon in target_cols:
		y_true = pd.to_numeric(merged[f"{horizon}_true"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
		y_true = np.clip(y_true, 0.0, None)

		y_pred = pd.to_numeric(merged[f"{horizon}_pred"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
		y_raw = pd.to_numeric(merged[f"{horizon}_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

		y_true_bin = (y_true > 0).astype(np.int8)
		y_pred_bin = (y_pred > 0).astype(np.int8)

		row: dict[str, float | str] = {
			"horizon": horizon,
			"f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
			"precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
			"recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
			"mae": float(mean_absolute_error(y_true, y_pred)),
			"rmse": float(root_mean_squared_error(y_true, y_pred)),
		}

		if np.unique(y_true_bin).size < 2:
			row["roc_auc"] = float("nan")
		else:
			row["roc_auc"] = float(roc_auc_score(y_true_bin, y_raw))

		nz = y_true > 0
		if nz.any():
			row["non_zero_mae"] = float(mean_absolute_error(y_true[nz], y_pred[nz]))
			row["non_zero_rmse"] = float(root_mean_squared_error(y_true[nz], y_pred[nz]))
		else:
			row["non_zero_mae"] = float("nan")
			row["non_zero_rmse"] = float("nan")

		metric_rows.append(row)

	metrics_df = pd.DataFrame(metric_rows)
	logging.info("Test metrics by horizon:\n%s", metrics_df.round(4).to_string(index=False))


def main() -> None:
	setup_logging()
	args = parse_args()
	prediction_df, raw_df, index_cols, target_cols = run_inference(args)

	if not args.no_eval:
		maybe_evaluate(prediction_df, raw_df, args.target_path, index_cols, target_cols)


if __name__ == "__main__":
	main()
