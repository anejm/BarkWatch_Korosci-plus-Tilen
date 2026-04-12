"""Run test-set inference using models/model.pkl.

This script:
1. Loads the trained model artifact and test split.
2. Builds the feature matrix with the same training-time rules.
3. Produces h1..h12 predictions.
4. Writes predictions/test_predictions.csv while preserving
   ['ggo', 'odsek', 'leto_mesec'].
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


INDEX_COLS = ["ggo", "odsek", "leto_mesec"]
TARGET_COLS = [f"h{i}" for i in range(1, 13)]
LEAKY_COLS = {"datum", "leto"}


def parse_args() -> argparse.Namespace:
	root = Path(__file__).resolve().parents[2]
	parser = argparse.ArgumentParser(description="Run multi-horizon test inference")
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
		help="Optional path for evaluation-only metrics",
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
	return candidate.select_dtypes(include=[np.number]).copy()


def positive_class_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
	proba = model.predict_proba(x)
	classes = model.named_steps["model"].classes_
	if 1 in classes:
		idx = int(np.where(classes == 1)[0][0])
		return proba[:, idx]
	return np.zeros(len(x), dtype=float)


def run_inference(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str]]:
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

	if x_test.shape[1] != len(feature_columns):
		raise ValueError("Could not align test features to training schema.")

	predictions: dict[str, np.ndarray] = {}
	for horizon in target_cols:
		if horizon not in models:
			raise ValueError(f"Model for horizon {horizon} not found in artifact.")

		model_info = models[horizon]
		constant_prediction = model_info.get("constant_prediction")
		if constant_prediction is not None:
			pred = np.full(len(x_test), int(constant_prediction), dtype=np.int8)
		else:
			estimator: Pipeline = model_info["estimator"]
			threshold = float(model_info.get("threshold", 0.5))
			prob = positive_class_probability(estimator, x_test)
			pred = (prob >= threshold).astype(np.int8)
		predictions[horizon] = pred

	pred_df = pd.concat(
		[
			test_df[index_cols].reset_index(drop=True),
			pd.DataFrame(predictions),
		],
		axis=1,
	)

	args.output_path.parent.mkdir(parents=True, exist_ok=True)
	pred_df.to_csv(args.output_path, index=False)
	logging.info("Saved predictions to %s", args.output_path)
	return pred_df, index_cols, target_cols


def maybe_evaluate(
	prediction_df: pd.DataFrame,
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

	merged = prediction_df.merge(
		target_df[index_cols + target_cols],
		on=index_cols,
		how="inner",
		suffixes=("_pred", "_true"),
	)
	if merged.empty:
		logging.info("Skipping evaluation: no overlapping rows between predictions and target.")
		return

	metric_rows: list[dict[str, float | str]] = []
	for horizon in target_cols:
		y_pred = pd.to_numeric(merged[f"{horizon}_pred"], errors="coerce").fillna(0).astype(int).to_numpy()
		y_true = (
			pd.to_numeric(merged[f"{horizon}_true"], errors="coerce").fillna(0.0).to_numpy() > 0.0
		).astype(int)
		metric_rows.append(
			{
				"horizon": horizon,
				"f1": float(f1_score(y_true, y_pred, zero_division=0)),
				"precision": float(precision_score(y_true, y_pred, zero_division=0)),
				"recall": float(recall_score(y_true, y_pred, zero_division=0)),
			}
		)

	metrics_df = pd.DataFrame(metric_rows)
	logging.info("Test classification metrics by horizon:\n%s", metrics_df.round(4).to_string(index=False))


def main() -> None:
	setup_logging()
	args = parse_args()
	pred_df, index_cols, target_cols = run_inference(args)

	if not args.no_eval:
		maybe_evaluate(pred_df, args.target_path, index_cols, target_cols)


if __name__ == "__main__":
	main()
