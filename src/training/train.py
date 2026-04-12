"""Train 12 horizon classifiers for bark-beetle prediction.

This script:
1. Loads train/val feature splits and target matrix.
2. Merges targets on ['ggo', 'odsek', 'leto_mesec'].
3. Trains one imbalance-aware binary classifier per horizon (h1..h12).
4. Tunes a decision threshold on validation data for best F1.
5. Logs validation metrics (F1, Precision, Recall, ROC-AUC).
6. Saves all trained artifacts to models/model.pkl.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


RANDOM_SEED = 42
INDEX_COLS = ["ggo", "odsek", "leto_mesec"]
TARGET_COLS = [f"h{i}" for i in range(1, 13)]
LEAKY_COLS = {"datum", "leto"}


def parse_args() -> argparse.Namespace:
	root = Path(__file__).resolve().parents[2]
	parser = argparse.ArgumentParser(description="Train multi-horizon imbalance-aware classifiers")
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
		default=300,
		help="Number of trees in each RandomForest classifier",
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


def _target_frame(target_df: pd.DataFrame, index_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
	require_columns(target_df, index_cols + target_cols, "target CSV")
	target_norm = normalize_index_columns(target_df, index_cols)
	subset = target_norm[index_cols + target_cols].copy()
	subset[target_cols] = subset[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
	return subset


def merge_features_with_target(
	feature_df: pd.DataFrame,
	target_df: pd.DataFrame,
	index_cols: list[str],
	target_cols: list[str],
	split_name: str,
) -> pd.DataFrame:
	require_columns(feature_df, index_cols, f"{split_name} features")
	feat_norm = normalize_index_columns(feature_df, index_cols)

	leakage_cols = [col for col in target_cols if col in feat_norm.columns]
	if leakage_cols:
		logging.warning(
			"%s already contains target horizons %s; dropping before merge.",
			split_name,
			leakage_cols,
		)
		feat_norm = feat_norm.drop(columns=leakage_cols)

	merged = feat_norm.merge(target_df[index_cols + target_cols], on=index_cols, how="inner")
	if merged.empty:
		raise ValueError(f"{split_name} merge produced 0 rows. Check index consistency.")

	logging.info(
		"%s rows: source=%d, merged=%d",
		split_name,
		len(feat_norm),
		len(merged),
	)
	return merged


def build_feature_matrix(df: pd.DataFrame, index_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
	drop_cols = set(index_cols) | set(target_cols) | LEAKY_COLS
	candidate = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
	numeric = candidate.select_dtypes(include=[np.number]).copy()
	if numeric.shape[1] == 0:
		raise ValueError("No numeric feature columns left after preprocessing.")
	return numeric


def to_binary_target(series: pd.Series) -> np.ndarray:
	values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
	return (values > 0.0).astype(np.int8)


def class_weight_dict(y: np.ndarray) -> dict[int, float] | None:
	unique = np.unique(y)
	if unique.size < 2:
		return None
	classes = np.array([0, 1], dtype=np.int8)
	weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
	return {0: float(weights[0]), 1: float(weights[1])}


def build_classifier(n_estimators: int, seed: int, weight_dict: dict[int, float] | None) -> Pipeline:
	clf = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=20,
		min_samples_leaf=2,
		n_jobs=-1,
		random_state=seed,
		class_weight=weight_dict,
	)
	return Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("model", clf),
		]
	)


def positive_class_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
	proba = model.predict_proba(x)
	classes = model.named_steps["model"].classes_
	if 1 in classes:
		idx = int(np.where(classes == 1)[0][0])
		return proba[:, idx]
	return np.zeros(len(x), dtype=float)


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
	best_t = 0.5
	best_f1 = -1.0
	for threshold in np.linspace(0.05, 0.95, 37):
		y_pred = (y_prob >= threshold).astype(np.int8)
		score = f1_score(y_true, y_pred, zero_division=0)
		if score > best_f1:
			best_f1 = score
			best_t = float(threshold)
	return best_t


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
	metrics: dict[str, float] = {
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"positive_rate_true": float(np.mean(y_true)),
		"positive_rate_pred": float(np.mean(y_pred)),
	}
	if np.unique(y_true).size < 2:
		metrics["roc_auc"] = float("nan")
	else:
		metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
	return metrics


def train_models(args: argparse.Namespace) -> None:
	train_raw = load_csv(args.train_path, "train split")
	val_raw = load_csv(args.val_path, "validation split")
	target_raw = load_csv(args.target_path, "target matrix")

	target_df = _target_frame(target_raw, INDEX_COLS, TARGET_COLS)

	train_df = merge_features_with_target(
		train_raw,
		target_df,
		INDEX_COLS,
		TARGET_COLS,
		split_name="train",
	)
	val_df = merge_features_with_target(
		val_raw,
		target_df,
		INDEX_COLS,
		TARGET_COLS,
		split_name="val",
	)

	x_train = build_feature_matrix(train_df, INDEX_COLS, TARGET_COLS)
	x_val = build_feature_matrix(val_df, INDEX_COLS, TARGET_COLS)

	x_train = x_train.reindex(sorted(x_train.columns), axis=1)
	x_val = x_val.reindex(columns=x_train.columns)

	model_bundle: dict[str, Any] = {
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
		y_train = to_binary_target(train_df[horizon])
		y_val = to_binary_target(val_df[horizon])

		unique_train = np.unique(y_train)
		if unique_train.size < 2:
			constant_prediction = int(unique_train[0])
			y_prob_val = np.full(len(y_val), float(constant_prediction), dtype=float)
			y_pred_val = np.full(len(y_val), constant_prediction, dtype=np.int8)
			threshold = 0.5
			artifact = {
				"estimator": None,
				"threshold": threshold,
				"constant_prediction": constant_prediction,
			}
		else:
			weights = class_weight_dict(y_train)
			estimator = build_classifier(
				n_estimators=args.n_estimators,
				seed=RANDOM_SEED + i,
				weight_dict=weights,
			)
			estimator.fit(x_train, y_train)
			y_prob_val = positive_class_probability(estimator, x_val)
			threshold = tune_threshold(y_val, y_prob_val)
			y_pred_val = (y_prob_val >= threshold).astype(np.int8)
			artifact = {
				"estimator": estimator,
				"threshold": threshold,
				"constant_prediction": None,
			}

		model_bundle["models"][horizon] = artifact
		metrics = evaluate_binary(y_val, y_pred_val, y_prob_val)
		metric_rows.append({"horizon": horizon, **metrics})

		logging.info(
			"[%02d/12] %s | F1=%.4f P=%.4f R=%.4f ROC-AUC=%s threshold=%.2f",
			i,
			horizon,
			metrics["f1"],
			metrics["precision"],
			metrics["recall"],
			"nan" if np.isnan(metrics["roc_auc"]) else f"{metrics['roc_auc']:.4f}",
			threshold,
		)

	metrics_df = pd.DataFrame(metric_rows)
	logging.info("Validation metrics by horizon:\n%s", metrics_df.round(4).to_string(index=False))

	macro = metrics_df[["f1", "precision", "recall", "roc_auc"]].mean(numeric_only=True)
	logging.info(
		"Validation macro averages | F1=%.4f Precision=%.4f Recall=%.4f ROC-AUC=%.4f",
		macro["f1"],
		macro["precision"],
		macro["recall"],
		macro["roc_auc"],
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
