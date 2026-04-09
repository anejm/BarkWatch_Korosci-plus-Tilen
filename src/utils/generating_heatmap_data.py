"""Generate a unified heatmap dataset from observed targets and predictions.

The output is a flat CSV with the schema:
['ggo', 'odsek_id', 'leto_mesec', 'target']
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["ggo", "odsek_id", "leto_mesec"]


def _normalize_month(value: str) -> str:
	return pd.Period(str(value), freq="M").strftime("%Y-%m")


def _load_observed_targets(path: Path) -> pd.DataFrame:
	frame = pd.read_csv(path)

	rename_map = {}
	if "odsek" in frame.columns:
		rename_map["odsek"] = "odsek_id"
	if "odseki_id" in frame.columns:
		rename_map["odseki_id"] = "odsek_id"

	frame = frame.rename(columns=rename_map)

	missing = [column for column in ["ggo", "odsek_id", "leto_mesec", "target"] if column not in frame.columns]
	if missing:
		raise ValueError(f"Missing required columns in {path}: {missing}")

	observed = frame.loc[:, ["ggo", "odsek_id", "leto_mesec", "target"]].copy()
	observed["leto_mesec"] = observed["leto_mesec"].map(_normalize_month)
	return observed


def _horizon_columns(frame: pd.DataFrame) -> list[tuple[int, str]]:
	horizons: list[tuple[int, str]] = []
	for column in frame.columns:
		match = re.fullmatch(r"h(\d+)(?:_pred)?", column)
		if match:
			horizons.append((int(match.group(1)), column))
	return sorted(horizons, key=lambda item: item[0])


def _load_predictions(path: Path) -> pd.DataFrame:
	frame = pd.read_csv(path)

	rename_map = {}
	if "odsek" in frame.columns:
		rename_map["odsek"] = "odsek_id"
	if "odseki_id" in frame.columns:
		rename_map["odseki_id"] = "odsek_id"

	frame = frame.rename(columns=rename_map)

	missing = [column for column in ["ggo", "odsek_id", "leto_mesec"] if column not in frame.columns]
	if missing:
		raise ValueError(f"Missing required columns in {path}: {missing}")

	horizon_columns = _horizon_columns(frame)
	if not horizon_columns:
		raise ValueError(f"No horizon columns found in {path}")

	prediction_rows = []
	for _, row in frame.iterrows():
		base_month = pd.Period(str(row["leto_mesec"]), freq="M")
		for horizon, column in horizon_columns:
			prediction_rows.append(
				{
					"ggo": row["ggo"],
					"odsek_id": row["odsek_id"],
					"leto_mesec": (base_month + horizon).strftime("%Y-%m"),
					"target": row[column],
				}
			)

	return pd.DataFrame(prediction_rows, columns=["ggo", "odsek_id", "leto_mesec", "target"])


def generate_heatmap_data(
	posek_path: Path,
	predictions_path: Path,
	output_path: Path,
) -> pd.DataFrame:
	observed = _load_observed_targets(posek_path)
	predicted = _load_predictions(predictions_path)

	combined = pd.concat([observed, predicted], ignore_index=True)
	combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="first")
	combined = combined.sort_values(KEY_COLUMNS).reset_index(drop=True)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	combined.to_csv(output_path, index=False)
	return combined


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate heatmap CSV data.")
	parser.add_argument(
		"--posek",
		type=Path,
		default=Path("data/processed/posek_processed.csv"),
		help="Path to posek_processed.csv.",
	)
	parser.add_argument(
		"--predictions",
		type=Path,
		default=Path("data/predictions/predictions.csv"),
		help="Path to predictions.csv.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/heatmap.csv"),
		help="Where to write the combined heatmap CSV.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	generate_heatmap_data(args.posek, args.predictions, args.output)


if __name__ == "__main__":
	main()
