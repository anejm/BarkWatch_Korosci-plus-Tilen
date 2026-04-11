# BarkWatch-arnes_hackathon2026

## Project Structure

```bash
.
├── api/                  # API keys
├── config/               # Configuration files
│   └── config.json
├── data/
│   ├── processed/        # Cleaned / preprocessed data ready for training
│   └── raw/              # Original raw data
│       ├── ARSO/
│       └── ZGS/
│           └── images/
├── frontend/             # Frontend code
├── models/               # Trained models saved for inference
├── notebooks/            # Jupyter notebooks for experimentation and prototyping
└── src/
    ├── data_processing/  # Scripts to preprocess, clean, or transform data
    ├── inference/        # Scripts to run model predictions
    ├── training/         # Scripts to train models
    └── utils/            # Utility functions and helper scripts

```

## Key Scripts

### Inference pipeline

**`src/extract_current_day_data.py`**
Reads the full test split (`data/processed/splits/test.csv`) and extracts the single most recent row for every `(ggo, odsek)` pair. Saves the result to `data/processed/current_state.csv`. Run this before `predict_the_future.py`.

**`src/predict_the_future.py`**
Loads the current-state snapshot and the trained models, then runs sequential two-stage inference for horizons h1–h12. Supports scenario overrides via CLI flags:
- `--padavine [malo|normalno|veliko]` — scale rainfall features (×0.25 / ×1 / ×2)
- `--temperatura [nizko|normalno|visoko]` — shift temperature features (−3 / 0 / +3 °C)
- `--h1 FACTOR` — multiply the h1 prediction before it feeds into subsequent models

Output: `data/predictions/future_predictions.csv` with columns `ggo, odsek, leto_mesec, h1_pred … h12_pred` (values in m³).

**`src/utils/generating_heatmap_data.py`**
Produces two flat tables for the frontend heatmap visualization:
- `heatmap_past_data.csv` — historical actual harvest reconstructed from `target.csv` (h1–h12 unpivoted, deduplicated to one row per parcel-month)
- `heatmap_future_predictions.csv` — future predictions from `future_predictions.csv` expanded to one row per parcel-month

Both share the schema `ggo, odsek_id, leto_mesec, target, is_a_prediction`.

### Training

**`src/training/train.py`**
Trains 12 sequential LightGBM two-stage models (one per horizon h1–h12). Each model uses base features plus predictions from all previous horizons. Stage 1 is a classifier (is harvest > 0?); Stage 2 is a Huber-loss regressor on non-zero rows. Thresholds are tuned via F-beta maximisation on the validation set (beta=0.7 for h8–h12 to suppress positive-bias compounding). Saves to `models/lgb_models.pkl`.

**`src/training/testing.py`**
Loads saved models and runs sequential inference on the test set. Evaluates MAE/RMSE per horizon (in m³). Saves predictions to `data/predictions/predictions.csv`.

**`models/model.py`**
Defines `TwoStageHorizonModel` (LGBMClassifier + LGBMRegressor) and the shared feature-engineering helpers `add_derived_features()` and `sanitize_columns()` used by both `train.py` and `testing.py`.

### Synthetic data

**`src/generating_synthetic_data/synthetic_pipeline.py`**
Generates synthetic monthly bark beetle population counts for every `(ggo, odsek)` parcel from 2007-01 to 2025-12 using a temperature-driven logistic growth model with harvest reduction, spatial diffusion via a neighbour graph, seasonality, and Gaussian noise. Output: `data/synthetic/bark_beetle_population.csv` (~12 M rows, columns `ggo, odsek, leto_mesec, bark_beetle_count`).

**`notebooks/synthetic_data_analysis.ipynb`**
Validates the synthetic beetle population model by measuring its correlation with real harvest data (`heatmap.csv`). Shows: direct scatter correlation (Pearson/Spearman), lead-lag cross-correlation across ±12 months, seasonal patterns, and a sample parcel time-series overlay.

