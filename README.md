# BarkWatch

**AI-powered bark beetle early-warning system for Slovenian forests.**

Built at the Arnes Hackathon 2026 by Korošci+Tilen.

---

## Overview

Slovenia is one of the most forested countries in Europe — roughly 58 % of its land is covered in forest. Bark beetles (*podlubniki*) are a major natural threat that can silently devastate large areas before a forester can respond. By the time visual damage appears, the infestation is already months old.

BarkWatch turns historical forest harvest records (*posek*) and meteorological observations into a 12-month forward-looking bark beetle risk forecast. Predictions are served through an interactive web map where foresters, planners, and researchers can explore the entire country at the individual forest sector (*odsek*) level.

**Web interface repo:** [BarkWatch_Arnes-Hackathon-2026_interface](../BarkWatch_Arnes-Hackathon-2026_interface)

---

## How it works

```
Raw data                 Processing              ML model               Frontend
──────────               ──────────              ────────               ────────
ZGS posek.csv      ──►  join_data.py  ──►  train.py (LightGBM)  ──►  server.py
ZGS odseki.gpkg          pipeline.py        12 two-stage models        MapLibre map
ZGS sestoji.gpkg                             h1 … h12 horizons          Chart.js
ARSO weather data   ──►  vreme_mesecno                          ──►  heatmap CSVs
Synthetic beetle pop ──►  synthetic_pipeline.py ──► synthetic models
```

### Data sources

| Source | Content |
|--------|---------|
| ZGS (*Zavod za gozdove Slovenije*) | Forest harvest volumes per sector per month (`posek.csv`), sector polygons (`odseki_gozdno.gpkg`), stand data (`sestoji.gpkg`) |
| ARSO (*Agencija RS za okolje*) | Monthly weather station measurements (temperature, precipitation) |
| Synthetic | Temperature-driven logistic growth model producing bark beetle population estimates for validation |

### Machine learning

The model is a **sequential two-stage LightGBM pipeline** — one model per forecast horizon (h1 – h12, i.e. 1 to 12 months ahead). Each horizon reuses predictions from all previous horizons as input features.

**Two-stage design per horizon:**
1. *Classifier* — is harvest volume > 0? (bark beetle activity present?)
2. *Regressor* — predict volume in m³ (Huber loss, non-zero rows only)

Thresholds are tuned via F-beta maximisation on the validation set. For longer horizons (h8 – h12) beta = 0.7 to suppress positive-bias compounding.

**Feature groups:**
- Lagged harvest volumes per sector and its neighbours
- Aggregated forest stand attributes (species mix, age, density)
- Monthly weather (temperature, precipitation) from the nearest weather station
- Derived time features (month, year, seasonal indicators)

---

## Repository structure

```
BarkWatch-arnes_hackathon2026/
├── config/
│   └── config.json                    # Shared configuration
├── data/
│   ├── raw/
│   │   ├── ARSO/                      # Raw weather station data
│   │   └── ZGS/                       # Raw harvest + GIS data
│   │       └── images/
│   ├── processed/                     # Cleaned data ready for training
│   │   ├── posek_processed.csv
│   │   ├── odseki_processed.csv
│   │   ├── sestoji_processed.csv
│   │   ├── vreme_mesecno.csv
│   │   ├── agg_posek_meritve.csv
│   │   ├── agg_posek_sosedi.csv
│   │   ├── target.csv                 # Final ML target table
│   │   ├── splits/                    # train / val / test splits
│   │   └── current_state.csv          # Latest snapshot for live inference
│   ├── synthetic/                     # Synthetic bark beetle population
│   └── predictions/                   # Model output CSVs
├── models/
│   ├── lgb_models.pkl                 # Trained LightGBM models (h1–h12)
│   └── model.py                       # TwoStageHorizonModel definition
├── notebooks/                         # Exploration and validation notebooks
├── scripts/                           # Helper scripts
├── src/
│   ├── data_processing/               # Raw → processed pipeline
│   ├── generating_synthetic_data/     # Synthetic population model
│   ├── training/
│   │   ├── train.py                   # Train all 12 horizon models
│   │   └── testing.py                 # Evaluate on test set (MAE / RMSE)
│   ├── inference/
│   ├── utils/
│   │   └── generating_heatmap_data.py # Export CSVs for the frontend
│   ├── extract_current_day_data.py    # Snapshot latest state for inference
│   ├── predict_the_future.py          # Run sequential inference h1–h12
│   └── pipeline.py                    # End-to-end processing pipeline
└── job_*.slurm                        # SLURM job scripts for HPC cluster
```

---

## Getting started

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+, LightGBM, pandas, numpy, scikit-learn, geopandas.

### Run the full pipeline

```bash
# 1. Process raw data
python src/pipeline.py

# 2. Train models
python src/training/train.py

# 3. Evaluate on test set
python src/training/testing.py

# 4. Generate heatmap export for the frontend
python src/extract_current_day_data.py
python src/predict_the_future.py
python src/utils/generating_heatmap_data.py
```

### Scenario overrides (inference)

```bash
python src/predict_the_future.py \
  --padavine veliko \       # rainfall: malo | normalno | veliko  (×0.25 / ×1 / ×2)
  --temperatura visoko \    # temperature: nizko | normalno | visoko  (−3 / 0 / +3 °C)
  --h1 1.5                  # multiply h1 prediction before feeding into h2–h12
```

### HPC / SLURM

Each processing stage has a corresponding SLURM job script:

| Script | Purpose |
|--------|---------|
| `job_pipeline_and_train.slurm` | Full pipeline + training on real data |
| `job_generate_synthetic_data.slurm` | Generate synthetic beetle population |
| `job_synthetic_pipeline_and_train.slurm` | Pipeline + training on synthetic data |
| `job_generate_heatmap.slurm` | Export heatmap CSVs (real predictions) |
| `job_generate_heatmap_synthetic.slurm` | Export heatmap CSVs (synthetic) |

---

## Key source files

### Data processing

| File | What it does |
|------|-------------|
| `src/data_processing/posek_processing.py` | Harvest CSV → monthly time series with features |
| `src/data_processing/odsek_processing.py` | GIS sector attributes → flat CSV |
| `src/data_processing/segment_processing.py` | Forest stand data → aggregated sector features |
| `src/data_processing/meritve_processing.py` | Weather measurements → monthly means |
| `src/data_processing/bliznje_vremenskepostaje.py` | Each sector → nearest weather station IDs |
| `src/data_processing/bliznji_odseki.py` | Each sector → list of neighbouring sector IDs |
| `src/data_processing/agg_posek_meritve.py` | Join harvest with weather |
| `src/data_processing/agg_posek_sosedi.py` | Add neighbour harvest features |
| `src/data_processing/join_data.py` | Merge everything into `target.csv` |

### Training & evaluation

| File | What it does |
|------|-------------|
| `src/training/train.py` | Trains 12 sequential two-stage LightGBM models; saves `lgb_models.pkl` |
| `src/training/testing.py` | Sequential inference on test set; reports MAE/RMSE per horizon |
| `models/model.py` | `TwoStageHorizonModel` class + `add_derived_features()` / `sanitize_columns()` |

### Inference

| File | What it does |
|------|-------------|
| `src/extract_current_day_data.py` | Extracts the latest row per `(ggo, odsek)` from the test split → `current_state.csv` |
| `src/predict_the_future.py` | Runs h1–h12 sequential inference; outputs `future_predictions.csv` |
| `src/utils/generating_heatmap_data.py` | Converts predictions + history into frontend-ready heatmap CSVs |

### Synthetic data

| File | What it does |
|------|-------------|
| `src/generating_synthetic_data/synthetic_pipeline.py` | Logistic growth + spatial diffusion model → ~12 M rows of synthetic beetle population |
| `notebooks/synthetic_data_analysis.ipynb` | Correlation analysis between synthetic and real harvest data |

---

## Output format

### `data/predictions/future_predictions.csv`

| Column | Description |
|--------|-------------|
| `ggo` | Forest district code (1–14) |
| `odsek` | Sector ID |
| `leto_mesec` | Forecast month (`YYYY-MM`) |
| `h1_pred … h12_pred` | Predicted harvest volume in m³ for horizons 1–12 |

### Frontend heatmap CSVs

Both files share the schema `ggo, odsek_id, leto_mesec, target, is_a_prediction`:

- `heatmap_past_data.csv` — historical actuals
- `heatmap_future_predictions.csv` — AI predictions

---

## Geographic hierarchy

```
GGO — Gozdnogospodarsko območje (Forest district)   14 districts
 └── GGE — Gozdnogospodarska enota (Forest unit)
      └── Odsek (Forest sector)                      ~42 000 sectors
```

---

## Team

Korošci + Tilen — Arnes Hackathon 2026

---

## License

See [LICENSE](LICENSE).
