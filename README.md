# BarkWatch-arnes_hackathon2026

## Project Structure

```bash
.
├── api/                  # API keys
├── config/               # Configuration files
│   └── config.json
├── data/                 # Datasets
│   ├── processed/        # Cleaned / preprocessed data ready for training
│   └── raw/              # Original raw data
│       ├── ARSO/         # Raw data from ARSO source
│       └── ZGS/          # Raw data from ZGS source
│           └── images/   # Image files from ZGS
├── frontend/             # Frontend code (React, Vue, etc.)
├── models/               # Trained models saved for inference
├── notebooks/            # Jupyter notebooks for experimentation and prototyping
└── src/                  # Source code for AI project
    ├── data_processing/  # Scripts to preprocess, clean, or transform data
    ├── inference/        # Scripts to run model predictions
    ├── training/         # Scripts to train models
    └── utils/            # Utility functions and helper scripts

```
