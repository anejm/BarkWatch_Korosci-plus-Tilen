# BarkWatch-arnes_hackathon2026

## Project Structure
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