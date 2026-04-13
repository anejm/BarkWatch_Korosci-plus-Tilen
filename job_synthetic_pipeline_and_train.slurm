#!/bin/bash
#SBATCH --job-name=barkwatch_synthetic
#SBATCH --output=logs/barkwatch_synthetic_%j.out
#SBATCH --error=logs/barkwatch_synthetic_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=all

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load Python/3.12.3-GCCcore-13.3.0

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

set -euo pipefail

mkdir -p logs data/synthetic/splits data/predictions models data/processed

# Limit threads to allocated CPUs — prevents LightGBM / OpenMP from grabbing
# all cores on a shared node.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------------------------------------------------------------------
# Virtual environment — recreate if Python version changed
# ---------------------------------------------------------------------------
PYTHON_VERSION=$(python --version 2>&1)
VENV_MARKER="venv/.python_version"

if [ ! -d "venv" ] || [ ! -f "$VENV_MARKER" ] || [ "$(cat $VENV_MARKER)" != "$PYTHON_VERSION" ]; then
    rm -rf venv
    python -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
    echo "$PYTHON_VERSION" > "$VENV_MARKER"
else
    source venv/bin/activate
fi

# ---------------------------------------------------------------------------
# Verify that upstream processed files exist (produced by job_pipeline_and_train)
# ---------------------------------------------------------------------------
for f in \
    data/processed/najblizji_odseki_postaje.csv \
    data/processed/posek_processed.csv \
    data/processed/vreme_mesecno.csv; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required upstream file missing: $f"
        echo "Run job_pipeline_and_train.slurm first to produce processed data."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Step 1: Generate synthetic bark beetle data per odsek
#         Input:  data/processed/posek_processed.csv
#                 data/processed/najblizji_odseki_postaje.csv
#                 data/raw/ARSO/vreme.csv
#         Output: data/synthetic/bark_beetle_by_odsek.csv
#         Skipped if the output already exists.
# ---------------------------------------------------------------------------
SYNTH_RAW="data/synthetic/bark_beetle_by_odsek.csv"

if [ -f "$SYNTH_RAW" ]; then
    echo "[$(date '+%H:%M:%S')] $SYNTH_RAW already exists — skipping generation."
else
    echo "[$(date '+%H:%M:%S')] Generating synthetic bark beetle data (generate_bark_beetle_csv.py) ..."
    python src/generating_synthetic_data/generate_bark_beetle_csv.py
    echo "[$(date '+%H:%M:%S')] Generation complete."
fi

# ---------------------------------------------------------------------------
# Step 2: Run the synthetic data pipeline
#         Reads:   data/synthetic/bark_beetle_by_odsek.csv
#                  data/processed/najblizji_odseki_postaje.csv  (for ggo + weather)
#         Produces:
#           data/synthetic/bark_beetle_processed.csv   — feature table
#           data/synthetic/bark_beetle_target.csv      — 12-horizon targets (log1p)
#           data/synthetic/splits/train_synthetic.csv
#           data/synthetic/splits/val_synthetic.csv
#           data/synthetic/splits/test_synthetic.csv
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running synthetic_pipeline.py ..."
python src/synthetic_pipeline.py
echo "[$(date '+%H:%M:%S')] Synthetic pipeline complete."

# ---------------------------------------------------------------------------
# Step 3: Train LightGBM regressor on synthetic data
#         Reads:   data/synthetic/splits/{train,val}_synthetic.csv
#                  data/synthetic/bark_beetle_target.csv
#         Produces: models/lgb_models_synthetic.pkl
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running train_synthetic.py ..."
python src/training/train_synthetic.py
echo "[$(date '+%H:%M:%S')] Training complete."

# ---------------------------------------------------------------------------
# Step 4: Run inference on the synthetic test set
#         Reads:   models/lgb_models_synthetic.pkl
#                  data/synthetic/splits/test_synthetic.csv
#                  data/synthetic/bark_beetle_target.csv  (optional, for eval)
#         Produces: data/predictions/predictions_synthetic.csv
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running test_synthetic.py ..."
python src/training/test_synthetic.py
echo "[$(date '+%H:%M:%S')] Inference complete."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "[$(date '+%H:%M:%S')] Job complete."
echo ""
echo "Artifacts ready for download:"
echo "  scp <user>@hpc.arnes.si:$WORKDIR/models/lgb_models_synthetic.pkl ."
echo "  scp <user>@hpc.arnes.si:$WORKDIR/data/predictions/predictions_synthetic.csv ."
echo "  scp <user>@hpc.arnes.si:$WORKDIR/data/synthetic/bark_beetle_target.csv ."
