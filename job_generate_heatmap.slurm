#!/bin/bash
#SBATCH --job-name=barkwatch_train
#SBATCH --output=logs/barkwatch_%j.out
#SBATCH --error=logs/barkwatch_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --partition=all

# Use this script when the pipeline has already been run and
# data/processed/splits/ + data/processed/target.csv already exist.

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load Python/3.12.3-GCCcore-13.3.0

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
# Generate
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Predicting Future ..."
python src/predict_the_future.py

echo "[$(date '+%H:%M:%S')] Generating Heatmap ..."
python src/utils/generating_heatmap_data.py


echo "[$(date '+%H:%M:%S')] Job complete."