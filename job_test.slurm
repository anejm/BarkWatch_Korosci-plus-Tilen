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

# Verify required data exists before spending queue time
for f in \
    data/processed/splits/train.csv \
    data/processed/splits/val.csv \
    data/processed/splits/test.csv \
    data/processed/target.csv; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file missing: $f"
        echo "Run job_pipeline_and_train.slurm first."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Test inference
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running testing.py ..."
python src/training/testing.py

echo "[$(date '+%H:%M:%S')] Job complete."
echo ""
echo "Artifacts ready for download:"
echo "  scp <user>@hpc.arnes.si:$WORKDIR/models/xgb_models.pkl ."
echo "  scp <user>@hpc.arnes.si:$WORKDIR/data/predictions/predictions.csv ."
