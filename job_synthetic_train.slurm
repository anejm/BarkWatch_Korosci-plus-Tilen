#!/bin/bash
#SBATCH --job-name=barkwatch_synthetic_train
#SBATCH --output=logs/barkwatch_synthetic_train_%j.out
#SBATCH --error=logs/barkwatch_synthetic_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=all

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load Python/3.12.3-GCCcore-13.3.0

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

set -euo pipefail

mkdir -p logs data/predictions models

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------------------------------------------------------------------------
# Virtual environment
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
# Verify that synthetic splits and target exist (from the pipeline job)
# ---------------------------------------------------------------------------
for f in \
    data/synthetic/splits/train_synthetic.csv \
    data/synthetic/splits/val_synthetic.csv \
    data/synthetic/splits/test_synthetic.csv \
    data/synthetic/synthetic_target.csv; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file missing: $f"
        echo "Run job_synthetic_pipeline_and_train.slurm first to generate synthetic data."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Step 1: Train model on synthetic data (two-stage: classifier + regressor)
#         Produces: models/lgb_models_synthetic.pkl
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running train.py --synthetic ..."
python src/training/train.py --synthetic
echo "[$(date '+%H:%M:%S')] Training complete."

# ---------------------------------------------------------------------------
# Step 2: Run inference on synthetic test set
#         Produces: data/predictions/predictions_synthetic.csv
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running testing.py --synthetic ..."
python src/training/testing.py --synthetic
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
