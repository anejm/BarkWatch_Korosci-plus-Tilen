#!/bin/bash
#SBATCH --job-name=barkwatch
#SBATCH --output=logs/barkwatch_%j.out
#SBATCH --error=logs/barkwatch_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=all

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load Python/3.12.3-GCCcore-13.3.0

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

set -euo pipefail

# Create logs dir if it doesn't exist
mkdir -p logs

# Set up virtual environment — recreate if Python version changed
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
# Step 5: Train model on train/val data
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running train.py ..."
python src/training/train.py 

# ---------------------------------------------------------------------------
# Step 6: Run inference on test data
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running testing.py ..."
python src/training/testing.py 

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Job complete."
echo ""
echo "Shortened files ready for download:"
echo "  data/processed/splits/train_short.csv"
echo "  data/processed/splits/val_short.csv"
echo "  data/processed/splits/test_short.csv"
echo ""
echo "Model artifact ready for download:"
echo "  models/xgb_models.pkl"
echo ""
echo "Predictions ready for download:"
echo "  data/predictions/predictions.csv"
echo ""
echo "To copy them to your local machine, run from your local terminal:"
echo "  scp <username>@hpc.arnes.si:$WORKDIR/data/processed/splits/*_short.csv ."
echo "  scp <username>@hpc.arnes.si:$WORKDIR/models/xgb_models.pkl ."
echo "  scp <username>@hpc.arnes.si:$WORKDIR/data/predictions/predictions.csv ."
