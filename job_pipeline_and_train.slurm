#!/bin/bash
#SBATCH --job-name=barkwatch
#SBATCH --output=logs/barkwatch_%j.out
#SBATCH --error=logs/barkwatch_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --partition=all

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load Python/3.12.3-GCCcore-13.3.0

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

set -euo pipefail

mkdir -p logs

# Limit threads to allocated CPUs — prevents XGBoost / OpenMP from grabbing
# all cores on a shared node, which can get the job killed on some clusters.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

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
# Step 1: Run full data pipeline
# Produces: data/processed/splits/{train,val,test}.csv
#           data/processed/target.csv  (12-horizon targets, log1p)
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running pipeline.py ..."
python src/pipeline.py --weather --sosedi --odseki

# ---------------------------------------------------------------------------
# Step 2: Create shortened CSVs (first 100 rows) for local inspection
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Creating shortened CSV files ..."

python - <<'EOF'
import csv, os

splits_dir = "data/processed/splits"
for name in ["train", "val", "test"]:
    src = os.path.join(splits_dir, f"{name}.csv")
    dst = os.path.join(splits_dir, f"{name}_short.csv")
    with open(src, newline="", encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(next(reader))
        for i, row in enumerate(reader):
            if i >= 100:
                break
            writer.writerow(row)
    print(f"  Saved {dst} (100 rows)")
EOF

# ---------------------------------------------------------------------------
# Step 3: Train model
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running train.py ..."
python src/training/train.py

# ---------------------------------------------------------------------------
# Step 4: Run inference on test set
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running testing.py ..."
python src/training/testing.py

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Job complete."
echo ""
echo "Artifacts ready for download:"
echo "  scp <user>@hpc.arnes.si:$WORKDIR/data/processed/splits/*_short.csv ."
echo "  scp <user>@hpc.arnes.si:$WORKDIR/models/xgb_models.pkl ."
echo "  scp <user>@hpc.arnes.si:$WORKDIR/data/predictions/predictions.csv ."
