#!/bin/bash
#SBATCH --job-name=barkwatch
#SBATCH --output=logs/barkwatch_%j.out
#SBATCH --error=logs/barkwatch_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load Python/3.12.3-GCCcore-13.3.0

WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

# Create logs dir if it doesn't exist
mkdir -p logs

# Set up virtual environment if not already present
if [ ! -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

# ---------------------------------------------------------------------------
# Step 1: Download ARSO weather data
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running arso_prenos.py ..."
python src/data_processing/arso_prenos.py

# ---------------------------------------------------------------------------
# Step 2: Run full data pipeline (produces train/val/test splits)
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running pipeline.py ..."
python src/pipeline.py

# ---------------------------------------------------------------------------
# Step 3: Create shortened CSVs (first 100 rows) for local inspection
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Creating shortened CSV files ..."

SPLITS_DIR="data/processed/splits"
SHORT_DIR="data/processed/splits"

python - <<'EOF'
import csv, os

splits_dir = "data/processed/splits"
files = ["train", "val", "test"]

for name in files:
    src = os.path.join(splits_dir, f"{name}.csv")
    dst = os.path.join(splits_dir, f"{name}_short.csv")

    with open(src, newline="", encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        for i, row in enumerate(reader):
            if i >= 100:
                break
            writer.writerow(row)

    print(f"  Saved {dst} ({100} rows)")
EOF

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
echo "To copy them to your local machine, run from your local terminal:"
echo "  scp <username>@hpc.arnes.si:$WORKDIR/data/processed/splits/*_short.csv ."
