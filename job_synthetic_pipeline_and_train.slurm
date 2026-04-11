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
# Verify that upstream processed files exist (from the main pipeline run)
# synthetic_pipeline.py now calls the full agg modules (weather, neighbours,
# odsek/segment), so sestoji_processed.csv is required in addition to the
# files needed by the raw population generator.
# ---------------------------------------------------------------------------
for f in \
    data/processed/odseki_processed.csv \
    data/processed/sestoji_processed.csv \
    data/processed/najblizji_odseki_postaje.csv \
    data/processed/vreme_mesecno.csv \
    data/processed/posek_processed.csv; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required upstream file missing: $f"
        echo "Run job_pipeline_and_train.slurm first to produce processed data."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Step 1: Generate synthetic bark beetle population timeseries
#         Skipped if bark_beetle_population.csv already exists.
# ---------------------------------------------------------------------------
SYNTH_RAW="data/synthetic/bark_beetle_population.csv"

if [ -f "$SYNTH_RAW" ]; then
    echo "[$(date '+%H:%M:%S')] $SYNTH_RAW already exists — skipping generation."
else
    echo "[$(date '+%H:%M:%S')] Generating synthetic bark beetle data ..."
    python src/generating_synthetic_data/generate_synthetic_data.py
    echo "[$(date '+%H:%M:%S')] Generation complete."
fi

# ---------------------------------------------------------------------------
# Step 2: Run the synthetic data pipeline
#         Produces:
#           data/synthetic/synthetic_posek_processed.csv
#           data/synthetic/synthetic_target.csv
#           data/processed/agg_posek_meritve.csv
#           data/processed/agg_posek_sosedi.csv
#           data/processed/agg_odseki_sestoji.csv
#           data/synthetic/splits/{train,val,test}_synthetic.csv
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running synthetic_pipeline.py ..."
python src/synthetic_pipeline.py
echo "[$(date '+%H:%M:%S')] Synthetic pipeline complete."

# ---------------------------------------------------------------------------
# Step 3: Train model on synthetic data (regressor only — no classifier)
#         Produces: models/lgb_models_synthetic.pkl
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Running train_synthetic.py ..."
python src/training/train_synthetic.py
echo "[$(date '+%H:%M:%S')] Training complete."

# ---------------------------------------------------------------------------
# Step 4: Run inference on synthetic test set
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
echo "  scp <user>@hpc.arnes.si:$WORKDIR/data/synthetic/synthetic_target.csv ."
