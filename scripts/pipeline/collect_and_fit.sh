#!/bin/bash
#SBATCH --job-name=olmix-collect-fit
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/collect-fit-%j.out
#SBATCH --error=logs/collect-fit-%j.err

# Join BPB scores to mixtures, then fit the regression if a fit config was given.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh

RUN_PREFIX=${RUN_PREFIX:?"set RUN_PREFIX"}
RESULTS_DIR=${RESULTS_DIR:?"set RESULTS_DIR"}
FIT_CONFIG=${FIT_CONFIG:-}

RUN_DIR="data/runs/${RUN_PREFIX}"

"$OLMIX_PYTHON" scripts/collect_bpb.py \
    --results-dir "$RESULTS_DIR" \
    --variants-json "$RUN_DIR/variants.json" \
    --output-dir "$RUN_DIR/olmix_fit"

if [[ -z "$FIT_CONFIG" ]]; then
    echo
    echo "No fit config given. To fit, point one at:"
    echo "  $RUN_DIR/olmix_fit/ratios.csv"
    echo "  $RUN_DIR/olmix_fit/metrics.csv"
    exit 0
fi

"$OLMIX_PYTHON" -m olmix fit --config "$FIT_CONFIG" --output-dir "$RUN_DIR/fit"
echo "Fit written to $RUN_DIR/fit"
