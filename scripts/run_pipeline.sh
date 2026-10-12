#!/bin/bash
# Run everything after training for one swarm: convert -> evaluate -> collect -> fit.
#
#   ./scripts/run_pipeline.sh <run_prefix> [<fit_config>]
#
# Run from the repository root, once the training array has finished.
#
# Each stage submits the next one, because the job IDs it needs to wait for do
# not exist until the stage before it has run. Dependencies use afterany rather
# than afterok: one model failing to convert should not throw away the results
# of the others, and collect_bpb.py only reports the variants it actually found.
set -euo pipefail

source env.sh

RUN_PREFIX=${1:?"usage: $0 <run_prefix> [fit_config], e.g. $0 stage2_mix_20260929_1153"}
FIT_CONFIG=${2:-}

RUN_DIR="data/runs/${RUN_PREFIX}"
[[ -d "$RUN_DIR/mixes" ]] || { echo "ERROR: no such run: $RUN_DIR" >&2; exit 1; }

NUM_VARIANTS=$(find "$RUN_DIR/mixes" -name "${RUN_PREFIX}-[0-9][0-9][0-9][0-9].txt" | wc -l)
(( NUM_VARIANTS > 0 )) || { echo "ERROR: no mix files in $RUN_DIR/mixes" >&2; exit 1; }

mkdir -p logs

echo "Swarm        : $RUN_PREFIX ($NUM_VARIANTS variants)"
echo "Fit config   : ${FIT_CONFIG:-(none, stopping after collect)}"
echo

# --- 1. convert every checkpoint, one array task per variant -----------------
convert_id=$(sbatch --parsable \
    --array="0-$((NUM_VARIANTS - 1))" \
    --export="ALL,RUN_PREFIX=${RUN_PREFIX}" \
    scripts/pipeline/convert_one.sh)
echo "convert      : job $convert_id (array 0-$((NUM_VARIANTS - 1)))"

# --- 2. schedule the BPB evals once every conversion has finished ------------
# This job runs eval_bpb.sh, which is what creates the eval jobs: the thing
# that needs to wait for the conversions is the submitter, not the submitted.
eval_id=$(sbatch --parsable \
    --dependency="afterany:${convert_id}" \
    --export="ALL,RUN_PREFIX=${RUN_PREFIX},FIT_CONFIG=${FIT_CONFIG}" \
    scripts/pipeline/schedule_evals.sh)
echo "schedule eval: job $eval_id (after $convert_id)"
echo
echo "The eval jobs, and the collect/fit job after them, are submitted by"
echo "job $eval_id once it runs. Watch with: squeue -u \$USER"
