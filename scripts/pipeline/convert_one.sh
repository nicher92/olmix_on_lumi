#!/bin/bash
#SBATCH --job-name=olmix-convert
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/convert-%A_%a.out
#SBATCH --error=logs/convert-%A_%a.err

# Convert one variant's final checkpoint to HuggingFace format.
# Submitted as an array by scripts/run_pipeline.sh; the array index selects the
# variant, the same way train-0.05B.sh uses it.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh

RUN_PREFIX=${RUN_PREFIX:?"set RUN_PREFIX (run_pipeline.sh does this)"}
EXP_NAME="${RUN_PREFIX}-$(printf "%04d" "${SLURM_ARRAY_TASK_ID:-0}")"

BASE="/flash/${PROJECT_ALLOCATION}/users/$USER/ablation_output"
CHECKPOINTS="$BASE/$EXP_NAME/checkpoints"
TRACKER="$CHECKPOINTS/latest_checkpointed_iteration.txt"

# Which iteration did this run actually reach? Megatron writes the tracker only
# after a checkpoint is fully saved, so it beats assuming a fixed number.
[[ -f "$TRACKER" ]] || { echo "no tracker file for $EXP_NAME; not trained?" >&2; exit 1; }
ITER_NAME=$(printf "iter_%07d" "$(tr -d '[:space:]' < "$TRACKER")")
CHECKPOINT_DIR="$CHECKPOINTS/$ITER_NAME"
[[ -d "$CHECKPOINT_DIR" ]] || { echo "missing $CHECKPOINT_DIR" >&2; exit 1; }

HF_OUTPUT="$BASE/hf_checkpoints/$EXP_NAME/${EXP_NAME}_hf_${ITER_NAME}"
if [[ -f "$HF_OUTPUT/config.json" ]]; then
    echo "$EXP_NAME already converted ($ITER_NAME)"
    exit 0
fi

echo "Converting $EXP_NAME at $ITER_NAME"
srun singularity exec \
    --bind "/pfs,/scratch,/flash" \
    "/scratch/${PROJECT_ALLOCATION}/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif" \
    "$SLURM_SUBMIT_DIR/scripts/megatron-to-hf.sh" \
    "$CHECKPOINT_DIR" "$HF_OUTPUT" "Qwen/Qwen3-0.6B" "openeurollm/tokenizer-256k" \
    "/flash/${PROJECT_ALLOCATION}/tools/Megatron-Bridge-utils" \
    "/flash/${PROJECT_ALLOCATION}/tools/Megatron-Bridge-LUMI"
