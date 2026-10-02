#!/bin/bash
#SBATCH --job-name=megatron-to-hf
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --time=00:25:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/megatron-to-hf-%j.out
#SBATCH --error=logs/megatron-to-hf-%j.err

# Run from the repository root: ./scripts/megatron-to-hf-lumi.sh

source env.sh

# LUMI container
CONTAINER="/scratch/${PROJECT_ALLOCATION}/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif"

# Directories to bind
BIND_DIRS="/pfs,/scratch,/flash,$(realpath /scratch/${PROJECT_ALLOCATION}),$(realpath /flash/${PROJECT_ALLOCATION})"

# Paths to Megatron-Bridge-LUMI and Megatron-Bridge-utils repos
BRIDGE_PATH="/flash/${PROJECT_ALLOCATION}/tools/Megatron-Bridge-LUMI"
UTILS_PATH="/flash/${PROJECT_ALLOCATION}/tools/Megatron-Bridge-utils"

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 INPUT_PATH OUTPUT_PATH HF_MODEL TOKENIZER" >&2
    exit 1
fi

# If this script is run without sbatch, invoke with sbatch here.
if [ -z $SLURM_JOB_ID ]; then
    sbatch "$0" "$@"
    exit
fi

singularity exec \
    --bind "$BIND_DIRS" \
    "$CONTAINER" \
    "${SLURM_SUBMIT_DIR}/scripts/megatron-to-hf.sh" "$@" "$UTILS_PATH" "$BRIDGE_PATH"
