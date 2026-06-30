#!/bin/bash

echo "Scanning nested-swarm checkpoints for iter_0022889..."

# Dynamically set the base path using the $USER variable
BASE_USER_DIR="/flash/project_462000963/users/$USER"

# Find all iter_0022889 folders across all nested-swarm directories
for CHECKPOINT_DIR in ${BASE_USER_DIR}/ablation_output/nested-swarm-*/checkpoints/iter_0022889; do

    # Extract the iteration name (iter_0022889)
    ITER_NAME=$(basename "$CHECKPOINT_DIR")

    # Extract the swarm name (e.g., nested-swarm-0018) from the path
    SWARM_NAME=$(echo "$CHECKPOINT_DIR" | grep -o "nested-swarm-[0-9]*")

    # Define the output path - incorporating SWARM_NAME to prevent overwriting
    HF_OUTPUT="${BASE_USER_DIR}/ablation_output/hf_checkpoints/${SWARM_NAME}/${SWARM_NAME}_hf_${ITER_NAME}"

    # Check if we already converted this specific swarm's checkpoint
    if [ -d "$HF_OUTPUT" ]; then
        echo "Skipping $SWARM_NAME / $ITER_NAME (Already converted)"
        continue
    fi

    echo "Submitting conversion for $SWARM_NAME : $ITER_NAME..."

    # Navigate to the user's directory and submit the dev-g job
    cd "/users/$USER/"
    
    ./megatron-to-hf-lumi.sh \
      "$CHECKPOINT_DIR" \
      "$HF_OUTPUT" \
      "Qwen/Qwen3-0.6B" \
      "openeurollm/tokenizer-256k"

    # Sleep for 360 seconds so we don't hammer the Slurm scheduler
    sleep 360
done

echo "Nested-swarm batch submission complete!"
