#!/bin/bash
# Run from the repository root: ./scripts/convert_olmix_models_to_hf.sh

echo "Sourcing env.sh.."
source env.sh

RUN_PREFIX=${1:?"usage: $0 <run_prefix>, e.g. stage2_mix_20260929_1153"}

echo "Scanning checkpoints with ${RUN_PREFIX} for iter_0022889..."

# Dynamically set the flash path using the $USER variable
BASE_USER_DIR="/flash/${PROJECT_ALLOCATION}/users/$USER"

# Find all iter_0022889 folders across all nested-swarm directories
for CHECKPOINT_DIR in ${BASE_USER_DIR}/ablation_output/${RUN_PREFIX}-[0-9][0-9][0-9][0-9]/checkpoints/iter_0022889; do

    ITER_NAME=$(basename "$CHECKPOINT_DIR")
    SWARM_NAME=$(echo "$CHECKPOINT_DIR" | grep -o "${RUN_PREFIX}-[0-9]*")
    HF_OUTPUT="${BASE_USER_DIR}/ablation_output/hf_checkpoints/${SWARM_NAME}/${SWARM_NAME}_hf_${ITER_NAME}"

    if [ -d "$HF_OUTPUT" ]; then
        echo "Skipping $SWARM_NAME / $ITER_NAME (Already converted)"
        continue
    fi

    echo "Submitting conversion for $SWARM_NAME : $ITER_NAME..."
    
      scripts/megatron-to-hf-lumi.sh \
      "$CHECKPOINT_DIR" \
      "$HF_OUTPUT" \
      "Qwen/Qwen3-0.6B" \
      "openeurollm/tokenizer-256k"

    sleep 360
done

echo "Nested-swarm batch submission complete!"
