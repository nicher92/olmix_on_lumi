#!/bin/bash

# Export a Megatron checkpoint to HuggingFace format with a custom tokenizer.
#
# Usage:
#   ./export_to_hf <input_path> <output_path> <hf_model> <tokenizer>
#                  [utils_path [bridge_path]]
#
# Arguments:
#   input_path   - Directory containing the Megatron checkpoint
#   output_path  - Output directory for the HuggingFace model
#   hf_model     - HF model name (e.g. Qwen/Qwen3-0.6B)
#   tokenizer    - HF tokenizer name (e.g. openai/gpt-oss-120b)
#   utils_path   - Root of Megatron-Bridge-utils repo (default: script dir)
#   bridge_path  - Root of Megatron-Bridge repo
#                  (default: utils_path/Megatron-Bridge)

# https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -euo pipefail

if (( $# < 4 || $# > 6 )); then
    echo "Usage: $0 INPUT_PATH OUTPUT_PATH HF_MODEL TOKENIZER [UTILS_PATH [BRIDGE_PATH]]" >&2
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_PATH="$2"
HF_MODEL="$3"
HF_TOKENIZER="$4"
UTILS_PATH="${5:-$SCRIPT_DIR}"
BRIDGE_PATH="${6:-$UTILS_PATH/Megatron-Bridge}"

if [ ! -d "$INPUT_PATH" ]; then
    echo "ERROR: not a directory: $INPUT_PATH" >&2
    exit 1
fi

if [ -e "$OUTPUT_PATH" ]; then
    echo "ERROR: $OUTPUT_PATH exists, not clobbering" >&2
    exit 1
fi

# Scripts
DUMMY_MODEL_SCRIPT="${UTILS_PATH}/create_dummy_model.py"
PATCH_SCRIPT="${UTILS_PATH}/export_custom_tokenizer_standalone.py"
CONVERT_SCRIPT="$BRIDGE_PATH/examples/conversion/convert_checkpoints.py"

if [[ ! -f "$DUMMY_MODEL_SCRIPT" ]]; then
    echo "ERROR: dummy model script not found in $DUMMY_MODEL_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$PATCH_SCRIPT" ]]; then
    echo "ERROR: patch script not found in $PATCH_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    echo "ERROR: conversion script not found in $CONVERT_SCRIPT" >&2
    exit 1
fi

# Configuration files
TOKENIZER_IN="${UTILS_PATH}/tokenizers/${HF_MODEL}"
TOKENIZER_OUT="${UTILS_PATH}/tokenizers/${HF_TOKENIZER}"
CONFIG="/users/niclhert/30m_config.json"
RUN_CONFIG="/users/niclhert/run_config.yaml"
#CONFIG="${UTILS_PATH}/configs/${HF_MODEL}"
#RUN_CONFIG="${UTILS_PATH}/templates/${HF_MODEL}/run_config.yaml"

if [ ! -e "$TOKENIZER_IN" ]; then
    echo "ERROR: tokenizer for $HF_MODEL not found in $TOKENIZER_IN" >&2
    exit 1
fi

if [ ! -e "$TOKENIZER_OUT" ]; then
    echo "ERROR: tokenizer $HF_TOKENIZER not found in $TOKENIZER_OUT" >&2
    exit 1
fi

if [ ! -e "$CONFIG" ]; then
    echo "ERROR: config for $HF_MODEL not found in $CONFIG" >&2
    exit 1
fi

if [[ ! -e "$RUN_CONFIG" ]]; then
    echo "ERROR: run config for $HF_MODEL not found in $RUN_CONFIG" >&2
    exit 1
fi

# Tmp dirs
DUMMY_HF_MODEL_PATH=$(mktemp -d)
TMP_MEGATRON_PATH=$(mktemp -d)

cleanup() {
  rm -rf "$DUMMY_HF_MODEL_PATH" "$TMP_MEGATRON_PATH"
}
trap cleanup EXIT


# AutoBridge.export_ckpt() requires a full model rather than just a
# config. We can get around this also in environments without internet
# access by having configs and tokenizers in this repo and creating a
# dummy model from these. (This could be made more flexible by falling
# back on attempting retrieval from HF if a config or tokenizer is not
# found)

echo "========================================"
echo "Step 1/4: Create dummy HF model"
echo "  config   : $CONFIG"
echo "  tokenizer: $TOKENIZER_IN"
echo "  output   : $DUMMY_HF_MODEL_PATH"
echo "========================================"

python3 "$DUMMY_MODEL_SCRIPT" "$CONFIG" "$TOKENIZER_IN" "$DUMMY_HF_MODEL_PATH"


# Megatron Bridge requires a run_config.yaml in the megatron
# checkpoint directory, but Megatron-LM does not write these. To get
# around this, we store run_config.yaml files in this repo. To also
# support different tokenizers, we template these for vocab size. To
# avoid changing the original checkpoint, we create a temporary copy
# and add the run_config.yaml there.

VOCAB_SIZE=$(python3 -c "from transformers import AutoTokenizer; print(len(AutoTokenizer.from_pretrained('$TOKENIZER_OUT')))")

echo
echo "========================================"
echo "Step 2/4: Create checkpoint copy with run_config.yaml"
echo "  original : $INPUT_PATH"
echo "  copy     : $TMP_MEGATRON_PATH"
echo "  template : $RUN_CONFIG"
echo "  vocabsize: $VOCAB_SIZE"
echo "========================================"

cp -r "$INPUT_PATH" "$TMP_MEGATRON_PATH"

# add in iter_ dir
TMP_MEGATRON_PATH="$TMP_MEGATRON_PATH/$(basename $INPUT_PATH)"

perl -pe "s/<<<VOCAB_SIZE>>>/$VOCAB_SIZE/" "$RUN_CONFIG" \
     > "$TMP_MEGATRON_PATH/run_config.yaml"

echo
echo "========================================"
echo "Step 3/4: Run conversion"
echo "  megatron : $TMP_MEGATRON_PATH"
echo "  hf model : $DUMMY_HF_MODEL_PATH"
echo "  output   : $OUTPUT_PATH"
echo "========================================"

export PYTHONPATH="${BRIDGE_PATH}/python-packages:${BRIDGE_PATH}/3rdparty/Megatron-LM:${BRIDGE_PATH}/src:$PYTHONPATH"

python "$CONVERT_SCRIPT" export \
    --megatron-path "$TMP_MEGATRON_PATH" \
    --hf-model "$DUMMY_HF_MODEL_PATH" \
    --hf-path "$OUTPUT_PATH"

echo
echo "========================================"
echo "Step 4/4: Patch tokenizer and config"
echo "  tokenizer: $TOKENIZER_OUT"
echo "  output   : $OUTPUT_PATH"
echo "========================================"

python "$PATCH_SCRIPT" \
    --hf-path "$OUTPUT_PATH" \
    --tokenizer-path "$TOKENIZER_OUT"
