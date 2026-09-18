#!/bin/bash
# Schedule BPB evaluations for one or more HuggingFace model directories.
#
# Usage:
#   ./scripts/eval_bpb.sh <model-dir> [<model-dir> ...]
#
# Examples:
#   # one checkpoint, quick check
#   ./scripts/eval_bpb.sh /flash/<project>/users/$USER/ablation_output/hf_checkpoints/nested-swarm-0000/nested-swarm-0000_hf_iter_0022889
#
#   # the whole swarm (shell globbing picks the checkpoints; a trailing glob
#   # like *_hf_iter_0022889 naturally leaves out any _saved backup copies)
#   ./scripts/eval_bpb.sh /flash/<project>/users/$USER/ablation_output/hf_checkpoints/nested-swarm-*/*_hf_iter_0022889
#
# Requires HF_HOME to be set to a dataset cache you can write to, e.g.
#   export HF_HOME=/scratch/<your-project>/cache/huggingface
#
# Set TASK_GROUPS=bpb-all to include MMLU STEM. Any other oellm-eval flags can
# be passed through the environment as EXTRA, e.g. EXTRA="--limit 32".

set -euo pipefail

[ -f ~/.hpc_secrets ] && source ~/.hpc_secrets
TASK_GROUPS="${TASK_GROUPS:-bpb-core}"

# HF_HOME must point at a dataset cache on a project you currently have. There
# is no sensible default: allocations change, and guessing wrong means either a
# silently cold cache or writing into someone else's project.
if [ -z "${HF_HOME:-}" ]; then
    echo "[error] HF_HOME is not set. Point it at a dataset cache, e.g." >&2
    echo "        export HF_HOME=/scratch/<your-project>/cache/huggingface" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    sed -n '2,18p' "$0" >&2
    exit 1
fi

# Each argument must be a directory holding a HuggingFace model. Checking up
# front turns a silent empty job list into an immediate error -- oellm-eval's
# own path expansion does not recognise this layout and would schedule nothing.
for dir in "$@"; do
    if ! ls "$dir"/*.safetensors >/dev/null 2>&1; then
        echo "[error] no *.safetensors in: $dir" >&2
        exit 1
    fi
done

MODELS=$(printf "%s," "$@")
MODELS="${MODELS%,}"

echo "Models      : $#"
echo "Task groups : $TASK_GROUPS"
echo "HF_HOME     : $HF_HOME"
echo

oellm-eval schedule \
    --models "$MODELS" \
    --task_groups "$TASK_GROUPS" \
    ${EXTRA:-}

echo
echo "When the jobs finish:"
echo "  python scripts/collect_bpb.py --results-dir \$EVAL_OUTPUT_DIR/<timestamp>"
