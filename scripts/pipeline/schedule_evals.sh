#!/bin/bash
#SBATCH --job-name=olmix-schedule-evals
#SBATCH --account=project_465002530
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/schedule-evals-%j.out
#SBATCH --error=logs/schedule-evals-%j.err

# Schedule BPB evals for every converted model of a swarm, then submit the
# collect/fit job that waits on them.
#
# This runs as a job rather than on the login node because it has to wait for
# the conversions, and because the eval job ID it needs does not exist until
# oellm-eval has run.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh

RUN_PREFIX=${RUN_PREFIX:?"set RUN_PREFIX"}
FIT_CONFIG=${FIT_CONFIG:-}

BASE="/flash/${PROJECT_ALLOCATION}/users/$USER/ablation_output"
shopt -s nullglob
MODELS=("$BASE/hf_checkpoints/${RUN_PREFIX}-"[0-9][0-9][0-9][0-9]/*_hf_iter_[0-9]*)
(( ${#MODELS[@]} > 0 )) || { echo "no converted models for $RUN_PREFIX" >&2; exit 1; }
echo "Found ${#MODELS[@]} converted model(s)"

# eval_bpb.sh prints "Submitted batch job <id>"; grab the number so the fit can
# wait for it. oellm-eval has no --parsable of its own.
out=$(./scripts/eval_bpb.sh "${MODELS[@]}" 2>&1 | tee /dev/stderr)
eval_id=$(echo "$out" | grep -oE "Submitted batch job [0-9]+" | tail -1 | grep -oE "[0-9]+")
[[ -n "$eval_id" ]] || { echo "could not find the eval job id in the output" >&2; exit 1; }

# oellm-eval names the results directory after the time it ran; it prints the
# path, so take it from there rather than guessing.
results_dir=$(echo "$out" | grep -oE "Evaluation directory: \S+" | tail -1 | awk '{print $NF}')
[[ -n "$results_dir" ]] || { echo "could not find the results directory" >&2; exit 1; }

echo "eval job     : $eval_id"
echo "results dir  : $results_dir"

collect_id=$(sbatch --parsable \
    --dependency="afterany:${eval_id}" \
    --export="ALL,RUN_PREFIX=${RUN_PREFIX},RESULTS_DIR=${results_dir},FIT_CONFIG=${FIT_CONFIG}" \
    scripts/pipeline/collect_and_fit.sh)
echo "collect+fit  : job $collect_id (after $eval_id)"
