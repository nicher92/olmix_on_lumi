#!/bin/bash
#SBATCH --job-name=olmix-30m-test

#SBATCH --partition=small-g # if not small-g we take the entire node even if we dont need it
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=7       # 7 CPUs assigned per task
#SBATCH --mem=120G              # Total memory for the job (or you can use --mem-per-gpu=60G)

#SBATCH --time=16:00:00               
#SBATCH --account=project_465002530
#SBATCH --output logs/%A_%a.out
#SBATCH --error logs/%A_%a.err


if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0" "$@"
    exit
fi

set -euo pipefail

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TASK_ID_PADDED=$(printf "%04d" $TASK_ID)
EXP_NAME="nested-swarm-${TASK_ID_PADDED}"
MIX_FILE="data/mixes/${EXP_NAME}.txt"


MEGATRON_DIR="/flash/project_462000963/tools/OpenEuroLLM-NVIDIA-Megatron-LM"

# --- Retained Slurm Rescheduling Protections (Array Safe) ---
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Determine the log ID format created by Slurm
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    LOG_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    LOG_ID="${SLURM_JOB_ID}"
fi

logfile_basename="${SLURM_JOB_NAME}-${LOG_ID}-${timestamp}"
mv -f "logs/${LOG_ID}.out" "logs/${logfile_basename}.out"
mv -f "logs/${LOG_ID}.err" "logs/${logfile_basename}.err"

if [[ -v SLURM_RESTART_COUNT ]]; then
    failed_node=$(grep 'Node failure' logs/latest.err | awk '{print $NF}')
    if [[ -z ${failed_node:+x} ]]; then
        echo "RUN RESTARTED but no node failure logged"
    else
        failed_node="${failed_node//$'\n'/ }"
        echo "RUN RESTARTED AFTER FAILURE OF NODE(s) $failed_node. Reason:"
        sacctmgr show event where node="$failed_node" format="NodeName,TimeStart,TimeEnd,State,Reason%100"
    fi
fi

ln -sf "${logfile_basename}.out" "logs/latest_${LOG_ID}.out"
ln -sf "${logfile_basename}.err" "logs/latest_${LOG_ID}.err"


module purge

CONTAINER=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
BIND_DIRS="/pfs,/scratch,/projappl,/project,/flash,/appl,/usr/lib64/libjansson.so.4,/usr/lib64/libcxi.so.1,/opt/cray,/var/spool/slurmd"
export PYTHONUSERBASE=""

c="fe"
BIND_MASK="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

BASE_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="/flash/project_462000963/users/$USER/ablation_output/${EXP_NAME}"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID"
WANDB_DIR="$OUTPUT_DIR/wandb"
source ~/.hpc_secrets

mkdir -p "$CHECKPOINT_PATH"

# --- Retained WandB Dashboard Trackers ---
WANDB_PROJECT="code-ablations-0.4B"
WANDB_EXP_NAME="${EXP_NAME}-mix"

LAUNCH_SCRIPT="$BASE_DIR/launch.sh"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Corrected Single-Node Distributed Network Variables ---
# Dynamically get the node's hostname (e.g., nid00XXXX), which natively resolves to the hsn IP!
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Generate a random port based on your Job ID to prevent collisions with other users on this shared node
export MASTER_PORT=$(( 10000 + (SLURM_JOBID % 20000) ))

export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=2
export HSA_ENABLE_SDMA=0

# Prefix matching: use the hsn network, but only the specific card Slurm gave you
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export HSA_FORCE_FINE_GRAIN_PCIE=1

export PYTHONWARNINGS=ignore
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

# ====================================================================
# MODEL AND PRETRAINING CONFIGURATION (30M Olmix Setup)
# ====================================================================
DATA_PATH=$(tr "\n" " " < "$MIX_FILE")
DATA_CACHE_PATH="/scratch/project_462000963/users/$USER/test_output/data_cache_${EXP_NAME}"
TOKENIZER_MODEL="openeurollm/tokenizer-256k"

# --- 30M Architecture Updates from Table 6 ---
NUM_LAYERS=4                 # Down from 8
HIDDEN_SIZE=256              # Down from 384
FFN_HIDDEN_SIZE=768          # Scaled down for SwiGLU (approx 8/3 * 256)
NUM_ATTENTION_HEADS=8        # Kept at 8
NUM_QUERY_GROUPS=8           # MHA (matches attention heads)
KV_CHANNELS=32               # head_dim (256 / 8 = 32)
# ---------------------------------------------
TIE_WORD_EMBEDDINGS=1
INIT_METHOD_STD=0.02
SEQ_LENGTH=2048              # per olmix paper
ROTARY_BASE=10000             
# ---------------------------------------------


# OPTIMIZER
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR=0.007
MIN_LR=0
LR_WARMUP_ITERS=500
COOLDOWN_FRACTION=1/5
CLIP_GRAD=1.0
WEIGHT_DECAY=0.1    # match scaling law setup

# PARALLELISM
PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=1
PROFILE=0

# TRAINING                     
FSDP=0
GLOBAL_BATCH_SIZE=64         
MICRO_BATCH_SIZE=8
RECOMPUTATION=0
TRAIN_TOKENS=3000000000      


confirm_unset() {
    local varname="$1"
    if [ -n "${!varname+x}" ]; then
	echo "Error: variable '$varname' should not be set." >&2
	exit 1
    fi
}
confirm_unset "TRAIN_ITERS"
confirm_unset "LR_DECAY_ITERS"
confirm_unset "LR_WSD_DECAY_ITERS"

divide_rounding_up() {
    echo $((($1+$2-1)/$2))
}

# Calculate TRAIN_ITERS from TRAIN_TOKENS
TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS=$(divide_rounding_up $TRAIN_TOKENS $ITER_TOKENS)

# Set LR_WSD_DECAY_ITERS based on COOLDOWN_FRACTION
LR_WSD_DECAY_ITERS=$((TRAIN_ITERS*${COOLDOWN_FRACTION}))

# LR_DECAY_ITERS is simply set to TRAIN_ITERS
LR_DECAY_ITERS=$TRAIN_ITERS


# --- Derived saving frequencies match step reductions ---
LOG_INTERVAL=1
SAVE_INTERVAL=2000
EVAL_INTERVAL=1000
EVAL_ITERS=100

# ====================================================================
# BUILDING COMMAND-LINE ARGUMENTS (Unchanged from production)
# ====================================================================
DATA_ARGS=(
    --data-path "$DATA_PATH"
    --data-cache-path "$DATA_CACHE_PATH"
    --split 99,1,0
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --make-vocab-size-divisible-by 128
    --dataloader-type cyclic
    --num-workers 2
)

MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
)

if [ "$NUM_QUERY_GROUPS" != "$NUM_ATTENTION_HEADS" ]; then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups $NUM_QUERY_GROUPS
    )
fi

if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    MODEL_ARGS+=(
        --untie-embeddings-and-output-weights
    )
fi

PARALLEL_ARGS=(
	--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
	--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
	--context-parallel-size $CONTEXT_PARALLEL_SIZE
	--sequence-parallel
	--use-distributed-optimizer
)


PROFILE_ARGS=()

MODEL_ARGS+=(
    --use-flash-attn
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
    --position-embedding-type rope
    --rotary-base $ROTARY_BASE
    --kv-channels $KV_CHANNELS
    --disable-bias-linear
    --init-method-std $INIT_METHOD_STD
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --qk-layernorm
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --bf16
    --swiglu
    --no-async-tensor-model-parallel-allreduce
    --no-masked-softmax-fusion
    --no-gradient-accumulation-fusion
    --no-bias-dropout-fusion
    --no-rope-fusion
    --distributed-timeout-minutes 30
    --overlap-grad-reduce
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 $ADAM_BETA1
    --adam-beta2 $ADAM_BETA2
    --adam-eps $ADAM_EPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style "WSD"
    --lr-wsd-decay-style "linear"
    --lr-warmup-iters $LR_WARMUP_ITERS
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad $CLIP_GRAD
    --weight-decay $WEIGHT_DECAY
)


OUTPUT_ARGS=(
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-queue-size 5
    --wandb-project "$WANDB_PROJECT"
    --wandb-exp-name "$WANDB_EXP_NAME"
    --wandb-save-dir "$WANDB_DIR"
    --log-throughput
    --log-progress
    --log-timers-to-tensorboard
    --log-interval $LOG_INTERVAL
)

if [ $PIPELINE_MODEL_PARALLEL_SIZE -gt 1 ] && [ $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE -gt 1 ]; then
    PARALLEL_ARGS+=(
        --num-layers-per-virtual-pipeline-stage $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE
    )
fi

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(
        --recompute-activations
        --recompute-granularity selective
    )
fi

CHECKPOINT_ARGS=(
    --ckpt-format torch_dist
    --load "$CHECKPOINT_PATH"
    --save "$CHECKPOINT_PATH"
    --save-interval $SAVE_INTERVAL
)

COMMAND=" \
    $MEGATRON_DIR/pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
"

echo '============= COMMAND: ============='
echo "$COMMAND"
echo '===================================='

echo "START $SLURM_JOBID: $(date)"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# --- Restored Strict Exec Pipeline ---
srun \
    --label \
    --cpu-bind=cores \
    singularity exec \
    -B "$BASE_DIR" \
    -B "$BIND_DIRS" \
    "$CONTAINER" \
    "$LAUNCH_SCRIPT" \
    $COMMAND

echo "END $SLURM_JOBID: $(date)"
