#!/bin/bash
#SBATCH --job-name=olmix-60m-test
#SBATCH --partition=small-g           # Aligned for single-GPU work
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # 1 task total
#SBATCH --gpus-per-task=1             # Request exactly 1 GPU (1 GCD)
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00               
#SBATCH --account=project_462000963
#SBATCH --output logs/%j.out
#SBATCH --error logs/%j.err

if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0" "$@"
    exit
fi

set -euo pipefail

# For this standalone test script, we hardcode the target variant
EXP_NAME="proxy_60M_test"
MIX_FILE="mixes/nested-swarm-0000.txt"

MEGATRON_DIR="/flash/project_462000963/tools/OpenEuroLLM-NVIDIA-Megatron-LM"

# --- Retained Slurm Rescheduling Protections ---
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
logfile_basename="${SLURM_JOB_NAME}-${SLURM_JOBID}-${timestamp}"
mv -f "logs/${SLURM_JOBID}.out" "logs/${logfile_basename}.out"
mv -f "logs/${SLURM_JOBID}.err" "logs/${logfile_basename}.err"

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

ln -sf "${logfile_basename}.out" "logs/latest.out"
ln -sf "${logfile_basename}.err" "logs/latest.err"

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
export WANDB_API_KEY="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"
export HF_TOKEN="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"


mkdir -p "$CHECKPOINT_PATH"

# --- Retained WandB Dashboard Trackers ---
WANDB_PROJECT="code-ablations-0.4B"
WANDB_EXP_NAME="${EXP_NAME}-mix"

LAUNCH_SCRIPT="$BASE_DIR/launch.sh"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Calibrated Single-GPU Distributed Network Variables ---
export MASTER_ADDR=localhost
export MASTER_PORT=9999
export WORLD_SIZE=1                   
export OMP_NUM_THREADS=2    
export HSA_ENABLE_SDMA=0

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTHONWARNINGS=ignore
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

# ====================================================================
# MODEL AND PRETRAINING CONFIGURATION (60M Olmix Setup)
# ====================================================================
DATA_PATH=$(tr "\n" " " < "$MIX_FILE")
DATA_CACHE_PATH="/scratch/project_462000963/users/$USER/test_output/data_cache_${EXP_NAME}"
TOKENIZER_MODEL="openeurollm/tokenizer-256k"

NUM_LAYERS=8
HIDDEN_SIZE=384
FFN_HIDDEN_SIZE=1024         
NUM_ATTENTION_HEADS=8
NUM_QUERY_GROUPS=8           
TIE_WORD_EMBEDDINGS=1
INIT_METHOD_STD=0.02
SEQ_LENGTH=2048              
ROTARY_BASE=10000            
KV_CHANNELS=48               

PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=1
PROFILE=0

LR=0.007                     
GLOBAL_BATCH_SIZE=64         
MICRO_BATCH_SIZE=8
RECOMPUTATION=0
TRAIN_TOKENS=6000000000      

# --- Safe Bash Calculation Engine ---
TRAIN_TOKENS=${TRAIN_TOKENS//_}
ITER_TOKENS=$((SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS=$(( (TRAIN_TOKENS + ITER_TOKENS - 1) / ITER_TOKENS ))
LR_WSD_DECAY_ITERS=$((TRAIN_ITERS / 5))
LR_DECAY_ITERS=$TRAIN_ITERS

# --- Derived saving frequencies match step reductions ---
LOG_INTERVAL=1
SAVE_INTERVAL=1000
EVAL_INTERVAL=500
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
    --distributed-timeout-minutes 10
    --overlap-grad-reduce
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --lr $LR
    --min-lr 0
    --lr-decay-style "WSD"
    --lr-wsd-decay-style "linear"
    --lr-warmup-iters 100
    --lr-decay-iters $LR_DECAY_ITERS
    --lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
    --clip-grad 1.0
    --weight-decay 0.1
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
