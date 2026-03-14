#!/bin/bash
# Training script for baseline_latent (DIT with latent conditioning via AdaLN or cross-attention)
# Uses baseline_latent/train_latent_dit.py or train_cross_dit.py
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Baseline Latent Training Script"
echo "=========================================="

# ============================================================
# ENVIRONMENT
# ============================================================
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# ============================================================
# DISTRIBUTED SETUP
# ============================================================
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [ -z "${MASTER_ADDR:-}" ]; then
  MASTER_ADDR=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || echo "127.0.0.1")
fi

# GPU detection
if [ -z "${NPROC_PER_NODE:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _cdevs <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE=0
    for _dev in "${_cdevs[@]}"; do
      [ -n "${_dev}" ] && NPROC_PER_NODE=$((NPROC_PER_NODE + 1))
    done
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  else
    NPROC_PER_NODE=1
  fi
fi

GLOBAL_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# ============================================================
# CONFIGURATION
# ============================================================
# CONDITIONING_TYPE: "adaln" (default) or "cross_attention"
CONDITIONING_TYPE="${CONDITIONING_TYPE:-adaln}"

if [ "${CONDITIONING_TYPE}" = "cross_attention" ]; then
  CONFIG_NAME="${CONFIG_NAME:-mdlm_cross_attention}"
  TRAIN_SCRIPT="baseline_latent/train_cross_dit.py"
else
  CONFIG_NAME="${CONFIG_NAME:-mdlm_latent}"
  TRAIN_SCRIPT="baseline_latent/train_latent_dit.py"
fi

RUN_NAME="${RUN_NAME:-baseline-latent-${CONDITIONING_TYPE}}"
TRAIN_BS="${TRAIN_BS:-32}"
EVAL_BS="${EVAL_BS:-${TRAIN_BS}}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
COMPILE="${COMPILE:-true}"
DTYPE="${DTYPE:-bf16}"
RESUME="${RESUME:-null}"
LATENT_DIM="${LATENT_DIM:-32}"
LATENT_DATA_ROOT="${LATENT_DATA_ROOT:-}"

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12000000
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

# ============================================================
# DISPLAY
# ============================================================
echo "  Config: ${CONFIG_NAME}"
echo "  Script: ${TRAIN_SCRIPT}"
echo "  Conditioning: ${CONDITIONING_TYPE}"
echo "  GPUs: ${NPROC_PER_NODE} x ${NNODES} = ${GLOBAL_WORLD_SIZE}"
echo "  Batch size/GPU: ${TRAIN_BS}"
echo "  LR: ${LEARNING_RATE}"
echo "  Latent dim: ${LATENT_DIM}"
echo "  Resume: ${RESUME}"
echo "=========================================="

# ============================================================
# DIRS
# ============================================================
mkdir -p baseline_latent/results/logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="baseline_latent/results/logs/train_${TIMESTAMP}.log"

# ============================================================
# HYDRA OVERRIDES
# ============================================================
OVERRIDES=(
  "logging.run_name=${RUN_NAME}"
  "logging.save_freq=${SAVE_FREQ}"
  "logging.log_freq=${LOG_FREQ}"
  "logging.eval_freq=${EVAL_FREQ}"
  "training.train_batch_size=${TRAIN_BS}"
  "training.eval_batch_size=${EVAL_BS}"
  "training.num_train_steps=${NUM_TRAIN_STEPS}"
  "training.compile_model=${COMPILE}"
  "training.dtype=${DTYPE}"
  "training.warmup_steps=${WARMUP_STEPS}"
  "training.resume=${RESUME}"
  "model.latent_dim=${LATENT_DIM}"
  "optimizer.lr=${LEARNING_RATE}"
  "optimizer.grad_clip_norm=${GRAD_CLIP_NORM}"
)

# Add data root override if provided
if [ -n "${LATENT_DATA_ROOT}" ]; then
  OVERRIDES+=("data.latent_data_root=${LATENT_DATA_ROOT}")
fi

# ============================================================
# LAUNCH
# ============================================================
echo "Launching baseline_latent training..."
echo "Logs: ${LOG_FILE}"

set +e
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  ${TRAIN_SCRIPT} \
  --config-name "${CONFIG_NAME}" \
  "${OVERRIDES[@]}" \
  2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
set -e

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Training completed successfully!"
else
  echo "Training failed with exit code ${EXIT_CODE}"
fi
echo "  Log: ${LOG_FILE}"
echo "=========================================="
exit $EXIT_CODE
