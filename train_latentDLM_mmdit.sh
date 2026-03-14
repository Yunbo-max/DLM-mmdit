#!/bin/bash
# Training script for latentDLM_mmdit (multimodal MMDiT with external mmdit package)
# Uses latentDLM_mmdit/train_mmdit_stable.py with Hydra config mmdit_stable
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "LatentDLM-MMDiT Training Script"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "=========================================="

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  echo "Activated .venv"
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
  echo "Activated venv"
fi

# ============================================================
# DISTRIBUTED SETUP
# ============================================================
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"

check_port_available() {
  local port=$1
  if command -v ss >/dev/null 2>&1; then
    ! ss -tuln | grep -q ":${port} "
  elif command -v netstat >/dev/null 2>&1; then
    ! netstat -tuln | grep -q ":${port} "
  else
    timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/${port}" 2>/dev/null && return 1 || return 0
  fi
}

find_available_port() {
  local start_port=${1:-29500}
  local max_attempts=100
  local port=$start_port
  for ((i=0; i<max_attempts; i++)); do
    if check_port_available "$port"; then
      echo "$port"
      return 0
    fi
    port=$((port + 1))
  done
  echo "ERROR: Could not find available port" >&2
  return 1
}

if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT=29500
fi

if [ "${NODE_RANK}" -eq 0 ]; then
  if ! check_port_available "${MASTER_PORT}"; then
    echo "Port ${MASTER_PORT} in use, finding available port..."
    MASTER_PORT=$(find_available_port "${MASTER_PORT}")
    echo "Using port: ${MASTER_PORT}"
  fi
fi

if [ -z "${MASTER_ADDR:-}" ]; then
  MASTER_ADDR=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || true)
  if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}' | head -n1 || true)
  fi
  if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR="127.0.0.1"
  fi
fi

# GPU detection
if [ -z "${NPROC_PER_NODE:-}" ]; then
  if [ "${NNODES}" -gt 1 ]; then
    NPROC_PER_NODE=8
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _cdevs <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE=0
    for _dev in "${_cdevs[@]}"; do
      if [ -n "${_dev}" ]; then
        NPROC_PER_NODE=$((NPROC_PER_NODE + 1))
      fi
    done
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  else
    NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
  fi
fi

if [ -z "${NPROC_PER_NODE}" ] || [ "${NPROC_PER_NODE}" -le 0 ]; then
  echo "ERROR: Could not determine GPU count. Set NPROC_PER_NODE explicitly."
  exit 1
fi

# GPU memory detection
GPU_MEMORY_GB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')
  GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
  echo "Detected ${NPROC_PER_NODE} GPUs, memory: ${GPU_MEMORY_GB}GB"
fi

GLOBAL_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Adaptive batch size
if [ -z "${TRAIN_BS:-}" ]; then
  if [ "${GPU_MEMORY_GB}" -ge 140 ]; then
    TRAIN_BS=8
  elif [ "${GPU_MEMORY_GB}" -ge 80 ]; then
    TRAIN_BS=4
  elif [ "${GPU_MEMORY_GB}" -ge 40 ]; then
    TRAIN_BS=2
  else
    TRAIN_BS=1
  fi
  echo "Adaptive batch size: ${TRAIN_BS} (based on ${GPU_MEMORY_GB}GB GPU)"
fi

EVAL_BS="${EVAL_BS:-${TRAIN_BS}}"

# Config
CONFIG_NAME="${CONFIG_NAME:-mmdit_stable}"
RUN_NAME="${RUN_NAME:-mmdit-qwen-32d-l2t-stable}"
SAVE_DIR="${SAVE_DIR:-/inspire/hdd/project/future-reading/zhangjiaquan-253108540222/jiaquan/latent/DLM-mmdit/latentDLM_mmdit/results/checkpoints}"
RESUME="${RESUME:-null}"

# Hyperparameters
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.5}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
LOG_FREQ="${LOG_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
LATENT_DIM="${LATENT_DIM:-32}"
LOSS_TYPE="${LOSS_TYPE:-l2t}"

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12000000
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=INFO

# ============================================================
# DISPLAY CONFIG
# ============================================================
echo ""
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "  GPUs: ${NPROC_PER_NODE} x ${NNODES} nodes = ${GLOBAL_WORLD_SIZE} total"
echo "  Config: ${CONFIG_NAME}"
echo "  Batch size/GPU: ${TRAIN_BS}"
echo "  LR: ${LEARNING_RATE}"
echo "  Loss type: ${LOSS_TYPE}"
echo "  Latent dim: ${LATENT_DIM}"
echo "  Resume: ${RESUME}"
echo "=========================================="
echo ""

# ============================================================
# CREATE DIRS
# ============================================================
mkdir -p "${SAVE_DIR}"
mkdir -p latentDLM_mmdit/results/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="latentDLM_mmdit/results/logs/train_${TIMESTAMP}_node${NODE_RANK}.log"

# ============================================================
# BUILD HYDRA OVERRIDES
# ============================================================
OVERRIDES=(
  "logging.run_name=${RUN_NAME}"
  "logging.save_dir=${SAVE_DIR}"
  "logging.save_freq=${SAVE_FREQ}"
  "logging.log_freq=${LOG_FREQ}"
  "logging.eval_freq=${EVAL_FREQ}"
  "training.train_batch_size=${TRAIN_BS}"
  "training.eval_batch_size=${EVAL_BS}"
  "training.num_train_steps=${NUM_TRAIN_STEPS}"
  "training.warmup_steps=${WARMUP_STEPS}"
  "training.resume=${RESUME}"
  "model.latent_dim=${LATENT_DIM}"
  "optimizer.lr=${LEARNING_RATE}"
  "optimizer.grad_clip_norm=${GRAD_CLIP_NORM}"
  "loss.loss_type=${LOSS_TYPE}"
)

# ============================================================
# LAUNCH TRAINING
# ============================================================
echo "Launching training..."
echo "Logs: ${LOG_FILE}"
echo ""

set +e

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  latentDLM_mmdit/train_mmdit_stable.py \
  --config-name "${CONFIG_NAME}" \
  --config-path "configs" \
  "${OVERRIDES[@]}" \
  2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
set -e

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Training completed successfully!"
else
  echo "Training failed with exit code ${EXIT_CODE}"
fi
echo "  Log: ${LOG_FILE}"
echo "  Checkpoints: ${SAVE_DIR}/${RUN_NAME}/"
echo "=========================================="

exit $EXIT_CODE
