#!/bin/bash
# File: train_mmdit_latent.sh
# Training script for mmdit_latent (fixed-latent conditioned text diffusion with MMDiT)
# Uses mmdit_latent/train_latent_dit.py with Hydra config mdlm_mmdit_latent
set -xeuo pipefail

echo "=========================================="
echo "MMDiT-Latent Training Script"
echo "=========================================="
echo "Fixed-latent conditioned text generation"
echo "DiT backbone swapped to MMDiT joint attention"
echo "=========================================="
echo ""

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# ============================================================
# DISTRIBUTED SETUP
# ============================================================
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"

# Function to check if a port is available
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

# Function to find an available port
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
  echo "ERROR: Could not find available port in range ${start_port}-$((start_port + max_attempts))" >&2
  return 1
}

# Set MASTER_PORT with automatic port selection
if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT=29500
fi

if [ "${NODE_RANK}" -eq 0 ]; then
  if ! check_port_available "${MASTER_PORT}"; then
    echo "Port ${MASTER_PORT} is already in use, finding available port..."
    MASTER_PORT=$(find_available_port "${MASTER_PORT}")
    if [ $? -ne 0 ]; then
      echo "ERROR: Could not find available port"
      exit 1
    fi
    echo "Using available port: ${MASTER_PORT}"
  fi
fi

# MASTER_ADDR detection
if [ -z "${MASTER_ADDR:-}" ]; then
  MASTER_ADDR=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || true)
  if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}' | head -n1)
  fi
  if [ -z "${MASTER_ADDR}" ]; then
    echo "ERROR: Could not determine MASTER_ADDR"
    exit 1
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

# Detect GPU memory for adaptive batch sizing
GPU_MEMORY_GB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
  GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')
  GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
  echo "Detected ${GPU_COUNT} GPUs, minimum memory: ${GPU_MEMORY_GB}GB"
fi

GLOBAL_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Adaptive batch size based on GPU memory
if [ -z "${TRAIN_BS:-}" ]; then
  if [ "${GPU_MEMORY_GB}" -ge 140 ]; then
    TRAIN_BS=32  # H200 141GB
  elif [ "${GPU_MEMORY_GB}" -ge 80 ]; then
    TRAIN_BS=16  # H100/A100 80GB
  elif [ "${GPU_MEMORY_GB}" -ge 40 ]; then
    TRAIN_BS=8   # A100 40GB
  elif [ "${GPU_MEMORY_GB}" -ge 24 ]; then
    TRAIN_BS=4   # RTX 4090/3090
  elif [ "${GPU_MEMORY_GB}" -ge 16 ]; then
    TRAIN_BS=2   # RTX 4080
  else
    TRAIN_BS=1   # Smaller GPUs
  fi
  echo "Adaptive batch size: ${TRAIN_BS} (based on ${GPU_MEMORY_GB}GB GPU)"
else
  echo "Using manual batch size: ${TRAIN_BS}"
fi

EVAL_BS="${EVAL_BS:-${TRAIN_BS}}"

# Gradient accumulation for effective batch size
if [ -z "${GRAD_ACCUM_STEPS:-}" ]; then
  EFFECTIVE_BS=$((TRAIN_BS * GLOBAL_WORLD_SIZE))
  if [ "${EFFECTIVE_BS}" -lt 32 ]; then
    GRAD_ACCUM_STEPS=$(( (32 + EFFECTIVE_BS - 1) / EFFECTIVE_BS ))
  else
    GRAD_ACCUM_STEPS=1
  fi
  echo "Gradient accumulation: ${GRAD_ACCUM_STEPS} (effective BS: $((EFFECTIVE_BS * GRAD_ACCUM_STEPS)))"
fi

# Training schedule
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-10000}"

# Scale schedule by world size
BASE_WORLD_SIZE="${BASE_WORLD_SIZE:-${NPROC_PER_NODE}}"
scale_value() {
  local base=$1
  echo $(( (base * BASE_WORLD_SIZE + GLOBAL_WORLD_SIZE - 1) / GLOBAL_WORLD_SIZE ))
}
NUM_TRAIN_STEPS=$(scale_value "${NUM_TRAIN_STEPS}")
SAVE_FREQ=$(scale_value "${SAVE_FREQ}")

# Hyperparameters
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
DTYPE="${DTYPE:-bf16}"
COMPILE="${COMPILE:-true}"
DATA_WORKERS="${DATA_WORKERS:-8}"

# Config and paths
CONFIG_NAME="${CONFIG_NAME:-mdlm_mmdit_latent}"
RUN_NAME="${RUN_NAME:-mmdit-latent-training}"
SAVE_DIR="${SAVE_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved}"
RESUME="${RESUME:-null}"

# Model
LATENT_DIM="${LATENT_DIM:-768}"
MODEL_SIZE="${MODEL_SIZE:-small}"  # tiny, small, base, 1B

# Data (override from config if needed)
LATENT_DATA_ROOT="${LATENT_DATA_ROOT:-}"

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export WANDB_DIR="./output_dir/wandb"
mkdir -p "${WANDB_DIR}"

# NCCL configuration
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12000000
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="/tmp/nccl_debug_mmdit_latent_${NODE_RANK}.log"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# ============================================================
# DISPLAY CONFIGURATION
# ============================================================
echo ""
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Hardware:"
echo "  Node Rank: ${NODE_RANK}/${NNODES}"
echo "  GPUs per node: ${NPROC_PER_NODE}"
echo "  Total GPUs: ${GLOBAL_WORLD_SIZE}"
echo "  GPU Memory: ${GPU_MEMORY_GB}GB"
echo ""
echo "Training:"
echo "  Config: ${CONFIG_NAME}"
echo "  Model size: ${MODEL_SIZE}"
echo "  Latent dim: ${LATENT_DIM}"
echo "  Batch size per GPU: ${TRAIN_BS}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((TRAIN_BS * GLOBAL_WORLD_SIZE * GRAD_ACCUM_STEPS))"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo "  Total steps: ${NUM_TRAIN_STEPS}"
echo "  Compile: ${COMPILE}"
echo "  Dtype: ${DTYPE}"
echo ""
echo "Schedule:"
echo "  Save freq: ${SAVE_FREQ}"
echo "  Log freq: ${LOG_FREQ}"
echo "  Eval freq: ${EVAL_FREQ}"
echo ""
echo "Network:"
echo "  Master Addr: ${MASTER_ADDR}"
echo "  Master Port: ${MASTER_PORT}"
echo ""
echo "Paths:"
echo "  Save dir: ${SAVE_DIR}"
echo "  Run name: ${RUN_NAME}"
echo "  Resume: ${RESUME}"
echo "=========================================="
echo ""

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo "Running pre-flight checks..."

# Check GPU availability
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. CUDA may not be available."
  exit 1
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "Found ${GPU_COUNT} GPUs"

if [ "${NPROC_PER_NODE}" -gt "${GPU_COUNT}" ]; then
  echo "ERROR: NPROC_PER_NODE (${NPROC_PER_NODE}) > available GPUs (${GPU_COUNT})"
  exit 1
fi

# Check config file
CONFIG_FILE="mmdit_latent/configs/${CONFIG_NAME}.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "WARNING: Config file not found: ${CONFIG_FILE}"
  echo "  Will try Hydra defaults"
else
  echo "Config file exists: ${CONFIG_FILE}"
fi

# Check data paths if overridden
if [ -n "${LATENT_DATA_ROOT}" ] && [ ! -d "${LATENT_DATA_ROOT}" ]; then
  echo "ERROR: Latent data root not found: ${LATENT_DATA_ROOT}"
  exit 1
fi

# Check master node connectivity (for non-master nodes)
if [ "${NODE_RANK}" -ne 0 ]; then
  if ! ping -c 1 -W 5 "${MASTER_ADDR}" >/dev/null 2>&1; then
    echo "WARNING: Cannot ping master node at ${MASTER_ADDR}"
  else
    echo "Master node is reachable"
  fi
fi

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "Pre-flight checks completed!"
echo "=========================================="
echo ""

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================
mkdir -p "${SAVE_DIR}"
mkdir -p train_logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_logs/train_mmdit_latent_${TIMESTAMP}_node${NODE_RANK}.log"

# ============================================================
# BUILD HYDRA OVERRIDES
# ============================================================
OVERRIDES=(
  "logging.run_name=${RUN_NAME}"
  "logging.save_dir=${SAVE_DIR}"
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
  "logging.save_freq=${SAVE_FREQ}"
  "logging.log_freq=${LOG_FREQ}"
  "logging.eval_freq=${EVAL_FREQ}"
)

# Add model size override
if [ "${MODEL_SIZE}" != "small" ]; then
  OVERRIDES+=("model=${MODEL_SIZE}")
fi

# Add data root override if provided
if [ -n "${LATENT_DATA_ROOT}" ]; then
  OVERRIDES+=("data.latent_data_root=${LATENT_DATA_ROOT}")
fi

# Add gradient accumulation if supported
if [ "${GRAD_ACCUM_STEPS}" -gt 1 ]; then
  OVERRIDES+=("training.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}")
fi

# ============================================================
# LAUNCH TRAINING
# ============================================================
echo "Launching training..."
echo "Logs will be saved to: ${LOG_FILE}"
echo ""

set +e  # Don't exit on error, we want to handle it

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  mmdit_latent/train_latent_dit.py \
  --config-name "${CONFIG_NAME}" \
  "${OVERRIDES[@]}" \
  2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
set -e

# ============================================================
# POST-TRAINING SUMMARY
# ============================================================
echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
  echo "Training completed successfully!"
else
  echo "Training failed with exit code ${EXIT_CODE}"
fi

echo "=========================================="
echo ""
echo "Summary:"
echo "  Exit code: ${EXIT_CODE}"
echo "  Log file: ${LOG_FILE}"
echo "  NCCL debug: /tmp/nccl_debug_mmdit_latent_${NODE_RANK}.log"
echo "  Checkpoints: ${SAVE_DIR}/${RUN_NAME}/"
echo ""

if [ $EXIT_CODE -ne 0 ]; then
  echo "Troubleshooting:"
  echo "  1. Check log file: ${LOG_FILE}"
  echo "  2. Check NCCL debug: /tmp/nccl_debug_mmdit_latent_${NODE_RANK}.log"
  echo "  3. Look for errors: grep -E 'ERROR|NaN' ${LOG_FILE}"
  echo "  4. Resume from checkpoint:"
  echo "     RESUME=${SAVE_DIR}/${RUN_NAME}/latest bash train_mmdit_latent.sh"
  echo ""
fi

echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Sample: CHECKPOINT=${SAVE_DIR}/${RUN_NAME}/latest bash sample_mmdit_latent.sh"
echo "  2. Eval loss: python mmdit_latent/eval/loss.py path=${SAVE_DIR}/${RUN_NAME}/latest"
echo "  3. Self-correction: python mmdit_latent/eval/self_correction.py path=${SAVE_DIR}/${RUN_NAME}/latest samples_path=samples.pt corrected_samples_path=corrected.pt"
echo "=========================================="

exit $EXIT_CODE
