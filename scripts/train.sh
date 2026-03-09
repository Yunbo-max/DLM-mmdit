#!/bin/bash
# File: train_qwen_english_l2t_stable.sh
# Improved training script with fixes for:
# 1. Distributed training connectivity issues
# 2. NaN gradient prevention
set -xeuo pipefail

echo "=========================================="
echo "MM-LDLM Stable Training Script"
echo "=========================================="
echo "This script includes fixes for:"
echo "  - Distributed training connectivity"
echo "  - NaN gradient prevention"
echo "=========================================="
echo ""

# Initialize environment
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
  # Check if port is in use using netstat or ss
  if command -v ss >/dev/null 2>&1; then
    ! ss -tuln | grep -q ":${port} "
  elif command -v netstat >/dev/null 2>&1; then
    ! netstat -tuln | grep -q ":${port} "
  else
    # Fallback: try to bind to the port
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

# Only check/change port on master node (rank 0)
if [ "${NODE_RANK}" -eq 0 ]; then
  if ! check_port_available "${MASTER_PORT}"; then
    echo "⚠ Port ${MASTER_PORT} is already in use, finding available port..."
    MASTER_PORT=$(find_available_port "${MASTER_PORT}")
    if [ $? -ne 0 ]; then
      echo "ERROR: Could not find available port"
      exit 1
    fi
    echo "✓ Using available port: ${MASTER_PORT}"
  else
    echo "✓ Port ${MASTER_PORT} is available"
  fi
fi

# Improved MASTER_ADDR detection
if [ -z "${MASTER_ADDR:-}" ]; then
  # Try to get the primary IP address
  MASTER_ADDR=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || true)
  if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}' | head -n1)
  fi
  if [ -z "${MASTER_ADDR}" ]; then
    echo "ERROR: Could not determine MASTER_ADDR"
    exit 1
  fi
fi

# NPROC_PER_NODE behavior
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
    NPROC_PER_NODE=$(python - <<'PY'
try:
    import torch
    count = torch.cuda.device_count()
except Exception:
    count = 0
print(count)
PY
)
  fi
fi

if [ -z "${NPROC_PER_NODE}" ] || [ "${NPROC_PER_NODE}" -le 0 ]; then
  echo "ERROR: Could not determine GPU count. Set NPROC_PER_NODE explicitly."
  exit 1
fi

# Detect GPU memory for adaptive batch sizing
# Use minimum GPU memory across all visible GPUs to ensure safe batch sizing
GPU_MEMORY_GB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
  # Get minimum memory across all GPUs (in MB), then convert to GB
  GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')
  GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
  echo "✓ Detected ${GPU_COUNT} GPUs, minimum memory: ${GPU_MEMORY_GB}GB"
fi

GLOBAL_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
BASE_WORLD_SIZE="${BASE_WORLD_SIZE:-${NPROC_PER_NODE}}"

echo ""
echo "=========================================="
echo "Adaptive Scaling Configuration"
echo "=========================================="
echo "Nodes: ${NNODES}"
echo "GPUs per node: ${NPROC_PER_NODE}"
echo "Total GPUs: ${GLOBAL_WORLD_SIZE}"
echo "GPU Memory: ${GPU_MEMORY_GB}GB"
echo "=========================================="
echo ""

if [ "${GLOBAL_WORLD_SIZE}" -le 0 ]; then
  echo "ERROR: GLOBAL_WORLD_SIZE must be > 0"
  exit 1
fi
if [ "${BASE_WORLD_SIZE}" -le 0 ]; then
  echo "ERROR: BASE_WORLD_SIZE must be > 0"
  exit 1
fi

scale_value() {
  local base=$1
  # ceil(base * BASE_WORLD_SIZE / GLOBAL_WORLD_SIZE)
  echo $(( (base * BASE_WORLD_SIZE + GLOBAL_WORLD_SIZE - 1) / GLOBAL_WORLD_SIZE ))
}

# ============================================================
# TRAINING SCHEDULE (scaled)
# ============================================================
L2T_BASE_STEPS="${L2T_BASE_STEPS:-1000000}"
L2T_BASE_SAVE_FREQ="${L2T_BASE_SAVE_FREQ:-50000}"
L2T_BASE_LOG_FREQ="${L2T_BASE_LOG_FREQ:-10000}"
L2T_BASE_EVAL_FREQ="${L2T_BASE_EVAL_FREQ:-10000}"

L2T_STEPS=$(scale_value "${L2T_BASE_STEPS}")
L2T_SAVE_FREQ=$(scale_value "${L2T_BASE_SAVE_FREQ}")
L2T_LOG_FREQ=$(scale_value "${L2T_BASE_LOG_FREQ}")
L2T_EVAL_FREQ=$(scale_value "${L2T_BASE_EVAL_FREQ}")

# ============================================================
# TRAINING CONFIGURATION (Adaptive defaults)
# ============================================================

# Adaptive batch size based on GPU memory
if [ -z "${L2T_TRAIN_BS:-}" ]; then
  if [ "${GPU_MEMORY_GB}" -ge 140 ]; then
    L2T_TRAIN_BS=6  # H200 141GB
  elif [ "${GPU_MEMORY_GB}" -ge 80 ]; then
    L2T_TRAIN_BS=4  # H100/A100 80GB
  elif [ "${GPU_MEMORY_GB}" -ge 40 ]; then
    L2T_TRAIN_BS=3  # A100 40GB
  elif [ "${GPU_MEMORY_GB}" -ge 24 ]; then
    L2T_TRAIN_BS=2  # RTX 4090/3090
  elif [ "${GPU_MEMORY_GB}" -ge 16 ]; then
    L2T_TRAIN_BS=2  # RTX 4080
  else
    L2T_TRAIN_BS=1  # Smaller GPUs
  fi
  echo "✓ Adaptive batch size: ${L2T_TRAIN_BS} (based on ${GPU_MEMORY_GB}GB GPU)"
else
  echo "✓ Using manual batch size: ${L2T_TRAIN_BS}"
fi

L2T_EVAL_BS="${L2T_EVAL_BS:-${L2T_TRAIN_BS}}"

# Adaptive gradient accumulation for effective batch size
if [ -z "${GRAD_ACCUM_STEPS:-}" ]; then
  # Target effective batch size of 32-64 across all GPUs
  EFFECTIVE_BS=$((L2T_TRAIN_BS * GLOBAL_WORLD_SIZE))
  if [ "${EFFECTIVE_BS}" -lt 32 ]; then
    GRAD_ACCUM_STEPS=$(( (32 + EFFECTIVE_BS - 1) / EFFECTIVE_BS ))
  else
    GRAD_ACCUM_STEPS=1
  fi
  echo "✓ Adaptive gradient accumulation: ${GRAD_ACCUM_STEPS} steps (effective BS: $((EFFECTIVE_BS * GRAD_ACCUM_STEPS)))"
else
  echo "✓ Using manual gradient accumulation: ${GRAD_ACCUM_STEPS}"
fi

# Adaptive learning rate scaling (linear scaling rule)
if [ -z "${LEARNING_RATE:-}" ]; then
  BASE_LR="5e-5"
  TOTAL_EFFECTIVE_BS=$((L2T_TRAIN_BS * GLOBAL_WORLD_SIZE * GRAD_ACCUM_STEPS))

  # Scale LR proportionally to batch size (with sqrt scaling for stability)
  if [ "${TOTAL_EFFECTIVE_BS}" -gt 32 ]; then
    # Use sqrt scaling for large batch sizes (more stable)
    SCALE_FACTOR=$(python -c "import math; print(f'{math.sqrt(${TOTAL_EFFECTIVE_BS} / 32):.6f}')")
    LEARNING_RATE=$(python -c "print(f'{float(\"${BASE_LR}\") * float(\"${SCALE_FACTOR}\"):.10f}')")
  else
    LEARNING_RATE="${BASE_LR}"
  fi
  echo "✓ Adaptive learning rate: ${LEARNING_RATE} (total effective BS: ${TOTAL_EFFECTIVE_BS})"
else
  echo "✓ Using manual learning rate: ${LEARNING_RATE}"
fi

# Adaptive warmup steps based on total steps
if [ -z "${WARMUP_STEPS:-}" ]; then
  # 2% of total steps for warmup
  WARMUP_STEPS=$(( L2T_STEPS / 50 ))
  if [ "${WARMUP_STEPS}" -lt 1000 ]; then
    WARMUP_STEPS=1000
  elif [ "${WARMUP_STEPS}" -gt 5000 ]; then
    WARMUP_STEPS=5000
  fi
  echo "✓ Adaptive warmup steps: ${WARMUP_STEPS}"
else
  echo "✓ Using manual warmup steps: ${WARMUP_STEPS}"
fi

L2T_CONFIG="${L2T_CONFIG:-mmdit_stable}"  # Use stable config by default
L2T_RUN_NAME="${L2T_RUN_NAME:-mmdit-qwen-32d-l2t-stable-multisteps}"
L2T_SAVE_DIR="${L2T_SAVE_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved}"
L2T_COMPILE="${L2T_COMPILE:-false}"
LATENT_DIM="${LATENT_DIM:-32}"
DTYPE="${DTYPE:-bf16}"
DATA_WORKERS="${DATA_WORKERS:-16}"
TOKEN_DIR="${TOKEN_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data/qwen-embeddings-32/tokens/train}"
LATENT_DIR="${LATENT_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data/qwen-embeddings-32/latents/train}"

# NaN-safe hyperparameters (can be overridden)
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.5}"  # Reduced from 1.0
LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-0.1}"  # Reduced from 1.0

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
# Reduce allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Disable wandb
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export WANDB_DIR="./output_dir/wandb"
mkdir -p "${WANDB_DIR}"

# NCCL configuration (relaxed timeouts)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12000000
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO  # Set to INFO for detailed error diagnostics
export NCCL_DEBUG_FILE="/tmp/nccl_debug_${NODE_RANK}.log"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Optional: Set network interface if needed
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1

# ============================================================
# DISPLAY CONFIGURATION
# ============================================================
echo ""
echo "=========================================="
echo "Adaptive Training Configuration"
echo "=========================================="
echo "Hardware:"
echo "  Node Rank: ${NODE_RANK}/${NNODES}"
echo "  GPUs per node: ${NPROC_PER_NODE}"
echo "  Total GPUs: ${GLOBAL_WORLD_SIZE}"
echo "  GPU Memory: ${GPU_MEMORY_GB}GB"
echo ""
echo "Adaptive Scaling:"
echo "  Batch size per GPU: ${L2T_TRAIN_BS}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Effective batch size: $((L2T_TRAIN_BS * GLOBAL_WORLD_SIZE * GRAD_ACCUM_STEPS))"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo ""
echo "Training Schedule:"
echo "  Total steps: ${L2T_STEPS}"
echo "  Save frequency: ${L2T_SAVE_FREQ}"
echo "  Log frequency: ${L2T_LOG_FREQ}"
echo "  Eval frequency: ${L2T_EVAL_FREQ}"
echo ""
echo "Network:"
echo "  Master Addr: ${MASTER_ADDR}"
echo "  Master Port: ${MASTER_PORT}"
echo ""
echo "Training Schedule (scaled):"
echo "  Total steps: ${L2T_STEPS}"
echo "  Save freq: ${L2T_SAVE_FREQ}"
echo "  Log freq: ${L2T_LOG_FREQ}"
echo "  Eval freq: ${L2T_EVAL_FREQ}"
echo ""
echo "Hyperparameters (NaN-safe):"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Gradient clip: ${GRAD_CLIP_NORM}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo "  Latent loss weight: ${LATENT_LOSS_WEIGHT}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Batch size: ${L2T_TRAIN_BS}"
echo "  Config: ${L2T_CONFIG}"
echo "=========================================="
echo ""

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo "Running pre-flight checks..."
echo ""

# Check 1: Verify GPU availability
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. CUDA may not be available."
  exit 1
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "✓ Found ${GPU_COUNT} GPUs"

if [ "${NPROC_PER_NODE}" -gt "${GPU_COUNT}" ]; then
  echo "ERROR: NPROC_PER_NODE (${NPROC_PER_NODE}) > available GPUs (${GPU_COUNT})"
  exit 1
fi

# Check 2: Verify data paths exist
if [ ! -d "${TOKEN_DIR}" ]; then
  echo "ERROR: Token directory does not exist: ${TOKEN_DIR}"
  exit 1
fi
echo "✓ Token directory exists: ${TOKEN_DIR}"

if [ ! -d "${LATENT_DIR}" ]; then
  echo "ERROR: Latent directory does not exist: ${LATENT_DIR}"
  exit 1
fi
echo "✓ Latent directory exists: ${LATENT_DIR}"

# Check 3: Verify config file exists
CONFIG_FILE="latentDLM_mmdit/configs/${L2T_CONFIG}.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "WARNING: Config file not found: ${CONFIG_FILE}"
  echo "  Will try to use default config"
else
  echo "✓ Config file exists: ${CONFIG_FILE}"
fi

# Check 4: Verify master node is reachable (for non-master nodes)
if [ "${NODE_RANK}" -ne 0 ]; then
  echo "Checking connectivity to master node..."
  if ! ping -c 1 -W 5 "${MASTER_ADDR}" >/dev/null 2>&1; then
    echo "WARNING: Cannot ping master node at ${MASTER_ADDR}"
    echo "  This may be normal if ICMP is blocked. Continuing..."
  else
    echo "✓ Master node is reachable"
  fi

  # Try to connect to the port
  if timeout 5 bash -c "cat < /dev/null > /dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then
    echo "✓ Master port ${MASTER_PORT} is accessible"
  else
    echo "WARNING: Cannot connect to master port ${MASTER_ADDR}:${MASTER_PORT}"
    echo "  This may cause training to fail. Check firewall settings."
  fi
fi

# Check 5: Verify PyTorch and NCCL
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ NCCL version: {torch.cuda.nccl.version() if torch.cuda.is_available() else \"N/A\"}')"

# Check 6: Verify master port is available (on master node)
if [ "${NODE_RANK}" -eq 0 ]; then
  if check_port_available "${MASTER_PORT}"; then
    echo "✓ Master port ${MASTER_PORT} is available"
  else
    echo "ERROR: Master port ${MASTER_PORT} is still in use after port selection"
    echo "  This should not happen. Please check for port conflicts."
    exit 1
  fi
fi

echo ""
echo "Pre-flight checks completed!"
echo "=========================================="
echo ""

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================
mkdir -p output_dir/l2t_logs
mkdir -p "${L2T_SAVE_DIR}"

# Create a log file for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_logs/train_${TIMESTAMP}_node${NODE_RANK}.log"
mkdir -p train_logs

# ============================================================
# LAUNCH TRAINING
# ============================================================
echo "Launching training..."
echo "Logs will be saved to: ${LOG_FILE}"
echo ""

# Launch with error handling
set +e  # Don't exit on error, we want to handle it

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  latentDLM_mmdit/train_mmdit_stable.py \
  --config-name "${L2T_CONFIG}" \
  logging.run_name="${L2T_RUN_NAME}" \
  logging.save_dir="${L2T_SAVE_DIR}" \
  training.train_batch_size="${L2T_TRAIN_BS}" \
  training.eval_batch_size="${L2T_EVAL_BS}" \
  training.num_train_steps="${L2T_STEPS}" \
  training.compile_model="${L2T_COMPILE}" \
  loss.loss_type="l2t" \
  model.latent_dim="${LATENT_DIM}" \
  training.dtype="${DTYPE}" \
  logging.save_freq="${L2T_SAVE_FREQ}" \
  logging.log_freq="${L2T_LOG_FREQ}" \
  logging.eval_freq="${L2T_EVAL_FREQ}" \
  data.num_workers="${DATA_WORKERS}" \
  data.token_dir="${TOKEN_DIR}" \
  data.latent_dir="${LATENT_DIR}" \
  optimizer.lr="${LEARNING_RATE}" \
  optimizer.grad_clip_norm="${GRAD_CLIP_NORM}" \
  training.warmup_steps="${WARMUP_STEPS}" \
  loss.latent_loss_weight="${LATENT_LOSS_WEIGHT}" \
  training.gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
set -e  # Re-enable exit on error

# ============================================================
# POST-TRAINING ANALYSIS
# ============================================================
echo ""
echo "=========================================="
echo "POST-TRAINING STABILITY ANALYSIS"
echo "=========================================="

# Check for the training log JSONL file
LOG_DIR="${L2T_SAVE_DIR}/${L2T_RUN_NAME}"
TRAINING_LOG="${LOG_DIR}/training_log.jsonl"

if [ -f "${TRAINING_LOG}" ]; then
  echo "✓ Found training log: ${TRAINING_LOG}"
  echo ""
  
  # Run stability analysis using Python
  python3 << 'ANALYSIS_EOF'
import json
import sys
import numpy as np
from pathlib import Path

# Try to import pandas for rolling statistics
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

log_file = Path("${TRAINING_LOG}")
if not log_file.exists():
    print("No training log found for analysis")
    sys.exit(0)

# Read all log entries
entries = []
with open(log_file) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

if not entries:
    print("No valid log entries found")
    sys.exit(0)

print("=" * 60)
print("LOSS STABILITY ANALYSIS")
print("=" * 60)

# Extract loss values
losses = [e.get("loss", 0) for e in entries]
steps = [e.get("step", 0) for e in entries]
lr_values = [e.get("lr", 0) for e in entries]

# Basic statistics
import numpy as np
losses_arr = np.array(losses)

print(f"Total steps logged: {len(entries)}")
print(f"Loss range: [{losses_arr.min():.4f}, {losses_arr.max():.4f}]")
print(f"Loss mean: {losses_arr.mean():.4f}")
print(f"Loss std: {losses_arr.std():.4f}")
print(f"Final loss: {losses_arr[-1]:.4f}")
print()

# Detect loss spikes (sudden increases > 2 std deviations)
if len(losses) >= 10:
    rolling_mean = pd.Series(losses).rolling(window=50, min_periods=1).mean() if HAS_PANDAS else None
    
    # Calculate spike thresholds
    mean_loss = losses_arr.mean()
    std_loss = losses_arr.std()
    spike_threshold = mean_loss + 2 * std_loss
    
    print(f"Spike detection threshold: {spike_threshold:.4f}")
    
    spikes = []
    for i, (loss, step) in enumerate(zip(losses, steps)):
        if loss > spike_threshold:
            spikes.append((step, loss))
    
    if spikes:
        print(f"⚠ Found {len(spikes)} loss spike(s)!")
        for step, loss in spikes[:5]:  # Show first 5
            print(f"  Step {step}: loss = {loss:.4f} (> {spike_threshold:.4f})")
    else:
        print("✓ No significant loss spikes detected")
else:
    print("Not enough data for spike detection (need >= 10 steps)")

print()

# Learning rate analysis
if lr_values:
    print("=" * 60)
    print("LEARNING RATE ANALYSIS")
    print("=" * 60)
    print(f"LR range: [{min(lr_values):.2e}, {max(lr_values):.2e}]")
    print(f"Final LR: {lr_values[-1]:.2e}")

print()

# Gradient statistics (if available)
print("=" * 60)
print("GRADIENT STATISTICS")
print("=" * 60)

grad_norms = [e.get("train/grad_norm", 0) for e in entries if "train/grad_norm" in e]
grad_stds = [e.get("train/grad_std", 0) for e in entries if "train/grad_std" in e]

if grad_norms:
    grad_norms_arr = np.array(grad_norms)
    print(f"Gradient norm range: [{grad_norms_arr.min():.4f}, {grad_norms_arr.max():.4f}]")
    print(f"Gradient norm mean: {grad_norms_arr.mean():.4f}")
    
    # Check for invalid gradients
    if (grad_norms_arr > 100).any():
        print(f"⚠ Warning: High gradient norms detected (> 100)")
    else:
        print("✓ Gradient norms within acceptable range")
else:
    print("Gradient norm data not available in log")

print()
ANALYSIS_EOF

  echo ""
fi

echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Training completed successfully!"
else
  echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "=========================================="
echo ""
echo "Summary:"
echo "  Exit code: ${EXIT_CODE}"
echo "  Log file: ${LOG_FILE}"
echo "  NCCL debug: /tmp/nccl_debug_${NODE_RANK}.log"
echo "  Checkpoints: ${L2T_SAVE_DIR}/${L2T_RUN_NAME}/"
echo ""

if [ $EXIT_CODE -ne 0 ]; then
  echo "Troubleshooting:"
  echo "  1. Check log file for errors: ${LOG_FILE}"
  echo "  2. Check NCCL debug log: /tmp/nccl_debug_${NODE_RANK}.log"
  echo "  3. Look for 'ERROR' or 'NaN' in logs:"
  echo "     grep -E 'ERROR|NaN' ${LOG_FILE}"
  echo "  4. Check if distributed setup failed:"
  echo "     grep -E 'DistNetworkError|Connection' ${LOG_FILE}"
  echo ""
fi

echo "=========================================="

exit $EXIT_CODE
