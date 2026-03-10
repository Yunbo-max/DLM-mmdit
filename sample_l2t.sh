#!/bin/bash
# File: sample_l2t.sh
# Sampling script for latentDLM_mmdit L2T (Latent-to-Text) generation
# Mirrors the style of train_qwen_english_l2t_stable.sh
set -xeuo pipefail

echo "=========================================="
echo "MM-LDLM L2T Sampling Script"
echo "=========================================="
echo "Generates text conditioned on latent vectors"
echo "using trained MMDiT checkpoint."
echo "=========================================="
echo ""

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# ============================================================
# CONFIGURATION (override via environment variables)
# ============================================================

# Checkpoint path (REQUIRED — set before running)
CHECKPOINT="${CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ]; then
  # Auto-detect latest checkpoint
  SAVE_DIR="${SAVE_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved}"
  RUN_NAME="${RUN_NAME:-mmdit-qwen-32d-l2t-stable}"
  CKPT_DIR="${SAVE_DIR}/${RUN_NAME}"

  if [ -d "${CKPT_DIR}/latest" ]; then
    CHECKPOINT="${CKPT_DIR}/latest/model.pth"
  elif [ -d "${CKPT_DIR}" ]; then
    # Find most recent .pth file
    CHECKPOINT=$(find "${CKPT_DIR}" -name "model.pth" -o -name "model_final.pth" | sort -t/ -k+1 | tail -1 || true)
  fi

  if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint found. Set CHECKPOINT=/path/to/model.pth"
    echo "  Searched in: ${CKPT_DIR}"
    exit 1
  fi
  echo "Auto-detected checkpoint: ${CHECKPOINT}"
fi

# Config path (auto-detect from checkpoint directory or use default)
CONFIG="${CONFIG:-}"
if [ -z "${CONFIG}" ]; then
  CKPT_PARENT=$(dirname "${CHECKPOINT}")
  if [ -f "${CKPT_PARENT}/config.yaml" ]; then
    CONFIG="${CKPT_PARENT}/config.yaml"
  else
    CONFIG="latentDLM_mmdit/configs/mmdit_stable.yaml"
  fi
fi

# Data paths
LATENT_DIR="${LATENT_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data/qwen-embeddings-32/latents/train}"

# Sampling parameters
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-128}"
STEPS="${STEPS:-1000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
ALGORITHM="${ALGORITHM:-reverse}"       # "reverse" or "ddim"
ETA="${ETA:-0.0}"                       # DDIM eta (0=deterministic, 1=stochastic)
WITH_LOSS_LOGGING="${WITH_LOSS_LOGGING:-true}"

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved/samples/${TIMESTAMP}}"

# ============================================================
# GPU DETECTION
# ============================================================
GPU_MEMORY_GB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
  GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')
  GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
  echo "Detected ${GPU_COUNT} GPUs, minimum memory: ${GPU_MEMORY_GB}GB"
fi

# Adaptive batch size for sampling
if [ -z "${BATCH_SIZE:-}" ] || [ "${BATCH_SIZE}" -eq 8 ]; then
  if [ "${GPU_MEMORY_GB}" -ge 140 ]; then
    BATCH_SIZE=32   # H200 141GB
  elif [ "${GPU_MEMORY_GB}" -ge 80 ]; then
    BATCH_SIZE=16   # H100/A100 80GB
  elif [ "${GPU_MEMORY_GB}" -ge 40 ]; then
    BATCH_SIZE=8    # A100 40GB
  elif [ "${GPU_MEMORY_GB}" -ge 24 ]; then
    BATCH_SIZE=4    # RTX 4090/3090
  else
    BATCH_SIZE=2    # Smaller GPUs
  fi
fi

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo ""
echo "Running pre-flight checks..."

# Check checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi
echo "Checkpoint: ${CHECKPOINT}"

# Check config exists
if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: Config not found: ${CONFIG}"
  exit 1
fi
echo "Config: ${CONFIG}"

# Check latent dir exists
if [ ! -d "${LATENT_DIR}" ]; then
  echo "ERROR: Latent directory not found: ${LATENT_DIR}"
  exit 1
fi
LATENT_COUNT=$(find "${LATENT_DIR}" -name "*.npy" | head -100 | wc -l)
echo "Latent directory: ${LATENT_DIR} (~${LATENT_COUNT}+ .npy files)"

# Check GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found, will use CPU (very slow)"
fi

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Sampling Configuration"
echo "=========================================="
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Config: ${CONFIG}"
echo "  Latent dir: ${LATENT_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""
echo "  Num samples: ${NUM_SAMPLES}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Seq length: ${SEQ_LEN}"
echo "  Steps: ${STEPS}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Algorithm: ${ALGORITHM}"
if [ "${ALGORITHM}" = "ddim" ]; then
  echo "  DDIM eta: ${ETA}"
fi
echo "  Loss logging: ${WITH_LOSS_LOGGING}"
echo "=========================================="
echo ""

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p sample_logs
LOG_FILE="sample_logs/sample_${TIMESTAMP}.log"

# ============================================================
# BUILD COMMAND
# ============================================================
SAMPLE_CMD="python latentDLM_mmdit/sample_l2t_fixed.py \
  --checkpoint ${CHECKPOINT} \
  --config ${CONFIG} \
  --npy_dir ${LATENT_DIR} \
  --num_samples ${NUM_SAMPLES} \
  --batch_size ${BATCH_SIZE} \
  --seq_len ${SEQ_LEN} \
  --steps ${STEPS} \
  --temperature ${TEMPERATURE} \
  --algorithm ${ALGORITHM} \
  --output_dir ${OUTPUT_DIR}"

if [ "${ALGORITHM}" = "ddim" ]; then
  SAMPLE_CMD="${SAMPLE_CMD} --eta ${ETA}"
fi

if [ "${WITH_LOSS_LOGGING}" = "true" ]; then
  SAMPLE_CMD="${SAMPLE_CMD} --with_loss_logging"
fi

# ============================================================
# LAUNCH SAMPLING
# ============================================================
echo "Launching sampling..."
echo "Logs: ${LOG_FILE}"
echo ""

set +e  # Don't exit on error, we want to handle it

eval "${SAMPLE_CMD}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
set -e

# ============================================================
# POST-SAMPLING ANALYSIS
# ============================================================
echo ""
echo "=========================================="
echo "POST-SAMPLING ANALYSIS"
echo "=========================================="

if [ -f "${OUTPUT_DIR}/results.json" ]; then
  SAMPLE_COUNT=$(python3 -c "import json; d=json.load(open('${OUTPUT_DIR}/results.json')); print(d.get('num_samples', 0))")
  echo "Generated ${SAMPLE_COUNT} samples"

  # Show first few samples
  echo ""
  echo "--- First 3 samples ---"
  python3 << PREVIEW_EOF
import json
with open("${OUTPUT_DIR}/results.json") as f:
    data = json.load(f)
for i, text in enumerate(data.get("texts", [])[:3]):
    print(f"  [{i+1}] {text[:200]}{'...' if len(text) > 200 else ''}")
    print()
PREVIEW_EOF
fi

# Show loss analysis if available
if [ -f "${OUTPUT_DIR}/sampling_metrics.json" ]; then
  echo ""
  echo "--- Sampling Metrics ---"
  python3 << METRICS_EOF
import json
import numpy as np

with open("${OUTPUT_DIR}/sampling_metrics.json") as f:
    metrics = json.load(f)

losses = metrics.get("losses", [])
accs = metrics.get("text_accuracies", [])
mask_ratios = metrics.get("mask_ratios", [])

if losses:
    losses = np.array(losses)
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Mean loss: {losses.mean():.4f}")
    print(f"  Min loss: {losses.min():.4f}")

if accs:
    accs = np.array(accs)
    print(f"  Final accuracy: {accs[-1]:.4f}")
    print(f"  Mean accuracy: {accs.mean():.4f}")

if mask_ratios:
    print(f"  Final mask ratio: {mask_ratios[-1]:.4f}")
METRICS_EOF
fi

echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
  echo "Sampling completed successfully!"
  echo ""
  echo "Output files:"
  echo "  Results:     ${OUTPUT_DIR}/results.json"
  echo "  Texts:       ${OUTPUT_DIR}/texts.txt"
  if [ "${WITH_LOSS_LOGGING}" = "true" ]; then
    echo "  Metrics:     ${OUTPUT_DIR}/sampling_metrics.json"
    echo "  Metrics log: ${OUTPUT_DIR}/sampling_metrics.jsonl"
  fi
  echo "  Log:         ${LOG_FILE}"
else
  echo "Sampling FAILED with exit code ${EXIT_CODE}"
  echo ""
  echo "Troubleshooting:"
  echo "  1. Check log: ${LOG_FILE}"
  echo "  2. Check for errors:"
  echo "     grep -E 'ERROR|Error|Traceback' ${LOG_FILE}"
  echo "  3. Check checkpoint compatibility:"
  echo "     python -c \"import torch; c=torch.load('${CHECKPOINT}', map_location='cpu', weights_only=False); print(list(c.keys()))\""
  echo "  4. Try with fewer samples:"
  echo "     NUM_SAMPLES=3 BATCH_SIZE=1 STEPS=100 bash sample_l2t.sh"
fi

echo "=========================================="

exit $EXIT_CODE
