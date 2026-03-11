#!/bin/bash
# File: sample_mmdit_latent.sh
# Sampling & eval script for mmdit_latent (fixed-latent conditioned text generation)
# All paths relative to repo root — copy to any machine and run directly.
set -xeuo pipefail

# Get the repo root directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "MMDiT-Latent Sampling & Evaluation Script"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "=========================================="
echo ""

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# ============================================================
# CONFIGURATION
# ============================================================

# Checkpoint (REQUIRED)
CHECKPOINT="${CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ]; then
  SAVE_DIR="${SAVE_DIR:-mmdit_latent/results/checkpoints}"
  RUN_NAME="${RUN_NAME:-mmdit-latent-training}"
  CKPT_DIR="${SAVE_DIR}/${RUN_NAME}"

  if [ -d "${CKPT_DIR}/latest" ]; then
    CHECKPOINT="${CKPT_DIR}/latest"
  elif [ -d "${CKPT_DIR}" ]; then
    CHECKPOINT=$(find "${CKPT_DIR}" -name "model.pt" -printf '%h\n' 2>/dev/null | sort | tail -1 || true)
  fi

  if [ -z "${CHECKPOINT}" ] || [ ! -d "${CHECKPOINT}" ]; then
    echo "ERROR: No checkpoint found. Set CHECKPOINT=/path/to/checkpoint_dir/"
    echo "  (directory containing model.pt, config.yaml, tokenizer files)"
    echo "  Searched in: ${CKPT_DIR}"
    exit 1
  fi
  echo "Auto-detected checkpoint: ${CHECKPOINT}"
fi

# Latent data for conditioning
LATENT_PATH="${LATENT_PATH:-}"  # .npy or .pt file with latent vectors

# Sampling parameters
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-128}"
MIN_P="${MIN_P:-0.0}"
MAX_LENGTH="${MAX_LENGTH:-}"  # Override generation length (e.g., 128, 512, 4096)

# Output — all relative
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-mmdit_latent/results/samples/${TIMESTAMP}}"
SAMPLES_PATH="${OUTPUT_DIR}/samples.pt"

# PPL evaluation
RUN_PPL="${RUN_PPL:-true}"
PPL_MODEL="${PPL_MODEL:-gpt2-large}"
PPL_TOKENIZER="${PPL_TOKENIZER:-gpt2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

# ============================================================
# GPU DETECTION & ADAPTIVE BATCH SIZE
# ============================================================
GPU_MEMORY_GB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
  GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')
  GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
  echo "Detected ${GPU_COUNT} GPUs, minimum memory: ${GPU_MEMORY_GB}GB"

  if [ "${BATCH_SIZE}" -eq 16 ]; then
    if [ "${GPU_MEMORY_GB}" -ge 140 ]; then
      BATCH_SIZE=64
    elif [ "${GPU_MEMORY_GB}" -ge 80 ]; then
      BATCH_SIZE=32
    elif [ "${GPU_MEMORY_GB}" -ge 40 ]; then
      BATCH_SIZE=16
    elif [ "${GPU_MEMORY_GB}" -ge 24 ]; then
      BATCH_SIZE=8
    else
      BATCH_SIZE=4
    fi
    echo "Adaptive batch size: ${BATCH_SIZE}"
  fi
fi

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo ""
echo "Running pre-flight checks..."

if [ ! -d "${CHECKPOINT}" ]; then
  echo "ERROR: Checkpoint directory not found: ${CHECKPOINT}"
  exit 1
fi
if [ ! -f "${CHECKPOINT}/model.pt" ]; then
  echo "ERROR: model.pt not found in checkpoint: ${CHECKPOINT}"
  exit 1
fi
echo "Checkpoint: ${CHECKPOINT}"

if [ -n "${LATENT_PATH}" ]; then
  if [ ! -f "${LATENT_PATH}" ]; then
    echo "ERROR: Latent file not found: ${LATENT_PATH}"
    exit 1
  fi
  echo "Latent path: ${LATENT_PATH}"
else
  echo "Latent path: (none - model will use null_latent)"
fi

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Sampling Configuration"
echo "=========================================="
echo "  Checkpoint:      ${CHECKPOINT}"
echo "  Latent path:     ${LATENT_PATH:-none}"
echo "  Num samples:     ${NUM_SAMPLES}"
echo "  Batch size:      ${BATCH_SIZE}"
echo "  Denoising steps: ${NUM_DENOISING_STEPS}"
echo "  Min-p:           ${MIN_P}"
echo "  Max length:      ${MAX_LENGTH:-model default}"
echo "  Output:          ${OUTPUT_DIR}"
echo "  Run PPL eval:    ${RUN_PPL}"
echo "=========================================="
echo ""

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p mmdit_latent/results/logs
LOG_FILE="mmdit_latent/results/logs/sample_${TIMESTAMP}.log"

# ============================================================
# STEP 1: GENERATE SAMPLES
# ============================================================
echo "Step 1: Generating ${NUM_SAMPLES} samples..."
echo ""

GENERATE_CMD="python mmdit_latent/eval/generate_samples.py \
  path=${CHECKPOINT} \
  batch_size=${BATCH_SIZE} \
  num_samples=${NUM_SAMPLES} \
  num_denoising_steps=${NUM_DENOISING_STEPS} \
  min_p=${MIN_P} \
  samples_path=${SAMPLES_PATH}"

if [ -n "${LATENT_PATH}" ]; then
  GENERATE_CMD="${GENERATE_CMD} latent_path=${LATENT_PATH}"
fi

if [ -n "${MAX_LENGTH}" ]; then
  GENERATE_CMD="${GENERATE_CMD} max_length=${MAX_LENGTH}"
fi

set +e
eval "${GENERATE_CMD}" 2>&1 | tee "${LOG_FILE}"
GEN_EXIT=$?
set -e

if [ $GEN_EXIT -ne 0 ]; then
  echo ""
  echo "ERROR: Sample generation failed with exit code ${GEN_EXIT}"
  echo "Check log: ${LOG_FILE}"
  exit $GEN_EXIT
fi

if [ ! -f "${SAMPLES_PATH}" ]; then
  echo "ERROR: Samples file not created: ${SAMPLES_PATH}"
  exit 1
fi

echo ""
echo "Generated samples saved to: ${SAMPLES_PATH}"

# ============================================================
# STEP 2: DECODE SAMPLES TO TEXT
# ============================================================
echo ""
echo "Step 2: Decoding samples to text..."

DECODED_PATH="${OUTPUT_DIR}/samples.json"

set +e
python mmdit_latent/eval/decode.py --path "${SAMPLES_PATH}" 2>&1 | tee -a "${LOG_FILE}"
DECODE_EXIT=$?
set -e

if [ $DECODE_EXIT -eq 0 ] && [ -f "${DECODED_PATH}" ]; then
  echo "Decoded to: ${DECODED_PATH}"

  echo ""
  echo "--- First 5 samples ---"
  python3 << PREVIEW_EOF
import json
with open("${DECODED_PATH}") as f:
    texts = json.load(f)
for i, text in enumerate(texts[:5]):
    print(f"  [{i+1}] {text[:200]}{'...' if len(text) > 200 else ''}")
    print()
print(f"Total: {len(texts)} samples")
PREVIEW_EOF
else
  echo "WARNING: Decoding failed, continuing to PPL eval on raw tokens"
fi

# ============================================================
# STEP 3: GENERATIVE PERPLEXITY (GPT-2)
# ============================================================
if [ "${RUN_PPL}" = "true" ]; then
  echo ""
  echo "Step 3: Computing generative perplexity..."

  METRICS_PATH="${OUTPUT_DIR}/metrics_ppl.json"

  set +e
  python mmdit_latent/eval/generative_ppl.py \
    samples_path="${SAMPLES_PATH}" \
    model_tokenizer="${PPL_TOKENIZER}" \
    pretrained_model="${PPL_MODEL}" \
    batch_size="${EVAL_BATCH_SIZE}" \
    metrics_path="${METRICS_PATH}" \
    torch_compile=false \
    2>&1 | tee -a "${LOG_FILE}"
  PPL_EXIT=$?
  set -e

  if [ $PPL_EXIT -eq 0 ] && [ -f "${METRICS_PATH}" ]; then
    echo ""
    echo "--- Perplexity Results ---"
    python3 << PPL_EOF
import json
with open("${METRICS_PATH}") as f:
    m = json.load(f)
print(f"  PPL:            {m.get('ppl', 'N/A'):.2f}")
print(f"  Avg NLL:        {m.get('avg_nll', 'N/A'):.4f}")
print(f"  Median NLL:     {m.get('median_nll', 'N/A'):.4f}")
print(f"  Accuracy:       {m.get('acc', 'N/A'):.4f}")
print(f"  Total tokens:   {m.get('tokens', 'N/A')}")
print(f"  Skipped:        {m.get('skipped_batches', 0)}")
PPL_EOF
  else
    echo "WARNING: PPL evaluation failed (exit ${PPL_EXIT})"
  fi
fi

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "=========================================="
echo "SAMPLING COMPLETE"
echo "=========================================="
echo "Output files:"
echo "  Samples (tokens):  ${SAMPLES_PATH}"
[ -f "${OUTPUT_DIR}/samples.json" ] && echo "  Samples (text):    ${OUTPUT_DIR}/samples.json"
[ -f "${OUTPUT_DIR}/metrics_ppl.json" ] && echo "  PPL metrics:       ${OUTPUT_DIR}/metrics_ppl.json"
echo "  Log:               ${LOG_FILE}"
echo "=========================================="

exit 0
