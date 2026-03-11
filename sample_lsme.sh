#!/bin/bash
# File: sample_lsme.sh
# LSME (Latent-Steered Masked Editing) sampling & evaluation script
# All paths relative to repo root — copy to any machine and run directly.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "LSME Sampling & Evaluation Script"
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
    exit 1
  fi
  echo "Auto-detected checkpoint: ${CHECKPOINT}"
fi

# Latent data & metadata for attribute encoding
LATENT_DIR="${LATENT_DIR:-mmdit_latent/data/latent_shards}"
METADATA_FILE="${METADATA_FILE:-mmdit_latent/data/metadata.json}"
if [ ! -d "${LATENT_DIR}" ] || [ ! -f "${METADATA_FILE}" ]; then
  echo "ERROR: Set LATENT_DIR and METADATA_FILE for attribute encoding"
  echo "  LATENT_DIR=${LATENT_DIR}"
  echo "  METADATA_FILE=${METADATA_FILE}"
  exit 1
fi

# LSME editing parameters
ATTRIBUTE="${ATTRIBUTE:-sentiment}"
TARGET_VALUE="${TARGET_VALUE:-positive}"
MASK_RATIO="${MASK_RATIO:-0.3}"
MASK_MODE="${MASK_MODE:-random}"
STEPS="${STEPS:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Input texts to edit
INPUT_FILE="${INPUT_FILE:-}"
INPUT_TEXT="${INPUT_TEXT:-}"
if [ -z "${INPUT_FILE}" ] && [ -z "${INPUT_TEXT}" ]; then
  echo "ERROR: Provide INPUT_FILE or INPUT_TEXT"
  exit 1
fi

# Evaluation settings
RUN_EVAL="${RUN_EVAL:-true}"
FLUENCY_MODEL="${FLUENCY_MODEL:-gpt2}"
CLASSIFIER="${CLASSIFIER:-distilbert-base-uncased-finetuned-sst-2-english}"

# Output — all relative
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-mmdit_latent/results/lsme/${TIMESTAMP}}"

# ============================================================
# GPU DETECTION & ADAPTIVE BATCH SIZE
# ============================================================
GPU_MEMORY_GB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
  GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')
  GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))

  if [ "${BATCH_SIZE}" -eq 32 ]; then
    if [ "${GPU_MEMORY_GB}" -ge 140 ]; then BATCH_SIZE=64
    elif [ "${GPU_MEMORY_GB}" -ge 80 ]; then BATCH_SIZE=32
    elif [ "${GPU_MEMORY_GB}" -ge 40 ]; then BATCH_SIZE=16
    elif [ "${GPU_MEMORY_GB}" -ge 24 ]; then BATCH_SIZE=8
    else BATCH_SIZE=4; fi
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
if [ ! -d "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}/model.pt" ]; then
  echo "ERROR: Invalid checkpoint: ${CHECKPOINT}"; exit 1
fi
if [ -n "${INPUT_FILE}" ] && [ ! -f "${INPUT_FILE}" ]; then
  echo "ERROR: Input file not found: ${INPUT_FILE}"; exit 1
fi

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "LSME Configuration"
echo "=========================================="
echo "  Checkpoint:    ${CHECKPOINT}"
echo "  Attribute:     ${ATTRIBUTE} -> ${TARGET_VALUE}"
echo "  Mask ratio:    ${MASK_RATIO}"
echo "  Steps:         ${STEPS}"
echo "  Max length:    ${MAX_LENGTH}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# ============================================================
# RUN LSME + EVAL
# ============================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p mmdit_latent/results/logs
LOG_FILE="mmdit_latent/results/logs/lsme_${TIMESTAMP}.log"

LSME_OUTPUT="${OUTPUT_DIR}/lsme_output.json"

LSME_CMD="python -m mmdit_latent.scripts.run_lsme \
  --checkpoint_path ${CHECKPOINT} \
  --latent_dir ${LATENT_DIR} \
  --metadata_file ${METADATA_FILE} \
  --attribute ${ATTRIBUTE} \
  --target_value ${TARGET_VALUE} \
  --mask_ratio ${MASK_RATIO} \
  --mask_mode ${MASK_MODE} \
  --steps ${STEPS} \
  --temperature ${TEMPERATURE} \
  --max_length ${MAX_LENGTH} \
  --batch_size ${BATCH_SIZE} \
  --output_file ${LSME_OUTPUT}"

if [ -n "${INPUT_FILE}" ]; then
  LSME_CMD="${LSME_CMD} --input_file ${INPUT_FILE}"
elif [ -n "${INPUT_TEXT}" ]; then
  LSME_CMD="${LSME_CMD} --input_text '${INPUT_TEXT}'"
fi

set +e
eval "${LSME_CMD}" 2>&1 | tee "${LOG_FILE}"
LSME_EXIT=$?
set -e

if [ $LSME_EXIT -ne 0 ]; then
  echo "ERROR: LSME failed (exit ${LSME_EXIT}). Check ${LOG_FILE}"; exit $LSME_EXIT
fi

if [ "${RUN_EVAL}" = "true" ] && [ -f "${LSME_OUTPUT}" ]; then
  EVAL_OUTPUT="${OUTPUT_DIR}/eval_results"
  mkdir -p "${EVAL_OUTPUT}"
  set +e
  python -m mmdit_latent.scripts.run_eval \
    --results_file "${LSME_OUTPUT}" \
    --output_dir "${EVAL_OUTPUT}" \
    --fluency_model "${FLUENCY_MODEL}" \
    --classifier_name "${CLASSIFIER}" \
    --device cuda \
    2>&1 | tee -a "${LOG_FILE}"
  set -e
fi

echo ""
echo "=========================================="
echo "LSME COMPLETE"
echo "=========================================="
echo "  Output: ${LSME_OUTPUT}"
echo "  Log:    ${LOG_FILE}"
echo "=========================================="

exit 0
