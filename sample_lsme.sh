#!/bin/bash
# File: sample_lsme.sh
# LSME (Latent-Steered Masked Editing) sampling & evaluation script
# Uses mmdit_latent/sample_lsme.py (LSMESampler) for attribute-steered text editing
# Then runs 6-pillar DLM-Eval Suite
set -xeuo pipefail

echo "=========================================="
echo "LSME Sampling & Evaluation Script"
echo "=========================================="
echo "Latent-Steered Masked Editing for"
echo "controllable text generation/editing."
echo "=========================================="
echo ""

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# ============================================================
# CONFIGURATION
# ============================================================

# Checkpoint (mmdit_latent format: directory with model.pt, config.yaml)
CHECKPOINT="${CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ]; then
  SAVE_DIR="${SAVE_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved}"
  RUN_NAME="${RUN_NAME:-mmdit-latent}"
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
LATENT_DIR="${LATENT_DIR:-}"
METADATA_FILE="${METADATA_FILE:-}"
if [ -z "${LATENT_DIR}" ] || [ -z "${METADATA_FILE}" ]; then
  echo "ERROR: Set LATENT_DIR and METADATA_FILE for attribute encoding"
  echo "  LATENT_DIR=/path/to/latent_npy_dir/"
  echo "  METADATA_FILE=/path/to/metadata.json"
  echo ""
  echo "  metadata.json format:"
  echo '  {"sample_001.npy": {"sentiment": "positive"}, ...}'
  exit 1
fi

# LSME editing parameters
ATTRIBUTE="${ATTRIBUTE:-sentiment}"
TARGET_VALUE="${TARGET_VALUE:-positive}"
MASK_RATIO="${MASK_RATIO:-0.3}"
MASK_MODE="${MASK_MODE:-random}"    # random, entropy, suffix
STEPS="${STEPS:-100}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Input texts to edit
INPUT_FILE="${INPUT_FILE:-}"
INPUT_TEXT="${INPUT_TEXT:-}"
if [ -z "${INPUT_FILE}" ] && [ -z "${INPUT_TEXT}" ]; then
  echo "ERROR: Provide INPUT_FILE or INPUT_TEXT"
  echo "  INPUT_FILE=/path/to/texts.txt (one per line)"
  echo "  INPUT_TEXT='This is a single text to edit.'"
  exit 1
fi

# Evaluation settings
RUN_EVAL="${RUN_EVAL:-true}"
FLUENCY_MODEL="${FLUENCY_MODEL:-gpt2}"
CLASSIFIER="${CLASSIFIER:-distilbert-base-uncased-finetuned-sst-2-english}"

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved/lsme_results/${TIMESTAMP}}"

# ============================================================
# GPU DETECTION & ADAPTIVE BATCH SIZE
# ============================================================
GPU_MEMORY_GB=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
  GPU_MEMORY_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')
  GPU_MEMORY_GB=$((GPU_MEMORY_MB / 1024))
  echo "Detected ${GPU_COUNT} GPUs, minimum memory: ${GPU_MEMORY_GB}GB"

  if [ "${BATCH_SIZE}" -eq 32 ]; then
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

if [ ! -d "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}/model.pt" ]; then
  echo "ERROR: Invalid checkpoint: ${CHECKPOINT}"
  echo "  Expected directory with model.pt, config.yaml"
  exit 1
fi
echo "Checkpoint: ${CHECKPOINT}"

if [ ! -d "${LATENT_DIR}" ]; then
  echo "ERROR: Latent directory not found: ${LATENT_DIR}"
  exit 1
fi
echo "Latent dir: ${LATENT_DIR}"

if [ ! -f "${METADATA_FILE}" ]; then
  echo "ERROR: Metadata file not found: ${METADATA_FILE}"
  exit 1
fi
echo "Metadata: ${METADATA_FILE}"

if [ -n "${INPUT_FILE}" ] && [ ! -f "${INPUT_FILE}" ]; then
  echo "ERROR: Input file not found: ${INPUT_FILE}"
  exit 1
fi

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "LSME Configuration"
echo "=========================================="
echo "  Checkpoint:    ${CHECKPOINT}"
echo "  Latent dir:    ${LATENT_DIR}"
echo "  Metadata:      ${METADATA_FILE}"
echo ""
echo "  Attribute:     ${ATTRIBUTE}"
echo "  Target value:  ${TARGET_VALUE}"
echo "  Mask ratio:    ${MASK_RATIO}"
echo "  Mask mode:     ${MASK_MODE}"
echo "  Steps:         ${STEPS}"
echo "  Temperature:   ${TEMPERATURE}"
echo "  Max length:    ${MAX_LENGTH}"
echo "  Batch size:    ${BATCH_SIZE}"
echo ""
if [ -n "${INPUT_FILE}" ]; then
  INPUT_COUNT=$(wc -l < "${INPUT_FILE}" | tr -d ' ')
  echo "  Input file:    ${INPUT_FILE} (${INPUT_COUNT} lines)"
else
  echo "  Input text:    ${INPUT_TEXT:0:80}..."
fi
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Run eval:      ${RUN_EVAL}"
echo "=========================================="
echo ""

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p sample_logs
LOG_FILE="sample_logs/lsme_${TIMESTAMP}.log"

# ============================================================
# STEP 1: RUN LSME EDITING
# ============================================================
echo "Step 1: Running LSME editing..."
echo ""

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
  echo ""
  echo "ERROR: LSME editing failed with exit code ${LSME_EXIT}"
  echo "Check log: ${LOG_FILE}"
  exit $LSME_EXIT
fi

if [ ! -f "${LSME_OUTPUT}" ]; then
  echo "ERROR: LSME output not created: ${LSME_OUTPUT}"
  exit 1
fi

# Show some results
echo ""
echo "--- LSME Editing Results ---"
python3 << SHOW_EOF
import json
with open("${LSME_OUTPUT}") as f:
    results = json.load(f)
print(f"Edited {len(results)} texts")
print()
for r in results[:3]:
    print(f"  Source:  {r['source'][:120]}{'...' if len(r['source']) > 120 else ''}")
    print(f"  Edited:  {r['edited'][:120]}{'...' if len(r['edited']) > 120 else ''}")
    print()
SHOW_EOF

# ============================================================
# STEP 2: RUN 6-PILLAR EVALUATION
# ============================================================
if [ "${RUN_EVAL}" = "true" ]; then
  echo ""
  echo "Step 2: Running DLM-Eval Suite (6 pillars)..."
  echo ""

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
  EVAL_EXIT=$?
  set -e

  if [ $EVAL_EXIT -eq 0 ] && [ -f "${EVAL_OUTPUT}/eval_results.json" ]; then
    echo ""
    echo "--- Evaluation Results ---"
    python3 << EVAL_EOF
import json
with open("${EVAL_OUTPUT}/eval_results.json") as f:
    e = json.load(f)

pillars = [
    ("Fluency", "fluency", ["ppl_mean"]),
    ("Controllability", "controllability", ["accuracy"]),
    ("Edit Quality", "edit_quality", ["bleu_mean", "rouge_l_mean", "bertscore_mean"]),
    ("Diversity", "diversity", ["distinct_1", "distinct_2", "self_bleu"]),
]

for name, key, metrics in pillars:
    if key in e:
        vals = ", ".join(f"{m}={e[key].get(m, 'N/A')}" for m in metrics)
        print(f"  {name}: {vals}")
EVAL_EOF
  else
    echo "WARNING: Evaluation failed (exit ${EVAL_EXIT})"
  fi
fi

# ============================================================
# STEP 3: MASK RATIO SWEEP (optional)
# ============================================================
RUN_SWEEP="${RUN_SWEEP:-false}"

if [ "${RUN_SWEEP}" = "true" ]; then
  echo ""
  echo "Step 3: Running mask ratio sweep..."
  echo ""

  SWEEP_DIR="${OUTPUT_DIR}/sweep"
  mkdir -p "${SWEEP_DIR}"

  for MR in 0.1 0.3 0.5 0.7; do
    echo "--- Mask ratio: ${MR} ---"
    SWEEP_OUTPUT="${SWEEP_DIR}/lsme_mr${MR}.json"

    SWEEP_CMD="python -m mmdit_latent.scripts.run_lsme \
      --checkpoint_path ${CHECKPOINT} \
      --latent_dir ${LATENT_DIR} \
      --metadata_file ${METADATA_FILE} \
      --attribute ${ATTRIBUTE} \
      --target_value ${TARGET_VALUE} \
      --mask_ratio ${MR} \
      --mask_mode ${MASK_MODE} \
      --steps ${STEPS} \
      --temperature ${TEMPERATURE} \
      --max_length ${MAX_LENGTH} \
      --batch_size ${BATCH_SIZE} \
      --output_file ${SWEEP_OUTPUT}"

    if [ -n "${INPUT_FILE}" ]; then
      SWEEP_CMD="${SWEEP_CMD} --input_file ${INPUT_FILE}"
    elif [ -n "${INPUT_TEXT}" ]; then
      SWEEP_CMD="${SWEEP_CMD} --input_text '${INPUT_TEXT}'"
    fi

    set +e
    eval "${SWEEP_CMD}" 2>&1 | tee -a "${LOG_FILE}"
    set -e

    # Evaluate each sweep point
    if [ "${RUN_EVAL}" = "true" ] && [ -f "${SWEEP_OUTPUT}" ]; then
      SWEEP_EVAL="${SWEEP_DIR}/eval_mr${MR}"
      mkdir -p "${SWEEP_EVAL}"
      set +e
      python -m mmdit_latent.scripts.run_eval \
        --results_file "${SWEEP_OUTPUT}" \
        --output_dir "${SWEEP_EVAL}" \
        --device cuda \
        2>&1 | tee -a "${LOG_FILE}"
      set -e
    fi
  done

  echo ""
  echo "Sweep results saved to: ${SWEEP_DIR}"
fi

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "=========================================="
echo "LSME COMPLETE"
echo "=========================================="
echo "Output files:"
echo "  LSME output:    ${LSME_OUTPUT}"
[ -d "${OUTPUT_DIR}/eval_results" ] && echo "  Eval results:    ${OUTPUT_DIR}/eval_results/eval_results.json"
[ "${RUN_SWEEP}" = "true" ] && echo "  Sweep results:   ${OUTPUT_DIR}/sweep/"
echo "  Log:             ${LOG_FILE}"
echo ""
echo "To run baselines for comparison:"
echo "  python -m mmdit_latent.scripts.run_baselines \\"
echo "    --baseline_results_dir results/baselines/ \\"
echo "    --output_dir results/comparison/"
echo ""
echo "To run geometry analysis:"
echo "  python -m mmdit_latent.scripts.run_geometry \\"
echo "    --checkpoint_path ${CHECKPOINT} \\"
echo "    --latent_dir ${LATENT_DIR} \\"
echo "    --metadata_file ${METADATA_FILE}"
echo "=========================================="

exit 0
