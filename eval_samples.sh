#!/bin/bash
# File: eval_samples.sh
# Evaluation script for generated samples (generative PPL + decode)
# Run after sample_l2t.sh to compute perplexity with GPT-2
set -xeuo pipefail

echo "=========================================="
echo "MM-LDLM Sample Evaluation Script"
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

# Path to generated samples (from sample_l2t.sh output or generate_samples.py)
SAMPLES_PATH="${SAMPLES_PATH:-}"
if [ -z "${SAMPLES_PATH}" ]; then
  echo "ERROR: Set SAMPLES_PATH=/path/to/samples.pt"
  echo "  This should be the .pt file from generate_samples.py or"
  echo "  the results.json from sample_l2t_fixed.py"
  exit 1
fi

# GPT-2 model for perplexity evaluation
PPL_MODEL="${PPL_MODEL:-gpt2-large}"
PPL_TOKENIZER="${PPL_TOKENIZER:-gpt2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

# Output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
METRICS_PATH="${METRICS_PATH:-$(dirname ${SAMPLES_PATH})/metrics_${TIMESTAMP}.json}"

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo "Running pre-flight checks..."

if [ ! -f "${SAMPLES_PATH}" ]; then
  echo "ERROR: Samples file not found: ${SAMPLES_PATH}"
  exit 1
fi
echo "Samples: ${SAMPLES_PATH}"
echo "PPL model: ${PPL_MODEL}"
echo "Metrics output: ${METRICS_PATH}"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "Evaluation Configuration"
echo "=========================================="
echo "  Samples: ${SAMPLES_PATH}"
echo "  PPL model: ${PPL_MODEL}"
echo "  PPL tokenizer: ${PPL_TOKENIZER}"
echo "  Eval batch size: ${EVAL_BATCH_SIZE}"
echo "  Metrics output: ${METRICS_PATH}"
echo "=========================================="
echo ""

# ============================================================
# CREATE LOG DIRECTORY
# ============================================================
mkdir -p eval_logs
LOG_FILE="eval_logs/eval_${TIMESTAMP}.log"

# ============================================================
# STEP 1: DECODE (if samples are .pt token tensors)
# ============================================================
SAMPLES_EXT="${SAMPLES_PATH##*.}"

if [ "${SAMPLES_EXT}" = "pt" ]; then
  echo "Step 1: Decoding token tensor to text..."

  DECODED_PATH="$(dirname ${SAMPLES_PATH})/$(basename ${SAMPLES_PATH} .pt).json"
  python latentDLM_mmdit/eval/decode.py --path "${SAMPLES_PATH}" 2>&1 | tee -a "${LOG_FILE}"

  if [ -f "${DECODED_PATH}" ]; then
    echo "Decoded to: ${DECODED_PATH}"
  else
    echo "WARNING: Decoded file not found at ${DECODED_PATH}, continuing with PPL on raw tokens"
  fi
  echo ""
fi

# ============================================================
# STEP 2: GENERATIVE PERPLEXITY (GPT-2)
# ============================================================
echo "Step 2: Computing generative perplexity..."
echo ""

set +e

python latentDLM_mmdit/eval/generative_ppl.py \
  samples_path="${SAMPLES_PATH}" \
  model_tokenizer="${PPL_TOKENIZER}" \
  pretrained_model="${PPL_MODEL}" \
  batch_size="${EVAL_BATCH_SIZE}" \
  metrics_path="${METRICS_PATH}" \
  torch_compile=false \
  2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?
set -e

# ============================================================
# POST-EVAL ANALYSIS
# ============================================================
echo ""
echo "=========================================="
echo "EVALUATION RESULTS"
echo "=========================================="

if [ -f "${METRICS_PATH}" ]; then
  python3 << RESULTS_EOF
import json

with open("${METRICS_PATH}") as f:
    m = json.load(f)

print(f"  Perplexity (PPL):  {m.get('ppl', 'N/A'):.2f}")
print(f"  Avg NLL:           {m.get('avg_nll', 'N/A'):.4f}")
print(f"  Median NLL:        {m.get('median_nll', 'N/A'):.4f}")
print(f"  Accuracy:          {m.get('acc', 'N/A'):.4f}")
print(f"  Total tokens:      {m.get('tokens', 'N/A')}")
print(f"  Skipped batches:   {m.get('skipped_batches', 0)}")
print(f"  Success rate:      {m.get('success_rate', 'N/A'):.1f}%")
RESULTS_EOF
else
  echo "  No metrics file found at ${METRICS_PATH}"
fi

echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
  echo "Evaluation completed successfully!"
  echo ""
  echo "Output files:"
  echo "  Metrics:  ${METRICS_PATH}"
  echo "  Log:      ${LOG_FILE}"
else
  echo "Evaluation FAILED with exit code ${EXIT_CODE}"
  echo ""
  echo "Troubleshooting:"
  echo "  1. Check log: ${LOG_FILE}"
  echo "  2. Verify samples file is valid:"
  echo "     python -c \"import torch; t=torch.load('${SAMPLES_PATH}', weights_only=True); print(t.shape, t.dtype)\""
  echo "  3. Try with smaller batch:"
  echo "     EVAL_BATCH_SIZE=1 bash eval_samples.sh"
fi

echo "=========================================="

exit $EXIT_CODE
