# MM-LDLM — Multimodal Latent Diffusion Language Models

This repository contains three codebases for latent-conditioned text diffusion using MMDiT (Multimodal Diffusion Transformer), plus baseline reproductions and data preprocessing tools.

## Repository Structure

```
MM-LDLM/
├── latentDLM_mmdit/     # Full joint MMDiT: dual diffusion (text masked + latent continuous)
├── mmdit_latent/         # Fixed-latent MMDiT + LSME editing (latent conditioning, text-only diffusion)
├── baseline/             # Baseline reproductions (HDLM, MDLM, AR, GIDD+)
├── baseline_latent/      # Latent-focused baselines (DIT with latent conditioning)
├── latentIMG_mmdit/      # Image+text MMDiT (continuous & discrete)
├── preprocessed_data/    # Data preprocessing & distributed latent extraction
├── scripts/              # Shared training/utility scripts
│
├── train_mmdit_latent.sh # Training: mmdit_latent
├── scripts/train.sh      # Training: latentDLM_mmdit
├── sample_l2t.sh         # Sampling: latentDLM_mmdit
├── sample_mmdit_latent.sh# Sampling + eval: mmdit_latent
├── sample_lsme.sh        # Editing + eval: lsme
└── eval_samples.sh       # Standalone PPL evaluation
```

### Codebase Overview

| Codebase | Model | Text Diffusion | Latent Diffusion | Use Case |
|----------|-------|---------------|------------------|----------|
| `latentDLM_mmdit/` | `MultimodalMMDiT` | Masked (MDLM) | Continuous (DDIM) | Joint text-latent generation |
| `mmdit_latent/` | `MMDiTWithLatentConditioning` | Masked (MDLM) | None (fixed input) | Latent-conditioned text generation + LSME editing |

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Preprocessing

Extract text latents using `preprocessed_data/prepare_data_multi_gpu.py`:

```bash
# SONAR embeddings (1024d)
torchrun --nproc_per_node=2 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext --latent-model sonar --batch-size 128 \
  --max-samples 10000000 --output-dir preprocessed_data/sonar_1024d_full

# E5 embeddings (1024d)
torchrun --nproc_per_node=1 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext --latent-model e5 --batch-size 256 \
  --max-samples 10000000 --output-dir preprocessed_data/e5_1024d_full

# Qwen embeddings
torchrun --nproc_per_node=1 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext --latent-model qwen --batch-size 8 \
  --max-samples 10000000 --output-dir preprocessed_data/qwen_1024d_full
```

Output structure:
```
output-dir/
├── texts/train/*.txt
├── latents/train/*.npy
├── train_data.json
└── validation_data.json
```

---

## 1. `latentDLM_mmdit/` — Full Joint MMDiT

Dual diffusion: text tokens use masked diffusion (MDLM), latents use continuous diffusion (DDIM). Both modalities attend to each other through MMDiT joint attention blocks.

**Model**: `MultimodalMMDiT` — forward signature: `(text_tokens, latents, text_timesteps, latent_timesteps)`

### Training

```bash
# Using the stable training script (adaptive GPU/batch, error handling, distributed)
bash scripts/train.sh

# Or manually:
torchrun --nproc_per_node=8 latentDLM_mmdit/train_mmdit_stable.py \
  --config-name mmdit_stable \
  logging.run_name="mmdit-l2t-training" \
  logging.save_dir="./saved" \
  training.train_batch_size=4 \
  loss.loss_type="l2t" \
  model.latent_dim=32 \
  training.dtype=bf16
```

Key environment variables for `scripts/train.sh`:
| Variable | Default | Description |
|----------|---------|-------------|
| `L2T_CONFIG` | `mmdit_stable` | Hydra config name |
| `L2T_TRAIN_BS` | auto | Per-GPU batch size (auto-detected from GPU memory) |
| `LATENT_DIM` | `32` | Latent dimension |
| `LEARNING_RATE` | auto | LR with sqrt scaling |
| `TOKEN_DIR` | — | Path to tokenized data |
| `LATENT_DIR` | — | Path to precomputed latents |

### Sampling

```bash
# Using shell script (auto-detects checkpoint, adaptive batch)
CHECKPOINT=/path/to/checkpoint bash sample_l2t.sh

# Or directly:
python latentDLM_mmdit/sample_l2t_fixed.py \
  --checkpoint /path/to/checkpoint \
  --num_samples 100 \
  --num_steps 128 \
  --output results.json
```

### Evaluation

```bash
# Generate token samples
python latentDLM_mmdit/eval/generate_samples.py \
  path=/path/to/checkpoint \
  num_samples=256 num_denoising_steps=128 \
  samples_path=samples.pt \
  latent_path=/path/to/latents.npy  # optional

# Decode to text
python latentDLM_mmdit/eval/decode.py --path samples.pt

# Generative PPL (GPT-2)
python latentDLM_mmdit/eval/generative_ppl.py \
  samples_path=samples.pt \
  pretrained_model=gpt2-large \
  model_tokenizer=gpt2 \
  batch_size=1 metrics_path=metrics.json

# Validation loss
python latentDLM_mmdit/eval/loss.py \
  path=/path/to/checkpoint batch_size=32

# Self-correction (GIDD-style iterative refinement)
python latentDLM_mmdit/eval/self_correction.py \
  path=/path/to/checkpoint \
  samples_path=samples.pt \
  corrected_samples_path=corrected.pt \
  latent_path=/path/to/latents.npy  # optional
```

**Checkpoint format**: Single `.pth` file with `model_state_dict` + `config` dict, loaded via `checkpoints_mmdit.load_full_model()`.

---

## 2. `mmdit_latent/` — Fixed-Latent MMDiT

Simpler approach: latents are **not** diffused. They serve as fixed context for text generation through MMDiT joint attention. The DiT backbone from the baseline is swapped to MMDiT.

**Model**: `MMDiTWithLatentConditioning` — forward signature: `(indices, sigma, latents=None, attention_mask=None)`

### Training

```bash
# Using shell script (adaptive GPU/batch, distributed, error handling)
bash train_mmdit_latent.sh

# Or manually:
torchrun --nproc_per_node=1 mmdit_latent/train_latent_dit.py \
  --config-name mdlm_mmdit_latent \
  logging.run_name="mmdit-latent-training" \
  logging.save_dir="./saved" \
  training.train_batch_size=16 \
  model.latent_dim=768 \
  training.dtype=bf16 \
  training.compile_model=true
```

Key environment variables for `train_mmdit_latent.sh`:
| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_NAME` | `mdlm_mmdit_latent` | Hydra config name |
| `TRAIN_BS` | auto | Per-GPU batch size |
| `MODEL_SIZE` | `small` | Model size: `tiny`, `small`, `base`, `1B` |
| `LATENT_DIM` | `768` | Latent dimension (match encoder) |
| `LATENT_DATA_ROOT` | from config | Path to data with text + latent pairs |
| `RESUME` | `null` | Checkpoint path to resume from |

### Sampling

```bash
# Full pipeline: generate -> decode -> PPL
CHECKPOINT=/path/to/checkpoint bash sample_mmdit_latent.sh

# With latent conditioning:
CHECKPOINT=/path/to/checkpoint \
LATENT_PATH=/path/to/latents.npy \
NUM_SAMPLES=100 \
  bash sample_mmdit_latent.sh
```

### Evaluation

```bash
# Generate samples
python mmdit_latent/eval/generate_samples.py \
  path=/path/to/checkpoint \
  num_samples=256 num_denoising_steps=128 \
  samples_path=samples.pt \
  latent_path=/path/to/latents.npy  # optional

# Decode to text
python mmdit_latent/eval/decode.py --path samples.pt

# Generative PPL
python mmdit_latent/eval/generative_ppl.py \
  samples_path=samples.pt \
  pretrained_model=gpt2-large \
  model_tokenizer=gpt2 \
  batch_size=1 metrics_path=metrics.json

# Validation loss
python mmdit_latent/eval/loss.py \
  path=/path/to/checkpoint batch_size=32

# Self-correction
python mmdit_latent/eval/self_correction.py \
  path=/path/to/checkpoint \
  samples_path=samples.pt \
  corrected_samples_path=corrected.pt \
  latent_path=/path/to/latents.npy  # optional

# Compare samples (LaTeX diff visualization)
python mmdit_latent/eval/compare_samples.py \
  --original samples.pt --corrected corrected.pt
```

**Checkpoint format**: Directory with `model.pt`, `config.yaml`, `noise_schedule.pt`, tokenizer files, loaded via `checkpoints.load_checkpoint()`.

---

## 3. LSME — Latent-Steered Masked Editing (in `mmdit_latent/`)

SDEdit-style controllable text editing using partial masking + latent steering through MMDiT. Uses a pretrained `mmdit_latent` checkpoint (no separate training needed). LSME code lives inside `mmdit_latent/` alongside the base model:

- `mmdit_latent/sample_lsme.py` — `LSMESampler` (core editing algorithm)
- `mmdit_latent/latent_utils/` — Attribute encoding, SLERP/LERP interpolation
- `mmdit_latent/evaluation/` — 6-pillar DLM-Eval Suite
- `mmdit_latent/scripts/` — Entry points (`run_lsme.py`, `run_eval.py`, `run_geometry.py`, `run_baselines.py`)
- `mmdit_latent/configs/lsme_*.yaml` — Experiment configs

### Editing

```bash
# Using shell script (editing + 6-pillar eval)
CHECKPOINT=/path/to/mmdit_latent/checkpoint \
LATENT_DIR=/path/to/latents \
METADATA=/path/to/metadata.json \
  bash sample_lsme.sh

# Or directly:
python -m mmdit_latent.scripts.run_lsme \
  --checkpoint /path/to/checkpoint \
  --config mmdit_latent/configs/lsme_yelp.yaml \
  --latent_dir /path/to/latents/ \
  --metadata_file /path/to/metadata.json \
  --output results/lsme_yelp/
```

### Evaluation (6-Pillar DLM-Eval Suite)

```bash
python -m mmdit_latent.scripts.run_eval \
  --results_file results/lsme_yelp/results.json \
  --target_attribute POSITIVE \
  --output_dir results/eval/
```

| Pillar | Metrics | Direction |
|--------|---------|-----------|
| 1. Fluency | Perplexity (GPT-2), Grammar errors | Lower PPL = better |
| 2. Controllability | Classifier accuracy, Attribute scores | Higher = better |
| 3. Edit Quality | BLEU, ROUGE-L, BERTScore, Edit distance | Higher BLEU = better |
| 4. Latent Geometry | SSS, MTS, Cluster separation | Higher = smoother |
| 5. Diversity | Self-BLEU, Distinct-1/2/3 | Lower Self-BLEU = more diverse |
| 6. Efficiency | Wall-clock time, NFE, Tokens/sec | Lower time = better |

Novel metrics:
- **SSS (Semantic Smoothness Score)**: Mean cosine similarity between adjacent texts along a latent interpolation path.
- **MTS (Monotonic Transition Score)**: Fraction of steps where classifier confidence increases monotonically.

### Latent Geometry Analysis

```bash
python -m mmdit_latent.scripts.run_geometry \
  --checkpoint /path/to/checkpoint \
  --latent_dir /path/to/latents/ \
  --metadata_file /path/to/metadata.json \
  --output_dir results/geometry/
```

---

## Shared Evaluation Scripts

The `eval_samples.sh` script runs standalone PPL evaluation on any `.pt` samples file:

```bash
SAMPLES_PATH=/path/to/samples.pt bash eval_samples.sh

# Options:
PPL_MODEL=gpt2-large      # Default: gpt2-large
EVAL_BATCH_SIZE=1          # Default: 1
```

All three codebases share the same evaluation pipeline from `baseline/eval/`:
1. **generate_samples.py** — Generate token samples via reverse diffusion
2. **decode.py** — Decode token IDs to text (JSON output)
3. **generative_ppl.py** — Compute perplexity using GPT-2 (NaN-safe)
4. **loss.py** — Evaluate validation loss using the trained model's trainer
5. **self_correction.py** — GIDD-style iterative token refinement with early stopping
6. **compare_samples.py** — LaTeX diff visualization between original and corrected samples

---

## Baselines

```bash
# MDLM baseline
torchrun --nproc_per_node=1 baseline/train.py \
  --config-name mdlm \
  logging.run_name="test-openwebtext" \
  data.dataset_name="openwebtext" \
  training.train_batch_size=4 \
  training.compile_model=False

# Latent DIT baseline
torchrun --nproc_per_node=1 baseline_latent/train_latent_dit.py \
  --config-name mdlm_latent \
  logging.run_name="latent-full" \
  model.latent_dim=768 \
  model.use_latent_conditioning=true \
  training.train_batch_size=32

# Cross-attention latent baseline
torchrun --nproc_per_node=1 baseline_latent/train_cross_dit.py \
  --config-name mdlm_cross_attention \
  logging.run_name="cross-attention-training" \
  training.train_batch_size=16
```

---

## Image MMDiT

```bash
# Continuous image+text MMDiT
python latentIMG_mmdit/train_image_continuous.py \
  --data-root /path/to/coco2014/images \
  --epochs 50 --batch-size 8 \
  --dim-text 1024 --dim-image 512

# Discrete image+text MMDiT
python latentIMG_mmdit/train_image_discrete.py \
  --data-root /path/to/coco2014/images \
  --epochs 50 --batch-size 8 \
  --dim-text 768 --dim-image 512
```

---

## Configuration

All training scripts use [Hydra](https://hydra.cc/) for configuration management. Configs are in `<codebase>/configs/` with the following structure:

```
configs/
├── <main_config>.yaml     # Main training config (defaults + overrides)
├── model/                  # Model size variants (tiny, small, base, 1B)
├── data/                   # Dataset configs (owt, lm1b, fineweb)
├── optimizer/              # Optimizer configs (adam, psgd)
├── logging/                # Logging settings
├── generate.yaml           # Sample generation config
├── self_correction.yaml    # Self-correction config
└── eval.yaml               # Loss evaluation config
```

Override any config value via CLI: `training.train_batch_size=16 model.latent_dim=768`

---

## Notes

- Verify embedding dimension matches between encoder and model config (`latent_dim`).
- For quick pipeline checks, set `num_samples=10` or `max_samples=10`.
- All shell scripts support `CUDA_VISIBLE_DEVICES` for GPU selection and auto-detect GPU count/memory.
- Resume training from checkpoint: set `RESUME=/path/to/checkpoint` (for `mmdit_latent`) or `training.resume=/path` (Hydra override).

## Acknowledgements

Built upon:
* [GIDD](https://github.com/dvruette/gidd/)
* [MDLM](https://github.com/kuleshov-group/mdlm)
* [ConGenBench](https://github.com/princeton-nlp/ConGenBench)
* [discrete-diffusion-guidance](https://github.com/naver-ai/discrete-diffusion-guidance)
* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
