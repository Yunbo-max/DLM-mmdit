# DLM-MMDiT — Latent-Conditioned Diffusion Language Models with MMDiT

This repository contains codebases for latent-conditioned text diffusion using MMDiT (Multimodal Diffusion Transformer).

## Quick Start (mmdit_latent)

All paths are relative — copy the repo to any machine and run directly.

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download models locally (one-time, ~16GB)
python -m mmdit_latent.download_models
# Saves to: mmdit_latent/data/models/bert-base-uncased/
#            mmdit_latent/data/models/Qwen3-Embedding-8B/

# 3. Preprocess data (Option C: chunk-based with local models)
#    Choose 4096 (full document) or 512 (fast experiments):

# 4096 seq len (default) → mmdit_latent/data/
python -m mmdit_latent.preprocess_data \
  --dataset Skylion007/openwebtext \
  --max_seq_len 4096 \
  --latent_dim 32 \
  --device cuda:0,cuda:1  # multi-GPU for faster encoding

# 512 seq len (faster) → mmdit_latent/data_512/
python -m mmdit_latent.preprocess_data \
  --dataset Skylion007/openwebtext \
  --max_seq_len 512 \
  --latent_dim 32 \
  --output_dir mmdit_latent/data_512 \
  --device cuda:0,cuda:1

# Note: --device cuda:0 for single GPU, cuda:0,cuda:1 for two GPUs
# If OOM, reduce --batch_size (default 256, try 64 or 32)

# 4. Train
# 4096 (default):
bash train_mmdit_latent.sh

# 512:
CONFIG_NAME=mdlm_mmdit_latent_512 \
LATENT_DATA_ROOT=mmdit_latent/data_512 \
  bash train_mmdit_latent.sh

# 5. Sample (any length <= max_seq_len)
CHECKPOINT=mmdit_latent/results/checkpoints/mmdit-latent-training/latest \
  bash sample_mmdit_latent.sh

# Generate shorter text (128 tokens) from same model:
CHECKPOINT=mmdit_latent/results/checkpoints/mmdit-latent-training/latest \
MAX_LENGTH=128 \
  bash sample_mmdit_latent.sh
```

## Directory Structure

```
DLM-mmdit/
├── mmdit_latent/                # Main package (self-contained)
│   ├── data/                    # Preprocessed data + local models
│   │   ├── models/              # Downloaded models (step 2)
│   │   │   ├── bert-base-uncased/
│   │   │   └── Qwen3-Embedding-8B/
│   │   ├── train_data.jsonl
│   │   ├── validation_data.jsonl
│   │   ├── latent_shards/       # shard_0000.npy, shard_0001.npy, ...
│   │   └── metadata.json
│   ├── results/                 # All outputs
│   │   ├── checkpoints/         # Training checkpoints
│   │   ├── samples/             # Generated samples
│   │   ├── lsme/                # LSME editing results
│   │   ├── eval/                # Evaluation results
│   │   ├── logs/                # Training/sampling logs
│   │   └── wandb/               # W&B logs
│   ├── models/                  # Model definitions
│   │   ├── mmdit_latent.py      # MMDiTWithLatentConditioning
│   │   ├── mmdit_block.py       # MMDiT joint attention block
│   │   └── dit.py               # Base DIT components
│   ├── configs/                 # Hydra configs
│   │   ├── mdlm_mmdit_latent.yaml      # 4096 seq len (default)
│   │   ├── mdlm_mmdit_latent_512.yaml  # 512 seq len (fast)
│   │   ├── lsme_yelp.yaml
│   │   └── ...
│   ├── eval/                    # Evaluation scripts
│   ├── scripts/                 # LSME entry points
│   ├── latent_utils/            # Attribute encoding, interpolation
│   ├── evaluation/              # 6-pillar DLM-Eval Suite
│   ├── download_models.py       # Download tokenizer + encoder locally
│   ├── preprocess_data.py       # Data preprocessing pipeline
│   ├── train_latent_dit.py      # Training script
│   ├── sampling.py              # Samplers (MDLM, GIDD, HDLM)
│   ├── trainer_latent.py        # Training loop
│   ├── data_simple.py           # Dataset (JSONL + sharded latents)
│   └── sample_lsme.py           # LSMESampler
│
├── baseline/                    # Baseline reproductions (HDLM, MDLM)
├── train_mmdit_latent.sh        # Training launcher
├── sample_mmdit_latent.sh       # Sampling launcher
├── sample_lsme.sh               # LSME editing launcher
└── README.md
```

## Data Preprocessing

**Step 2: Download models first** (if not already done):
```bash
python -m mmdit_latent.download_models
```
This saves `bert-base-uncased` and `Qwen3-Embedding-8B` into `mmdit_latent/data/models/`. All preprocessing and training commands use these local copies by default — no internet needed after this step.

**Step 3: Preprocess** using Option C (chunk-based): documents are split into `max_seq_len`-token chunks by BERT tokenizer, each chunk gets its own Qwen3-Embedding-8B latent (32-dim). This ensures tight latent-text alignment.

```bash
# Default: 4096-token chunks → mmdit_latent/data/
python -m mmdit_latent.preprocess_data \
  --dataset Skylion007/openwebtext \
  --max_seq_len 4096 \
  --device cuda:0,cuda:1 \
  --batch_size 64

# 512-token chunks for faster training
python -m mmdit_latent.preprocess_data \
  --dataset Skylion007/openwebtext \
  --max_seq_len 512 \
  --output_dir mmdit_latent/data_512 \
  --device cuda:0,cuda:1 \
  --batch_size 64

# From local text file
python -m mmdit_latent.preprocess_data \
  --text_file /path/to/texts.txt \
  --device cuda:0,cuda:1 \
  --batch_size 64

# Limit for testing
python -m mmdit_latent.preprocess_data \
  --dataset Skylion007/openwebtext \
  --max_docs 10000
```

Output format:
- **JSONL** index (lazy loading, handles 1B+ samples)
- **Sharded latents** (100K samples per `.npy` shard, memory-mapped)

## Training

```bash
# 4096 seq len (default config)
bash train_mmdit_latent.sh

# 512 seq len (faster)
CONFIG_NAME=mdlm_mmdit_latent_512 bash train_mmdit_latent.sh

# Custom data path
LATENT_DATA_ROOT=mmdit_latent/data_512 \
CONFIG_NAME=mdlm_mmdit_latent_512 \
  bash train_mmdit_latent.sh

# Resume from checkpoint
RESUME=mmdit_latent/results/checkpoints/mmdit-latent-training/latest \
  bash train_mmdit_latent.sh
```

Environment variables:
| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG_NAME` | `mdlm_mmdit_latent` | Hydra config name (4096) or `mdlm_mmdit_latent_512` |
| `TRAIN_BS` | auto | Per-GPU batch size (auto from GPU memory) |
| `MODEL_SIZE` | `small` | `tiny`, `small`, `base`, `1B` |
| `LATENT_DIM` | `32` | Must match preprocessing |
| `LATENT_DATA_ROOT` | `mmdit_latent/data` | Path to preprocessed data |
| `SAVE_DIR` | `mmdit_latent/results/checkpoints` | Checkpoint save directory |
| `RESUME` | `null` | Checkpoint to resume from |

## Sampling

Train with `max_seq_len=4096`, generate any length <= 4096:

```bash
# Default length (uses model's max_seq_len)
CHECKPOINT=mmdit_latent/results/checkpoints/mmdit-latent-training/latest \
  bash sample_mmdit_latent.sh

# Short text (128 tokens)
CHECKPOINT=... MAX_LENGTH=128 bash sample_mmdit_latent.sh

# Long text (4096 tokens)
CHECKPOINT=... MAX_LENGTH=4096 bash sample_mmdit_latent.sh

# With latent conditioning
CHECKPOINT=... LATENT_PATH=my_latent.npy bash sample_mmdit_latent.sh

# Or directly via Python
python mmdit_latent/eval/generate_samples.py \
  path=mmdit_latent/results/checkpoints/mmdit-latent-training/latest \
  num_samples=100 num_denoising_steps=128 \
  max_length=512 \
  latent_path=my_latent.npy \
  samples_path=mmdit_latent/results/samples/my_samples.pt
```

## LSME — Latent-Steered Masked Editing

SDEdit-style text editing using partial masking + latent steering. Uses a pretrained mmdit_latent checkpoint.

```bash
CHECKPOINT=mmdit_latent/results/checkpoints/mmdit-latent-training/latest \
LATENT_DIR=mmdit_latent/data/latent_shards \
METADATA_FILE=mmdit_latent/data/metadata.json \
INPUT_TEXT="The food was terrible and the service was slow." \
ATTRIBUTE=sentiment TARGET_VALUE=positive \
  bash sample_lsme.sh
```

### 6-Pillar Evaluation

| Pillar | Metrics | Direction |
|--------|---------|-----------|
| Fluency | PPL (GPT-2), Grammar | Lower PPL = better |
| Controllability | Classifier accuracy | Higher = better |
| Edit Quality | BLEU, ROUGE-L, BERTScore | Higher = better |
| Latent Geometry | SSS, MTS | Higher = smoother |
| Diversity | Self-BLEU, Distinct-1/2/3 | Lower Self-BLEU = better |
| Efficiency | Time, NFE, Tokens/sec | Lower time = better |

## Evaluation

```bash
# Validation loss
python mmdit_latent/eval/loss.py \
  path=mmdit_latent/results/checkpoints/mmdit-latent-training/latest

# Self-correction
python mmdit_latent/eval/self_correction.py \
  path=mmdit_latent/results/checkpoints/mmdit-latent-training/latest \
  samples_path=mmdit_latent/results/samples/samples.pt \
  corrected_samples_path=mmdit_latent/results/samples/corrected.pt
```

## Model Architecture

**MMDiTWithLatentConditioning**: Text tokens and latent tokens attend to each other via joint attention at every block.

- Text pathway: BERT vocab embedding + positional encoding
- Latent pathway: `Linear(32→1536) → LayerNorm → GELU → Linear(1536→768)`
- Joint attention: MMDiT blocks with per-modality FFN + AdaLN
- Output: text logits only (latent stream discarded)

Forward: `model(token_ids, timestep, latents=None, attention_mask=None)`

## Configuration

Hydra configs in `mmdit_latent/configs/`:

| Config | max_seq_len | Batch size | Use case |
|--------|-------------|------------|----------|
| `mdlm_mmdit_latent` | 4096 | 4 | Full document generation |
| `mdlm_mmdit_latent_512` | 512 | 32 | Short text, fast training |

Override any value: `training.train_batch_size=16 model.latent_dim=64`

## Acknowledgements

Built upon [GIDD](https://github.com/dvruette/gidd/), [MDLM](https://github.com/kuleshov-group/mdlm), [ConGenBench](https://github.com/princeton-nlp/ConGenBench).
