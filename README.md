# DLM-MMDiT — Latent-Conditioned Diffusion Language Models

This repository contains multiple approaches for latent-conditioned text diffusion:

| Method | Directory | Architecture | Latent Injection |
|--------|-----------|-------------|-----------------|
| **MMDiT-Latent** | `mmdit_latent/` | MMDiT + Hyper-Connections | Joint attention (2nd modality) |
| **LatentDLM-MMDiT** | `latentDLM_mmdit/` | External MMDiT package | Joint attention + dual heads |
| **Baseline** | `baseline/` | DIT | None (unconditional) |
| **Baseline-Latent** | `baseline_latent/` | DIT + AdaLN or Cross-Attention | AdaLN modulation / Cross-attn |

## Training Scripts

### MMDiT-Latent (main approach)

```bash
# Default (1 GPU, auto batch size)
bash train_mmdit_latent.sh

# 1B model
CUDA_VISIBLE_DEVICES=0 MODEL_SIZE=1B CONFIG_NAME=mdlm_mmdit_latent \
  TRAIN_BS=8 LEARNING_RATE=3e-4 COMPILE=false \
  bash train_mmdit_latent.sh

# Resume
RESUME=mmdit_latent/results/checkpoints/mmdit-latent-training/latest \
  bash train_mmdit_latent.sh
```

### LatentDLM-MMDiT (external MMDiT, l2t mode)

```bash
# L2T mode with mmdit_stable config (default)
CUDA_VISIBLE_DEVICES=0 CONFIG_NAME=mmdit_stable LOSS_TYPE=l2t \
  bash train_latentDLM_mmdit.sh

# Custom settings
CUDA_VISIBLE_DEVICES=0 CONFIG_NAME=mmdit_stable LOSS_TYPE=l2t \
  TRAIN_BS=4 LEARNING_RATE=5e-5 GRAD_CLIP_NORM=0.5 \
  bash train_latentDLM_mmdit.sh

# 1 GPU test
CUDA_VISIBLE_DEVICES=0 CONFIG_NAME=mmdit_stable LOSS_TYPE=l2t \
  TRAIN_BS=4 bash train_latentDLM_mmdit.sh
```

Note: Requires `pip install mmdit` (installs hyper-connections automatically).
Data paths are configured in `latentDLM_mmdit/configs/mmdit_stable.yaml`.

### Baseline (no latent)

```bash
# MDLM baseline
CUDA_VISIBLE_DEVICES=0 CONFIG_NAME=mdlm bash train_baseline.sh

# Custom batch/LR
CUDA_VISIBLE_DEVICES=0 TRAIN_BS=32 LEARNING_RATE=5e-4 \
  bash train_baseline.sh
```

### Baseline-Latent (DIT + latent)

```bash
# AdaLN conditioning (default)
CUDA_VISIBLE_DEVICES=0 bash train_baseline_latent.sh

# Cross-attention conditioning
CUDA_VISIBLE_DEVICES=0 CONDITIONING_TYPE=cross_attention \
  bash train_baseline_latent.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all | GPUs to use |
| `CONFIG_NAME` | varies | Hydra config name |
| `MODEL_SIZE` | `small` | `tiny`, `small`, `base`, `1B` |
| `TRAIN_BS` | auto | Per-GPU batch size |
| `LEARNING_RATE` | config default | Learning rate |
| `GRAD_CLIP_NORM` | config default | Gradient clipping |
| `WARMUP_STEPS` | config default | LR warmup steps |
| `COMPILE` | `true` | torch.compile |
| `RESUME` | `null` | Checkpoint path to resume |
| `LATENT_DATA_ROOT` | config default | Path to preprocessed data |
| `LATENT_DIM` | `32` | Latent embedding dimension |

## Data Preprocessing

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m mmdit_latent.preprocess_data \
  --dataset Skylion007/openwebtext \
  --max_seq_len 4096 \
  --latent_dim 32 \
  --batch_size 64

# Multi-GPU (no NCCL needed)
CUDA_VISIBLE_DEVICES=0 python -m mmdit_latent.preprocess_data \
  --dataset Skylion007/openwebtext \
  --worker_rank 0 --num_workers 2 \
  --output_dir mmdit_latent/data_part0

CUDA_VISIBLE_DEVICES=1 python -m mmdit_latent.preprocess_data \
  --dataset Skylion007/openwebtext \
  --worker_rank 1 --num_workers 2 \
  --output_dir mmdit_latent/data_part1

# Merge
python -m mmdit_latent.merge_shards \
  --inputs mmdit_latent/data_part0 mmdit_latent/data_part1 \
  --output mmdit_latent/data

# Cleanup bad entries (if preprocessing was interrupted)
python mmdit_latent/cleanup_jsonl.py /path/to/data
```

## Directory Structure

```
DLM-mmdit/
├── mmdit_latent/                # MMDiT with latent as 2nd modality
│   ├── models/
│   │   ├── mmdit_latent.py      # MMDiTWithLatentConditioning
│   │   ├── mmdit_block.py       # MMDiT joint attention blocks
│   │   ├── hyper_connections.py # Vendored hyper-connections (no pip needed)
│   │   └── dit.py               # Base DIT components
│   ├── configs/
│   │   ├── mdlm_mmdit_latent.yaml      # 4096 seq len
│   │   └── mdlm_mmdit_latent_512.yaml  # 512 seq len
│   ├── data/                    # Preprocessed data + local models
│   ├── results/                 # Checkpoints, samples, logs
│   ├── train_latent_dit.py      # Training script
│   ├── preprocess_data.py       # Data preprocessing
│   └── cleanup_jsonl.py         # Fix corrupted data
│
├── latentDLM_mmdit/             # External MMDiT package approach
│   ├── models/multimodal_mmdit.py
│   ├── improved_trainer_stable.py
│   ├── train_mmdit_stable.py
│   └── configs/mmdit_stable.yaml
│
├── baseline/                    # MDLM/HDLM without latent
│   ├── train.py
│   └── configs/
│
├── baseline_latent/             # DIT + latent (AdaLN / cross-attn)
│   ├── train_latent_dit.py      # AdaLN training
│   ├── train_cross_dit.py       # Cross-attention training
│   └── configs/
│
├── train_mmdit_latent.sh        # MMDiT-Latent launcher
├── train_latentDLM_mmdit.sh     # LatentDLM-MMDiT launcher
├── train_baseline.sh            # Baseline launcher
├── train_baseline_latent.sh     # Baseline-Latent launcher
└── README.md
```

## Architecture Comparison

| | Baseline DIT | Baseline-Latent (AdaLN) | Baseline-Latent (Cross-Attn) | MMDiT-Latent |
|---|---|---|---|---|
| **Latent injection** | None | Layer norm modulation | Text→latent cross-attention | Joint attention |
| **Latent is a modality** | No | No | No | Yes |
| **Attention type** | Self-attn | Self-attn | Self + cross-attn | Joint attn |
| **Hyper-connections** | No | No | No | Yes (2 streams) |
| **QK-RMSNorm** | No | No | No | Yes |

## Acknowledgements

Built upon [GIDD](https://github.com/dvruette/gidd/), [MDLM](https://github.com/kuleshov-group/mdlm), [MMDiT](https://github.com/lucidrains/mmdit), [Hyper-Connections](https://github.com/lucidrains/hyper-connections).
