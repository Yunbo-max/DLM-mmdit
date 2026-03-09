# LSME — Latent-Steered Masked Editing

SDEdit-style text editing for discrete masked diffusion language models using partial masking + latent replacement + joint denoising through MMDiT.

## Overview

LSME (Latent-Steered Masked Editing) enables controllable text editing by:
1. Partially masking source tokens (controlled by `mask_ratio`)
2. Computing an entry timestep from the mask ratio
3. Running reverse diffusion from that timestep with a target attribute latent as clean conditioning through MMDiT joint attention blocks

This produces edited text that preserves unmasked content while steering the regenerated tokens toward the target attribute (e.g., positive sentiment, formal style, different topic).

## Installation

```bash
# Create environment
conda create -n lsme python=3.10 -y && conda activate lsme

# Install dependencies
pip install -r requirements.txt

# OR install individual packages
pip install torch>=2.0 transformers datasets sentence-transformers
pip install evaluate nltk scikit-learn numpy tqdm omegaconf
```

## Project Structure

```
lsme/
├── __init__.py                          # Module aliasing for self-contained packaging
├── README.md                            # This file
│
├── sample_lsme.py                       # Core: LSMESampler (LSME editing algorithm)
│
├── data/                                # Phase 3A: Data preprocessing
│   ├── __init__.py                      #   (adapted from MDLM repo)
│   ├── dataloader.py                    #   Dataset loading, tokenization, DataLoaders
│   └── preprocessing.py                 #   Text cleaning, detokenization
│
├── latent_utils/                        # Latent space utilities
│   ├── __init__.py
│   ├── attribute_encoder.py             #   Attribute → latent centroid encoding
│   └── interpolation.py                 #   SLERP, LERP, directional editing
│
├── evaluation/                          # Phase 3C: Evaluation (6-pillar DLM-Eval Suite)
│   ├── __init__.py
│   ├── eval_suite.py                    #   DLMEvalSuite — unified 6-pillar runner
│   ├── metrics/
│   │   ├── fluency.py                   #   Pillar 1: PPL (GPT-2), grammar errors
│   │   ├── controllability.py           #   Pillar 2: Classifier accuracy
│   │   ├── edit_quality.py              #   Pillar 3: BLEU, ROUGE-L, BERTScore
│   │   ├── latent_geometry.py           #   Pillar 4: SSS, MTS (novel metrics)
│   │   ├── diversity.py                 #   Pillar 5: Self-BLEU, Distinct-n
│   │   └── efficiency.py               #   Pillar 6: Wall-clock, NFE, tokens/sec
│   └── benchmarks/
│       ├── yelp_sentiment.py            #   Yelp: negative → positive
│       ├── amazon_topic.py              #   Amazon: electronics → books
│       └── formality.py                 #   GYAFC: informal → formal
│
├── scripts/                             # Entry points
│   ├── run_lsme.py                      #   Run LSME editing
│   ├── run_eval.py                      #   Run DLM-Eval Suite
│   ├── run_geometry.py                  #   Run latent geometry analysis
│   └── run_baselines.py                 #   Run baseline comparison
│
├── configs/                             # Hydra-style YAML configs
│   ├── lsme_yelp.yaml                   #   Yelp sentiment editing experiment
│   ├── lsme_amazon.yaml                 #   Amazon topic transfer experiment
│   └── eval.yaml                        #   Evaluation suite settings
│
└── mmdit_latent/                        # Bundled: MMDiT training + inference codebase
    ├── models/
    │   ├── mmdit_latent.py              #   MMDiTWithLatentConditioning
    │   ├── mmdit_block.py               #   Joint attention block
    │   └── dit.py                       #   Base DIT model
    ├── sampling.py                      #   Sampler, MDLMSampler, get_sampler()
    ├── checkpoints.py                   #   load_checkpoint()
    ├── diffusion_process.py             #   Noise schedules, masking
    ├── loss.py                          #   MDLM/GIDD/HDLM loss functions
    ├── data_simple.py                   #   Training data loader
    ├── utils.py                         #   sample_categorical(), etc.
    ├── train_latent_dit.py              #   Training entry point
    ├── trainer_latent.py                #   Training loop
    └── configs/                         #   Model/data/optimizer configs
        ├── mdlm_mmdit_latent.yaml       #     Base: 768d, 12 blocks, 12 heads
        ├── model/{tiny,small,base,1B}.yaml
        ├── data/{owt,lm1b,fineweb}.yaml
        └── optimizer/{adam,psgd}.yaml
```

### Code Organization (3 Phases)

| Phase | Directory | Source | Description |
|-------|-----------|--------|-------------|
| Data Preprocessing | `data/` | Adapted from [MDLM](https://github.com/kuleshov-group/mdlm) | Tokenization, dataset loading, text cleaning |
| Methodology (Novel) | `sample_lsme.py`, `latent_utils/` | Written from scratch | LSME algorithm, attribute encoding, latent interpolation |
| Evaluation | `evaluation/` | Adapted from [ConGenBench](https://github.com/princeton-nlp/ConGenBench), [discrete-diffusion-guidance](https://github.com/naver-ai/discrete-diffusion-guidance) | 6-pillar eval suite, benchmarks |

## Quick Start

### 1. Train the MMDiT model

```bash
python -m lsme.mmdit_latent.train_latent_dit \
    --config-name mdlm_mmdit_latent \
    --config-path lsme/mmdit_latent/configs
```

### 2. Run LSME editing (Yelp sentiment)

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint path/to/checkpoint.pt \
    --config lsme/configs/lsme_yelp.yaml \
    --latent_dir path/to/latents/ \
    --metadata_file path/to/metadata.json \
    --output results/lsme_yelp/
```

### 3. Evaluate results

```bash
python -m lsme.scripts.run_eval \
    --results_file results/lsme_yelp/results.json \
    --target_attribute POSITIVE \
    --output_dir results/eval/
```

### 4. Run baselines

```bash
python -m lsme.scripts.run_baselines \
    --baseline_results_dir results/baselines/ \
    --output_dir results/comparison/
```

### 5. Latent geometry analysis

```bash
python -m lsme.scripts.run_geometry \
    --checkpoint path/to/checkpoint.pt \
    --latent_dir path/to/latents/ \
    --metadata_file path/to/metadata.json \
    --output_dir results/geometry/
```

## Dependencies

### Bundled (included in this directory)

| Module | Source | Why bundled |
|--------|--------|-------------|
| `mmdit_latent/` | `../mmdit_latent/` (project-local) | Core model architecture, training, and inference — LSME extends its sampling logic |

### External (pip install)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0 | Deep learning framework |
| transformers | >=4.30 | GPT-2 (PPL), DistilBERT (classifier), tokenizers |
| datasets | >=2.14 | HuggingFace datasets (Yelp, Amazon) |
| sentence-transformers | >=2.2 | all-MiniLM-L6-v2 (sentence embeddings for SSS) |
| evaluate | >=0.4 | BERTScore metric |
| nltk | >=3.8 | BLEU, Self-BLEU |
| scikit-learn | >=1.0 | Silhouette score, t-SNE |
| numpy | >=1.24 | Array operations |
| omegaconf | >=2.3 | Config loading |
| tqdm | >=4.65 | Progress bars |

### Models (downloaded on first use)

| Model | Size | Source | Purpose |
|-------|------|--------|---------|
| gpt2 | 548 MB | HuggingFace | Perplexity evaluation (Pillar 1) |
| distilbert-base-uncased-finetuned-sst-2-english | 268 MB | HuggingFace | Sentiment classification (Pillar 2) |
| all-MiniLM-L6-v2 | 91 MB | sentence-transformers | Sentence embeddings for SSS (Pillar 4) |

## Datasets

| Dataset | Task | Size | Download |
|---------|------|------|----------|
| Yelp Review Polarity | Sentiment editing (neg→pos) | ~560K test | `datasets.load_dataset("yelp_polarity")` |
| Amazon Reviews | Topic transfer (electronics→books) | ~400K test | `datasets.load_dataset("amazon_polarity")` |
| GYAFC | Formality transfer (informal→formal) | ~50K | Manual download required |

## Baselines

| Method | Type | Runner |
|--------|------|--------|
| MDLM | Unconditional generation | `scripts/run_baselines.py` |
| ReMDM | Remasking (editing without latent) | `scripts/run_baselines.py` |
| LatentOps | ODE-based latent traversal | `scripts/run_baselines.py` |
| DiffusER | Edit-based diffusion | `scripts/run_baselines.py` |

## Ablation Study

| Variant | Config key | What changes |
|---------|-----------|--------------|
| Full LSME | `mask_ratio=0.3` | All components |
| mask_ratio sweep | `mask_ratio=[0.1,0.3,0.5,0.7]` | Amount of masking |
| mask_mode | `mask_mode=[random,entropy,suffix]` | Masking strategy |
| No latent steering | Remove target latent | Editing without attribute control |

## Configs

| Config | Purpose |
|--------|---------|
| `configs/lsme_yelp.yaml` | Yelp sentiment editing (mask_ratio, steps, target, eval models) |
| `configs/lsme_amazon.yaml` | Amazon topic transfer (electronics→books) |
| `configs/eval.yaml` | DLM-Eval Suite settings (fluency model, classifier, encoder) |

## Evaluation: 6-Pillar DLM-Eval Suite

| Pillar | Metrics | Direction |
|--------|---------|-----------|
| 1. Fluency | Perplexity (GPT-2), Grammar errors | Lower PPL = better |
| 2. Controllability | Classifier accuracy, Attribute scores | Higher = better |
| 3. Edit Quality | BLEU, ROUGE-L, BERTScore, Edit distance | Higher BLEU = better |
| 4. Latent Geometry | SSS, MTS, Cluster separation | Higher = smoother |
| 5. Diversity | Self-BLEU, Distinct-1/2/3 | Lower Self-BLEU = more diverse |
| 6. Efficiency | Wall-clock time, NFE, Tokens/sec | Lower time = better |

**Novel metrics (our contribution):**
- **SSS (Semantic Smoothness Score)**: Mean cosine similarity between adjacent texts along a latent interpolation path. 1.0 = smooth, 0.0 = abrupt.
- **MTS (Monotonic Transition Score)**: Fraction of steps where classifier confidence increases monotonically. 1.0 = ideal, 0.5 = random.

## Hardware Requirements

- GPU: 1x NVIDIA A100 40GB (for training), 1x V100 16GB (for inference/eval)
- RAM: 32GB+
- Disk: ~5GB for models + datasets + checkpoints

## Reused Code Attribution

| Component | Source Repo | What was adapted |
|-----------|------------|------------------|
| Data loading | [MDLM](https://github.com/kuleshov-group/mdlm) | Tokenizer loading, dataset pipeline, detokenizers |
| PPL evaluation | [discrete-diffusion-guidance](https://github.com/naver-ai/discrete-diffusion-guidance) | Chunked perplexity computation for long sequences |
| Diversity metrics | [ConGenBench](https://github.com/princeton-nlp/ConGenBench) | Per-sentence Distinct-N metric variant |
