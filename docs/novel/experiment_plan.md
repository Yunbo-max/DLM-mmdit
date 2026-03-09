# LSME Experiment Plan

**Project:** LSME (Latent-Steered Masked Editing)
**Mode:** MANUAL -- copy-paste commands, update status after each run
**Start date:** 2026-03-03
**Estimated duration:** 7 days (2026-03-03 to 2026-03-09)

---

## Experiment Index

| ID | Category | Description | Table | Day |
|----|----------|-------------|-------|-----|
| EXP-B01 | Baseline | MDLM (unconditional) on Yelp | Tab 4 | 1 |
| EXP-B02 | Baseline | ReMDM on Yelp | Tab 4 | 1 |
| EXP-B03 | Baseline | LatentOps on Yelp | Tab 4 | 1 |
| EXP-B04 | Baseline | DiffusER on Yelp | Tab 4 | 1 |
| EXP-B05 | Baseline | PLANNER on Yelp | Tab 4 | 1 |
| EXP-B06 | Baseline | LD4LG on Yelp | Tab 4 | 1 |
| EXP-B07 | Baseline | LatentOps on Amazon | Tab 5 | 1 |
| EXP-B08 | Baseline | DiffusER on Amazon | Tab 5 | 1 |
| EXP-B09 | Baseline | LatentOps on GYAFC | Tab 6 | 1 |
| EXP-B10 | Baseline | DiffusER on GYAFC | Tab 6 | 1 |
| EXP-M01 | Main | LSME r=0.3 on Yelp (sentiment neg->pos) | Tab 4 | 2 |
| EXP-M02 | Main | LSME r=0.5 on Yelp (sentiment neg->pos) | Tab 4 | 2 |
| EXP-M03 | Main | LSME r=0.7 on Yelp (sentiment neg->pos) | Tab 4 | 2 |
| EXP-M04 | Main | LSME r=0.3 on Amazon (topic elec->books) | Tab 5 | 2 |
| EXP-M05 | Main | LSME r=0.5 on Amazon (topic elec->books) | Tab 5 | 2 |
| EXP-M06 | Main | LSME r=0.3 on GYAFC (formality inf->formal) | Tab 6 | 3 |
| EXP-M07 | Main | LSME r=0.5 on GYAFC (formality inf->formal) | Tab 6 | 3 |
| EXP-M08 | Main | Eval + comparison on Yelp (all methods) | Tab 4 | 3 |
| EXP-M09 | Main | Eval + comparison on Amazon | Tab 5 | 3 |
| EXP-M10 | Main | Eval + comparison on GYAFC | Tab 6 | 3 |
| EXP-A01 | Ablation | Mask ratio r=0.1 on Yelp | Tab 7 | 4 |
| EXP-A02 | Ablation | Mask ratio r=0.3 on Yelp (reuse EXP-M01) | Tab 7 | 4 |
| EXP-A03 | Ablation | Mask ratio r=0.5 on Yelp (reuse EXP-M02) | Tab 7 | 4 |
| EXP-A04 | Ablation | Mask ratio r=0.7 on Yelp (reuse EXP-M03) | Tab 7 | 4 |
| EXP-A05 | Ablation | Mask ratio r=0.9 on Yelp | Tab 7 | 4 |
| EXP-A06 | Ablation | Mask mode: random (reuse EXP-M01) | Tab 8 | 4 |
| EXP-A07 | Ablation | Mask mode: entropy on Yelp | Tab 8 | 4 |
| EXP-A08 | Ablation | Mask mode: suffix on Yelp | Tab 8 | 4 |
| EXP-A09 | Ablation | Steps: 10 on Yelp | Tab 9 | 5 |
| EXP-A10 | Ablation | Steps: 50 on Yelp | Tab 9 | 5 |
| EXP-A11 | Ablation | Steps: 100 (reuse EXP-M01) | Tab 9 | 5 |
| EXP-A12 | Ablation | Steps: 500 on Yelp | Tab 9 | 5 |
| EXP-A13 | Ablation | Temperature: 0.5 on Yelp | Tab 10 | 5 |
| EXP-A14 | Ablation | Temperature: 0.8 on Yelp | Tab 10 | 5 |
| EXP-A15 | Ablation | Temperature: 1.0 (reuse EXP-M01) | Tab 10 | 5 |
| EXP-A16 | Ablation | Temperature: 1.2 on Yelp | Tab 10 | 5 |
| EXP-A17 | Ablation | Latent source: zeros on Yelp | Tab 11 | 5 |
| EXP-A18 | Ablation | Latent source: random on Yelp | Tab 11 | 5 |
| EXP-A19 | Ablation | Latent source: centroid (reuse EXP-M01) | Tab 11 | 5 |
| EXP-A20 | Ablation | Latent source: nearest neighbor on Yelp | Tab 11 | 5 |
| EXP-A21 | Ablation | Latent source: directional alpha=1.0 on Yelp | Tab 11 | 5 |
| EXP-G01 | Geometry | SSS/MTS: Neg->Pos (Yelp sentiment) | Tab 12 | 6 |
| EXP-G02 | Geometry | SSS/MTS: Elec->Books (Amazon topic) | Tab 12 | 6 |
| EXP-G03 | Geometry | SSS/MTS: Informal->Formal (GYAFC) | Tab 12 | 6 |
| EXP-G04 | Geometry | Cluster separation + variance ratio (Yelp) | Tab 12 | 6 |
| EXP-E01 | Efficiency | LSME 100 steps profiling (Yelp) | Tab 13 | 7 |
| EXP-E02 | Efficiency | LSME 50 steps profiling (Yelp) | Tab 13 | 7 |
| EXP-E03 | Efficiency | Baseline efficiency comparison | Tab 13 | 7 |

---

## Shared Paths

```bash
# Set these once at the start of each session:
export PROJ=/Users/yunbo/Documents/Agent/DLM-mmdit
export CKPT=checkpoints/mmdit_latent
export LATENT_DIR=data/latents
export META=data/latents/metadata.json
export YELP_INPUT=data/yelp_negatives.txt
export AMAZON_INPUT=data/amazon_electronics.txt
export GYAFC_INPUT=data/gyafc_informal.txt
```

---

## Prerequisites

Before starting any experiments, verify setup:

```bash
cd $PROJ

# 1. Check that lsme package is importable
python -c "from lsme.sample_lsme import LSMESampler; print('LSME OK')"

# 2. Check that checkpoint exists
ls -la $CKPT/

# 3. Check that latent data exists
ls -la $LATENT_DIR/*.npy | head -5
cat $META | python -m json.tool | head -20

# 4. Check that input files exist
wc -l $YELP_INPUT $AMAZON_INPUT $GYAFC_INPUT

# 5. Create results directory structure
mkdir -p results/{EXP-B01,EXP-B02,EXP-B03,EXP-B04,EXP-B05,EXP-B06,EXP-B07,EXP-B08,EXP-B09,EXP-B10}
mkdir -p results/{EXP-M01,EXP-M02,EXP-M03,EXP-M04,EXP-M05,EXP-M06,EXP-M07,EXP-M08,EXP-M09,EXP-M10}
mkdir -p results/{EXP-A01,EXP-A02,EXP-A03,EXP-A04,EXP-A05,EXP-A06,EXP-A07,EXP-A08}
mkdir -p results/{EXP-A09,EXP-A10,EXP-A11,EXP-A12,EXP-A13,EXP-A14,EXP-A15,EXP-A16}
mkdir -p results/{EXP-A17,EXP-A18,EXP-A19,EXP-A20,EXP-A21}
mkdir -p results/{EXP-G01,EXP-G02,EXP-G03,EXP-G04}
mkdir -p results/{EXP-E01,EXP-E02,EXP-E03}
mkdir -p results/baselines/{yelp,amazon,gyafc}
```

---

## Day 1 (2026-03-03): Baselines

All 10 baseline runs can execute in parallel. Each baseline must produce a JSON file in the format `[{"source": "...", "edited": "..."}, ...]` with 500 samples.

> **NOTE:** Baselines require their own repos/environments to be set up first. The commands below assume you have already generated baseline outputs and placed them in `results/baselines/`. If not, run each baseline's own generation script first (see experiments.tex Section 2.2 for repos), then evaluate with our suite.

### Step 1a: Generate baseline outputs (if not already done)

Each baseline should produce 500 edited/generated texts on the same test inputs. Place the JSON output files as follows:

```
results/baselines/yelp/mdlm.json       # EXP-B01
results/baselines/yelp/remdm.json      # EXP-B02
results/baselines/yelp/latentops.json   # EXP-B03
results/baselines/yelp/diffuser.json    # EXP-B04
results/baselines/yelp/planner.json     # EXP-B05
results/baselines/yelp/ld4lg.json       # EXP-B06
results/baselines/amazon/latentops.json # EXP-B07
results/baselines/amazon/diffuser.json  # EXP-B08
results/baselines/gyafc/latentops.json  # EXP-B09
results/baselines/gyafc/diffuser.json   # EXP-B10
```

### Step 1b: Evaluate baseline outputs through DLM-Eval Suite

**EXP-B01 through EXP-B06 (Yelp baselines):**

```bash
# EXP-B01..B06: Evaluate all Yelp baselines at once
python -m lsme.scripts.run_baselines \
    --baseline_results_dir results/baselines/yelp/ \
    --output_dir results/EXP-B01/eval/ \
    --target_attribute POSITIVE \
    --device cuda
```

**EXP-B07 through EXP-B08 (Amazon baselines):**

```bash
# EXP-B07..B08: Evaluate Amazon baselines
python -m lsme.scripts.run_baselines \
    --baseline_results_dir results/baselines/amazon/ \
    --output_dir results/EXP-B07/eval/ \
    --target_attribute books \
    --device cuda
```

**EXP-B09 through EXP-B10 (GYAFC baselines):**

```bash
# EXP-B09..B10: Evaluate GYAFC baselines
python -m lsme.scripts.run_baselines \
    --baseline_results_dir results/baselines/gyafc/ \
    --output_dir results/EXP-B09/eval/ \
    --target_attribute formal \
    --device cuda
```

### Day 1 Update

After all baselines complete, update this table:

| ID | Status | Attr-Acc | PPL | BLEU | BERTScore | Notes |
|----|--------|----------|-----|------|-----------|-------|
| EXP-B01 | [ ] | — | — | — | — | |
| EXP-B02 | [ ] | — | — | — | — | |
| EXP-B03 | [ ] | — | — | — | — | |
| EXP-B04 | [ ] | — | — | — | — | |
| EXP-B05 | [ ] | — | — | — | — | |
| EXP-B06 | [ ] | — | — | — | — | |
| EXP-B07 | [ ] | — | — | — | — | |
| EXP-B08 | [ ] | — | — | — | — | |
| EXP-B09 | [ ] | — | — | — | — | |
| EXP-B10 | [ ] | — | — | — | — | |

---

## Day 2 (2026-03-04): Main LSME Runs -- Yelp and Amazon

### EXP-M01: LSME r=0.3 on Yelp (sentiment neg->pos)

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-M01/output.json \
    --batch_size 32 \
    --device cuda
```

### EXP-M02: LSME r=0.5 on Yelp

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.5 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-M02/output.json \
    --batch_size 32 \
    --device cuda
```

### EXP-M03: LSME r=0.7 on Yelp

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.7 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-M03/output.json \
    --batch_size 32 \
    --device cuda
```

### EXP-M04: LSME r=0.3 on Amazon (topic electronics->books)

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute topic \
    --target_value books \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $AMAZON_INPUT \
    --output_file results/EXP-M04/output.json \
    --batch_size 32 \
    --device cuda
```

### EXP-M05: LSME r=0.5 on Amazon

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute topic \
    --target_value books \
    --mask_ratio 0.5 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $AMAZON_INPUT \
    --output_file results/EXP-M05/output.json \
    --batch_size 32 \
    --device cuda
```

### Day 2 Update

After all Day 2 runs complete, update this table:

| ID | Status | Samples | Runtime | Notes |
|----|--------|---------|---------|-------|
| EXP-M01 | [ ] | — | — | |
| EXP-M02 | [ ] | — | — | |
| EXP-M03 | [ ] | — | — | |
| EXP-M04 | [ ] | — | — | |
| EXP-M05 | [ ] | — | — | |

---

## Day 3 (2026-03-05): Main LSME Runs -- GYAFC + Evaluation for All Main Results

### EXP-M06: LSME r=0.3 on GYAFC (formality informal->formal)

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute formality \
    --target_value formal \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $GYAFC_INPUT \
    --output_file results/EXP-M06/output.json \
    --batch_size 32 \
    --device cuda
```

### EXP-M07: LSME r=0.5 on GYAFC

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute formality \
    --target_value formal \
    --mask_ratio 0.5 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $GYAFC_INPUT \
    --output_file results/EXP-M07/output.json \
    --batch_size 32 \
    --device cuda
```

### EXP-M08: Evaluate all Yelp results (LSME + baselines)

Run eval on each LSME result, then compare with baselines:

```bash
# Eval LSME r=0.3
python -m lsme.scripts.run_eval \
    --results_file results/EXP-M01/output.json \
    --output_dir results/EXP-M01/eval/ \
    --device cuda

# Eval LSME r=0.5
python -m lsme.scripts.run_eval \
    --results_file results/EXP-M02/output.json \
    --output_dir results/EXP-M02/eval/ \
    --device cuda

# Eval LSME r=0.7
python -m lsme.scripts.run_eval \
    --results_file results/EXP-M03/output.json \
    --output_dir results/EXP-M03/eval/ \
    --device cuda
```

### EXP-M09: Evaluate all Amazon results

```bash
# Eval LSME r=0.3
python -m lsme.scripts.run_eval \
    --results_file results/EXP-M04/output.json \
    --output_dir results/EXP-M04/eval/ \
    --device cuda

# Eval LSME r=0.5
python -m lsme.scripts.run_eval \
    --results_file results/EXP-M05/output.json \
    --output_dir results/EXP-M05/eval/ \
    --device cuda
```

### EXP-M10: Evaluate all GYAFC results

```bash
# Eval LSME r=0.3
python -m lsme.scripts.run_eval \
    --results_file results/EXP-M06/output.json \
    --output_dir results/EXP-M06/eval/ \
    --device cuda

# Eval LSME r=0.5
python -m lsme.scripts.run_eval \
    --results_file results/EXP-M07/output.json \
    --output_dir results/EXP-M07/eval/ \
    --device cuda
```

### Day 3 Update

| ID | Status | Attr-Acc | PPL | BLEU | BERTScore | Distinct-2 | Notes |
|----|--------|----------|-----|------|-----------|------------|-------|
| EXP-M01 | [ ] | — | — | — | — | — | |
| EXP-M02 | [ ] | — | — | — | — | — | |
| EXP-M03 | [ ] | — | — | — | — | — | |
| EXP-M04 | [ ] | — | — | — | — | — | |
| EXP-M05 | [ ] | — | — | — | — | — | |
| EXP-M06 | [ ] | — | — | — | — | — | |
| EXP-M07 | [ ] | — | — | — | — | — | |
| EXP-M08 | [ ] | — | — | — | — | — | Yelp comparison |
| EXP-M09 | [ ] | — | — | — | — | — | Amazon comparison |
| EXP-M10 | [ ] | — | — | — | — | — | GYAFC comparison |

---

## Day 4 (2026-03-06): Ablations -- Mask Ratio + Mask Mode

### Mask Ratio Ablation (Tab 7)

EXP-A02 (r=0.3), EXP-A03 (r=0.5), EXP-A04 (r=0.7) reuse EXP-M01, EXP-M02, EXP-M03 outputs. Only need to run r=0.1 and r=0.9.

**EXP-A01: Mask ratio r=0.1 on Yelp**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.1 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A01/output.json \
    --batch_size 32 \
    --device cuda
```

**EXP-A02: Reuse EXP-M01 (r=0.3)**

```bash
# No generation needed -- symlink or copy from EXP-M01
ln -sf ../../EXP-M01/output.json results/EXP-A02/output.json
```

**EXP-A03: Reuse EXP-M02 (r=0.5)**

```bash
ln -sf ../../EXP-M02/output.json results/EXP-A03/output.json
```

**EXP-A04: Reuse EXP-M03 (r=0.7)**

```bash
ln -sf ../../EXP-M03/output.json results/EXP-A04/output.json
```

**EXP-A05: Mask ratio r=0.9 on Yelp**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.9 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A05/output.json \
    --batch_size 32 \
    --device cuda
```

**Evaluate mask ratio ablation:**

```bash
# EXP-A01 eval
python -m lsme.scripts.run_eval \
    --results_file results/EXP-A01/output.json \
    --output_dir results/EXP-A01/eval/ \
    --device cuda

# EXP-A05 eval
python -m lsme.scripts.run_eval \
    --results_file results/EXP-A05/output.json \
    --output_dir results/EXP-A05/eval/ \
    --device cuda

# EXP-A02, A03, A04 already evaluated in Day 3 (EXP-M01, M02, M03)
```

### Mask Mode Ablation (Tab 8)

EXP-A06 (random) reuses EXP-M01. Only need entropy and suffix.

**EXP-A06: Reuse EXP-M01 (random mask mode)**

```bash
ln -sf ../../EXP-M01/output.json results/EXP-A06/output.json
```

**EXP-A07: Mask mode entropy on Yelp (r=0.3)**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode entropy \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A07/output.json \
    --batch_size 32 \
    --device cuda
```

**EXP-A08: Mask mode suffix on Yelp (r=0.3)**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode suffix \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A08/output.json \
    --batch_size 32 \
    --device cuda
```

**Evaluate mask mode ablation:**

```bash
python -m lsme.scripts.run_eval \
    --results_file results/EXP-A07/output.json \
    --output_dir results/EXP-A07/eval/ \
    --device cuda

python -m lsme.scripts.run_eval \
    --results_file results/EXP-A08/output.json \
    --output_dir results/EXP-A08/eval/ \
    --device cuda
```

### Day 4 Update

| ID | Status | Attr-Acc | PPL | BLEU | BERTScore | Notes |
|----|--------|----------|-----|------|-----------|-------|
| EXP-A01 | [ ] | — | — | — | — | r=0.1 |
| EXP-A02 | [ ] | — | — | — | — | r=0.3 (=EXP-M01) |
| EXP-A03 | [ ] | — | — | — | — | r=0.5 (=EXP-M02) |
| EXP-A04 | [ ] | — | — | — | — | r=0.7 (=EXP-M03) |
| EXP-A05 | [ ] | — | — | — | — | r=0.9 |
| EXP-A06 | [ ] | — | — | — | — | random (=EXP-M01) |
| EXP-A07 | [ ] | — | — | — | — | entropy |
| EXP-A08 | [ ] | — | — | — | — | suffix |

---

## Day 5 (2026-03-07): Ablations -- Steps, Temperature, Latent Source

### Diffusion Steps Ablation (Tab 9)

EXP-A11 (100 steps) reuses EXP-M01.

**EXP-A09: Steps=10 on Yelp**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.3 \
    --steps 10 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A09/output.json \
    --batch_size 32 \
    --device cuda
```

**EXP-A10: Steps=50 on Yelp**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.3 \
    --steps 50 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A10/output.json \
    --batch_size 32 \
    --device cuda
```

**EXP-A11: Reuse EXP-M01 (Steps=100)**

```bash
ln -sf ../../EXP-M01/output.json results/EXP-A11/output.json
```

**EXP-A12: Steps=500 on Yelp**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.3 \
    --steps 500 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A12/output.json \
    --batch_size 32 \
    --device cuda
```

**Evaluate steps ablation:**

```bash
python -m lsme.scripts.run_eval \
    --results_file results/EXP-A09/output.json \
    --output_dir results/EXP-A09/eval/ \
    --device cuda

python -m lsme.scripts.run_eval \
    --results_file results/EXP-A10/output.json \
    --output_dir results/EXP-A10/eval/ \
    --device cuda

python -m lsme.scripts.run_eval \
    --results_file results/EXP-A12/output.json \
    --output_dir results/EXP-A12/eval/ \
    --device cuda
```

### Temperature Ablation (Tab 10)

EXP-A15 (T=1.0) reuses EXP-M01.

**EXP-A13: Temperature=0.5 on Yelp**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 0.5 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A13/output.json \
    --batch_size 32 \
    --device cuda
```

**EXP-A14: Temperature=0.8 on Yelp**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 0.8 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A14/output.json \
    --batch_size 32 \
    --device cuda
```

**EXP-A15: Reuse EXP-M01 (T=1.0)**

```bash
ln -sf ../../EXP-M01/output.json results/EXP-A15/output.json
```

**EXP-A16: Temperature=1.2 on Yelp**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.2 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A16/output.json \
    --batch_size 32 \
    --device cuda
```

**Evaluate temperature ablation:**

```bash
python -m lsme.scripts.run_eval \
    --results_file results/EXP-A13/output.json \
    --output_dir results/EXP-A13/eval/ \
    --device cuda

python -m lsme.scripts.run_eval \
    --results_file results/EXP-A14/output.json \
    --output_dir results/EXP-A14/eval/ \
    --device cuda

python -m lsme.scripts.run_eval \
    --results_file results/EXP-A16/output.json \
    --output_dir results/EXP-A16/eval/ \
    --device cuda
```

### Latent Source Ablation (Tab 11)

> **NOTE:** The `run_lsme.py` script currently uses `--target_value` to select the centroid latent source. For the non-centroid ablation variants (zeros, random, nearest-neighbor, directional), you will need to either:
> (a) Add a `--latent_source` flag to `run_lsme.py`, or
> (b) Write a short wrapper script that calls `AttributeLatentEncoder` methods directly.
>
> The commands below assume option (a) has been implemented with values: `centroid`, `zeros`, `random`, `nearest_neighbor`, `directional`.

EXP-A19 (centroid) reuses EXP-M01.

**EXP-A17: Latent source = zeros (no latent)**

```bash
# REQUIRES: --latent_source zeros flag added to run_lsme.py
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --latent_source zeros \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A17/output.json \
    --batch_size 32 \
    --device cuda
```

**EXP-A18: Latent source = random**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --latent_source random \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A18/output.json \
    --batch_size 32 \
    --device cuda
```

**EXP-A19: Reuse EXP-M01 (centroid)**

```bash
ln -sf ../../EXP-M01/output.json results/EXP-A19/output.json
```

**EXP-A20: Latent source = nearest neighbor**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --latent_source nearest_neighbor \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A20/output.json \
    --batch_size 32 \
    --device cuda
```

**EXP-A21: Latent source = directional (alpha=1.0)**

```bash
python -m lsme.scripts.run_lsme \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --target_value positive \
    --latent_source directional \
    --directional_alpha 1.0 \
    --mask_ratio 0.3 \
    --steps 100 \
    --temperature 1.0 \
    --mask_mode random \
    --input_file $YELP_INPUT \
    --output_file results/EXP-A21/output.json \
    --batch_size 32 \
    --device cuda
```

**Evaluate latent source ablation:**

```bash
python -m lsme.scripts.run_eval \
    --results_file results/EXP-A17/output.json \
    --output_dir results/EXP-A17/eval/ \
    --device cuda

python -m lsme.scripts.run_eval \
    --results_file results/EXP-A18/output.json \
    --output_dir results/EXP-A18/eval/ \
    --device cuda

python -m lsme.scripts.run_eval \
    --results_file results/EXP-A20/output.json \
    --output_dir results/EXP-A20/eval/ \
    --device cuda

python -m lsme.scripts.run_eval \
    --results_file results/EXP-A21/output.json \
    --output_dir results/EXP-A21/eval/ \
    --device cuda
```

### Day 5 Update

| ID | Status | Attr-Acc | PPL | BLEU | BERTScore | Notes |
|----|--------|----------|-----|------|-----------|-------|
| EXP-A09 | [ ] | — | — | — | — | 10 steps |
| EXP-A10 | [ ] | — | — | — | — | 50 steps |
| EXP-A11 | [ ] | — | — | — | — | 100 steps (=M01) |
| EXP-A12 | [ ] | — | — | — | — | 500 steps |
| EXP-A13 | [ ] | — | — | — | — | T=0.5 |
| EXP-A14 | [ ] | — | — | — | — | T=0.8 |
| EXP-A15 | [ ] | — | — | — | — | T=1.0 (=M01) |
| EXP-A16 | [ ] | — | — | — | — | T=1.2 |
| EXP-A17 | [ ] | — | — | — | — | zeros |
| EXP-A18 | [ ] | — | — | — | — | random |
| EXP-A19 | [ ] | — | — | — | — | centroid (=M01) |
| EXP-A20 | [ ] | — | — | — | — | nearest neighbor |
| EXP-A21 | [ ] | — | — | — | — | directional |

---

## Day 6 (2026-03-08): Latent Geometry Analysis

### EXP-G01: SSS/MTS on Yelp (negative -> positive)

```bash
python -m lsme.scripts.run_geometry \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute sentiment \
    --source_value negative \
    --target_value positive \
    --n_pairs 10 \
    --n_points 10 \
    --n_samples 5 \
    --output_dir results/EXP-G01/ \
    --device cuda
```

### EXP-G02: SSS/MTS on Amazon (electronics -> books)

```bash
python -m lsme.scripts.run_geometry \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute topic \
    --source_value electronics \
    --target_value books \
    --n_pairs 10 \
    --n_points 10 \
    --n_samples 5 \
    --output_dir results/EXP-G02/ \
    --device cuda
```

### EXP-G03: SSS/MTS on GYAFC (informal -> formal)

```bash
python -m lsme.scripts.run_geometry \
    --checkpoint_path $CKPT \
    --latent_dir $LATENT_DIR \
    --metadata_file $META \
    --attribute formality \
    --source_value informal \
    --target_value formal \
    --n_pairs 10 \
    --n_points 10 \
    --n_samples 5 \
    --output_dir results/EXP-G03/ \
    --device cuda
```

### EXP-G04: Cluster separation and variance ratio (Yelp)

> Cluster separation metrics are computed as part of `run_geometry`. EXP-G01 already produces silhouette score and variance ratio for Yelp. If you need them separately for Amazon and GYAFC, re-run EXP-G02 and EXP-G03 -- those scripts also compute cluster_separation and variance_ratio.

```bash
# Verify cluster metrics exist in EXP-G01 output
cat results/EXP-G01/geometry_results.json | python -m json.tool
```

### Day 6 Update

| ID | Status | SSS | MTS | Silhouette | Var Ratio | Interp PPL | Notes |
|----|--------|-----|-----|------------|-----------|------------|-------|
| EXP-G01 | [ ] | — | — | — | — | — | Yelp sentiment |
| EXP-G02 | [ ] | — | — | — | — | — | Amazon topic |
| EXP-G03 | [ ] | — | — | — | — | — | GYAFC formality |
| EXP-G04 | [ ] | — | — | — | — | — | Cluster metrics |

---

## Day 7 (2026-03-09): Efficiency Profiling + Final Compilation

### EXP-E01: LSME 100 steps efficiency profiling

```bash
# Time LSME at 100 steps on Yelp (batch_size=32, 512 tokens)
python -c "
import time, torch, json
from pathlib import Path
from mmdit_latent.checkpoints import load_checkpoint
from lsme.sample_lsme import LSMESampler
from lsme.latent_utils.attribute_encoder import AttributeLatentEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, ns, tok, cfg = load_checkpoint('$CKPT', device=device)
model.eval()
sampler = LSMESampler(model, tok, ns)
encoder = AttributeLatentEncoder('$LATENT_DIR', '$META')
encoder.compute_attribute_centroids('sentiment')
z_target = encoder.get_target_latent('sentiment', 'positive', device=device)

with open('$YELP_INPUT') as f:
    texts = [l.strip() for l in f if l.strip()][:32]

# Warmup
_ = sampler.edit_from_text(texts[:4], z_target.unsqueeze(0).expand(4,-1),
                           mask_ratio=0.3, steps=10, show_progress=False)

# Profile
torch.cuda.synchronize()
t0 = time.perf_counter()
edited, _ = sampler.edit_from_text(texts, z_target.unsqueeze(0).expand(len(texts),-1),
                                    mask_ratio=0.3, steps=100, show_progress=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

total_tokens = len(texts) * 512
tps = total_tokens / elapsed
mem = torch.cuda.max_memory_allocated() / 1e9
params = sum(p.numel() for p in model.parameters()) / 1e6

results = {
    'method': 'LSME_100steps',
    'params_M': round(params, 1),
    'nfe': 100,
    'tps': round(tps, 1),
    'gpu_mem_gb': round(mem, 2),
    'wall_seconds': round(elapsed, 2),
    'batch_size': len(texts),
    'seq_len': 512,
}
Path('results/EXP-E01').mkdir(parents=True, exist_ok=True)
with open('results/EXP-E01/efficiency.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
"
```

### EXP-E02: LSME 50 steps efficiency profiling

```bash
# Same as EXP-E01 but with steps=50
python -c "
import time, torch, json
from pathlib import Path
from mmdit_latent.checkpoints import load_checkpoint
from lsme.sample_lsme import LSMESampler
from lsme.latent_utils.attribute_encoder import AttributeLatentEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, ns, tok, cfg = load_checkpoint('$CKPT', device=device)
model.eval()
sampler = LSMESampler(model, tok, ns)
encoder = AttributeLatentEncoder('$LATENT_DIR', '$META')
encoder.compute_attribute_centroids('sentiment')
z_target = encoder.get_target_latent('sentiment', 'positive', device=device)

with open('$YELP_INPUT') as f:
    texts = [l.strip() for l in f if l.strip()][:32]

# Warmup
_ = sampler.edit_from_text(texts[:4], z_target.unsqueeze(0).expand(4,-1),
                           mask_ratio=0.3, steps=10, show_progress=False)

# Profile
torch.cuda.synchronize()
t0 = time.perf_counter()
edited, _ = sampler.edit_from_text(texts, z_target.unsqueeze(0).expand(len(texts),-1),
                                    mask_ratio=0.3, steps=50, show_progress=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

total_tokens = len(texts) * 512
tps = total_tokens / elapsed
mem = torch.cuda.max_memory_allocated() / 1e9
params = sum(p.numel() for p in model.parameters()) / 1e6

results = {
    'method': 'LSME_50steps',
    'params_M': round(params, 1),
    'nfe': 50,
    'tps': round(tps, 1),
    'gpu_mem_gb': round(mem, 2),
    'wall_seconds': round(elapsed, 2),
    'batch_size': len(texts),
    'seq_len': 512,
}
Path('results/EXP-E02').mkdir(parents=True, exist_ok=True)
with open('results/EXP-E02/efficiency.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
"
```

### EXP-E03: Baseline efficiency comparison

> Baseline efficiency metrics (Params, NFE, TPS, GPU Mem) must be measured within each baseline's own environment. Record them manually into the table below after running each baseline.

```bash
# Collect all efficiency results into one table
python -c "
import json, glob
from pathlib import Path

all_results = {}
for f in sorted(glob.glob('results/EXP-E0*/efficiency.json')):
    with open(f) as fh:
        data = json.load(fh)
        all_results[data['method']] = data

# Print table
print(f\"{'Method':<20} {'Params(M)':>10} {'NFE':>6} {'TPS':>10} {'GPU(GB)':>10}\")
print('-' * 60)
for m, d in all_results.items():
    print(f\"{m:<20} {d['params_M']:>10.1f} {d['nfe']:>6} {d['tps']:>10.1f} {d['gpu_mem_gb']:>10.2f}\")
"
```

### Day 7 Update

| ID | Status | Params (M) | NFE | TPS | GPU Mem (GB) | Notes |
|----|--------|------------|-----|-----|--------------|-------|
| EXP-E01 | [ ] | — | 100 | — | — | LSME 100 steps |
| EXP-E02 | [ ] | — | 50 | — | — | LSME 50 steps |
| EXP-E03 | [ ] | — | — | — | — | Baselines |

---

## Multi-Seed Protocol

The experiments above run with a single seed. After initial results look reasonable, re-run the main experiments (EXP-M01 through EXP-M07) with 3 seeds to get mean +/- std:

```bash
# Example: EXP-M01 with 3 seeds
for SEED in 42 123 456; do
    python -m lsme.scripts.run_lsme \
        --checkpoint_path $CKPT \
        --latent_dir $LATENT_DIR \
        --metadata_file $META \
        --attribute sentiment \
        --target_value positive \
        --mask_ratio 0.3 \
        --steps 100 \
        --temperature 1.0 \
        --mask_mode random \
        --input_file $YELP_INPUT \
        --output_file results/EXP-M01/seed${SEED}/output.json \
        --batch_size 32 \
        --device cuda

    python -m lsme.scripts.run_eval \
        --results_file results/EXP-M01/seed${SEED}/output.json \
        --output_dir results/EXP-M01/seed${SEED}/eval/ \
        --device cuda
done
```

> **NOTE:** `run_lsme.py` does not currently have a `--seed` flag. You will need to either: (a) add `--seed` to the argparser and call `torch.manual_seed(seed)` at the start of main(), or (b) set `PYTHONHASHSEED` and `torch.manual_seed()` via an environment wrapper.

---

## How to Update This Plan After Each Run

After completing each experiment:

1. **Fill in the status column** with `[x]` (done), `[!]` (error), or `[-]` (skipped).
2. **Record key metrics** from the eval JSON output:
   ```bash
   # Example: read EXP-M01 eval results
   cat results/EXP-M01/eval/eval_results.json | python -m json.tool
   ```
3. **Copy the numbers** into the corresponding day's update table above.
4. **Note any issues** in the Notes column (OOM, NaN, unexpected results).
5. **After all days are complete**, transfer the results into `experiments.tex` by replacing the `\tbd{--}` placeholders.

### Quick Results Extraction Script

Run this after any eval completes to extract key metrics:

```bash
python -c "
import json, sys
f = sys.argv[1]
with open(f) as fh:
    r = json.load(fh)
ctrl = r.get('controllability', {})
flu = r.get('fluency', {})
eq = r.get('edit_quality', {})
div = r.get('diversity', {})
print(f\"Attr-Acc:   {ctrl.get('accuracy', 'N/A')}\")
print(f\"PPL:        {flu.get('ppl_mean', 'N/A')}\")
print(f\"BLEU:       {eq.get('bleu_mean', 'N/A')}\")
print(f\"BERTScore:  {eq.get('bertscore_f1_mean', 'N/A')}\")
print(f\"ROUGE-L:    {eq.get('rouge_l_mean', 'N/A')}\")
print(f\"Edit-Dist:  {eq.get('edit_distance_mean', 'N/A')}\")
print(f\"Distinct-2: {div.get('distinct_2', 'N/A')}\")
print(f\"Self-BLEU:  {div.get('self_bleu', 'N/A')}\")
" results/EXP-M01/eval/eval_results.json
```

---

## Mapping: Experiment IDs -> experiments.tex Tables

| experiments.tex Table | Label | Experiment IDs |
|----------------------|-------|----------------|
| Tab 4: Yelp Sentiment Transfer | `tab:yelp_results` | EXP-B01..B06, EXP-M01..M03 |
| Tab 5: Amazon Topic Transfer | `tab:amazon_results` | EXP-B07..B08, EXP-M04..M05 |
| Tab 6: GYAFC Formality Transfer | `tab:gyafc_results` | EXP-B09..B10, EXP-M06..M07 |
| Tab 7: Mask Ratio Ablation | `tab:ablation_mask_ratio` | EXP-A01..A05 |
| Tab 8: Mask Mode Ablation | `tab:ablation_mask_mode` | EXP-A06..A08 |
| Tab 9: Steps Ablation | `tab:ablation_steps` | EXP-A09..A12 |
| Tab 10: Temperature Ablation | `tab:ablation_temperature` | EXP-A13..A16 |
| Tab 11: Latent Source Ablation | `tab:ablation_latent_source` | EXP-A17..A21 |
| Tab 12: Latent Geometry | `tab:geometry` | EXP-G01..G04 |
| Tab 13: Efficiency | `tab:efficiency` | EXP-E01..E03 |

---

## Known Blockers / TODO Before Starting

1. **`--latent_source` flag:** Must be added to `lsme/scripts/run_lsme.py` before EXP-A17, A18, A20, A21. The `AttributeLatentEncoder` already has `get_directional_target()` and `get_nearest_neighbor()` methods, so only the CLI wiring is needed.

2. **`--seed` flag:** Must be added to `lsme/scripts/run_lsme.py` before running multi-seed experiments. Add `torch.manual_seed(args.seed)` and `np.random.seed(args.seed)` at the start of `main()`.

3. **Baseline output generation:** Baseline repos (MDLM, ReMDM, LatentOps, DiffusER, PLANNER, LD4LG) must be cloned, set up, and run on the same 500 test inputs. Their outputs must be saved as JSON in the `results/baselines/` directory structure.

4. **Input data preparation:** The test input files (`$YELP_INPUT`, `$AMAZON_INPUT`, `$GYAFC_INPUT`) must each contain exactly 500 lines of source text.

5. **Checkpoint availability:** Verify that `$CKPT` points to a trained MMDiT checkpoint that works with the current `load_checkpoint()` function.

---

## Summary

| Day | Category | New Runs | Reused | Total Commands |
|-----|----------|----------|--------|----------------|
| 1 | Baselines | 10 | 0 | 3 eval batches |
| 2 | Main (Yelp+Amazon) | 5 | 0 | 5 generate |
| 3 | Main (GYAFC+Eval) | 2 | 5 | 2 gen + 7 eval |
| 4 | Ablation (mask) | 4 | 4 | 4 gen + 4 eval |
| 5 | Ablation (steps/temp/src) | 8 | 5 | 8 gen + 7 eval |
| 6 | Geometry | 3 | 0 | 3 geometry |
| 7 | Efficiency | 2 | 0 | 2 profile + 1 summary |
| **Total** | | **34** | **14** | **46** |
