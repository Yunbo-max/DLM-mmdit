# HDLM — Hierarchical Diffusion Language Model

## Overview

HDLM introduces a **3-state hierarchical diffusion** process: tokens transition through an intermediate **cluster** level before reaching `[MASK]`. The forward process is `token -> cluster -> [MASK]`, where clusters are semantic groupings of vocabulary tokens (e.g., 64 clusters). This allows the reverse process to first predict the cluster, then refine to the exact token, capturing hierarchical structure in the vocabulary.

## Architecture

**Model**: `DIT` (Diffusion Transformer) from `baseline/models/dit.py`

| Parameter | Value |
|---|---|
| Hidden size | 768 |
| Cond dim (timestep embedding) | 128 |
| Transformer blocks | 12 (`DDiTBlockWithMask`) |
| Attention heads | 12 |
| MLP ratio | 4x (3072 intermediate) |
| Dropout | 0.0 |
| Max sequence length | 512 |
| Positional encoding | Rotary (RoPE) |
| Conditioning | Adaptive LayerNorm (AdaLN) on sinusoidal timestep embeddings |

**Dual output heads**: Unlike MDLM/GIDD, HDLM uses `cluster_size > 0`, which activates two output heads in `DIT.forward()`:
1. `output_layer` — predicts token-level logits (over `vocab_size + cluster_size`, rounded to 128)
2. `output_layer_clusters` — auxiliary cluster prediction head (same shape)

Both are `DDitFinalLayer` modules. The model returns `(logits, logits_clusters)`.

### Vocabulary Extension

The effective vocabulary is extended to `vocab_size + cluster_size`. Cluster IDs occupy indices `[vocab_size, vocab_size + cluster_size)`. A `cluster_dict` tensor maps each vocabulary token to its cluster ID.

## Noise Schedule

**Class**: `HierarchicalDiffusion` in `baseline/diffusion_process.py`

The forward process uses a **3-level Markov chain**: token -> cluster -> [MASK], controlled by a power-law schedule with parameter `gamma`.

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 1.0 | Power-law exponent for the noise schedule |
| `p_perturb` | 0.0 | Inter-cluster perturbation probability (`1 - xi`) |
| `p_uniform` | 0.0 | Uniform noise floor |
| `cluster_size` | 64 | Number of semantic clusters |
| `cluster_dict_path` | `.pt` file | Maps each vocab token to its cluster ID |

### Forward Transition Probabilities

```
t_gamma = t^gamma
t1m_gamma = (1-t)^gamma

alpha_t = t1m_gamma                    # stay-as-original-token probability

# For gamma == 1:
c_t = -(1-t) * log(1-t)               # transition to cluster probability
m_t = t + (1-t) * log(1-t)            # transition to [MASK] probability

# For gamma != 1:
c_t = gamma * (1-t) * (1 - (1-t)^(gamma-1)) / (gamma - 1)
m_t = 1 - c_t - alpha_t
```

The full transition probabilities are:
- `P(z_t = x | x) = alpha_t` (stay as token)
- `P(z_t = cluster(x) | x) = c_t` (degrade to cluster, with optional p_perturb cross-cluster noise)
- `P(z_t = [MASK] | x) = m_t` (fully mask)

**Forward sampling** (`sample_zt`):
1. Create one-hot for `x` and cluster `c = cluster_dict[x]` (both in extended vocab)
2. If `p_perturb > 0`: add cross-cluster noise to the cluster distribution
3. `probs = alpha_t * one_hot(x) + c_t * one_hot(cluster) + m_t * one_hot([MASK])`
4. Sample `z_t ~ Categorical(probs)`

## Loss Function

**Class**: `HDLMLoss` in `baseline/loss.py`

HDLM uses a **dual loss** with separate terms for cluster prediction and token prediction, with different weighting schemes.

### Configuration Options

| Parameter | Default | Description |
|---|---|---|
| `simplified` | False | Use MDLM-like loss (True) or GIDD-like strict ELBO (False) |
| `mask_only` | True | Only compute auxiliary loss on `[MASK]` positions |
| `cluster_loss_weight` | 1.0 | Weight for cluster prediction loss |
| `token_loss_weight` | 1.0 | Weight for token prediction loss |
| `auxiliary_loss_weight` | 0.0 | Weight for auxiliary cluster head loss |
| `force_transitting_within` | True | Constrain token predictions within the predicted cluster |
| `force_transitting_between` | False | Allow cross-cluster transitions weighted by xi |
| `hard_training` | False | Use CE on tokens instead of cluster NLL for cluster loss |
| `loss_weighting` | dynamic | Weight schedule: `dynamic`, `clip`, `empirical`, or `none` |

### ELBO Weights

```python
alpha_ratio = -gamma / (1 - t)

# Mask -> cluster weight
weights_mask = d/dt[beta_pi_mask] / beta_pi_mask(t)

# Cluster -> token weight
weights_clusters = d/dt[alpha_t] / beta_pi_clusters(t)
```

### Loss Computation (Non-Simplified / GIDD-like)

For positions where `z_t` is:
- **`[MASK]`**: Compute cluster prediction loss via `NLL(cluster_probs, true_cluster)`
  - `cluster_probs = einsum(softmax(logits)[:vocab_size], cluster_matrix)`
- **Cluster ID**: Compute token prediction loss via `CE(logits_normalized, input_ids)` with force-transitting constraints
- **Original token**: No loss (already correct)

**Force-transitting** (when `z_t` is a cluster): Adjusts token logits so that probability mass is concentrated within the cluster:
```python
probs = softmax(logits)
in_cluster_adjustment = xi / sum(probs * in_cluster_mask)
out_cluster_adjustment = (1 - xi) / sum(probs * ~in_cluster_mask)
probs = where(in_cluster, probs * in_cluster_adjustment, probs * out_cluster_adjustment)
```

Final loss: `loss = loss_clusters * w_mask + loss_tokens * w_clusters`

If `auxiliary_loss_weight > 0`: adds `CE(logits_clusters, cluster_dict[input_ids])` on masked positions.

### Loss Computation (Simplified / MDLM-like)

Only computed on `[MASK]` positions:
- `loss_clusters = xi * -log(p(correct_cluster) + cross_cluster_correction)`
- `loss_tokens = CE(logits, input_ids)` on cluster positions, split into in-cluster and out-cluster

## Sampling

**Class**: `HDLMSampler` in `baseline/sampling.py`

Supports two parameterizations (`sampling_parameterization` config):

### MDLM Parameterization (default)

Threshold-based sampling mimicking MDLM's copy-flag approach, extended to 3 states:

1. Initialize `z_T ~ prior` (all `[MASK]`)
2. For each step:
   - Run model -> `(logits, logits_clusters)`
   - Apply force-transitting constraints on cluster positions
   - **Mask positions** (`z_t == [MASK]`):
     - Compute threshold: `(beta_mask(t) - beta_mask(s)) / beta_mask(t)`
     - With probability `threshold`, transition from `[MASK]` to a sampled cluster (via Gumbel noise)
   - **Cluster positions** (`z_t` in `[vocab_size, vocab_size + cluster_size)`):
     - Compute threshold: `(alpha_s - alpha_t) / beta_clusters(t)`
     - With probability `threshold`, transition from cluster to a sampled token
   - **Token positions**: Copy through (never change)
3. Last step: deterministic argmax for all non-token positions

### GIDD Parameterization

Uses the exact posterior `q(z_{t-1} | z_t, x)` like `GiddSampler`, but extended to the 3-state space:
- Combines `beta_pi_mask` and `beta_pi_clusters` into a single `beta_pi`
- Computes `q_ts * q_s / q_zt` and samples categorically
- Last step: argmax with copy-flag

Both parameterizations support Gumbel noise temperature and `min_p` filtering.

## Configuration

From `baseline/configs/hdlm-small-cluster_64-gamma_1.0-xi_1.0.yaml`:

```yaml
model:
  type: diffusion
  diffusion_process: hdlm
  p_uniform: 0.0
  t_eps: 1e-4
  cluster_dict_path: hdlm/clusters/semantic_cluster_dict_gidd-small-p_unif-0.0_64.pt
  cluster_embed_path: hdlm/clusters/semantic_centroids_gidd-small-p_unif-0.0_64.pt
  cluster_size: 64
  gamma: 1.0
  p_perturb: 0.0        # xi = 1 - p_perturb = 1.0

training:
  train_batch_size: 64
  eval_batch_size: 64
  num_train_steps: 1_000_000
  lr_schedule: cosine
  warmup_steps: 10000
  low_discrepancy_sampling: true
  dtype: bf16
  compile_model: true

loss:
  loss_type: hdlm
  loss_weighting: dynamic
  min_loss_weight: 0.0
  max_loss_weight: 2.0
  simplified: false
  mask_only: true
  cluster_loss_weight: 1.0
  token_loss_weight: 1.0
  force_transitting_within: true
  force_transitting_between: false
  hard_training: false

# Sampling defaults
sampling_parameterization: mdlm
temperature: 1.0
num_denoising_steps: 512
min_p: 0.0
```

**Optimizer** (from `baseline/configs/optimizer/adam.yaml`):
- Adam with lr=5e-4, betas=(0.9, 0.99), eps=1e-9
- Weight decay: 0.02
- Gradient clip norm: 1.0

## Key Files

| File | Contents |
|---|---|
| `baseline/configs/hdlm-small-cluster_64-gamma_1.0-xi_1.0.yaml` | Main config (also variants with different cluster sizes, gamma, xi) |
| `baseline/configs/model/small.yaml` | Architecture hyperparameters (768d, 12 blocks, 12 heads) |
| `baseline/configs/optimizer/adam.yaml` | Optimizer settings |
| `baseline/diffusion_process.py` | `HierarchicalDiffusion` class |
| `baseline/loss.py` | `HDLMLoss` class |
| `baseline/sampling.py` | `HDLMSampler` class |
| `baseline/models/dit.py` | `DIT` model with dual output heads |
| `baseline/modeling.py` | Model + tokenizer instantiation |
| `baseline/trainer.py` | `DiffusionTrainer` (shared with MDLM/GIDD) |
