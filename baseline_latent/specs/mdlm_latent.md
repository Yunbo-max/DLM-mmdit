# MDLM-Latent — Masked Diffusion with AdaLN Latent Conditioning

## Overview

MDLM-Latent extends the MDLM discrete diffusion model by conditioning on a **latent vector** (e.g., from T5 encoder) via **Adaptive LayerNorm (AdaLN) modulation**. The latent is projected into per-block shift/scale/gate modulation parameters that are combined additively with the standard timestep-based AdaLN. This allows the diffusion model to be guided by a continuous semantic representation while keeping the same masked diffusion forward/reverse process.

## Architecture

**Model**: `DITWithLatentConditioning` in `baseline_latent/models/dit_latent.py`

| Parameter | Config Value |
|---|---|
| Hidden size | 768 |
| Cond dim | 768 (overridden from default 128) |
| Transformer blocks | 12 (`LatentConditionedDDiTBlock`) |
| Attention heads | 12 |
| MLP ratio | 4x (3072 intermediate) |
| Dropout | 0.1 |
| Max sequence length | 512 |
| Positional encoding | Rotary (RoPE) |
| Latent dim | 768 (T5 encoder dimension) |
| Cluster size | 0 (no cluster head) |

### Latent Encoder

The latent vector is projected to a conditioning signal via a 2-layer MLP:

```python
self.latent_encoder = nn.Sequential(
    nn.Linear(latent_dim, cond_dim * 4),   # 768 -> 3072
    nn.SiLU(),
    nn.Linear(cond_dim * 4, cond_dim * 2)  # 3072 -> 1536 (scale + shift)
)
```

The output is split into `(latent_shift, latent_scale)`, each of dimension `cond_dim`.

### LatentConditionedDDiTBlock

Each block extends `DDiTBlockWithMask` with an additional `latent_adaLN_modulation` linear layer (zero-initialized). The conditioning works by:

1. Compute **base modulation** from timestep: `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = adaLN_modulation(c)`
2. Compute **latent modulation** from timestep: `(lat_shift_msa, ..., lat_gate_mlp) = latent_adaLN_modulation(c)`
3. **Combine additively**, scaled by the latent signal:
   ```
   shift_msa = shift_msa_base + latent_scale * lat_shift_msa
   scale_msa = scale_msa_base + latent_scale * lat_scale_msa
   gate_msa  = gate_msa_base  + latent_scale * lat_gate_msa
   ```
   (Same for MLP modulation parameters)

When `latents=None`, the block falls back to standard `DDiTBlockWithMask` behavior.

### Forward Pass

```python
def forward(self, indices, sigma, latents=None, attention_mask=None):
    x = self.vocab_embed(indices)
    c = silu(sigma_map(sigma))

    latent_cond = None
    if latents is not None:
        if latents.dim() == 3:
            latents = latents.squeeze(1)    # [B, 1, D] -> [B, D]
        latent_cond = self.latent_encoder(latents)  # [B, 2*cond_dim]

    for block in self.blocks:
        x = block(x, rotary_cos_sin, c, latent_cond=latent_cond, attention_mask=attention_mask)

    return self.output_layer(x, c)
```

## Noise Schedule

**Class**: `MaskedDiffusion` in `baseline_latent/diffusion_process.py`

Identical to the standard MDLM noise schedule:

```
sigma(t) = -log(1 - (1 - eps) * t)
move_chance = 1 - exp(-sigma(t))
```

Each token is independently replaced with `[MASK]` with probability `move_chance`.

## Loss Function

**Class**: `MDLMLoss` in `baseline_latent/loss.py`

Identical to the standard MDLM loss — weighted cross-entropy on masked positions only:

```
weight = dsigma(t) / expm1(sigma(t)) * is_masked
loss = weight * CE(logits, input_ids)
```

Reduced via `tokenmean`.

## Training

**Trainer**: `LatentConditioningTrainer` in `baseline_latent/train_latent_dit.py`

The training loop:
1. Sample timestep `t` via `sample_t()` (low-discrepancy sampling)
2. Corrupt tokens: `z_t = noise_schedule.sample_zt(input_ids, t)`
3. Extract latent from batch: `latents = batch.get("latent", None)`
4. Forward: `logits = model(z_t, t, latents=latents, attention_mask=...)`
5. Compute MDLM loss on logits
6. Backpropagate

Additional logged metrics when latents present: `latent_norm`, `latent_std`.

## Data

**Class**: `SimpleLatentDataset` in `baseline_latent/data_simple.py`

Loads JSON files with text + latent paths:
- Each JSON entry has `"text"` and `"latent_path"` (path to `.npy` file)
- Latent `.npy` files contain T5-encoded vectors of shape `[1, 768]` or `[768]`
- The collate function pads variable-length latent sequences and stacks them

## Configuration

From `baseline_latent/configs/mdlm_latent.yaml`:

```yaml
model:
  type: diffusion
  diffusion_process: mdlm
  use_latent_conditioning: true
  conditioning_type: "adaln"
  latent_dim: 768
  cluster_size: 0
  hidden_size: 768
  n_blocks: 12
  n_heads: 12
  cond_dim: 768        # Larger than default 128 to accommodate latent info
  max_seq_len: 512
  dropout: 0.1
  t_eps: 1e-4

training:
  train_batch_size: 32
  eval_batch_size: 32
  num_train_steps: 1_000_000
  lr_schedule: cosine
  warmup_steps: 5000
  low_discrepancy_sampling: true
  dtype: bf16
  compile_model: true

loss:
  loss_type: mdlm
  loss_scale: 1.0
  reduction: tokenmean
```

**Optimizer** (from `baseline_latent/configs/optimizer/adam.yaml`):
- Adam with lr=5e-4, betas=(0.9, 0.99), eps=1e-9
- Weight decay: 0.02, gradient clip norm: 1.0

### Key Differences from Vanilla MDLM

| Parameter | Vanilla MDLM | MDLM-Latent |
|---|---|---|
| `cond_dim` | 128 | 768 |
| `dropout` | 0.0 | 0.1 |
| `warmup_steps` | 10000 | 5000 |
| `train_batch_size` | 64 | 32 |
| Block type | `DDiTBlockWithMask` | `LatentConditionedDDiTBlock` |
| Latent input | None | T5 embeddings (768d) |

## Key Files

| File | Contents |
|---|---|
| `baseline_latent/configs/mdlm_latent.yaml` | Main config |
| `baseline_latent/models/dit_latent.py` | `DITWithLatentConditioning`, `LatentConditionedDDiTBlock` |
| `baseline_latent/diffusion_process.py` | `MaskedDiffusion` (same as baseline) |
| `baseline_latent/loss.py` | `MDLMLoss` (same as baseline) |
| `baseline_latent/train_latent_dit.py` | Training script with `LatentConditioningTrainer` |
| `baseline_latent/modeling_latent.py` | Model selection logic (dispatches to AdaLN or cross-attn) |
| `baseline_latent/data_simple.py` | `SimpleLatentDataset` + collate with latent loading |
| `baseline_latent/sampling.py` | `MDLMSampler` (same as baseline; latent not yet wired for sampling) |
