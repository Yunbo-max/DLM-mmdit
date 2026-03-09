# MDLM-Cross — Masked Diffusion with Cross-Attention Latent Conditioning

## Overview

MDLM-Cross extends the MDLM discrete diffusion model by conditioning on a **latent sequence** via **cross-attention**. Unlike the AdaLN variant (MDLM-Latent), which compresses the latent into a single global vector, cross-attention allows the model to attend to a variable-length latent sequence, preserving positional and structural information from the encoder. Cross-attention blocks alternate with regular self-attention blocks (every other block). An optional **latent prediction head** provides an auxiliary MSE loss for predicting the latent from the denoised representation.

## Architecture

**Model**: `DITWithCrossAttention` in `baseline_latent/models/dit_cross_attention.py`

| Parameter | Config Value |
|---|---|
| Hidden size | 768 |
| Cond dim | 768 |
| Transformer blocks | 12 (alternating `CrossAttentionDDiTBlock` and `DDiTBlockWithMask`) |
| Attention heads | 12 |
| MLP ratio | 4x (3072 intermediate) |
| Dropout | 0.1 |
| Max sequence length | 512 |
| Positional encoding | Rotary (RoPE) for self-attention |
| Latent dim | 768 |
| Cluster size | 0 |
| Cross-attention frequency | Every 2 blocks (even-indexed blocks) |
| Latent prediction | Enabled (`predict_latents: true`) |
| Latent prediction weight | 0.1 |

### Block Layout

The 12 blocks alternate between two types:
- **Even blocks (0, 2, 4, 6, 8, 10)**: `CrossAttentionDDiTBlock` — self-attention + cross-attention + MLP
- **Odd blocks (1, 3, 5, 7, 9, 11)**: `DDiTBlockWithMask` — self-attention + MLP (no cross-attention)

### Latent Projection

Latents are projected before cross-attention via a 2-layer MLP:

```python
self.latent_projection = nn.Sequential(
    nn.Linear(latent_dim, hidden_size * 2),   # 768 -> 1536
    nn.SiLU(),
    nn.Linear(hidden_size * 2, hidden_size)   # 1536 -> 768
)
```

### CrossAttentionDDiTBlock

Each cross-attention block has three sub-layers:

**1. Self-Attention** (text-to-text):
- AdaLN modulation from timestep: `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`
- Standard bidirectional self-attention with RoPE
- Gated residual: `x = x_skip + gate_msa * dropout(attn_out(attn))`

**2. Cross-Attention** (text-to-latent):
- Query from text: `q = cross_attn_q(LayerNorm(x))`, modulated by AdaLN shift/scale
- Key/Value from latents: `k, v = cross_attn_kv(projected_latents).chunk(2)`
- Standard scaled dot-product attention with optional `latent_mask`
- Gated residual: `x = x_cross_skip + gate_msa * dropout(cross_attn_out(cross_attn))`

**3. MLP**:
- AdaLN-modulated, GELU activation
- Gated residual: `x = x_mlp_skip + gate_mlp * dropout(mlp(x))`

Note: Cross-attention reuses `gate_msa` and the same `shift_msa/scale_msa` for its LayerNorm modulation.

### Latent Prediction Head

An optional auxiliary head predicts the latent from the model's internal representations:

```python
self.latent_prediction_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size * 2),  # 768 -> 1536
    nn.SiLU(),
    nn.Linear(hidden_size * 2, latent_dim)    # 1536 -> 768
)
```

When `return_latent_pred=True`:
- Each block's output is mean-pooled over sequence length: `x.mean(dim=1)`
- All block representations are stacked and averaged: `combined = stack(preds).mean(dim=1)`
- The prediction head maps this to latent space: `latent_pred = head(combined)`

### Forward Pass

```python
def forward(self, indices, sigma, latents=None, latent_mask=None,
            attention_mask=None, return_latent_pred=False):
    x = self.vocab_embed(indices)
    c = silu(sigma_map(sigma))

    projected_latents = self.latent_projection(latents) if latents is not None else None

    for block in self.blocks:
        if hasattr(block, 'cross_attn_q') and projected_latents is not None:
            x = block(x, rotary, c, latents=projected_latents,
                      latent_mask=latent_mask, attention_mask=attention_mask)
        else:
            x = block(x, rotary, c, attention_mask=attention_mask)

    x1 = self.output_layer(x, c)

    if return_latent_pred:
        latent_pred = self.latent_prediction_head(combined_representations)
        return x1, latent_pred

    return x1
```

## Noise Schedule

**Class**: `MaskedDiffusion` in `baseline_latent/diffusion_process.py`

Identical to standard MDLM:

```
sigma(t) = -log(1 - (1 - eps) * t)
move_chance = 1 - exp(-sigma(t))
```

## Loss Function

**Primary**: `MDLMLoss` — weighted cross-entropy on masked positions (same as MDLM).

**Auxiliary**: MSE latent prediction loss, computed in `CrossAttentionTrainer`:

```python
if latent_pred is not None and latents is not None:
    latent_loss = MSE(latent_pred, latents.mean(dim=1))
    loss = loss + latent_prediction_weight * latent_loss   # weight = 0.1
```

Total loss: `L = L_mdlm + 0.1 * L_latent_pred`

## Training

**Trainer**: `CrossAttentionTrainer` in `baseline_latent/train_cross_dit.py`

The training loop:
1. Sample timestep `t`, corrupt tokens to `z_t`
2. Extract `latents` and `latent_mask` from batch
3. Forward with cross-attention: `model(z_t, t, latents=latents, latent_mask=latent_mask, return_latent_pred=True)`
4. Compute MDLM loss on token logits
5. Add MSE latent prediction loss (weighted by 0.1)
6. Backpropagate

Key difference from MDLM-Latent: `compile_model: false` in config (cross-attention can be buggy with `torch.compile`).

## Data

Same as MDLM-Latent: `SimpleLatentDataset` loading JSON + `.npy` latent files.

The collate function also produces `latent_mask` for variable-length latent sequences (padded latents get mask=0).

## Configuration

From `baseline_latent/configs/mdlm_cross_attention.yaml` (inherits from `mdlm_latent`):

```yaml
# Inherits all base settings from mdlm_latent.yaml, then overrides:

model:
  type: diffusion
  diffusion_process: mdlm
  use_latent_conditioning: true
  latent_dim: 768
  cluster_size: 0
  hidden_size: 768
  n_blocks: 12
  n_heads: 12
  cond_dim: 768
  max_seq_len: 512
  dropout: 0.1
  t_eps: 1e-4

  # Cross-attention specific
  conditioning_type: "cross_attention"
  cross_attention_frequency: 2
  predict_latents: true
  latent_prediction_weight: 0.1

training:
  train_batch_size: 32
  eval_batch_size: 32
  num_train_steps: 1_000_000
  compile_model: false         # Disabled for cross-attention compatibility

loss:
  loss_type: mdlm
  loss_scale: 1.0
  reduction: tokenmean
```

**Optimizer**: Same as MDLM-Latent (Adam, lr=5e-4, wd=0.02).

### Key Differences from MDLM-Latent (AdaLN)

| Aspect | MDLM-Latent (AdaLN) | MDLM-Cross (Cross-Attention) |
|---|---|---|
| Conditioning mechanism | Global latent vector -> AdaLN shift/scale | Latent sequence -> cross-attention Q/K/V |
| Latent granularity | Single vector per sample | Variable-length sequence per sample |
| Block type | `LatentConditionedDDiTBlock` (all 12) | Alternating `CrossAttentionDDiTBlock` (6) + `DDiTBlockWithMask` (6) |
| Extra parameters | `latent_adaLN_modulation` per block | `cross_attn_q`, `cross_attn_kv`, `cross_attn_out`, `cross_attn_norm` per cross-attn block |
| Latent prediction | No | Yes (auxiliary MSE loss, weight 0.1) |
| `torch.compile` | Enabled | Disabled |
| Latent masking | N/A | Supports `latent_mask` for padded sequences |

## Key Files

| File | Contents |
|---|---|
| `baseline_latent/configs/mdlm_cross_attention.yaml` | Main config (inherits from `mdlm_latent`) |
| `baseline_latent/models/dit_cross_attention.py` | `DITWithCrossAttention`, `CrossAttentionDDiTBlock` |
| `baseline_latent/diffusion_process.py` | `MaskedDiffusion` (same as baseline) |
| `baseline_latent/loss.py` | `MDLMLoss` (same as baseline) |
| `baseline_latent/train_cross_dit.py` | Training script with `CrossAttentionTrainer` |
| `baseline_latent/modeling_latent.py` | Model selection (dispatches based on `conditioning_type`) |
| `baseline_latent/data_simple.py` | `SimpleLatentDataset` + collate with latent loading |
| `baseline_latent/sampling.py` | `MDLMSampler` (latent not yet wired for sampling) |
