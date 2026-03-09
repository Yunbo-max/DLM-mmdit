# MDLM-MMDiT-Latent — Masked Diffusion with Joint-Attention Latent Conditioning

## Overview

MDLM-MMDiT-Latent extends the MDLM discrete diffusion model by conditioning on a **latent vector** (e.g., from T5 encoder) via **MMDiT joint attention**. Instead of injecting the latent through AdaLN modulation parameters, the latent becomes a **second modality** that participates in joint attention alongside the text tokens at every transformer block. This gives the latent direct, fine-grained influence on text representations through cross-attention within the joint attention mechanism.

## Architecture

**Model**: `MMDiTWithLatentConditioning` in `mmdit_latent/models/mmdit_latent.py`

| Parameter | Config Value |
|---|---|
| Hidden size (text) | 768 |
| Hidden size (latent) | 768 (`latent_hidden_size`) |
| Cond dim | 768 |
| Transformer blocks | 12 (`MMDiTBlock`) |
| Attention heads | 12 |
| MLP ratio | 4x (3072 intermediate) |
| Dropout | 0.1 |
| Max sequence length | 512 |
| Positional encoding | Learned (text only) |
| Latent dim (input) | 768 (T5 encoder dimension) |
| Cluster size | 0 (no cluster head) |

### Latent Encoder

The latent vector is projected to the latent hidden size via a 2-layer MLP:

```python
self.latent_encoder = nn.Sequential(
    nn.Linear(latent_dim, hidden_size * 2),   # 768 -> 1536
    nn.GELU(),
    nn.Linear(hidden_size * 2, latent_hidden_size),  # 1536 -> 768
)
```

A learned `null_latent` parameter (shape `[1, 1, latent_hidden_size]`) is used when no latent is provided.

### MMDiTBlock

Each block performs joint attention across two modalities (text + latent) followed by per-modality feedforward networks, all conditioned on the diffusion timestep.

**Structure per block:**

1. **Adaptive LayerNorm** (per-modality) — conditioned on timestep `c`
2. **Joint Attention** — text and latent tokens attend to each other via concatenated QKV
3. **Post-attention gating** — timestep-conditioned per-modality scale factors
4. **Residual connection**
5. **Adaptive LayerNorm** (per-modality) — conditioned on timestep `c`
6. **Per-modality FeedForward** — separate FFN for text and latent streams
7. **Post-FFN gating** — timestep-conditioned per-modality scale factors
8. **Residual connection**

### JointAttention

The joint attention module concatenates tokens from all modalities along the sequence dimension before computing attention, then splits outputs back per-modality:

```
Q, K, V = concat([text_qkv, latent_qkv], dim=seq)
attn_out = scaled_dot_product_attention(Q, K, V, mask)
text_out, latent_out = split(attn_out)
```

Each modality has its own `to_qkv` and `to_out` projections. Optional QK-RMSNorm is supported.

### Forward Pass

```python
def forward(self, indices, sigma, latents=None, attention_mask=None):
    # 1. Timestep conditioning
    c = silu(sigma_map(sigma))

    # 2. Text tokens: embed + positional
    text_tokens = vocab_embed(indices) + text_pos_embed[:, :S]

    # 3. Latent tokens: encode or use null
    if latents is not None:
        latent_tokens = latent_encoder(latents)   # (B, L, latent_hidden_size)
    else:
        latent_tokens = null_latent.expand(B, -1, -1)

    # 4. Run through MMDiT blocks (joint attention + per-modality FFN)
    for block in self.blocks:
        text_tokens, latent_tokens = block(
            modality_tokens=(text_tokens, latent_tokens),
            modality_masks=(text_mask, latent_mask),
            time_cond=c,
        )

    # 5. Output layer (text only — discard latent stream)
    return output_layer(text_tokens, c)
```

## Noise Schedule

**Class**: `MaskedDiffusion` in `mmdit_latent/diffusion_process.py`

Identical to standard MDLM:

```
sigma(t) = -log(1 - (1 - eps) * t)
move_chance = 1 - exp(-sigma(t))
```

## Loss Function

**Class**: `MDLMLoss` in `mmdit_latent/loss.py`

Standard MDLM weighted cross-entropy on masked positions:

```
weight = dsigma(t) / expm1(sigma(t)) * is_masked
loss = weight * CE(logits, input_ids)
```

## Configuration

From `mmdit_latent/configs/mdlm_mmdit_latent.yaml`:

```yaml
model:
  type: diffusion
  diffusion_process: mdlm
  use_latent_conditioning: true
  conditioning_type: "mmdit"
  latent_dim: 768
  latent_hidden_size: 768
  cluster_size: 0
  qk_rmsnorm: false
  hidden_size: 768
  n_blocks: 12
  n_heads: 12
  cond_dim: 768
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

## Key Differences from AdaLN Approach

| Aspect | AdaLN (old) | MMDiT (current) |
|---|---|---|
| Conditioning mechanism | Additive shift/scale/gate modulation on each block's LayerNorm | Joint attention — latent tokens attend with text tokens |
| Latent influence | Indirect (modulates normalization statistics) | Direct (cross-attention at every layer) |
| Latent representation | Single vector → 6 modulation params per block | Token sequence in shared attention space |
| Block type | `LatentConditionedDDiTBlock` | `MMDiTBlock` (per-modality FFN + joint attention) |
| Positional encoding | Rotary (RoPE) | Learned positional embeddings (text only) |
| Model file | `dit_latent.py` (deleted) | `mmdit_latent.py` + `mmdit_block.py` |
| Null conditioning | Falls back to base AdaLN (additive zero) | Uses learned `null_latent` token |

## Key Files

| File | Contents |
|---|---|
| `mmdit_latent/configs/mdlm_mmdit_latent.yaml` | Main config |
| `mmdit_latent/models/mmdit_latent.py` | `MMDiTWithLatentConditioning` — top-level model |
| `mmdit_latent/models/mmdit_block.py` | `MMDiTBlock`, `JointAttention`, `AdaptiveLayerNorm` |
| `mmdit_latent/models/dit.py` | `TimestepEmbedder`, `EmbeddingLayer`, `DDitFinalLayer` (shared) |
| `mmdit_latent/diffusion_process.py` | `MaskedDiffusion` (same as baseline) |
| `mmdit_latent/loss.py` | `MDLMLoss` (same as baseline) |
| `mmdit_latent/train_latent_dit.py` | Training script with `LatentConditioningTrainer` |
| `mmdit_latent/modeling_latent.py` | Model selection logic (dispatches to MMDiT, vanilla DIT, or autoregressive) |
| `mmdit_latent/data_simple.py` | `SimpleLatentDataset` + collate with latent loading |
