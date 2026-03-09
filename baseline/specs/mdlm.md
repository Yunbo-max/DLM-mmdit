# MDLM — Masked Diffusion Language Model

## Overview

MDLM is a discrete diffusion model that operates on a **2-state space**: each token is either its original vocabulary token or a special `[MASK]` token. The forward process progressively masks tokens with increasing probability over time `t in [0, 1]`, and the reverse process learns to unmask them. This is the simplest diffusion baseline in the codebase.

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

Each `DDiTBlockWithMask` block applies:
1. AdaLN-modulated self-attention (with optional attention masking for padding)
2. AdaLN-modulated FFN (GELU activation)
3. Gated residual connections (6 modulation parameters per block: shift/scale/gate for attention and MLP)

The output head is `DDitFinalLayer`: LayerNorm + AdaLN modulation + linear projection to vocabulary size.

For MDLM, `cluster_size=0`, so there is no auxiliary cluster output head.

## Noise Schedule

**Class**: `MaskedDiffusion` in `baseline/diffusion_process.py`

The forward process is parameterized by a masking rate `sigma(t)`:

```
sigma(t) = -log(1 - (1 - eps) * t)
dsigma(t) = (1 - eps) / (1 - (1 - eps) * t)
```

where `eps = 1e-4` (configurable via `t_eps`).

**Forward sampling** (`sample_zt`):
- Compute `move_chance = 1 - exp(-sigma(t))`
- Each token is independently replaced with `[MASK]` with probability `move_chance`
- `z_t = [MASK]` if `rand < move_chance`, else `z_t = x` (original token)

**Prior**: At `t=1`, essentially all tokens are `[MASK]`.

**Transition probabilities** (`probs_at_t`):
- `alpha_t = exp(-sigma(t))` (probability of staying as original token)
- `P(z_t = x | x) = alpha_t`, `P(z_t = [MASK] | x) = 1 - alpha_t`

## Loss Function

**Class**: `MDLMLoss` in `baseline/loss.py`

The loss is a **weighted cross-entropy on masked positions only**:

1. Mask out the `[MASK]` logit: `logits[..., mask_id] = -1e6`
2. Normalize logits via log-softmax
3. Zero out logits at non-masked positions (copy-through for unmasked tokens)
4. Compute cross-entropy: `rec_loss = CE(logits, input_ids)`
5. Weight by the ELBO coefficient:
   ```
   weight = dsigma(t) / expm1(sigma(t)) * is_masked
   ```
6. Final ELBO loss: `loss = weight * rec_loss`

The loss is reduced via `tokenmean`: sum over tokens weighted by `attention_mask`, divided by total valid tokens.

## Sampling

**Class**: `MDLMSampler` in `baseline/sampling.py`

Reverse process (generation):
1. Initialize `z_T ~ prior` (all `[MASK]` tokens)
2. Create time schedule: `ts = linspace(t_eps, 1 - t_eps, num_steps + 1)`
3. For each step `i` from `num_steps-1` down to `0`:
   - Run model: `logits = model(z_t, t_i)`
   - Mask out `[MASK]` logit
   - If `i == 0` (last step): `z_0 = argmax(logits)`
   - Otherwise:
     - Compute `move_chance_t` and `move_chance_{t-1}` from sigma
     - Mixing: `probs = softmax(logits) * (move_chance_t - move_chance_{t-1})` for token positions
     - `probs[:, :, mask_id] = move_chance_{t-1}` (probability of staying masked)
     - `probs /= move_chance_t` (normalize)
     - Sample `z_{t-1} ~ Categorical(probs)`
   - **Copy flag**: tokens already unmasked are never re-masked: `z_t = copy_flag * z_t + (1 - copy_flag) * z_{t-1}`

Supports optional `min_p` filtering to remove low-probability tokens.

## Configuration

From `baseline/configs/mdlm.yaml`:

```yaml
model:
  type: diffusion
  diffusion_process: mdlm
  t_eps: 1e-4

training:
  train_batch_size: 64
  eval_batch_size: 64
  num_train_steps: 1_131_000
  lr_schedule: cosine
  warmup_steps: 10000
  low_discrepancy_sampling: true
  dtype: bf16
  compile_model: true

loss:
  loss_type: mdlm
  loss_scale: 1.0
  reduction: tokenmean
```

**Optimizer** (from `baseline/configs/optimizer/adam.yaml`):
- Adam with lr=5e-4, betas=(0.9, 0.99), eps=1e-9
- Weight decay: 0.02
- Gradient clip norm: 1.0

## Key Files

| File | Contents |
|---|---|
| `baseline/configs/mdlm.yaml` | Main config |
| `baseline/configs/model/small.yaml` | Architecture hyperparameters (768d, 12 blocks, 12 heads) |
| `baseline/configs/optimizer/adam.yaml` | Optimizer settings |
| `baseline/diffusion_process.py` | `MaskedDiffusion` class |
| `baseline/loss.py` | `MDLMLoss` class |
| `baseline/sampling.py` | `MDLMSampler` class |
| `baseline/models/dit.py` | `DIT` model, `DDiTBlockWithMask`, `DDitFinalLayer` |
| `baseline/modeling.py` | Model + tokenizer instantiation |
| `baseline/trainer.py` | `DiffusionTrainer` (shared with GIDD/HDLM) |
