# GIDD — Generalized Interpolating Discrete Diffusion

## Overview

GIDD is a **hybrid continuous-discrete diffusion** model. Unlike MDLM's binary mask/unmask, GIDD allows tokens to transition to *any* vocabulary token during the forward process (not just `[MASK]`), governed by a continuous interpolation between the data distribution and a stationary distribution. It uses a KL-divergence-based loss with log-ratio correction and dynamic SNR-based weighting.

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

For GIDD, `cluster_size=0` (no auxiliary output head). The single output head projects to vocabulary size.

## Noise Schedule

**Class**: `HybridDiffusion` in `baseline/diffusion_process.py`

The forward process interpolates between the data distribution and a stationary prior with a power-law schedule. Key parameters:

- **gamma** (default 1.0): Power-law exponent controlling the noise schedule shape
- **p_uniform** (default 0.0): Minimum uniform noise floor
- **clip_noise** (default 20): Clipping threshold for numerical stability

### Stationary Distribution Parameter

```python
log_B = -log(1 + (1 - p_uniform) / p_uniform * vocab_size / 2)
```

`B = exp(log_B)` controls the stationary distribution: higher `[MASK]` probability, with remaining probability spread across all other tokens.

### Forward Transition Probabilities

Given time `t in [eps, 1-eps]`:

```
t_gamma = t^gamma
t1m_gamma = (1-t)^gamma
c_t = sqrt(t_gamma) * sqrt(t1m_gamma) * B
C_t = t_gamma + t1m_gamma + (V - 2) * c_t

alpha_t = (t1m_gamma - c_t) / C_t          # stay-as-original probability
beta_pi = (t_gamma * mask + c_t * (1-mask)) / C_t   # transition to other states
```

Where `V` is the vocabulary size and `mask` is a one-hot indicator for `[MASK]`.

**Forward sampling** (`sample_zt`):
- Convert `input_ids` to one-hot
- Compute `probs = alpha_t * one_hot(x) + beta_pi`
- Sample `z_t ~ Categorical(probs)`

### Logits Transformation

The model's output logits are transformed for the noise schedule:

```python
xi_t = gamma / 2 * log((1-t) / t).clip(-clip_noise, clip_noise)
logits = features * (xi_t - log_B) + log_B
logits[..., mask_id] = -xi_t
```

## Loss Function

**Class**: `GiddLoss` in `baseline/loss.py`

The loss combines **KL divergence** with a **log-ratio correction** and importance weighting.

### Weight Computation

For each position, the loss weight depends on whether `z_t` equals the original token `x`:

```
alpha_ratio = d/dt[alpha_hat] / alpha_hat - d/dt[C_t] / C_t
omega_t = (d/dt[pi_hat] - d/dt[alpha_hat] / alpha_hat * pi_hat) / C_t

# ELBO weights:
w_elbo = (1 - is_x) * omega_t / pi_beta + is_x * omega_t / (alpha + pi_beta)
```

**Loss weighting** (`config.loss.loss_weighting`):
- `"clip"`: Clip weights to `[min_loss_weight, max_loss_weight]`
- `"dynamic"` (default): SNR-based dynamic weighting:
  ```
  log_snr = -log(alpha / (1 - alpha)).clip(-20, 20)
  x_scale = B * exp(gamma / 2 * log_snr)
  w = (1 - is_x) * ((1 - is_mask) + 2 * is_mask) + is_x * x_scale
  w = clip(w, min_loss_weight, max_loss_weight)
  ```

### Loss Computation

```python
# KL divergence between true and predicted transition probabilities
log_q_t = noise_schedule.probs_at_t(one_hot(x), t).log()
log_p_t = noise_schedule.probs_at_t(softmax(logits), t).log()
kl_loss = KL(log_p_t || log_q_t).sum(-1)

# Log-ratio correction at z_t
log_ratio = log_q(z_t) - log_p(z_t)
correction = -log_ratio + exp(log_ratio)

# Final loss
loss = w * (kl_loss + correction)
elbo = w_elbo * (kl_loss + correction) + alpha_ratio
```

The `[MASK]` logit is forced to `-inf` before computing loss (`logits[..., mask_id] = -inf`).

## Sampling

**Class**: `GiddSampler` in `baseline/sampling.py`

Reverse process using the exact conditional `q(z_{t-1} | z_t, x)`:

1. Initialize `z_T ~ prior` (all `[MASK]`)
2. Time schedule: `ts = linspace(0, 1, num_steps + 1)`, scaled by `(1 - 2*t_eps) * ts + t_eps`
3. For each step `i` from `num_steps-1` down to `0`:
   - Run model: `logits = model(z_t, t_i)`, mask out `[MASK]` logit
   - Compute forward probabilities at `t` and `s = t_{i-1}`:
     ```
     q_s = probs_at_t(softmax(logits), s)
     q_t = probs_at_t(softmax(logits), t)
     q_zt = q_t.gather(z_t)
     ```
   - Compute transition ratios:
     ```
     alpha_ts = alpha_t / alpha_s
     beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s
     q_ts = alpha_ts * one_hot(z_t) + beta_pi_ts_at_zt
     ```
   - Posterior: `q(z_{t-1} | z_t, x) = q_ts * q_s / q_zt`
   - Sample `z_{t-1} ~ Categorical(q_st)`

Supports optional `min_p` filtering and `torch.compile` for the denoising step.

## Configuration

From `baseline/configs/gidd.yaml`:

```yaml
model:
  type: diffusion
  diffusion_process: gidd
  p_uniform: 0.0
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
  loss_type: gidd
  loss_weighting: dynamic
  min_loss_weight: 0.0
  max_loss_weight: 2.0
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
| `baseline/configs/gidd.yaml` | Main config |
| `baseline/configs/model/small.yaml` | Architecture hyperparameters (768d, 12 blocks, 12 heads) |
| `baseline/configs/optimizer/adam.yaml` | Optimizer settings |
| `baseline/diffusion_process.py` | `HybridDiffusion` class |
| `baseline/loss.py` | `GiddLoss` class |
| `baseline/sampling.py` | `GiddSampler` class |
| `baseline/models/dit.py` | `DIT` model (shared with MDLM/HDLM) |
| `baseline/modeling.py` | Model + tokenizer instantiation |
| `baseline/trainer.py` | `DiffusionTrainer` (shared with MDLM/HDLM) |
