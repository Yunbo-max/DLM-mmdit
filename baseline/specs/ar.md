# AR — Autoregressive Baseline

## Overview

The autoregressive (AR) baseline is a standard **GPT-style left-to-right language model**. It uses causal attention to generate tokens one at a time, conditioned on all previously generated tokens. There is no diffusion process, noise schedule, or denoising — this serves as a non-diffusion reference point for comparing perplexity and generation quality.

## Architecture

**Model**: `LlamaForCausalLM` from HuggingFace Transformers (not the DIT backbone)

| Parameter | Value |
|---|---|
| Hidden size | 768 |
| Intermediate size | 3072 (4x hidden) |
| Transformer layers | 12 |
| Attention heads | 12 |
| Max position embeddings | 512 |
| Attention implementation | Flash Attention 2 (if available), else SDPA |

Instantiated in `baseline/modeling.py` via `LlamaConfig`:

```python
cfg = LlamaConfig(
    vocab_size=len(tokenizer),
    num_hidden_layers=config.model.n_blocks,
    hidden_size=config.model.hidden_size,
    intermediate_size=4 * config.model.hidden_size,
    num_attention_heads=config.model.n_heads,
    max_position_embeddings=config.model.max_seq_len,
    attn_implementation="flash_attention_2" if has_flash_attn else "sdpa",
)
model = LlamaForCausalLM(cfg)
```

Key differences from the diffusion models:
- No timestep conditioning (no `TimestepEmbedder` or AdaLN)
- **Causal attention** (autoregressive mask) instead of bidirectional
- Uses HuggingFace's built-in RoPE, RMSNorm, and SiLU-gated MLP (Llama architecture)

## Noise Schedule

**None**. The AR model has no noise schedule (`get_noise_schedule` returns `None` for `type == "autoregressive"`).

## Loss Function

**Standard cross-entropy** via `nn.CrossEntropyLoss(reduction="none")`.

Computed in `AutoregressiveTrainer.forward()`:

```python
labels = input_ids[:, 1:]              # shifted right (next-token prediction)
loss_mask = attention_mask[:, :-1]     # mask padding

logits = model(input_ids, attention_mask, use_cache=False).logits[:, :-1]
loss = CrossEntropyLoss(logits.transpose(1, 2), labels)

# Token-mean reduction with distributed support
total_loss = (loss * loss_mask).sum()
total_tokens = loss_mask.sum()
loss = total_loss / total_tokens
```

The loss is the negative log-likelihood per token, equivalent to perplexity when exponentiated.

**Distributed training**: When using multiple GPUs, `total_tokens` is all-reduced across workers and normalized by world size to ensure consistent gradient scaling.

## Sampling

**Class**: `AutoregressiveSampler` in `baseline/sampling.py`

Standard token-by-token generation:

1. Initialize:
   - `input_ids[:, 0] = BOS` (uses `cls_token_id` or `bos_token_id`)
   - Fill remaining positions with `EOS` (uses `sep_token_id` or `eos_token_id`)
   - `attention_mask[:, 0] = 1`, rest zeros
2. For each position `i` from `1` to `max_length`:
   - `logits = model(input_ids, use_cache=False).logits[:, i-1]`
   - `probs = softmax(logits)`
   - Sample `next_token ~ Categorical(probs)`
   - If sample is done (hit EOS), pad with `pad_token_id`
   - Set `input_ids[:, i] = next_token`
   - Track done status; break early if all sequences finished
3. Return `input_ids`

Note: `use_cache=False` is used (no KV cache optimization), so each step reprocesses the full sequence. The `num_denoising_steps` parameter is accepted but ignored (generation always runs for `max_length` steps).

## Configuration

From `baseline/configs/ar.yaml`:

```yaml
model:
  type: autoregressive

training:
  train_batch_size: 64
  eval_batch_size: 64
  num_train_steps: 1_000_000
  lr_schedule: cosine
  warmup_steps: 10000
  dtype: bf16
  compile_model: true

loss:
  loss_type: ar
  loss_scale: 1.0
  reduction: tokenmean    # not used for AR loss
```

**Optimizer** (from `baseline/configs/optimizer/adam.yaml`):
- Adam with lr=5e-4, betas=(0.9, 0.99), eps=1e-9
- Weight decay: 0.02
- Gradient clip norm: 1.0

Notable differences from diffusion configs:
- No `diffusion_process`, `t_eps`, or `low_discrepancy_sampling`
- `num_train_steps` is 1M (vs 1.131M for MDLM/GIDD)
- `loss.reduction` field exists but is not used by `AutoregressiveTrainer`

## Key Files

| File | Contents |
|---|---|
| `baseline/configs/ar.yaml` | Main config |
| `baseline/configs/model/small.yaml` | Architecture hyperparameters (768d, 12 blocks, 12 heads) |
| `baseline/configs/optimizer/adam.yaml` | Optimizer settings |
| `baseline/modeling.py` | `LlamaForCausalLM` instantiation via `get_model()` |
| `baseline/sampling.py` | `AutoregressiveSampler` class |
| `baseline/trainer.py` | `AutoregressiveTrainer` class |
| `baseline/loss.py` | Returns `nn.CrossEntropyLoss(reduction="none")` for AR |
