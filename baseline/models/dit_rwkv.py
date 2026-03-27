# File: baseline/models/dit_rwkv.py
"""
Bi-RWKV Diffusion Language Model.

Replaces DiT's self-attention + FFN with:
  - Bi-WKV (bidirectional weighted key-value) from Diffusion-RWKV
  - Channel-Mix (RWKV-style gated FFN)

Both wrapped with adaLN modulation, matching the original DDiTBlock interface.

Adapted from Diffusion-RWKV (2D image) to 1D text sequences.
Reference: https://github.com/feizc/Diffusion-RWKV
"""

import math
import os
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------
# CUDA kernel for Bi-WKV
# ---------------------------------------------------------------------------

T_MAX = 8192  # max sequence length supported

try:
    from torch.utils.cpp_extension import load
    wkv_cuda = load(
        name="wkv",
        sources=[
            os.path.join(os.path.dirname(__file__), "cuda", "wkv_op.cpp"),
            os.path.join(os.path.dirname(__file__), "cuda", "wkv_cuda.cu"),
        ],
        verbose=True,
        extra_cuda_cflags=[
            '-res-usage', '--maxrregcount 60', '--use_fast_math',
            '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'
        ],
    )
    has_wkv_cuda = True
except Exception as e:
    print(f"WARNING: Could not compile WKV CUDA kernel: {e}")
    print("Falling back to pure PyTorch WKV (slower)")
    has_wkv_cuda = False


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        bf_mode = (w.dtype == torch.bfloat16)
        half_mode = (w.dtype == torch.half)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if bf_mode:
            y = y.bfloat16()
        elif half_mode:
            y = y.half()
        return y

    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        bf_mode = (w.dtype == torch.bfloat16)
        half_mode = (w.dtype == torch.half)
        wkv_cuda.backward(
            B, T, C,
            w.float().contiguous(), u.float().contiguous(),
            k.float().contiguous(), v.float().contiguous(),
            gy.float().contiguous(),
            gw, gu, gk, gv,
        )
        if bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw, gu, gk.bfloat16(), gv.bfloat16())
        elif half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw, gu, gk.half(), gv.half())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


def wkv_pytorch(B, T, C, w, u, k, v):
    """Pure PyTorch fallback for WKV (slow, for debugging/CPU)."""
    w = w.float()
    u = u.float()
    k = k.float()
    v = v.float()
    output = torch.zeros((B, T, C), device=k.device, dtype=torch.float32)
    for b in range(B):
        for c in range(C):
            a = 0.0
            bb = 0.0  # renamed to avoid shadow
            for t in range(T):
                kk = k[b, t, c]
                vv = v[b, t, c]
                ww = w[c] if w.dim() == 1 else w[b, c] if w.dim() == 2 else w[c]
                uu = u[c] if u.dim() == 1 else u[b, c] if u.dim() == 2 else u[c]
                wkv = (a + torch.exp(uu + kk) * vv) / (bb + torch.exp(uu + kk))
                output[b, t, c] = wkv
                a = a * torch.exp(ww) + torch.exp(kk) * vv
                bb = bb * torch.exp(ww) + torch.exp(kk)
    return output


# ---------------------------------------------------------------------------
# 1D q_shift (adapted from 2D version for text sequences)
# ---------------------------------------------------------------------------

def q_shift_1d(input, shift_pixel=1, gamma=1/4):
    """
    Shift channels along the sequence dimension for local context mixing.

    Original 2D version shifts in 4 directions (left/right/up/down).
    1D adaptation: shift forward and backward along sequence, keep rest unchanged.

    Args:
        input: (B, T, C) tensor
        shift_pixel: how many positions to shift
        gamma: fraction of channels to shift in each direction (1/4 each)
    """
    B, T, C = input.shape
    output = torch.zeros_like(input)

    # Shift 1/4 channels forward (right)
    c1 = int(C * gamma)
    output[:, shift_pixel:, :c1] = input[:, :T - shift_pixel, :c1]

    # Shift 1/4 channels backward (left)
    c2 = int(C * gamma * 2)
    output[:, :T - shift_pixel, c1:c2] = input[:, shift_pixel:, c1:c2]

    # Keep remaining 1/2 channels unchanged
    output[:, :, c2:] = input[:, :, c2:]

    return output


# ---------------------------------------------------------------------------
# RWKV Spatial Mix (Bi-WKV attention replacement) — 1D
# ---------------------------------------------------------------------------

class RWKV_SpatialMix_1D(nn.Module):
    """Bi-WKV attention for 1D text sequences."""

    def __init__(self, n_embd, n_layer, layer_id, shift_pixel=1,
                 channel_gamma=1/4, key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma

        # RWKV-style init
        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

            # Decay and first parameters
            decay_speed = torch.ones(n_embd)
            for h in range(n_embd):
                decay_speed[h] = -5 + 8 * (h / (n_embd - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.spatial_decay = nn.Parameter(decay_speed)

            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(n_embd)]) * 0.5)
            self.spatial_first = nn.Parameter(torch.ones(n_embd) * math.log(0.3) + zigzag)

            # Token mixing parameters
            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()

        # 1D token mixing via shift
        if self.shift_pixel > 0:
            xx = q_shift_1d(x, self.shift_pixel, self.channel_gamma)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk, xv, xr = x, x, x

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        # Bi-WKV computation
        if has_wkv_cuda and x.is_cuda:
            wkv_out = RUN_CUDA(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
        else:
            wkv_out = wkv_pytorch(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)

        if self.key_norm is not None:
            wkv_out = self.key_norm(wkv_out)

        x = sr * wkv_out
        x = self.output(x)
        return x


# ---------------------------------------------------------------------------
# RWKV Channel Mix (RWKV FFN replacement) — 1D
# ---------------------------------------------------------------------------

class RWKV_ChannelMix_1D(nn.Module):
    """RWKV-style gated FFN with squared ReLU for 1D text sequences."""

    def __init__(self, n_embd, n_layer, layer_id, shift_pixel=1,
                 channel_gamma=1/4, hidden_rate=4, key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma

        # RWKV-style init
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        if self.shift_pixel > 0:
            xx = q_shift_1d(x, self.shift_pixel, self.channel_gamma)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk, xr = x, x

        k = self.key(xk)
        k = torch.square(torch.relu(k))  # Squared ReLU activation
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(xr)) * kv
        return x


# ---------------------------------------------------------------------------
# Helpers from dit.py
# ---------------------------------------------------------------------------

def modulate_fused(x, shift, scale):
    return x * (1 + scale) + shift


def bias_dropout_add_scale(x, bias, scale, residual, prob, training):
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


# ---------------------------------------------------------------------------
# DRWKVBlock — drop-in replacement for DDiTBlock
# ---------------------------------------------------------------------------

class DRWKVBlock(nn.Module):
    """
    Bi-RWKV block with adaLN modulation.

    Replaces DDiTBlock:
      adaLN → Self-Attention → adaLN → FFN
    With:
      adaLN → Bi-WKV → adaLN → Channel-Mix
    """

    def __init__(self, dim, n_heads, cond_dim, n_layer, layer_id,
                 mlp_ratio=4, dropout=0.1, key_norm=False):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.dropout = dropout

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Bi-WKV replaces self-attention
        self.att = RWKV_SpatialMix_1D(
            n_embd=dim, n_layer=n_layer, layer_id=layer_id,
            key_norm=key_norm,
        )

        # Channel-Mix replaces FFN
        self.ffn = RWKV_ChannelMix_1D(
            n_embd=dim, n_layer=n_layer, layer_id=layer_id,
            hidden_rate=mlp_ratio, key_norm=key_norm,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # adaLN modulation (same as DDiTBlock)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, rotary_cos_sin, c, attention_mask=None, seqlens=None):
        # rotary_cos_sin is unused (RWKV doesn't use rotary embeddings)
        # attention_mask is unused (WKV handles full sequence natively)

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Bi-WKV attention
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        x = self.att(x)
        x = x_skip + gate_msa * self.dropout1(x)

        # Channel-Mix FFN
        x_skip = x
        x = modulate_fused(self.norm2(x), shift_mlp, scale_mlp)
        x = self.ffn(x)
        x = x_skip + gate_mlp * self.dropout2(x)

        return x


# ---------------------------------------------------------------------------
# Reuse from dit.py
# ---------------------------------------------------------------------------

from .dit import (
    TimestepEmbedder,
    EmbeddingLayer,
    DDitFinalLayer,
    Rotary,
)


# ---------------------------------------------------------------------------
# DRWKV — full model matching DIT interface
# ---------------------------------------------------------------------------

class DRWKV(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """
    Diffusion-RWKV for text: Bi-WKV + Channel-Mix with adaLN.

    Drop-in replacement for DIT — same forward signature.
    """

    def __init__(self, config, vocab_size: int, cluster_size: int = 100):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size
        self.cluster_size = cluster_size
        self.rounded_vocab_size = vocab_size + cluster_size + (128 - (vocab_size + cluster_size) % 128) % 128

        hidden_size = config.model.hidden_size
        cond_dim = config.model.cond_dim
        n_heads = config.model.n_heads
        n_blocks = config.model.n_blocks
        dropout = config.model.dropout
        mlp_ratio = getattr(config.model, 'mlp_ratio', 4)
        key_norm = getattr(config.model, 'key_norm', False)

        self.vocab_embed = EmbeddingLayer(hidden_size, self.rounded_vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        # Keep rotary for interface compat (unused by RWKV blocks)
        self.rotary_emb = Rotary(
            hidden_size // n_heads,
            max_seq_len=config.model.max_seq_len,
        )

        blocks = []
        for i in range(n_blocks):
            blocks.append(DRWKVBlock(
                dim=hidden_size,
                n_heads=n_heads,
                cond_dim=cond_dim,
                n_layer=n_blocks,
                layer_id=i,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                key_norm=key_norm,
            ))
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            hidden_size, self.rounded_vocab_size, cond_dim)

        if cluster_size > 0:
            self.output_layer_clusters = DDitFinalLayer(
                hidden_size, self.rounded_vocab_size, cond_dim)
        else:
            self.output_layer_clusters = None

    def forward(self, indices, sigma, attention_mask=None):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        for block in self.blocks:
            x = block(x, rotary_cos_sin, c, attention_mask=attention_mask)

        x1 = self.output_layer(x, c)

        if self.cluster_size > 0:
            x2 = self.output_layer_clusters(x, c)
            return x1, x2

        return x1
