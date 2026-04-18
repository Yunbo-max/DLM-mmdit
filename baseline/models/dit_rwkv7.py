# File: baseline/models/dit_rwkv7.py
"""
Bidirectional RWKV-7 "Goose" for Diffusion Language Models (MDLM).

RWKV-7 key innovations over RWKV-4:
  1. Matrix-valued state (H, N, N) — much higher information capacity
  2. Data-dependent decay: w = f(x) — model learns when to forget
  3. In-context learning: state = state * w + state @ a @ b + v @ k
     The "state @ a @ b" term is self-attention on the state itself!
  4. Value residual across layers
  5. LoRA for data-dependent parameters (efficient)

Adapted for MDLM:
  - Bidirectional scan (forward + backward) for masked diffusion
  - adaLN modulation for timestep conditioning
  - Drop-in replacement for DDiTBlock/DIT

Reference: https://github.com/BlinkDL/RWKV-LM (RWKV-v7)
"""

import math
import os

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------
# CUDA kernel for WKV-7
# ---------------------------------------------------------------------------

HEAD_SIZE = 64  # RWKV-7 default, don't change

try:
    from torch.utils.cpp_extension import load
    wkv7_cuda = load(
        name="wkv7g",
        sources=[
            os.path.join(os.path.dirname(__file__), "cuda_v7", "wkv7_op.cpp"),
            os.path.join(os.path.dirname(__file__), "cuda_v7", "wkv7.cu"),
        ],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
            "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}",
        ],
        is_python_module=False,
    )
    has_wkv7_cuda = True
except Exception as e:
    print(f"WARNING: Could not compile WKV-7 CUDA kernel: {e}")
    print("Using pure PyTorch fallback (much slower)")
    has_wkv7_cuda = False


class WKV7_CUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(t.is_contiguous() for t in [r, w, k, v, a, b])
            y = torch.empty((B, T, C), device=r.device, dtype=r.dtype,
                            memory_format=torch.contiguous_format)
            torch.ops.wkv7g.forward(B, T, C, H, r, w, k, v, a, b, y)
            return y


def rwkv7_op(r, w, k, v, a, b, use_cuda=True):
    """Run WKV-7 operation with CUDA or PyTorch fallback."""
    if has_wkv7_cuda and use_cuda and r.is_cuda:
        return WKV7_CUDA.apply(r, w, k, v, a, b)
    else:
        return rwkv7_pytorch(r, w, k, v, a, b)


def rwkv7_pytorch(r, w, k, v, a, b):
    """Pure PyTorch fallback for WKV-7 (slow, for debugging)."""
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)

    return out.view(B, T, C).to(dtype=r.dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate_fused(x, shift, scale):
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


# ---------------------------------------------------------------------------
# Bidirectional RWKV-7 TimeMix
# ---------------------------------------------------------------------------

class BiRWKV7_TimeMix(nn.Module):
    """
    Bidirectional RWKV-7 TimeMix for diffusion LMs.

    Runs WKV-7 forward AND backward, then combines outputs.
    Data-dependent decay + in-context learning + matrix state.
    """

    def __init__(self, n_embd, n_layer, layer_id,
                 d_decay_lora=64, d_aaa_lora=64, d_mv_lora=32, d_gate_lora=128):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.head_size = HEAD_SIZE
        self.n_head = n_embd // HEAD_SIZE
        assert n_embd % HEAD_SIZE == 0

        H = self.n_head
        N = HEAD_SIZE
        C = n_embd

        # Token mixing parameters (learnable interpolation with shifted tokens)
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        # Data-dependent decay (via LoRA)
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, d_decay_lora))
        self.w2 = nn.Parameter(torch.empty(d_decay_lora, C))

        # In-context learning rate (via LoRA)
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, d_aaa_lora))
        self.a2 = nn.Parameter(torch.empty(d_aaa_lora, C))

        # Value residual (via LoRA)
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, d_mv_lora))
        self.v2 = nn.Parameter(torch.empty(d_mv_lora, C))

        # Output gate (via LoRA)
        self.g1 = nn.Parameter(torch.empty(C, d_gate_lora))
        self.g2 = nn.Parameter(torch.empty(d_gate_lora, C))

        # Key normalization and in-context learning
        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        # Bidirectional shift: forward shift + backward shift
        self.shift_fwd = nn.ZeroPad2d((0, 0, 1, -1))  # shift right (causal)
        self.shift_bwd = nn.ZeroPad2d((0, 0, -1, 1))  # shift left (anti-causal)

        # Projections
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        # Combine forward + backward
        self.bi_merge = nn.Linear(C * 2, C, bias=False)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        C = self.n_embd
        layer_id = self.layer_id
        n_layer = self.n_layer

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

            # Fancy init from RWKV-7
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k.copy_(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v.copy_(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            decay_speed = torch.ones(C)
            for h in range(C):
                decay_speed[h] = -7 + 5 * (h / max(C - 1, 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0.copy_(decay_speed.reshape(1, 1, C) + 0.5)
            nn.init.orthogonal_(self.w1, gain=0.1)
            nn.init.zeros_(self.w2)

            self.a0.copy_(torch.zeros(1, 1, C))
            nn.init.orthogonal_(self.a1, gain=0.1)
            nn.init.zeros_(self.a2)

            self.v0.copy_(torch.zeros(1, 1, C) + 1.0)
            nn.init.orthogonal_(self.v1, gain=0.1)
            nn.init.zeros_(self.v2)

            nn.init.orthogonal_(self.g1, gain=0.1)
            nn.init.zeros_(self.g2)

            self.k_k.copy_(torch.ones(1, 1, C) * 0.85)
            self.k_a.copy_(torch.ones(1, 1, C))
            self.r_k.copy_(torch.zeros(self.n_head, HEAD_SIZE) + 0.1)

            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.bi_merge.weight)

    def _run_one_direction(self, x, xx, v_first, dtype):
        """Run RWKV-7 in one direction."""
        B, T, C = x.size()
        H = self.n_head

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        out = rwkv7_op(
            r.to(dtype), w.to(dtype), k.to(dtype), v.to(dtype),
            (-kk).to(dtype), (kk * a).to(dtype),
        )
        out = self.ln_x(out.view(B * T, C)).view(B, T, C)

        # Bonus: direct r*k interaction
        out = out + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        out = out * g

        return out, v_first

    def forward(self, x, v_first):
        B, T, C = x.size()
        dtype = x.dtype

        # Forward direction: shift right
        xx_fwd = self.shift_fwd(x) - x
        out_fwd, v_first = self._run_one_direction(x, xx_fwd, v_first, dtype)

        # Backward direction: shift left
        xx_bwd = self.shift_bwd(x) - x
        out_bwd, _ = self._run_one_direction(x, xx_bwd, v_first, dtype)

        # Merge bidirectional outputs
        out = self.bi_merge(torch.cat([out_fwd, out_bwd], dim=-1))
        out = self.output(out)

        return out, v_first


# ---------------------------------------------------------------------------
# RWKV-7 ChannelMix (same as original, with bidirectional shift)
# ---------------------------------------------------------------------------

class BiRWKV7_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4):
        super().__init__()
        self.n_embd = n_embd
        dim_ffn = n_embd * hidden_rate

        self.shift_fwd = nn.ZeroPad2d((0, 0, 1, -1))
        self.shift_bwd = nn.ZeroPad2d((0, 0, -1, 1))

        with torch.no_grad():
            ratio = 1.0 - (layer_id / n_layer)
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio))

        self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.value = nn.Linear(dim_ffn, n_embd, bias=False)

    def forward(self, x):
        # Bidirectional shift: average forward and backward
        xx = (self.shift_fwd(x) + self.shift_bwd(x)) / 2.0 - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


# ---------------------------------------------------------------------------
# Bi-RWKV-7 Block with adaLN
# ---------------------------------------------------------------------------

class BiRWKV7Block(nn.Module):
    """
    Bi-RWKV-7 block with adaLN modulation for diffusion.

    Structure:
        adaLN → Bi-RWKV-7 TimeMix → residual
        adaLN → Bi-RWKV-7 ChannelMix → residual
    """

    def __init__(self, n_embd, n_heads, cond_dim, n_layer, layer_id,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.layer_id = layer_id
        self.dropout_val = dropout

        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        if layer_id == 0:
            self.ln0 = LayerNorm(n_embd)

        self.att = BiRWKV7_TimeMix(n_embd, n_layer, layer_id)
        self.ffn = BiRWKV7_ChannelMix(n_embd, n_layer, layer_id, hidden_rate=mlp_ratio)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # adaLN modulation (6 params: shift/scale/gate for att + ffn)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * n_embd, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, rotary_cos_sin, c, v_first, attention_mask=None, seqlens=None):
        # rotary_cos_sin unused (RWKV-7 doesn't use rotary embeddings)

        if self.layer_id == 0:
            x = self.ln0(x)

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # TimeMix (Bi-RWKV-7)
        x_skip = x
        xx = modulate_fused(self.ln1(x), shift_msa, scale_msa)
        xx, v_first = self.att(xx, v_first)
        x = x_skip + gate_msa * self.dropout1(xx)

        # ChannelMix
        x_skip = x
        xx = modulate_fused(self.ln2(x), shift_mlp, scale_mlp)
        xx = self.ffn(xx)
        x = x_skip + gate_mlp * self.dropout2(xx)

        return x, v_first


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
# DRWKV7 — full model matching DIT interface
# ---------------------------------------------------------------------------

class DRWKV7(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """
    Bidirectional RWKV-7 for text diffusion (MDLM).

    Key advantages:
      - Matrix-valued state (N x N per head) — much richer than RWKV-4's vector
      - Data-dependent decay — learns when to forget (critical for random masking)
      - In-context learning — state self-attention
      - O(T) complexity
      - Bidirectional for masked diffusion

    Drop-in replacement for DIT — same forward(indices, sigma, attention_mask).
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

        # hidden_size must be divisible by HEAD_SIZE (64)
        assert hidden_size % HEAD_SIZE == 0, f"hidden_size ({hidden_size}) must be divisible by HEAD_SIZE ({HEAD_SIZE})"

        self.vocab_embed = EmbeddingLayer(hidden_size, self.rounded_vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        # Keep rotary for interface compat (unused by RWKV-7)
        self.rotary_emb = Rotary(
            hidden_size // n_heads,
            max_seq_len=config.model.max_seq_len,
        )

        blocks = []
        for i in range(n_blocks):
            blocks.append(BiRWKV7Block(
                n_embd=hidden_size,
                n_heads=n_heads,
                cond_dim=cond_dim,
                n_layer=n_blocks,
                layer_id=i,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
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

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, rotary_cos_sin, c, v_first, attention_mask=attention_mask)

        x1 = self.output_layer(x, c)

        if self.cluster_size > 0:
            x2 = self.output_layer_clusters(x, c)
            return x1, x2

        return x1
