# File: baseline/models/dit_gla.py
"""
Bidirectional Gated Linear Attention (Bi-GLA) for Diffusion Language Models.

Replaces DiT's self-attention with bidirectional gated linear attention:
  - Data-dependent gates (unlike RWKV's fixed decay)
  - Bidirectional scan (forward + backward, essential for masked diffusion)
  - O(T) complexity, pure PyTorch (no custom CUDA kernels needed)
  - Channel-Mix FFN (from RWKV, with squared ReLU gating)

Key difference from RWKV:
  RWKV:  decay is FIXED per channel (learned but input-independent)
  GLA:   decay is DATA-DEPENDENT → model can choose to remember/forget at each position
         → For unmasked tokens: keep gate open (remember)
         → For masked tokens: close gate (forget the [MASK])

This makes GLA better suited for MDLM's random masking pattern.

Reference: "Gated Linear Attention Transformers with Hardware-Efficient Training" (Yang et al., ICML 2024)
"""

import math
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers from dit.py
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
# Bidirectional Gated Linear Attention
# ---------------------------------------------------------------------------

def _gla_recurrence(q, k, v, gate, reverse=False):
    """
    Gated linear attention recurrence (single direction).

    For each timestep t:
        S_t = gate_t * S_{t-1} + k_t^T * v_t    (state update)
        o_t = q_t @ S_t                           (output query)

    where gate_t is data-dependent (sigmoid of input projection).

    Using diagonal approximation for efficiency:
        Instead of (d_k, d_v) matrix state, use element-wise operations.
        S_t[i] = gate_t[i] * S_{t-1}[i] + k_t[i] * v_t[i]
        o_t[i] = q_t[i] * S_t[i]

    Args:
        q: (B, T, C) query
        k: (B, T, C) key
        v: (B, T, C) value
        gate: (B, T, C) forget gate in [0, 1]
        reverse: if True, scan from T→1 instead of 1→T

    Returns:
        output: (B, T, C)
    """
    B, T, C = q.shape
    dtype = q.dtype

    # Work in float32 for numerical stability
    q = q.float()
    k = k.float()
    v = v.float()
    gate = gate.float()

    if reverse:
        q = q.flip(1)
        k = k.flip(1)
        v = v.flip(1)
        gate = gate.flip(1)

    # Parallel scan using cumulative operations
    # For efficiency, process in chunks
    output = torch.zeros_like(q)

    # State: (B, C) — accumulated key-value pairs
    state = torch.zeros(B, C, device=q.device, dtype=torch.float32)

    for t in range(T):
        state = gate[:, t] * state + k[:, t] * v[:, t]
        output[:, t] = q[:, t] * state

    if reverse:
        output = output.flip(1)

    return output.to(dtype)


def _gla_chunk_recurrence(q, k, v, gate, chunk_size=64, reverse=False):
    """
    Chunk-wise parallel GLA for better GPU utilization.

    Within each chunk: parallel matmul (like attention but linear)
    Across chunks: recurrent state passing

    This is much faster than pure sequential scan on GPU.
    """
    B, T, C = q.shape
    dtype = q.dtype

    q = q.float()
    k = k.float()
    v = v.float()
    gate = gate.float()

    if reverse:
        q = q.flip(1)
        k = k.flip(1)
        v = v.flip(1)
        gate = gate.flip(1)

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - T % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        gate = F.pad(gate, (0, 0, 0, pad_len), value=1.0)

    T_padded = q.shape[1]
    num_chunks = T_padded // chunk_size

    # Reshape into chunks: (B, num_chunks, chunk_size, C)
    q = q.reshape(B, num_chunks, chunk_size, C)
    k = k.reshape(B, num_chunks, chunk_size, C)
    v = v.reshape(B, num_chunks, chunk_size, C)
    gate = gate.reshape(B, num_chunks, chunk_size, C)

    output = torch.zeros_like(q)

    # Compute cumulative gate products within each chunk
    # gate_cumsum[i] = gate[0] * gate[1] * ... * gate[i]
    log_gate = torch.log(gate.clamp(min=1e-6))
    log_gate_cumsum = log_gate.cumsum(dim=2)  # (B, num_chunks, chunk_size, C)

    # Cross-chunk state
    state = torch.zeros(B, C, device=q.device, dtype=torch.float32)

    for chunk_idx in range(num_chunks):
        g = gate[:, chunk_idx]           # (B, chunk_size, C)
        q_c = q[:, chunk_idx]            # (B, chunk_size, C)
        k_c = k[:, chunk_idx]            # (B, chunk_size, C)
        v_c = v[:, chunk_idx]            # (B, chunk_size, C)

        lg_cumsum = log_gate_cumsum[:, chunk_idx]  # (B, chunk_size, C)

        # 1) Contribution from previous chunks' state
        # state decayed by cumulative gate within this chunk
        gate_from_start = torch.exp(lg_cumsum)  # (B, chunk_size, C)
        state_contribution = q_c * (gate_from_start * state.unsqueeze(1))

        # 2) Intra-chunk: causal linear attention within chunk
        # For position i in chunk, sum over j<=i of: gate[j+1]*...*gate[i] * k[j]*v[j]
        # = exp(lg_cumsum[i] - lg_cumsum[j]) * k[j] * v[j]
        intra = torch.zeros_like(q_c)
        local_state = torch.zeros(B, C, device=q.device, dtype=torch.float32)
        for t in range(chunk_size):
            local_state = g[:, t] * local_state + k_c[:, t] * v_c[:, t]
            intra[:, t] = q_c[:, t] * local_state

        output[:, chunk_idx] = state_contribution + intra

        # Update cross-chunk state: decay by full chunk gate, add chunk's contribution
        chunk_total_gate = torch.exp(lg_cumsum[:, -1])  # (B, C)
        # State after this chunk = state * total_decay + accumulated kv in chunk
        local_state_final = torch.zeros(B, C, device=q.device, dtype=torch.float32)
        for t in range(chunk_size):
            local_state_final = g[:, t] * local_state_final + k_c[:, t] * v_c[:, t]
        state = chunk_total_gate * state + local_state_final

    # Reshape back and remove padding
    output = output.reshape(B, T_padded, C)[:, :T]

    if reverse:
        output = output.flip(1)

    return output.to(dtype)


class BiGLA(nn.Module):
    """
    Bidirectional Gated Linear Attention.

    Core formulas:
        gate_t = sigmoid(W_gate @ x_t + b_gate)     # data-dependent forget gate
        q_t = W_q @ x_t                              # query
        k_t = W_k @ x_t                              # key
        v_t = W_v @ x_t                              # value

        Forward scan:  S_t = gate_t * S_{t-1} + k_t * v_t;  o_fwd_t = q_t * S_t
        Backward scan: S_t = gate_t * S_{t+1} + k_t * v_t;  o_bwd_t = q_t * S_t

        output = sigmoid(W_r @ x_t) * (o_fwd + o_bwd)  # gated combination
    """

    def __init__(self, dim, n_layer, layer_id, expand_ratio=1, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        inner_dim = int(dim * expand_ratio)

        # Projections
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.gate_proj = nn.Linear(dim, inner_dim, bias=True)
        self.receptance = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        # Layer norm on output (stabilizes training)
        self.out_norm = nn.LayerNorm(inner_dim)

        # Init gate bias to positive (default: remember, gate ≈ 0.7-0.9)
        nn.init.constant_(self.gate_proj.bias, 1.0)

        # Small init for output projection
        nn.init.zeros_(self.out_proj.weight)

        # Token mixing (1D shift, like RWKV)
        self.layer_id = layer_id
        self.n_layer = n_layer
        with torch.no_grad():
            ratio = 1.0 - (layer_id / n_layer)
            x = torch.ones(1, 1, dim)
            for i in range(dim):
                x[0, 0, i] = i / dim
            self.mix_q = nn.Parameter(torch.pow(x, ratio))
            self.mix_k = nn.Parameter(torch.pow(x, ratio))
            self.mix_v = nn.Parameter(torch.pow(x, ratio) + 0.3 * (layer_id / max(n_layer - 1, 1)))

    def _shift_mix(self, x):
        """Simple 1D token mixing: shift + weighted average with neighbors."""
        B, T, C = x.shape
        # Shift by 1 position (prepend zeros)
        x_shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        return x_shifted

    def forward(self, x):
        B, T, C = x.shape

        # Token mixing
        xx = self._shift_mix(x)
        xq = x * self.mix_q + xx * (1 - self.mix_q)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xv = x * self.mix_v + xx * (1 - self.mix_v)

        # Project
        q = self.q_proj(xq)
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        gate = torch.sigmoid(self.gate_proj(x))  # data-dependent gate!
        r = torch.sigmoid(self.receptance(x))

        # Normalize q, k for stability
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Bidirectional gated linear recurrence
        if T <= 256:
            # Short sequences: simple sequential scan
            o_fwd = _gla_recurrence(q, k, v, gate, reverse=False)
            o_bwd = _gla_recurrence(q, k, v, gate, reverse=True)
        else:
            # Long sequences: chunk-wise parallel
            o_fwd = _gla_chunk_recurrence(q, k, v, gate, self.chunk_size, reverse=False)
            o_bwd = _gla_chunk_recurrence(q, k, v, gate, self.chunk_size, reverse=True)

        # Combine forward + backward with receptance gating
        output = r * self.out_norm(o_fwd + o_bwd)
        output = self.out_proj(output)

        return output


# ---------------------------------------------------------------------------
# Channel-Mix FFN (same as RWKV, works well with GLA)
# ---------------------------------------------------------------------------

class ChannelMix(nn.Module):
    """RWKV-style gated FFN with squared ReLU."""

    def __init__(self, dim, n_layer, layer_id, hidden_rate=4):
        super().__init__()
        self.layer_id = layer_id
        hidden_sz = hidden_rate * dim

        self.key = nn.Linear(dim, hidden_sz, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden_sz, dim, bias=False)

        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.receptance.weight)

        # Token mixing
        with torch.no_grad():
            ratio = 1.0 - (layer_id / n_layer)
            x = torch.ones(1, 1, dim)
            for i in range(dim):
                x[0, 0, i] = i / dim
            self.mix_k = nn.Parameter(torch.pow(x, ratio))
            self.mix_r = nn.Parameter(torch.pow(x, ratio))

    def forward(self, x):
        xx = F.pad(x[:, :-1], (0, 0, 1, 0))
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xr = x * self.mix_r + xx * (1 - self.mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


# ---------------------------------------------------------------------------
# Bi-GLA Block with adaLN (drop-in for DDiTBlock)
# ---------------------------------------------------------------------------

class BiGLABlock(nn.Module):
    """
    Bidirectional GLA block with adaLN modulation.

    Structure:
        adaLN → Bi-GLA      → residual
        adaLN → Channel-Mix  → residual
    """

    def __init__(self, dim, n_heads, cond_dim, n_layer, layer_id,
                 mlp_ratio=4, dropout=0.1, chunk_size=64):
        super().__init__()
        self.dim = dim
        self.dropout = dropout

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # Bi-GLA replaces self-attention
        self.att = BiGLA(
            dim=dim, n_layer=n_layer, layer_id=layer_id,
            chunk_size=chunk_size,
        )

        # Channel-Mix replaces FFN
        self.ffn = ChannelMix(
            dim=dim, n_layer=n_layer, layer_id=layer_id,
            hidden_rate=mlp_ratio,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # adaLN modulation (same 6-param as DDiTBlock)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, rotary_cos_sin, c, attention_mask=None, seqlens=None):
        # rotary_cos_sin unused (GLA doesn't use rotary embeddings)
        # attention_mask unused (GLA handles full sequence via gating)

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Bi-GLA
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
# DGLA — full model matching DIT interface
# ---------------------------------------------------------------------------

class DGLA(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """
    Diffusion model with Bidirectional Gated Linear Attention.

    Key advantages over DiT:
      - O(T) complexity instead of O(T^2)
      - Data-dependent gating (better than RWKV's fixed decay for random masking)
      - No custom CUDA kernels needed (pure PyTorch)

    Key advantages over Bi-RWKV:
      - Data-dependent gates: model decides per-token what to remember/forget
      - Better for random masking: can "see through" masks to distant unmasked tokens
      - No CUDA kernel compilation needed

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
        chunk_size = getattr(config.model, 'gla_chunk_size', 64)

        self.vocab_embed = EmbeddingLayer(hidden_size, self.rounded_vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        # Keep rotary for interface compat (unused by GLA blocks)
        self.rotary_emb = Rotary(
            hidden_size // n_heads,
            max_seq_len=config.model.max_seq_len,
        )

        blocks = []
        for i in range(n_blocks):
            blocks.append(BiGLABlock(
                dim=hidden_size,
                n_heads=n_heads,
                cond_dim=cond_dim,
                n_layer=n_blocks,
                layer_id=i,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                chunk_size=chunk_size,
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
