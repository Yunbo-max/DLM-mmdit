"""
Self-contained MMDiT building blocks for N-modality joint attention.

Inlines all dependencies (no external mmdit, x_transformers, or hyper_connections packages).
Based on mmdit-0.3.0/mmdit/mmdit_pytorch.py and mmdit_generalized_pytorch.py.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# ---------------------------------------------------------------------------
# RMSNorm (inline)
# ---------------------------------------------------------------------------

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x.float(), dim=-1).to(x.dtype) * self.gamma * self.scale


# ---------------------------------------------------------------------------
# MultiHeadRMSNorm  (from mmdit_pytorch.py:37-44)
# ---------------------------------------------------------------------------

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


# ---------------------------------------------------------------------------
# FeedForward (inline)
# ---------------------------------------------------------------------------

class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Residual (inline — simple identity residual)
# ---------------------------------------------------------------------------

class Residual(Module):
    """Simple residual wrapper: returns (x, add_residual_fn)."""
    def __init__(self, num_streams=1, dim=None):
        super().__init__()
        # num_streams / dim args accepted for API compat but unused in simple mode

    def forward(self, x):
        return x, lambda res: x + res


# ---------------------------------------------------------------------------
# AdaptiveLayerNorm  (from mmdit_generalized_pytorch.py:38-77)
# ---------------------------------------------------------------------------

class AdaptiveLayerNorm(Module):
    def __init__(self, dim, dim_cond=None):
        super().__init__()
        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        self.ln = nn.LayerNorm(dim, elementwise_affine=not has_cond)

        if has_cond:
            cond_linear = nn.Linear(dim_cond, dim * 2)

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear,
            )

            nn.init.zeros_(cond_linear.weight)
            nn.init.constant_(cond_linear.bias[:dim], 1.)
            nn.init.zeros_(cond_linear.bias[dim:])

    def forward(self, x, cond=None):
        assert not (exists(cond) ^ self.has_cond)
        x = self.ln(x)
        if self.has_cond:
            gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
            x = x * gamma + beta
        return x


# ---------------------------------------------------------------------------
# JointAttention  (from mmdit_pytorch.py:48-165, with F.scaled_dot_product_attention)
# ---------------------------------------------------------------------------

class JointAttention(Module):
    def __init__(
        self,
        *,
        dim_inputs: tuple[int, ...],
        dim_head=64,
        heads=8,
        qk_rmsnorm=False,
        dropout=0.0,
    ):
        super().__init__()

        dim_inner = dim_head * heads
        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs
        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = ModuleList(
            [nn.Linear(d, dim_inner * 3, bias=False) for d in dim_inputs]
        )
        self.to_out = ModuleList(
            [nn.Linear(dim_inner, d, bias=False) for d in dim_inputs]
        )

        self.qk_rmsnorm = qk_rmsnorm
        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads=heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads=heads) for _ in range(num_inputs)])
        else:
            self.q_rmsnorms = [None] * num_inputs
            self.k_rmsnorms = [None] * num_inputs

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: tuple[Tensor, ...],
        masks: tuple[Tensor | None, ...] | None = None,
    ):
        device = inputs[0].device
        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(
            inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms
        ):
            qkv = to_qkv(x)
            # reshape to (3, B, H, N, D)
            qkv = rearrange(qkv, 'b n (three h d) -> three b h n d', three=3, h=self.heads)

            if self.qk_rmsnorm:
                q, k, v = qkv[0], qkv[1], qkv[2]
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device=device, dtype=torch.bool)
            all_masks.append(mask)

        # Concatenate along sequence dim: (3, B, H, N_total, D)
        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        q, k, v = all_qkvs[0], all_qkvs[1], all_qkvs[2]

        # Build boolean attention mask: (B, 1, 1, N_total) for broadcasting
        attn_mask = all_masks.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_total)
        # We need key mask: positions where key is valid
        # For SDPA: True means attend, but we pass it as attn_mask
        # Actually SDPA bool mask: True = attend. Shape (B, 1, N_q, N_kv)
        # Since all queries attend to valid keys: expand key mask
        attn_mask = all_masks[:, None, None, :]  # (B, 1, 1, N_kv) — broadcast over heads and queries

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = self.attn_dropout(out)

        # Merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Unpack per modality
        outs = unpack(out, packed_shape, 'b * d')

        # Separate output projections per modality
        all_outs = []
        for o, to_out in zip(outs, self.to_out):
            all_outs.append(to_out(o))

        return tuple(all_outs)


# ---------------------------------------------------------------------------
# MMDiTBlock  (from mmdit_generalized_pytorch.py:81-204)
# ---------------------------------------------------------------------------

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_modalities: tuple[int, ...],
        dim_cond=None,
        dim_head=64,
        heads=8,
        qk_rmsnorm=False,
        dropout=0.0,
        ff_kwargs: dict = dict(),
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        # Simple residual connections
        self.attn_residual_fns = ModuleList([Residual(dim=d) for d in dim_modalities])
        self.ff_residual_fns = ModuleList([Residual(dim=d) for d in dim_modalities])

        # Optional time conditioning → post-branch gammas
        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            cond_linear = nn.Linear(dim_cond, sum(dim_modalities) * 2)
            self.to_post_branch_gammas = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear,
            )
            nn.init.zeros_(cond_linear.weight)
            nn.init.constant_(cond_linear.bias, 1.)

        # Adaptive layer norms for attention
        self.attn_layernorms = ModuleList(
            [AdaptiveLayerNorm(d, dim_cond=dim_cond) for d in dim_modalities]
        )

        # Joint attention
        self.joint_attn = JointAttention(
            dim_inputs=dim_modalities,
            dim_head=dim_head,
            heads=heads,
            qk_rmsnorm=qk_rmsnorm,
            dropout=dropout,
        )

        # Feedforward per modality
        self.ff_layernorms = ModuleList(
            [AdaptiveLayerNorm(d, dim_cond=dim_cond) for d in dim_modalities]
        )
        self.feedforwards = ModuleList(
            [FeedForward(d, **ff_kwargs) for d in dim_modalities]
        )

    def forward(
        self,
        *,
        modality_tokens: tuple[Tensor, ...],
        modality_masks: tuple[Tensor | None, ...] | None = None,
        time_cond=None,
    ):
        assert len(modality_tokens) == self.num_modalities
        assert not (exists(time_cond) ^ self.has_cond)

        ln_kwargs = dict()
        if self.has_cond:
            ln_kwargs = dict(cond=time_cond)
            gammas = self.to_post_branch_gammas(time_cond)
            attn_gammas, ff_gammas = gammas.chunk(2, dim=-1)

        # --- Attention ---
        modality_tokens, modality_residual_fns = tuple(
            zip(*[res_fn(tok) for res_fn, tok in zip(self.attn_residual_fns, modality_tokens)])
        )
        modality_tokens = [ln(tok, **ln_kwargs) for tok, ln in zip(modality_tokens, self.attn_layernorms)]

        modality_tokens = self.joint_attn(inputs=modality_tokens, masks=modality_masks)

        if self.has_cond:
            split_attn_gammas = attn_gammas.split(self.dim_modalities, dim=-1)
            modality_tokens = [tok * g for tok, g in zip(modality_tokens, split_attn_gammas)]

        modality_tokens = [add_res(tok) for add_res, tok in zip(modality_residual_fns, modality_tokens)]

        # --- Feedforward ---
        modality_tokens, modality_residual_fns = tuple(
            zip(*[res_fn(tok) for res_fn, tok in zip(self.ff_residual_fns, modality_tokens)])
        )
        modality_tokens = [ln(tok, **ln_kwargs) for tok, ln in zip(modality_tokens, self.ff_layernorms)]
        modality_tokens = [ff(tok) for tok, ff in zip(modality_tokens, self.feedforwards)]

        if self.has_cond:
            split_ff_gammas = ff_gammas.split(self.dim_modalities, dim=-1)
            modality_tokens = [tok * g for tok, g in zip(modality_tokens, split_ff_gammas)]

        modality_tokens = [add_res(tok) for add_res, tok in zip(modality_residual_fns, modality_tokens)]

        return modality_tokens
