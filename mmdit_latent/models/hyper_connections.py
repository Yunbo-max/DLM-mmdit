"""
Manifold-Constrained Hyper-Connections (mHCv2).

Vendored from hyper-connections==0.4.9 (MIT License) by lucidrains.
Paper: https://arxiv.org/abs/2409.19606

This file is self-contained — no external hyper_connections or torch_einops_utils needed.
"""

from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce

# helper functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def add(x, y):
    return x + y

# sinkhorn

def l1norm(t, dim):
    return F.normalize(t, p=1, dim=dim)

def sinkhorn_knopps(log_alpha, iters=20):
    if iters <= 0:
        return log_alpha

    assert log_alpha.shape[-2] == log_alpha.shape[-1]

    dtype = log_alpha.dtype
    log_alpha = log_alpha.float()
    log_alpha = log_alpha - log_alpha.amax(dim=-2, keepdim=True).detach()
    alpha = log_alpha.exp()

    for _ in range(iters):
        alpha = l1norm(alpha, dim=-2)
        alpha = l1norm(alpha, dim=-1)

    return alpha.to(dtype)

# expand / reduce stream functions

def get_expand_reduce_stream_functions(num_streams, disable=False, **kwargs):
    if disable:
        return (nn.Identity(), nn.Identity())

    expand_fn = Reduce('... d -> ... s d', 'repeat', s=num_streams)
    reduce_fn = Reduce('... s d -> ... d', 'sum')
    return expand_fn, reduce_fn

# norms

class _RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)

# residual base class

class Residual(Module):
    def __init__(self, *args, branch=None, residual_transform=None, **kwargs):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(self, residuals):
        return residuals, residuals, dict()

    def depth_connection(self, branch_output, residuals):
        return branch_output + self.residual_transform(residuals)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)

# manifold-constrained hyper-connections

class ManifoldConstrainedHyperConnections(Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch=None,
        layer_index=None,
        dropout=0.,
        residual_transform=None,
        add_branch_out_to_residual=True,
        num_input_views=1,
        depth_residual_fn=add,
        num_fracs=1,
        sinkhorn_iters=20,
        num_dynamic_alpha_proposals=1,
        use_triton_sinkhorn=False,
        **kwargs,
    ):
        super().__init__()
        self.branch = branch

        assert num_fracs >= 1
        self.num_fracs = num_fracs
        self.has_fracs = num_fracs > 1

        self.split_fracs = Rearrange('b ... (f d) -> b ... f d', f=num_fracs)
        self.merge_fracs = Rearrange('b ... f d -> b ... (f d)')

        assert divisible_by(dim, num_fracs)
        dim //= num_fracs

        self.maybe_mix_streams = None  # Not used without torch_einops_utils

        self.norm = _RMSNorm(dim)

        assert num_residual_streams > 0
        self.num_residual_streams = num_residual_streams
        init_residual_index = default(layer_index, randrange(num_residual_streams)) % num_residual_streams

        num_residual_streams_fracs = num_residual_streams * num_fracs
        num_input_views_fracs = num_input_views * num_fracs

        self.num_fracs = num_fracs
        assert num_input_views >= 1
        self.num_input_views = num_input_views

        self.has_dynamic_alpha_proposals = num_dynamic_alpha_proposals > 1
        self.num_dynamic_alpha_proposals = num_dynamic_alpha_proposals

        # width connection (alpha)
        init_alpha0 = torch.zeros((num_residual_streams_fracs, num_input_views_fracs))
        init_alpha0[init_residual_index, :] = 1.
        self.static_alpha = nn.Parameter(cat((init_alpha0, torch.eye(num_residual_streams_fracs)), dim=1))
        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(num_dynamic_alpha_proposals, dim, num_residual_streams_fracs + num_input_views_fracs))
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

        # depth connection (beta)
        self.add_branch_out_to_residual = add_branch_out_to_residual
        if add_branch_out_to_residual:
            self.static_beta = nn.Parameter(torch.ones(num_residual_streams, num_fracs, 1))
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim, num_fracs))
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.residual_mix_constraint_fn = partial(sinkhorn_knopps, iters=sinkhorn_iters)
        self.dropout = nn.Dropout(dropout)
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

    def width_connection(self, residuals):
        streams, fracs = self.num_residual_streams, self.num_fracs

        residuals = self.residual_transform(residuals)
        residuals = self.split_fracs(residuals)

        normed = self.norm(residuals)

        dtype = residuals.dtype
        normed = normed.float()

        wc_weight = einsum(normed, self.dynamic_alpha_fn.float(), '... d, p d mix -> p ... mix')
        wc_weight = rearrange(wc_weight, '... s1 f2 mix -> ... (s1 f2) mix')

        pre_branch_scale = repeat(self.pre_branch_scale.float(), '1 -> s', s=self.num_fracs)
        residual_scale = repeat(self.residual_scale.float(), '1 -> s', s=self.num_fracs * streams)
        alpha_scale = cat((pre_branch_scale, residual_scale))
        alpha_scale = repeat(alpha_scale, 'n -> (v n)', v=self.num_input_views)

        dynamic_alpha = wc_weight * alpha_scale
        alpha = dynamic_alpha + self.static_alpha.float()

        alpha_pre, alpha_residual = alpha[..., :self.num_input_views * self.num_fracs], alpha[..., self.num_input_views * self.num_fracs:]
        alpha_pre = alpha_pre.sigmoid()
        alpha_residual = self.residual_mix_constraint_fn(alpha_residual)
        alpha = cat((alpha_pre, alpha_residual), dim=-1)

        if self.has_dynamic_alpha_proposals:
            alpha = reduce(alpha, 'p ... -> ...', 'mean')
        else:
            alpha = rearrange(alpha, '1 ... -> ...')

        alpha = rearrange(alpha, '... (s f) t -> ... s f t', s=streams)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = normed @ self.dynamic_beta_fn.float()
            dynamic_beta = dc_weight * self.h_post_scale.float()
            beta = dynamic_beta + self.static_beta.float()
            beta = beta.sigmoid() * 2

        mix_h = einsum(alpha, residuals.float(), '... s f tf, ... s f d -> ... tf d')
        mix_h = rearrange(mix_h, '... (t f) d -> ... t f d', f=fracs)

        if self.num_input_views == 1:
            branch_input, residuals = mix_h[..., 0, :, :], mix_h[..., 1:, :, :]
        else:
            branch_input, residuals = mix_h[..., :self.num_input_views, :, :], mix_h[..., self.num_input_views:, :, :]
            branch_input = rearrange(branch_input, 'b ... v f d -> v b ... f d')

        branch_input = self.merge_fracs(branch_input)
        residuals = rearrange(residuals, 'b ... s f d -> b ... s (f d)')

        branch_input, residuals = tuple(t.to(dtype) for t in (branch_input, residuals))
        if exists(beta):
            beta = beta.to(dtype)

        return branch_input, residuals, dict(beta=beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        assert self.add_branch_out_to_residual

        branch_output = self.split_fracs(branch_output)
        dtype = residuals.dtype

        output = einsum(branch_output.float(), beta.float(), 'b ... f1 d, b ... s f1 f2 -> b ... s f2 d')
        output = self.merge_fracs(output)

        residuals = self.depth_residual_fn(output.to(dtype), residuals)
        return self.dropout(residuals)

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


ManifoldConstrainedHyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
