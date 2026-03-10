"""
MMDiT-based model with latent conditioning as a second modality in joint attention.

Instead of conditioning via AdaLN additive modulation (baseline approach), the latent
becomes a second modality in MMDiT's joint attention mechanism.
"""

import math

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import (
    TimestepEmbedder,
    EmbeddingLayer,
    DDitFinalLayer,
    modulate_fused,
)
from .mmdit_block import MMDiTBlock


class MMDiTWithLatentConditioning(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """
    Replaces the DIT backbone with MMDiT for N-modality joint attention.

    Text tokens and latent tokens attend to each other via joint attention
    at every block, with per-modality FFN and adaptive layer norms conditioned
    on the diffusion timestep.
    """

    def __init__(self, config, vocab_size: int, latent_dim: int = 768, cluster_size: int = 100):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.cluster_size = cluster_size
        self.rounded_vocab_size = vocab_size + cluster_size + (128 - (vocab_size + cluster_size) % 128) % 128

        hidden_size = config.hidden_size
        cond_dim = config.cond_dim
        n_heads = config.n_heads
        n_blocks = config.n_blocks
        dropout = config.get("dropout", 0.1)
        max_seq_len = config.get("max_seq_len", 512)
        qk_rmsnorm = config.get("qk_rmsnorm", False)
        latent_hidden_size = config.get("latent_hidden_size", hidden_size)

        self.hidden_size = hidden_size
        self.latent_hidden_size = latent_hidden_size

        # --- Text pathway ---
        self.vocab_embed = EmbeddingLayer(hidden_size, self.rounded_vocab_size)
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        nn.init.trunc_normal_(self.text_pos_embed, std=0.02)

        # --- Timestep conditioning ---
        self.sigma_map = TimestepEmbedder(cond_dim)

        # --- Latent pathway ---
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, latent_hidden_size),
        )
        self.null_latent = nn.Parameter(torch.zeros(1, 1, latent_hidden_size))
        nn.init.trunc_normal_(self.null_latent, std=0.02)

        # --- MMDiT blocks ---
        dim_head = hidden_size // n_heads
        blocks = []
        for _ in range(n_blocks):
            blocks.append(MMDiTBlock(
                dim_modalities=(hidden_size, latent_hidden_size),
                dim_cond=cond_dim,
                dim_head=dim_head,
                heads=n_heads,
                qk_rmsnorm=qk_rmsnorm,
                dropout=dropout,
            ))
        self.blocks = nn.ModuleList(blocks)

        # --- Output layer (text only) ---
        self.output_layer = DDitFinalLayer(
            hidden_size, self.rounded_vocab_size, cond_dim
        )

        if cluster_size > 0:
            self.output_layer_clusters = DDitFinalLayer(
                hidden_size, self.rounded_vocab_size, cond_dim
            )
        else:
            self.output_layer_clusters = None

    def forward(self, indices, sigma, latents=None, attention_mask=None):
        """
        Forward pass — signature identical to DITWithLatentConditioning.

        Args:
            indices:  (B, S) token indices (possibly noised)
            sigma:    (B,) diffusion timesteps
            latents:  (B, latent_dim) or (B, 1, latent_dim) or None
            attention_mask: (B, S) with 1 for valid tokens, 0 for pad

        Returns:
            logits (B, S, vocab_size), or (logits, cluster_logits) if cluster_size > 0
        """
        B, S = indices.shape

        # --- Timestep conditioning ---
        sigma_float = sigma.float() if sigma.dtype != torch.float32 else sigma
        c = F.silu(self.sigma_map(sigma_float))
        c = c.to(dtype=self.vocab_embed.embedding.dtype)

        # --- Text tokens ---
        text_tokens = self.vocab_embed(indices) + self.text_pos_embed[:, :S]

        # --- Latent tokens ---
        if latents is not None:
            if latents.dim() == 2:
                latents = latents.unsqueeze(1)  # (B, 1, latent_dim)
            latent_tokens = self.latent_encoder(latents)  # (B, L, latent_hidden_size)
        else:
            latent_tokens = self.null_latent.expand(B, -1, -1)  # (B, 1, latent_hidden_size)

        # --- Build masks ---
        text_mask = attention_mask.bool() if attention_mask is not None else None
        # Latent tokens are always valid
        latent_mask = torch.ones(B, latent_tokens.shape[1], device=indices.device, dtype=torch.bool)

        # --- Run through MMDiT blocks ---
        for block in self.blocks:
            text_tokens, latent_tokens = block(
                modality_tokens=(text_tokens, latent_tokens),
                modality_masks=(text_mask, latent_mask),
                time_cond=c,
            )

        # --- Output (text only, discard latent) ---
        x1 = self.output_layer(text_tokens, c)

        if self.cluster_size > 0 and self.output_layer_clusters is not None:
            x2 = self.output_layer_clusters(text_tokens, c)
            return x1, x2

        return x1
