"""
MMDiT-based model with latent conditioning as a second modality in joint attention.

Uses the external mmdit package (from mmdit.mmdit_generalized_pytorch) instead of
internal copies. This ensures consistency with latentDLM_mmdit and uses the
latest hyper-connections implementation.
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

try:
    from mmdit.mmdit_generalized_pytorch import MMDiT

    has_external_mmdit = True
    print("Using external mmdit package (mmdit.mmdit_generalized_pytorch.MMDiT)")
except ImportError as e:
    print(f"ERROR: Could not import external mmdit package: {e}")
    print("Please install: pip install mmdit")
    raise


class MMDiTWithLatentConditioning(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """
    Replaces the DIT backbone with MMDiT for N-modality joint attention.

    Text tokens and latent tokens attend to each other via joint attention
    at every block, with per-modality FFN and adaptive layer norms conditioned
    on the diffusion timestep.

    Uses external mmdit.MMDiT from the mmdit package.
    """

    def __init__(
        self, config, vocab_size: int, latent_dim: int = 768, cluster_size: int = 100
    ):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.cluster_size = cluster_size
        self.rounded_vocab_size = vocab_size

        hidden_size = config.hidden_size
        cond_dim = config.cond_dim
        n_heads = config.n_heads
        n_blocks = config.n_blocks
        dropout = config.get("dropout", 0.1)
        max_seq_len = config.get("max_seq_len", 512)
        qk_rmsnorm = config.get("qk_rmsnorm", True)
        num_residual_streams = config.get("num_residual_streams", 2)
        latent_hidden_size = config.get("latent_hidden_size", hidden_size)

        self.hidden_size = hidden_size
        self.latent_hidden_size = latent_hidden_size
        self.num_residual_streams = num_residual_streams

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

        # --- MMDiT backbone (external package) ---
        self.mmdit = MMDiT(
            depth=n_blocks,
            dim_modalities=(
                hidden_size,
                latent_hidden_size,
            ),
            dim_cond=cond_dim,
            dim_head=hidden_size // n_heads,
            heads=n_heads,
            qk_rmsnorm=qk_rmsnorm,
            num_residual_streams=num_residual_streams,
        )

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

        print(f"Initialized MMDiTWithLatentConditioning with external mmdit:")
        print(f"  Vocab size: {self.rounded_vocab_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  MMDiT depth: {n_blocks}")
        print(f"  Residual streams: {num_residual_streams}")

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
        model_dtype = self.vocab_embed.embedding.dtype
        sigma_dtype = sigma.float() if sigma.dtype != torch.float32 else sigma
        sigma_bf16 = sigma_dtype.to(dtype=model_dtype)
        c = F.silu(self.sigma_map(sigma_bf16))

        # --- Text tokens ---
        text_tokens = self.vocab_embed(indices) + self.text_pos_embed[:, :S]

        # --- Latent tokens ---
        if latents is not None:
            if latents.dim() == 2:
                latents = latents.unsqueeze(1)  # (B, 1, latent_dim)
            latent_tokens = self.latent_encoder(latents)  # (B, L, latent_hidden_size)
        else:
            latent_tokens = self.null_latent.expand(
                B, -1, -1
            )  # (B, 1, latent_hidden_size)

        # --- Build masks ---
        text_mask = attention_mask.bool() if attention_mask is not None else None
        latent_mask = torch.ones(
            B, latent_tokens.shape[1], device=indices.device, dtype=torch.bool
        )

        # --- Run through MMDiT blocks (external handles expand/reduce internally) ---
        text_tokens, latent_tokens = self.mmdit(
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
