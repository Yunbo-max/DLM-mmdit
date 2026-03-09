"""Implementation of the JointAttention for two modalities."""

# pylint: disable=not-callable

from typing import Tuple

import torch
from einops import rearrange
from torch import nn


class JointAttention(nn.Module):
    """
    Implements the Joint Attention mechanism for handling cross-modal attention
    between textual and image inputs.

    Args:
        dim_txt (int): Dimensionality of the text input features.
        dim_img (int): Dimensionality of the image input features.
        n_heads (int, optional): Number of attention heads. Defaults to 8.
        dim_head (int, optional): Dimensionality of each attention head. Defaults to 64.
        qk_rmsnorm (bool, optional): If True, applies RMS normalization on query and key. Defaults to False.
    """

    def __init__(
        self,
        dim_txt: int,
        dim_img: int,
        n_heads: int = 8,
        dim_head: int = 64,
        qk_rmsnorm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Calculate hidden size
        self.n_heads = n_heads
        self.hidden_size = self.n_heads * dim_head

        # Pre-attention linear layers for text and image
        self.pre_attn_txt_linear = nn.Linear(
            dim_txt, 3 * self.hidden_size
        )  # 3 for q,k,v
        self.pre_attn_img_linear = nn.Linear(
            dim_img, 3 * self.hidden_size
        )  # 3 for q,k,v

        # Normalization
        if qk_rmsnorm:
            self.txt_ln_q_norm = nn.RMSNorm(self.hidden_size)
            self.txt_ln_k_norm = nn.RMSNorm(self.hidden_size)

            self.img_ln_q_norm = nn.RMSNorm(self.hidden_size)
            self.img_ln_k_norm = nn.RMSNorm(self.hidden_size)
        else:
            self.txt_ln_q_norm = nn.Identity()  # type: ignore[assignment]
            self.txt_ln_k_norm = nn.Identity()  # type: ignore[assignment]

            self.img_ln_q_norm = nn.Identity()  # type: ignore[assignment]
            self.img_ln_k_norm = nn.Identity()  # type: ignore[assignment]

        # Post-attention linear layers for text and image
        self.post_attn_txt_linear = nn.Linear(self.hidden_size, dim_txt)
        self.post_attn_img_linear = nn.Linear(self.hidden_size, dim_img)

    def forward(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for JointAttention module. Computes attention between text (c) and image (x).

        Args:
            c (torch.Tensor): Text input tensor.
            x (torch.Tensor): Image input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The attended output for text and image.
        """

        # Pre attention of c
        c_pre_attn_linear = self.pre_attn_txt_linear(c)
        c_q, c_k, c_v = c_pre_attn_linear.chunk(
            3, dim=-1
        )  # Split along the last dimension (for query, key, value)
        c_opt_norm_q = self.txt_ln_q_norm(c_q)
        c_opt_norm_k = self.txt_ln_k_norm(c_k)

        # Pre attention of x
        x_pre_attn_linear = self.pre_attn_img_linear(x)
        x_q, x_k, x_v = x_pre_attn_linear.chunk(
            3, dim=-1
        )  # Split along the last dimension (for query, key, value)
        x_opt_norm_q = self.img_ln_q_norm(x_q)
        x_opt_norm_k = self.img_ln_k_norm(x_k)

        # Create qkv
        q_concat = torch.concat([c_opt_norm_q, x_opt_norm_q], dim=1)
        k_concat = torch.concat([c_opt_norm_k, x_opt_norm_k], dim=1)
        v_concat = torch.concat([c_v, x_v], dim=1)

        # Split into heads
        q = rearrange(q_concat, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k_concat, "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(v_concat, "b s (h d) -> b h s d", h=self.n_heads)

        # Attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Merge heads
        merged = rearrange(attn_output, "b h s d -> b s (h d)")

        # Split merged attention output back into text and image parts
        c_len = c.shape[1]
        x_len = x.shape[1]
        c_merged_out, x_merged_out = (
            merged[:, :c_len, :],
            merged[:, c_len : c_len + x_len, :],
        )

        # Post attention of c
        c_post_attn_linear = self.post_attn_txt_linear(c_merged_out)

        # Post attention of x
        x_post_attn_linear = self.post_attn_img_linear(x_merged_out)

        return c_post_attn_linear, x_post_attn_linear
