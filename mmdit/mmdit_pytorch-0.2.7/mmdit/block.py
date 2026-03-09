"""Core implementation of a Diffusion Transformer with two modulaities"""

from typing import Tuple

import torch
from einops import rearrange
from torch import nn

from mmdit.attention import JointAttention


class Modulation(nn.Module):
    """
    A modulation layer that generates modulating factors for text and image features.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_size (int): Dimensionality of the hidden layer.
        n_mods (int): Number of modulation factors.
    """

    def __init__(self, *, input_dim: int, hidden_size: int, n_mods: int, **kwargs):
        super().__init__(**kwargs)

        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(input_dim, n_mods * hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the modulation layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Modulated tensor.
        """
        return self.modulation(x)


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) for processing text and image features after attention.

    Args:
        input_dim (int): Dimensionality of the input features.
    """

    def __init__(self, *, input_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        return self.mlp(x)


class MMDiTBlock(nn.Module):
    """
    Implements a single block of the Multi-modal Diffusion Transformer (MMDiT), combining text and image processing.

    Args:
        dim_txt (int): Dimensionality of the text input features.
        dim_img (int): Dimensionality of the image input features.
        dim_timestep (int): Dimensionality of the timestep input.
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        qk_rmsnorm (bool, optional): If True, applies RMS normalization on query and key. Defaults to False.
    """

    def __init__(
        self,
        *,
        dim_txt: int,
        dim_img: int,
        dim_timestep: int,
        num_heads: int = 8,
        qk_rmsnorm: bool = False,
    ):
        super().__init__()

        assert (
            dim_txt + dim_img
        ) % num_heads == 0, "(dim_txt + dim_img) is not fully divisible by num_head"

        self.n_mod = 6

        # Modulation layers
        self.txt_modulation = Modulation(
            input_dim=dim_timestep, hidden_size=dim_txt, n_mods=self.n_mod
        )
        self.img_modulation = Modulation(
            input_dim=dim_timestep, hidden_size=dim_img, n_mods=self.n_mod
        )

        # Pre-attention Layers norms
        self.txt_pre_attn_layer_norm = nn.LayerNorm(dim_txt)
        self.img_pre_attn_layer_norm = nn.LayerNorm(dim_img)

        # pre-attn linear + rmsnorm and post-attn linear are part of JointAttention
        self.joint_attn = JointAttention(
            dim_txt=dim_txt, dim_img=dim_img, qk_rmsnorm=qk_rmsnorm
        )

        # Post-attention Layers norms
        self.txt_post_attn_layer_norm = nn.LayerNorm(dim_txt)
        self.img_post_attn_layer_norm = nn.LayerNorm(dim_img)

        # MLP layers
        self.txt_mlp = MLP(input_dim=dim_txt)
        self.img_mlp = MLP(input_dim=dim_img)

    def forward(
        self, c: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MMDiT block.

        Args:
            c (torch.Tensor): Text input tensor.
            x (torch.Tensor): Image input tensor.
            y (torch.Tensor): Timestep or context tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The processed outputs for text and image.
        """

        c_mod = self.txt_modulation(y)
        x_mod = self.img_modulation(y)

        alpha_c, beta_c, gamma_c, delta_c, epsilon_c, zeta_c = rearrange(
            c_mod, "b (n d) -> b n d", n=self.n_mod
        ).chunk(self.n_mod, dim=1)
        alpha_x, beta_x, gamma_x, delta_x, epsilon_x, zeta_x = rearrange(
            x_mod, "b (n d) -> b n d", n=self.n_mod
        ).chunk(self.n_mod, dim=1)

        # Pre-attention of c
        c_layer_norm = self.txt_pre_attn_layer_norm(c)
        c_mod_out = alpha_c * c_layer_norm + beta_c

        # Pre-attention of x
        x_layer_norm = self.img_pre_attn_layer_norm(x)
        x_mod_out = alpha_x * x_layer_norm + beta_x

        # Attention
        c_attn_out, x_attn_out = self.joint_attn(c_mod_out, x_mod_out)

        # Post attention of c
        c_gamma_out = gamma_c * c_attn_out
        c_residual_out = c + c_gamma_out

        c_residual_layer_norm = self.txt_post_attn_layer_norm(c_residual_out)
        c_residual_mod_out = delta_c * c_residual_layer_norm + epsilon_c

        c_mlp_out = self.txt_mlp(c_residual_mod_out)
        c_zeta_out = zeta_c * c_mlp_out

        c_out = c + c_zeta_out

        # Post attention of x
        x_gamma_out = gamma_x * x_attn_out
        x_residual_out = x + x_gamma_out

        x_residual_layer_norm = self.img_post_attn_layer_norm(x_residual_out)
        x_residual_mod_out = delta_x * x_residual_layer_norm + epsilon_x

        x_mlp_out = self.img_mlp(x_residual_mod_out)
        x_zeta_out = zeta_x * x_mlp_out

        x_out = x + x_zeta_out

        return c_out, x_out
