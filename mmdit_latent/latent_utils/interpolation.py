"""
Latent interpolation utilities for LSME.

Provides spherical and linear interpolation between latent vectors
for latent geometry analysis and attribute steering.
"""

import torch
import torch.nn.functional as F


def slerp(z_a, z_b, alpha):
    """
    Spherical linear interpolation between two latent vectors.

    Args:
        z_a: (..., D) starting latent
        z_b: (..., D) ending latent
        alpha: float or Tensor, interpolation factor in [0, 1]

    Returns:
        z_interp: (..., D) interpolated latent
    """
    z_a_norm = F.normalize(z_a, dim=-1)
    z_b_norm = F.normalize(z_b, dim=-1)

    # Cosine of angle between vectors
    cos_omega = (z_a_norm * z_b_norm).sum(-1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(cos_omega)

    # Fall back to lerp for nearly-parallel vectors
    sin_omega = torch.sin(omega)
    is_small = (sin_omega.abs() < 1e-6)

    # SLERP formula
    scale_a = torch.sin((1 - alpha) * omega) / (sin_omega + 1e-10)
    scale_b = torch.sin(alpha * omega) / (sin_omega + 1e-10)

    # Preserve original magnitudes
    mag_a = z_a.norm(dim=-1, keepdim=True)
    mag_b = z_b.norm(dim=-1, keepdim=True)
    target_mag = (1 - alpha) * mag_a + alpha * mag_b

    z_interp = scale_a * z_a_norm + scale_b * z_b_norm
    z_interp = F.normalize(z_interp, dim=-1) * target_mag

    # Use lerp where slerp is numerically unstable
    if is_small.any():
        z_lerp = lerp(z_a, z_b, alpha)
        z_interp = torch.where(is_small, z_lerp, z_interp)

    return z_interp


def lerp(z_a, z_b, alpha):
    """
    Linear interpolation between two latent vectors.

    Args:
        z_a: (..., D) starting latent
        z_b: (..., D) ending latent
        alpha: float or Tensor, interpolation factor in [0, 1]

    Returns:
        z_interp: (..., D) interpolated latent
    """
    return (1 - alpha) * z_a + alpha * z_b


def interpolation_path(z_a, z_b, n_points=10, method="slerp"):
    """
    Generate a sequence of interpolated latent vectors.

    Args:
        z_a: (D,) or (1, D) starting latent
        z_b: (D,) or (1, D) ending latent
        n_points: int, number of interpolation points
        method: "slerp" or "lerp"

    Returns:
        path: (n_points, D) interpolated latents
        alphas: (n_points,) interpolation factors
    """
    interp_fn = slerp if method == "slerp" else lerp
    alphas = torch.linspace(0, 1, n_points)

    if z_a.dim() == 1:
        z_a = z_a.unsqueeze(0)
    if z_b.dim() == 1:
        z_b = z_b.unsqueeze(0)

    path = torch.stack([interp_fn(z_a, z_b, a.item()).squeeze(0) for a in alphas])
    return path, alphas


def directional_edit(z_source, z_pos, z_neg, alpha=1.0):
    """
    Directional latent editing: z_source + alpha * (z_pos - z_neg).

    Preserves source content better than hard centroid replacement.

    Args:
        z_source: (..., D) source latent
        z_pos: (..., D) positive attribute centroid
        z_neg: (..., D) negative attribute centroid
        alpha: float, edit strength

    Returns:
        z_edited: (..., D) edited latent
    """
    direction = z_pos - z_neg
    return z_source + alpha * direction
