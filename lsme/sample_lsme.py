"""
LSME Sampler — Latent-Steered Masked Editing for discrete masked diffusion LMs.

Solves gap: Gap 4 — no DLM supports SDEdit-style editing with continuous latent steering.
Inspired by: SDEdit (image editing) + ReMDM (remasking) + LatentOps (latent manipulation)

Zero architecture changes — uses existing MMDiTWithLatentConditioning as-is.
~100 lines of core editing logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm

from mmdit_latent.utils import sample_categorical


class LSMESampler(nn.Module):
    """
    SDEdit-style text editing for discrete masked DLMs.

    Given input text and a target latent vector, partially masks text tokens
    and joint-denoises them conditioned on the target latent.

    Input:
        text_tokens:   (B, L) int64 — original text token ids
        target_latent: (B, D) float32 — target latent for desired attribute
        mask_ratio:    float in [0, 1] — edit strength
        steps:         int — reverse diffusion steps from entry point

    Output:
        edited_tokens: (B, L) int64 — edited text token ids
        edit_mask:     (B, L) bool — which positions were masked/edited
    """

    def __init__(self, model, tokenizer, noise_schedule, t_eps=1e-4):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.t_eps = t_eps
        self.mask_id = tokenizer.mask_token_id

    def _get_sigmas(self, t, eps=1e-4):
        """Compute MDLM noise schedule sigmas (matches MDLMSampler)."""
        # dsigma: derivative of sigma w.r.t. t
        # sigma: cumulative noise level
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clamp(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clamp(eps, 1))
        return dsigma, sigma

    def _create_mask(self, text_tokens, mask_ratio, mask_mode="random"):
        """
        Create a binary mask indicating which positions to edit.

        Args:
            text_tokens: (B, L) token ids
            mask_ratio: float, fraction of tokens to mask
            mask_mode: "random", "entropy", or "suffix"

        Returns:
            mask: (B, L) bool — True where tokens should be masked
        """
        B, L = text_tokens.shape
        device = text_tokens.device

        if mask_mode == "random":
            # Each position masked independently with probability mask_ratio
            mask = torch.rand(B, L, device=device) < mask_ratio

        elif mask_mode == "entropy":
            # Mask positions where model is most uncertain
            # Run a forward pass at t=0 (clean) to get logits, then pick
            # high-entropy positions
            with torch.no_grad():
                sigma_zero = torch.full((B,), self.t_eps, device=device)
                logits = self.model(text_tokens, sigma_zero)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = logits.softmax(-1)  # (B, L, V)
                entropy = -(probs * (probs + 1e-10).log()).sum(-1)  # (B, L)
            num_to_mask = max(1, int(mask_ratio * L))
            # Top-k highest entropy positions per sample
            _, topk_indices = entropy.topk(num_to_mask, dim=-1)
            mask = torch.zeros(B, L, device=device, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)

        elif mask_mode == "suffix":
            # Mask the last mask_ratio * L positions
            num_to_mask = max(1, int(mask_ratio * L))
            mask = torch.zeros(B, L, device=device, dtype=torch.bool)
            mask[:, L - num_to_mask:] = True

        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

        return mask

    @torch.no_grad()
    def edit(
        self,
        text_tokens,
        target_latent,
        mask_ratio=0.3,
        steps=100,
        temperature=1.0,
        mask_mode="random",
        attention_mask=None,
        show_progress=True,
    ):
        """
        LSME-Edit: SDEdit-style text editing via partial masking + latent steering.

        Args:
            text_tokens:    (B, L) original text token ids
            target_latent:  (B, D) target latent encoding desired attribute
            mask_ratio:     float in [0, 1], controls edit strength
            steps:          int, reverse diffusion steps from entry point
            temperature:    float, sampling temperature (1.0 = standard)
            mask_mode:      str, "random" / "entropy" / "suffix"
            attention_mask: (B, L) optional, 1=valid 0=pad
            show_progress:  bool, show tqdm progress bar

        Returns:
            edited_tokens: (B, L) int64
            edit_mask:     (B, L) bool
        """
        B, L = text_tokens.shape
        device = text_tokens.device

        # --- 1. PARTIAL MASKING ---
        edit_mask = self._create_mask(text_tokens, mask_ratio, mask_mode)
        z_t = text_tokens.clone()
        z_t[edit_mask] = self.mask_id

        # --- 2. COMPUTE ENTRY TIMESTEP ---
        # Map mask_ratio to the diffusion schedule
        t_entry = mask_ratio
        eps = self.t_eps

        # Partial schedule: from t_entry down to eps
        ts = torch.linspace(eps, t_entry, steps + 1, device=device)  # (steps+1,)

        # Ensure target_latent has right shape: (B, 1, D) for model
        if target_latent.dim() == 2:
            target_latent = target_latent.unsqueeze(1)  # (B, 1, D)

        # --- 3. REVERSE DIFFUSION from t_entry → 0 ---
        for i in tqdm.trange(steps - 1, -1, -1, desc="LSME editing",
                             disable=not show_progress, dynamic_ncols=True):
            t = ts[i + 1].unsqueeze(0).expand(B)   # (B,)
            tm1 = ts[i].unsqueeze(0).expand(B)      # (B,)

            # Forward pass: text tokens + TARGET latent (clean conditioning)
            logits = self.model(z_t, t, latents=target_latent,
                                attention_mask=attention_mask)
            if isinstance(logits, tuple):
                logits = logits[0]  # (B, L, V) — text logits only

            # Block mask token from being sampled
            logits[..., self.mask_id] = -1e6

            if i == 0:
                # Last step: argmax for deterministic output
                z_tm1 = logits.argmax(-1)  # (B, L)
            else:
                # MDLM posterior sampling (matches MDLMSampler logic)
                _, sigma_t = self._get_sigmas(t, eps=eps)
                _, sigma_tm1 = self._get_sigmas(tm1, eps=eps)

                move_chance_t = 1 - torch.exp(-sigma_t)      # (B,)
                move_chance_tm1 = 1 - torch.exp(-sigma_tm1)  # (B,)
                move_chance_t = move_chance_t[:, None, None]      # (B, 1, 1)
                move_chance_tm1 = move_chance_tm1[:, None, None]  # (B, 1, 1)

                probs = logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                probs[:, :, self.mask_id] = move_chance_tm1[:, :, 0]
                probs = probs / move_chance_t

                # Apply temperature
                if temperature != 1.0:
                    log_probs = (probs + 1e-10).log() / temperature
                    probs = log_probs.softmax(-1)

                z_tm1 = sample_categorical(probs)  # (B, L)

            # Copy flag: preserve positions that were NOT masked
            copy_flag = (z_t != self.mask_id).to(z_t.dtype)
            z_t = (copy_flag * z_t + (1 - copy_flag) * z_tm1).long()

        return z_t, edit_mask

    @torch.no_grad()
    def edit_from_text(
        self,
        texts,
        target_latent,
        mask_ratio=0.3,
        steps=100,
        temperature=1.0,
        mask_mode="random",
        max_length=512,
        show_progress=True,
        decode=True,
    ):
        """
        Convenience method: tokenize text strings, edit, optionally decode.

        Args:
            texts: list of str, input texts to edit
            target_latent: (B, D) target latent
            mask_ratio, steps, temperature, mask_mode: see edit()
            max_length: int, max token length
            decode: bool, if True return decoded strings

        Returns:
            If decode: (edited_texts: list[str], edit_mask: (B, L) bool)
            Else: (edited_tokens: (B, L) int64, edit_mask: (B, L) bool)
        """
        device = next(self.model.parameters()).device

        # Tokenize
        encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        text_tokens = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        edited_tokens, edit_mask = self.edit(
            text_tokens, target_latent,
            mask_ratio=mask_ratio, steps=steps,
            temperature=temperature, mask_mode=mask_mode,
            attention_mask=attention_mask,
            show_progress=show_progress,
        )

        if decode:
            edited_texts = self.tokenizer.batch_decode(
                edited_tokens, skip_special_tokens=True
            )
            return edited_texts, edit_mask
        return edited_tokens, edit_mask
