# File: latentDLM_mmdit/sample_l2t_fixed.py
"""
L2T Sampling with Loss Logging

Latent-to-Text generation using reverse diffusion with loss tracking.
Similar to training loop but in inference mode.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import sys
import os
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from latentDLM_mmdit.models.multimodal_mmdit import MultimodalMMDiT
from latentDLM_mmdit.modeling_mmdit import get_tokenizer
from latentDLM_mmdit.diffusion_process import MaskedDiffusion, sample_t
from latentDLM_mmdit.continuous_diffusion import ContinuousDiffusion
from latentDLM_mmdit.utils import sample_categorical


class ReverseDiffusionSampler:
    def __init__(self, checkpoint_path, config_path=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        print(f"Loading checkpoint from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        elif "config" in checkpoint and checkpoint["config"] is not None:
            config = checkpoint["config"]
            print("Using config from checkpoint")
        else:
            raise ValueError("No config provided or found in checkpoint")

        # Get tokenizer
        self.tokenizer = get_tokenizer(config)
        self.mask_token_id = self.tokenizer.mask_token_id
        print(f"✓ Tokenizer loaded: vocab_size={len(self.tokenizer)}")

        # Determine vocab size from checkpoint
        vocab_size = None
        if "model_state_dict" in checkpoint:
            for key in checkpoint["model_state_dict"]:
                if "text_head.weight" in key:
                    vocab_size = checkpoint["model_state_dict"][key].shape[0]
                    print(f"Found vocab_size={vocab_size} in checkpoint")
                    break

        if vocab_size is None:
            vocab_size = len(self.tokenizer)
            print(f"Using tokenizer vocab_size={vocab_size}")

        # Create noise schedules first
        self.text_noise_schedule = MaskedDiffusion(self.tokenizer)

        # Create model with correct architecture
        latent_dim = config["model"].get("latent_dim", 32)
        self.model = MultimodalMMDiT(
            config=config["model"],
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            cluster_size=0,
        ).to(device)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

            # Strip prefix (model., module., etc.) and separate trainer weights
            model_state_dict = {}
            trainer_state_dict = {}

            for k, v in state_dict.items():
                # Remove common prefixes
                if k.startswith("model."):
                    k = k[6:]
                elif k.startswith("module."):
                    k = k[7:]
                elif k.startswith("trainer.model."):
                    k = k[14:]
                elif k.startswith("_orig_mod."):
                    k = k[10:]

                # Check if this belongs to text_noise_schedule
                if k.startswith("text_noise_schedule."):
                    trainer_state_dict[k] = v
                else:
                    model_state_dict[k] = v

            # Load text_noise_schedule if we have its parameters
            if "text_noise_schedule.log_prior" in trainer_state_dict:
                self.text_noise_schedule.log_prior = trainer_state_dict[
                    "text_noise_schedule.log_prior"
                ]
                self.text_noise_schedule.log_gamma = trainer_state_dict[
                    "text_noise_schedule.log_gamma"
                ]
                self.text_noise_schedule.log_B = trainer_state_dict[
                    "text_noise_schedule.log_B"
                ]
                print("✓ Loaded text_noise_schedule from checkpoint")

            # Save weight keys comparison to file for verification
            keys_file = os.path.join(os.path.dirname(checkpoint_path), "weight_keys_comparison.txt")
            with open(keys_file, "w") as kf:
                kf.write("CHECKPOINT STATE_DICT KEYS (after prefix stripping):\n")
                kf.write("=" * 100 + "\n")
                for k, v in sorted(model_state_dict.items()):
                    kf.write(f"  {k:60s} {str(v.shape):>20s} {v.dtype}\n")
                kf.write(f"\nTotal checkpoint keys: {len(model_state_dict)}\n")
                kf.write("\n\nMODEL EXPECTED KEYS:\n")
                kf.write("=" * 100 + "\n")
                for k, v in sorted(self.model.state_dict().items()):
                    kf.write(f"  {k:60s} {str(v.shape):>20s} {v.dtype}\n")
                kf.write(f"\nTotal model keys: {len(self.model.state_dict())}\n")

                # Show mismatches
                ckpt_keys = set(model_state_dict.keys())
                model_keys = set(self.model.state_dict().keys())
                missing_in_ckpt = model_keys - ckpt_keys
                unexpected_in_ckpt = ckpt_keys - model_keys
                kf.write("\n\nMISSING IN CHECKPOINT (model needs but checkpoint doesn't have):\n")
                kf.write("=" * 100 + "\n")
                for k in sorted(missing_in_ckpt):
                    kf.write(f"  {k}\n")
                kf.write(f"\nTotal missing: {len(missing_in_ckpt)}\n")
                kf.write("\n\nUNEXPECTED IN CHECKPOINT (checkpoint has but model doesn't need):\n")
                kf.write("=" * 100 + "\n")
                for k in sorted(unexpected_in_ckpt):
                    kf.write(f"  {k}\n")
                kf.write(f"\nTotal unexpected: {len(unexpected_in_ckpt)}\n")

                # Check shape mismatches
                kf.write("\n\nSHAPE MISMATCHES:\n")
                kf.write("=" * 100 + "\n")
                shape_mismatches = 0
                for k in sorted(ckpt_keys & model_keys):
                    ckpt_shape = model_state_dict[k].shape
                    model_shape = self.model.state_dict()[k].shape
                    if ckpt_shape != model_shape:
                        kf.write(f"  {k:60s} ckpt={str(ckpt_shape):>20s} model={str(model_shape):>20s}\n")
                        shape_mismatches += 1
                if shape_mismatches == 0:
                    kf.write("  (none)\n")
                kf.write(f"\nTotal shape mismatches: {shape_mismatches}\n")

            print(f"Saved weight keys comparison to: {keys_file}")

            # Load model weights directly (LatentEncoder now matches checkpoint)
            missing, unexpected = self.model.load_state_dict(
                model_state_dict, strict=False
            )
            print(
                f"\nModel loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )

            if len(missing) > 0:
                print(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                print(f"Unexpected keys: {unexpected}")
        self.model.eval()

        # Create continuous diffusion for latents
        self.latent_diffusion = ContinuousDiffusion(
            num_timesteps=config["model"].get("latent_timesteps", 1000),
            beta_schedule=config["model"].get("latent_beta_schedule", "cosine"),
            parameterization=config["model"].get("latent_parameterization", "epsilon"),
            device=device,
        )

        print(f"✓ Sampler initialized")

    def _get_sigmas(self, t, eps=1e-4):
        """Compute dsigma and sigma for the MDLM noise schedule."""
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
        return dsigma, sigma

    @torch.no_grad()
    def sample_l2t(self, latents, seq_len=128, steps=1000, temperature=1.0,
                   target_tokens=None, target_attention_mask=None):
        """
        Correct MDLM reverse diffusion sampling for L2T mode.
        Uses conditional transition p(x_{t-1} | x_t, x_0_pred) with copy-flag
        to ensure monotonic unmasking.

        Args:
            target_tokens: Optional [batch, seq_len] ground truth token ids
                           for computing cross-entropy loss (same as training).
            target_attention_mask: Optional [batch, seq_len] attention mask from
                           the target text tokenization. 1=real token, 0=padding.
                           Matches training where padding positions have mask=0.
        """
        batch_size = latents.shape[0]
        device = self.device
        eps = 1e-4

        # Ensure latents are correct shape [batch, latent_dim]
        if latents.dim() == 1:
            latents = latents.unsqueeze(0)
        latents = latents.to(device)

        if target_tokens is not None:
            target_tokens = target_tokens.to(device)

        # Attention mask: use target's mask if provided (matches training),
        # otherwise all-ones (all positions are real)
        if target_attention_mask is not None:
            attention_mask = target_attention_mask.bool().to(device)
        else:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=device
            )

        # Initialize z_t: [MASK] where attention_mask=1, [PAD] where attention_mask=0
        # This matches training where padding positions stay as PAD tokens
        z_t = torch.full(
            (batch_size, seq_len), self.tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        z_t[attention_mask] = self.mask_token_id

        # Latent conditioning (clean, t=0)
        latent_t = torch.zeros(batch_size, device=device)

        # Timestep schedule: evenly spaced from t_eps to 1-t_eps, iterated in reverse
        ts = torch.linspace(eps, 1 - eps, steps + 1, device=device).unsqueeze(-1)

        pbar = tqdm(range(steps - 1, -1, -1), desc="Reverse diffusion")
        for i in pbar:
            t = ts[i]
            tm1 = ts[max(0, i - 1)]

            t_vec = t.expand(batch_size)
            tm1_vec = tm1.expand(batch_size)

            # Model predicts logits for x₀
            text_logits, _ = self.model(
                text_tokens=z_t,
                latents=latents.unsqueeze(1),
                text_timesteps=t_vec,
                latent_timesteps=latent_t,
                attention_mask=attention_mask,
            )

            # Apply temperature
            if temperature != 1.0:
                text_logits = text_logits / temperature

            # Compute loss on masked positions (same as training)
            text_loss = 0.0
            text_accuracy = 0.0
            current_mask = (z_t == self.mask_token_id)
            if current_mask.any():
                if target_tokens is not None:
                    # Cross-entropy vs ground truth (same as training)
                    vocab_size = text_logits.shape[-1]
                    text_loss_unmasked = F.cross_entropy(
                        text_logits.view(-1, vocab_size),
                        target_tokens.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(batch_size, seq_len)
                    mask_sum = current_mask.sum().float().clamp(min=1)
                    text_loss = (text_loss_unmasked * current_mask).sum() / mask_sum
                    text_loss = text_loss.item()

                    # Accuracy on masked positions (same as training)
                    pred_tokens = torch.argmax(text_logits, dim=-1)
                    text_accuracy = (
                        (pred_tokens[current_mask] == target_tokens[current_mask])
                        .float()
                        .mean()
                        .item()
                    )

            # Suppress mask token in model predictions
            text_logits[..., self.mask_token_id] = -1e6

            if i == 0:
                # Final step: take argmax
                z_tm1 = text_logits.argmax(-1)
            else:
                # Compute conditional transition probabilities
                _, sigma_t = self._get_sigmas(t_vec, eps=eps)
                _, sigma_tm1 = self._get_sigmas(tm1_vec, eps=eps)

                move_chance_t = 1 - torch.exp(-sigma_t)
                move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
                move_chance_t = move_chance_t[:, None, None]
                move_chance_tm1 = move_chance_tm1[:, None, None]

                probs = text_logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                probs[:, :, self.mask_token_id] = move_chance_tm1[:, :, 0]
                probs = probs / move_chance_t

                z_tm1 = sample_categorical(probs)

            # Copy flag: keep already-unmasked tokens unchanged
            copy_flag = (z_t != self.mask_token_id).to(z_t.dtype)
            z_t = copy_flag * z_t + (1 - copy_flag) * z_tm1

            # Show progress (same format as training) — update every step
            mask_ratio = (z_t == self.mask_token_id).float().mean().item()
            pbar.set_postfix({
                "t": f"{t.item():.3f}",
                "Loss": f"{text_loss:.4f}",
                "Acc": f"{text_accuracy:.4f}",
                "mask%": f"{mask_ratio * 100:.1f}%",
            })

            # Print detailed log every 10% of steps
            if i % max(1, steps // 10) == 0:
                print(f"  Step {steps - i:4d}/{steps} | t={t.item():.4f} | "
                      f"Loss={text_loss:.4f} | Acc={text_accuracy:.4f} | "
                      f"mask%={mask_ratio * 100:.1f}%")

        return z_t

    @torch.no_grad()
    def sample_l2t_ddim(
        self, latents, seq_len=128, steps=100, temperature=1.0, eta=0.0,
        target_tokens=None, target_attention_mask=None
    ):
        """
        DDIM-style sampling for faster generation with copy-flag.
        eta=0: deterministic, eta=1: stochastic (like DDPM).
        Only masked positions are updated; already-unmasked tokens are preserved.

        Args:
            target_tokens: Optional [batch, seq_len] ground truth token ids
                           for computing cross-entropy loss (same as training).
            target_attention_mask: Optional [batch, seq_len] attention mask from
                           the target text tokenization. Matches training.
        """
        batch_size = latents.shape[0]
        device = self.device
        eps = 1e-4

        # Ensure latents are correct shape
        if latents.dim() == 1:
            latents = latents.unsqueeze(0)
        latents = latents.to(device)

        if target_tokens is not None:
            target_tokens = target_tokens.to(device)

        # Attention mask: use target's mask if provided (matches training)
        if target_attention_mask is not None:
            attention_mask = target_attention_mask.bool().to(device)
        else:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=device
            )

        # Initialize z_t: [MASK] where attention_mask=1, [PAD] where attention_mask=0
        z_t = torch.full(
            (batch_size, seq_len), self.tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        z_t[attention_mask] = self.mask_token_id
        latent_t = torch.zeros(batch_size, device=device)

        # Timestep schedule: evenly spaced from t_eps to 1-t_eps, iterated in reverse
        ts = torch.linspace(eps, 1 - eps, steps + 1, device=device).unsqueeze(-1)

        pbar = tqdm(range(steps - 1, -1, -1), desc="DDIM sampling")
        for i in pbar:
            t = ts[i]
            tm1 = ts[max(0, i - 1)]

            t_vec = t.expand(batch_size)
            tm1_vec = tm1.expand(batch_size)

            # Model predicts logits for x₀
            text_logits, _ = self.model(
                text_tokens=z_t,
                latents=latents.unsqueeze(1),
                text_timesteps=t_vec,
                latent_timesteps=latent_t,
                attention_mask=attention_mask,
            )

            if temperature != 1.0:
                text_logits = text_logits / temperature

            # Compute loss on masked positions (same as training)
            text_loss = 0.0
            text_accuracy = 0.0
            current_mask = (z_t == self.mask_token_id)
            if current_mask.any() and target_tokens is not None:
                vocab_size = text_logits.shape[-1]
                text_loss_unmasked = F.cross_entropy(
                    text_logits.view(-1, vocab_size),
                    target_tokens.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(batch_size, seq_len)
                mask_sum = current_mask.sum().float().clamp(min=1)
                text_loss = (text_loss_unmasked * current_mask).sum() / mask_sum
                text_loss = text_loss.item()

                pred_tokens = torch.argmax(text_logits, dim=-1)
                text_accuracy = (
                    (pred_tokens[current_mask] == target_tokens[current_mask])
                    .float()
                    .mean()
                    .item()
                )

            # Suppress mask token in model predictions
            text_logits[..., self.mask_token_id] = -1e6

            if i == 0:
                # Final step: take argmax
                z_tm1 = text_logits.argmax(-1)
            else:
                # Compute conditional transition probabilities
                _, sigma_t = self._get_sigmas(t_vec, eps=eps)
                _, sigma_tm1 = self._get_sigmas(tm1_vec, eps=eps)

                move_chance_t = 1 - torch.exp(-sigma_t)
                move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
                move_chance_t = move_chance_t[:, None, None]
                move_chance_tm1 = move_chance_tm1[:, None, None]

                probs = text_logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                probs[:, :, self.mask_token_id] = move_chance_tm1[:, :, 0]
                probs = probs / move_chance_t

                if eta > 0:
                    z_tm1 = sample_categorical(probs)
                else:
                    z_tm1 = probs.argmax(-1)

            # Copy flag: keep already-unmasked tokens unchanged
            copy_flag = (z_t != self.mask_token_id).to(z_t.dtype)
            z_t = copy_flag * z_t + (1 - copy_flag) * z_tm1

            # Show progress (same format as training)
            mask_ratio = (z_t == self.mask_token_id).float().mean().item()
            pbar.set_postfix({
                "t": f"{t.item():.3f}",
                "Loss": f"{text_loss:.4f}",
                "Acc": f"{text_accuracy:.4f}",
                "mask%": f"{mask_ratio * 100:.1f}%",
            })

        return z_t

    @torch.no_grad()
    def sample_l2t_with_loss_logging(
        self, latents, target_texts=None, seq_len=128, steps=1000, temperature=1.0
    ):
        """
        Correct MDLM reverse diffusion with loss logging.

        Args:
            latents: Latent embeddings [batch, latent_dim]
            target_texts: Optional ground truth texts for loss computation
            seq_len: Maximum sequence length
            steps: Number of reverse diffusion steps
            temperature: Sampling temperature

        Returns:
            tokens: Generated token sequences
            losses: List of text losses per step (if target_texts provided)
        """
        batch_size = latents.shape[0]
        device = self.device
        eps = 1e-4

        # Ensure latents are correct shape [batch, latent_dim]
        if latents.dim() == 1:
            latents = latents.unsqueeze(0)
        latents = latents.to(device)

        # Encode target texts if provided
        target_tokens = None
        attention_mask = None
        if target_texts is not None:
            tokenized = self.tokenizer(
                target_texts,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )
            target_tokens = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].bool().to(device)
            print(f"Encoded {len(target_texts)} target texts for loss computation")

        # Fallback to all-ones if no target texts
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=device
            )

        # Initialize z_t: [MASK] where attention_mask=1, [PAD] where attention_mask=0
        z_t = torch.full(
            (batch_size, seq_len), self.tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        z_t[attention_mask] = self.mask_token_id

        # Latent conditioning (clean, t=0)
        latent_t = torch.zeros(batch_size, device=device)

        # Timestep schedule: evenly spaced from t_eps to 1-t_eps, iterated in reverse
        ts = torch.linspace(eps, 1 - eps, steps + 1, device=device).unsqueeze(-1)

        # Loss tracking
        losses = []
        mask_ratios = []
        text_accuracies = []

        pbar = tqdm(range(steps - 1, -1, -1), desc="Reverse diffusion (with loss logging)")
        for i in pbar:
            t = ts[i]
            tm1 = ts[max(0, i - 1)]

            t_vec = t.expand(batch_size)
            tm1_vec = tm1.expand(batch_size)

            # Model predicts logits for x₀
            text_logits, _ = self.model(
                text_tokens=z_t,
                latents=latents.unsqueeze(1),
                text_timesteps=t_vec,
                latent_timesteps=latent_t,
                attention_mask=attention_mask,
            )

            # Apply temperature
            if temperature != 1.0:
                text_logits = text_logits / temperature

            # Compute text loss if target texts provided (before suppressing mask token)
            text_loss = None
            text_accuracy = 0.0
            if target_tokens is not None:
                vocab_size = text_logits.shape[-1]
                current_mask = z_t == self.mask_token_id

                if current_mask.any():
                    text_loss_unmasked = F.cross_entropy(
                        text_logits.view(-1, vocab_size),
                        target_tokens.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(batch_size, seq_len)

                    mask_sum = current_mask.sum().float().clamp(min=1)
                    text_loss = (text_loss_unmasked * current_mask).sum() / mask_sum

                    pred_tokens = torch.argmax(text_logits, dim=-1)
                    text_accuracy = (
                        (pred_tokens[current_mask] == target_tokens[current_mask])
                        .float()
                        .mean()
                        .item()
                    )

                losses.append(text_loss.item() if text_loss is not None else 0.0)
                text_accuracies.append(text_accuracy)

            # Suppress mask token in model predictions
            text_logits[..., self.mask_token_id] = -1e6

            if i == 0:
                # Final step: take argmax
                z_tm1 = text_logits.argmax(-1)
            else:
                # Compute conditional transition probabilities
                _, sigma_t = self._get_sigmas(t_vec, eps=eps)
                _, sigma_tm1 = self._get_sigmas(tm1_vec, eps=eps)

                move_chance_t = 1 - torch.exp(-sigma_t)
                move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
                move_chance_t = move_chance_t[:, None, None]
                move_chance_tm1 = move_chance_tm1[:, None, None]

                probs = text_logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                probs[:, :, self.mask_token_id] = move_chance_tm1[:, :, 0]
                probs = probs / move_chance_t

                z_tm1 = sample_categorical(probs)

            # Copy flag: keep already-unmasked tokens unchanged
            copy_flag = (z_t != self.mask_token_id).to(z_t.dtype)
            z_t = copy_flag * z_t + (1 - copy_flag) * z_tm1

            # Track mask ratio
            mask_ratio = (z_t == self.mask_token_id).float().mean().item()
            mask_ratios.append(mask_ratio)

            # Log progress
            if i % max(1, steps // 10) == 0:
                pbar.set_postfix(
                    {
                        "t": f"{t.item():.3f}",
                        "mask%": f"{mask_ratio * 100:.1f}%",
                        "loss": f"{text_loss.item() if text_loss is not None else 0.0:.4f}",
                        "acc": f"{text_accuracy:.4f}",
                    }
                )

        return z_t, {
            "losses": losses,
            "mask_ratios": mask_ratios,
            "text_accuracies": text_accuracies,
        }

    @torch.no_grad()
    def sample_l2t_ddim_with_loss_logging(
        self,
        latents,
        target_texts=None,
        seq_len=128,
        steps=100,
        temperature=1.0,
        eta=0.0,
    ):
        """
        DDIM-style sampling with loss logging and copy-flag.

        Args:
            latents: Latent embeddings [batch, latent_dim]
            target_texts: Optional ground truth texts for loss computation
            seq_len: Maximum sequence length
            steps: Number of DDIM steps
            temperature: Sampling temperature
            eta: DDIM eta parameter (0=deterministic, 1=stochastic)

        Returns:
            tokens: Generated token sequences
            losses: List of text losses per step (if target_texts provided)
        """
        batch_size = latents.shape[0]
        device = self.device
        eps = 1e-4

        # Ensure latents are correct shape
        if latents.dim() == 1:
            latents = latents.unsqueeze(0)
        latents = latents.to(device)

        # Encode target texts if provided
        target_tokens = None
        attention_mask = None
        if target_texts is not None:
            tokenized = self.tokenizer(
                target_texts,
                padding="max_length",
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )
            target_tokens = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].bool().to(device)
            print(f"Encoded {len(target_texts)} target texts for loss computation")

        # Fallback to all-ones if no target texts
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=device
            )

        # Initialize z_t: [MASK] where attention_mask=1, [PAD] where attention_mask=0
        z_t = torch.full(
            (batch_size, seq_len), self.tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        z_t[attention_mask] = self.mask_token_id
        latent_t = torch.zeros(batch_size, device=device)

        # Timestep schedule: evenly spaced from t_eps to 1-t_eps, iterated in reverse
        ts = torch.linspace(eps, 1 - eps, steps + 1, device=device).unsqueeze(-1)

        losses = []
        mask_ratios = []
        text_accuracies = []

        pbar = tqdm(range(steps - 1, -1, -1), desc="DDIM sampling (with loss logging)")
        for i in pbar:
            t = ts[i]
            tm1 = ts[max(0, i - 1)]

            t_vec = t.expand(batch_size)
            tm1_vec = tm1.expand(batch_size)

            # Model predicts logits for x₀
            text_logits, _ = self.model(
                text_tokens=z_t,
                latents=latents.unsqueeze(1),
                text_timesteps=t_vec,
                latent_timesteps=latent_t,
                attention_mask=attention_mask,
            )

            if temperature != 1.0:
                text_logits = text_logits / temperature

            # Compute text loss if target texts provided (before suppressing mask token)
            text_loss = None
            text_accuracy = 0.0
            if target_tokens is not None:
                vocab_size = text_logits.shape[-1]
                current_mask = z_t == self.mask_token_id

                if current_mask.any():
                    text_loss_unmasked = F.cross_entropy(
                        text_logits.view(-1, vocab_size),
                        target_tokens.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(batch_size, seq_len)

                    mask_sum = current_mask.sum().float().clamp(min=1)
                    text_loss = (text_loss_unmasked * current_mask).sum() / mask_sum

                    pred_tokens = torch.argmax(text_logits, dim=-1)
                    text_accuracy = (
                        (pred_tokens[current_mask] == target_tokens[current_mask])
                        .float()
                        .mean()
                        .item()
                    )

                losses.append(text_loss.item() if text_loss is not None else 0.0)
                text_accuracies.append(text_accuracy)

            # Suppress mask token in model predictions
            text_logits[..., self.mask_token_id] = -1e6

            if i == 0:
                # Final step: take argmax
                z_tm1 = text_logits.argmax(-1)
            else:
                # Compute conditional transition probabilities
                _, sigma_t = self._get_sigmas(t_vec, eps=eps)
                _, sigma_tm1 = self._get_sigmas(tm1_vec, eps=eps)

                move_chance_t = 1 - torch.exp(-sigma_t)
                move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
                move_chance_t = move_chance_t[:, None, None]
                move_chance_tm1 = move_chance_tm1[:, None, None]

                probs = text_logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                probs[:, :, self.mask_token_id] = move_chance_tm1[:, :, 0]
                probs = probs / move_chance_t

                if eta > 0:
                    z_tm1 = sample_categorical(probs)
                else:
                    z_tm1 = probs.argmax(-1)

            # Copy flag: keep already-unmasked tokens unchanged
            copy_flag = (z_t != self.mask_token_id).to(z_t.dtype)
            z_t = copy_flag * z_t + (1 - copy_flag) * z_tm1

            mask_ratio = (z_t == self.mask_token_id).float().mean().item()
            mask_ratios.append(mask_ratio)

            if i % max(1, steps // 10) == 0:
                pbar.set_postfix(
                    {
                        "t": f"{t.item():.3f}",
                        "mask%": f"{mask_ratio * 100:.1f}%",
                        "loss": f"{text_loss.item() if text_loss is not None else 0.0:.4f}",
                        "acc": f"{text_accuracy:.4f}",
                    }
                )

        return z_t, {
            "losses": losses,
            "mask_ratios": mask_ratios,
            "text_accuracies": text_accuracies,
        }

    def decode(self, tokens):
        """Decode tokens to text"""
        texts = []
        for t in tokens.cpu().numpy():
            valid = []
            for tok in t:
                if tok in [
                    self.tokenizer.pad_token_id,
                    self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id,
                    self.mask_token_id,
                ]:
                    continue
                valid.append(tok)

            if valid:
                text = self.tokenizer.decode(valid, skip_special_tokens=True).strip()
                texts.append(text)
            else:
                texts.append("")
        return texts

    def load_latents(self, npy_dir, num_samples=None):
        """Load .npy latent files and corresponding .txt files.

        Expects directory structure:
            <base>/latents/train/<name>.npy
            <base>/texts/train/<name>.txt

        Returns:
            latents: Tensor [N, latent_dim]
            texts: List of strings (or None if texts dir not found)
        """
        import glob

        files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))

        if num_samples and len(files) > num_samples:
            import random

            files = random.sample(files, num_samples)
        elif num_samples:
            files = files[:num_samples]

        # Find texts directory: npy_dir=.../latents/train/ -> .../texts/train/
        npy_dir_abs = os.path.abspath(npy_dir)
        texts_dir = npy_dir_abs.replace("/latents/", "/texts/")
        has_texts = os.path.isdir(texts_dir)
        if has_texts:
            print(f"Found texts directory: {texts_dir}")
        else:
            print(f"No texts directory found at {texts_dir}, loss will not be computed")

        latents = []
        texts = []
        for f in tqdm(files, desc="Loading latents"):
            data = np.load(f)
            latent_dim = 32  # From your config
            if data.shape[0] >= latent_dim:
                data = data[:latent_dim]
            else:
                data = np.pad(data, (0, latent_dim - data.shape[0]))
            latents.append(torch.from_numpy(data).float())

            # Load corresponding text
            if has_texts:
                basename = os.path.splitext(os.path.basename(f))[0]
                txt_path = os.path.join(texts_dir, basename + ".txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as tf:
                        texts.append(tf.read().strip())
                else:
                    texts.append("")

        if latents:
            return torch.stack(latents, dim=0), texts if has_texts else None

        print(f"No .npy files found, creating random latents")
        return torch.randn(num_samples or 3, 32), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--npy_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of reverse diffusion steps"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--output_dir", default="./reverse_diffusion_output")
    parser.add_argument("--algorithm", choices=["reverse", "ddim"], default="reverse")
    parser.add_argument(
        "--eta", type=float, default=0.0, help="DDIM eta (0=deterministic)"
    )
    parser.add_argument(
        "--with_loss_logging",
        action="store_true",
        help="Compute and log text loss during sampling",
    )
    parser.add_argument(
        "--target_texts",
        type=str,
        default=None,
        help="Path to file with target texts for loss computation",
    )

    args = parser.parse_args()

    # Create sampler
    sampler = ReverseDiffusionSampler(args.checkpoint, args.config)

    # Load latents and corresponding texts (auto-discovered from sibling texts/ dir)
    latents, loaded_texts = sampler.load_latents(args.npy_dir, args.num_samples)
    print(f"Loaded {latents.shape[0]} latents")

    # Use --target_texts file if provided, otherwise use auto-discovered texts
    target_texts = None
    if args.target_texts and os.path.exists(args.target_texts):
        with open(args.target_texts, "r") as f:
            target_texts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(target_texts)} target texts from {args.target_texts}")
    elif loaded_texts:
        target_texts = loaded_texts
        print(f"Using {len(target_texts)} texts auto-loaded from texts/ directory")

    # Generate
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_texts = []
    all_results = {
        "losses": [],
        "mask_ratios": [],
        "text_accuracies": [],
    }

    for i in range(0, latents.shape[0], args.batch_size):
        batch = latents[i : i + args.batch_size]
        print(f"\nGenerating batch {i // args.batch_size + 1}")

        if args.with_loss_logging:
            if args.algorithm == "reverse":
                tokens, results = sampler.sample_l2t_with_loss_logging(
                    batch,
                    target_texts=target_texts[i : i + args.batch_size]
                    if target_texts
                    else None,
                    seq_len=args.seq_len,
                    steps=args.steps,
                    temperature=args.temperature,
                )
            else:  # ddim
                tokens, results = sampler.sample_l2t_ddim_with_loss_logging(
                    batch,
                    target_texts=target_texts[i : i + args.batch_size]
                    if target_texts
                    else None,
                    seq_len=args.seq_len,
                    steps=args.steps,
                    temperature=args.temperature,
                    eta=args.eta,
                )

            # Store results
            all_results["losses"].extend(results.get("losses", []))
            all_results["mask_ratios"].extend(results.get("mask_ratios", []))
            all_results["text_accuracies"].extend(results.get("text_accuracies", []))
        else:
            # Tokenize target texts for loss computation and attention mask
            batch_target_tokens = None
            batch_attention_mask = None
            if target_texts:
                batch_texts = target_texts[i : i + args.batch_size]
                tokenized = sampler.tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=args.seq_len,
                    return_tensors="pt",
                )
                batch_target_tokens = tokenized["input_ids"]
                batch_attention_mask = tokenized["attention_mask"]

            if args.algorithm == "reverse":
                tokens = sampler.sample_l2t(
                    batch,
                    seq_len=args.seq_len,
                    steps=args.steps,
                    temperature=args.temperature,
                    target_tokens=batch_target_tokens,
                    target_attention_mask=batch_attention_mask,
                )
            else:  # ddim
                tokens = sampler.sample_l2t_ddim(
                    batch,
                    seq_len=args.seq_len,
                    steps=args.steps,
                    temperature=args.temperature,
                    eta=args.eta,
                    target_tokens=batch_target_tokens,
                    target_attention_mask=batch_attention_mask,
                )

        texts = sampler.decode(tokens)
        all_texts.extend(texts)

        for j, text in enumerate(texts):
            print(f"\nSample {i + j + 1}:")
            if target_texts:
                print(f"  [TARGET] {target_texts[i + j][:200]}")
            print(f"  [GENERATED] {text[:200]}")

    # Save results
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "texts": all_texts,
                "num_samples": len(all_texts),
                "parameters": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save loss logging data if enabled
    if args.with_loss_logging:
        loss_log = {
            "losses": all_results["losses"],
            "mask_ratios": all_results["mask_ratios"],
            "text_accuracies": all_results["text_accuracies"],
        }
        with open(output_dir / "sampling_metrics.json", "w", encoding="utf-8") as f:
            json.dump(loss_log, f, indent=2)

        # Also save as JSONL (like training log)
        with open(output_dir / "sampling_metrics.jsonl", "w", encoding="utf-8") as f:
            for step, loss in enumerate(all_results["losses"]):
                log_entry = {
                    "step": step,
                    "loss": loss,
                    "mask_ratio": all_results["mask_ratios"][step],
                    "text_accuracy": all_results["text_accuracies"][step],
                }
                f.write(json.dumps(log_entry) + "\n")

        print(f"\n✓ Saved sampling metrics to {output_dir}")

    with open(output_dir / "texts.txt", "w", encoding="utf-8") as f:
        for i, text in enumerate(all_texts):
            f.write(f"Sample {i + 1}:\n{text}\n\n")

    print(f"\nSaved {len(all_texts)} samples to {output_dir}")


if __name__ == "__main__":
    main()
