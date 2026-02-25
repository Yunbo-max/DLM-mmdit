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

            # Separate model weights from trainer weights
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

            # Remap latent_encoder keys to match current architecture
            remapped_state_dict = {}

            for k, v in model_state_dict.items():
                if k.startswith("latent_encoder.projection."):
                    parts = k.split(".")
                    if len(parts) >= 4:
                        idx = int(parts[2])
                        param_type = parts[3]

                        # Map based on index
                        if idx == 0:  # First Linear (32 -> 2048)
                            new_k = f"latent_encoder.adapter.0.{param_type}"
                            remapped_state_dict[new_k] = v
                            print(f"Mapping {k} -> {new_k} [{v.shape}]")

                        elif idx == 1:  # Skip GELU parameters
                            print(f"Skipping {k} (GELU parameters)")
                            continue

                        elif idx == 3:  # Second Linear (2048 -> 1024)
                            # Map to adapter.2
                            new_k = f"latent_encoder.adapter.2.{param_type}"
                            remapped_state_dict[new_k] = v
                            print(f"Mapping {k} -> {new_k} [{v.shape}]")
                else:
                    remapped_state_dict[k] = v

            # Initialize missing layers
            with torch.no_grad():
                # Initialize LayerNorm
                if hasattr(self.model, "latent_encoder") and hasattr(
                    self.model.latent_encoder, "norm"
                ):
                    if "latent_encoder.norm.weight" not in remapped_state_dict:
                        remapped_state_dict["latent_encoder.norm.weight"] = (
                            torch.ones_like(self.model.latent_encoder.norm.weight)
                        )
                        remapped_state_dict["latent_encoder.norm.bias"] = (
                            torch.zeros_like(self.model.latent_encoder.norm.bias)
                        )
                        print("✓ Initialized missing LayerNorm")

                # Initialize residual_proj from scratch
                if hasattr(self.model, "latent_encoder") and hasattr(
                    self.model.latent_encoder, "residual_proj"
                ):
                    if "latent_encoder.residual_proj.weight" not in remapped_state_dict:
                        remapped_state_dict["latent_encoder.residual_proj.weight"] = (
                            torch.randn_like(
                                self.model.latent_encoder.residual_proj.weight
                            )
                            * 0.02
                        )
                        remapped_state_dict["latent_encoder.residual_proj.bias"] = (
                            torch.zeros_like(
                                self.model.latent_encoder.residual_proj.bias
                            )
                        )
                        print("✓ Initialized missing residual_proj with random weights")

            # Load model weights
            missing, unexpected = self.model.load_state_dict(
                remapped_state_dict, strict=False
            )
            print(
                f"\nModel loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )

            if len(missing) > 0:
                print(f"First 5 missing keys: {missing[:5]}")
            if len(unexpected) > 0:
                print(f"First 5 unexpected keys: {unexpected[:5]}")
        self.model.eval()

        # Create continuous diffusion for latents
        self.latent_diffusion = ContinuousDiffusion(
            num_timesteps=config["model"].get("latent_timesteps", 1000),
            beta_schedule=config["model"].get("latent_beta_schedule", "cosine"),
            parameterization=config["model"].get("latent_parameterization", "epsilon"),
            device=device,
        )

        print(f"✓ Sampler initialized")

    @torch.no_grad()
    def sample_l2t(self, latents, seq_len=128, steps=1000, temperature=1.0):
        """
        True reverse diffusion sampling for L2T mode.
        Follows the reverse process from t=1.0 (fully masked) to t=0.0 (clean).
        """
        batch_size = latents.shape[0]
        device = self.device

        # Ensure latents are correct shape [batch, latent_dim]
        if latents.dim() == 1:
            latents = latents.unsqueeze(0)
        latents = latents.to(device)

        # Start from fully masked tokens (t=1.0)
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )

        # Attention mask (all tokens valid)
        attention_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=device
        )

        # Latent conditioning (clean, t=0)
        latent_t = torch.zeros(batch_size, device=device)

        # Reverse time steps (from 1.0 to 0.0)
        eps = 1e-4
        timesteps = torch.linspace(1.0 - eps, eps, steps, device=device)

        pbar = tqdm(timesteps, desc="Reverse diffusion")
        for i, t in enumerate(pbar):
            t_vec = torch.full((batch_size,), float(t), device=device)

            # Model predicts p(x₀ | x_t, z)
            text_logits, _ = self.model(
                text_tokens=x,
                latents=latents.unsqueeze(1),
                text_timesteps=t_vec,
                latent_timesteps=latent_t,
                attention_mask=attention_mask,
            )

            # Apply temperature
            if temperature != 1.0:
                text_logits = text_logits / temperature

            # Convert to probabilities for x₀
            probs_x0 = torch.softmax(text_logits, dim=-1)

            # Get transition probabilities for x_{t-Δt}
            if i < steps - 1:
                t_next = timesteps[i + 1]
                t_next_vec = torch.full((batch_size,), float(t_next), device=device)

                # Get logits for x_{t-Δt}
                logits_xt_next = self.text_noise_schedule.logits_at_t(
                    probs_x0, t_next_vec
                )
                probs_xt_next = torch.softmax(logits_xt_next, dim=-1)
            else:
                # Final step: sample from p(x₀)
                probs_xt_next = probs_x0

            # Sample new tokens
            probs_flat = probs_xt_next.reshape(-1, probs_xt_next.shape[-1])
            sampled_flat = torch.multinomial(probs_flat, 1)
            x = sampled_flat.view(batch_size, seq_len)

            # Show progress
            if i % max(1, steps // 10) == 0:
                mask_ratio = (x == self.mask_token_id).float().mean().item()
                pbar.set_postfix({"t": f"{t:.3f}", "mask%": f"{mask_ratio * 100:.1f}%"})

        return x

    @torch.no_grad()
    def sample_l2t_ddim(
        self, latents, seq_len=128, steps=100, temperature=1.0, eta=0.0
    ):
        """
        DDIM-style sampling for faster generation.
        eta=0: deterministic, eta=1: stochastic (like DDPM)
        """
        batch_size = latents.shape[0]
        device = self.device

        # Ensure latents are correct shape
        if latents.dim() == 1:
            latents = latents.unsqueeze(0)
        latents = latents.to(device)

        # Start from fully masked
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )

        attention_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=device
        )
        latent_t = torch.zeros(batch_size, device=device)

        # DDIM timesteps (subsampled)
        eps = 1e-4
        all_timesteps = torch.linspace(1.0 - eps, eps, steps + 1, device=device)

        pbar = tqdm(range(steps), desc="DDIM sampling")
        for i in pbar:
            t = all_timesteps[i]
            t_next = all_timesteps[i + 1]

            t_vec = torch.full((batch_size,), float(t), device=device)

            # Model predicts logits for x₀
            text_logits, _ = self.model(
                text_tokens=x,
                latents=latents.unsqueeze(1),
                text_timesteps=t_vec,
                latent_timesteps=latent_t,
                attention_mask=attention_mask,
            )

            if temperature != 1.0:
                text_logits = text_logits / temperature

            # Get probabilities for x₀
            probs_x0 = torch.softmax(text_logits, dim=-1)

            # Get probabilities for x_t
            probs_xt = self.text_noise_schedule.probs_at_t(probs_x0, t_vec)

            # Get probabilities for x_{t_next} (deterministic or stochastic)
            if eta > 0:
                # Stochastic step
                probs_xt_next = self.text_noise_schedule.probs_at_t(probs_x0, t_next)
            else:
                # Deterministic: take argmax of x₀ prediction
                x0_pred = torch.argmax(probs_x0, dim=-1)
                # Apply noise to get x_{t_next}
                probs_xt_next = self.text_noise_schedule.probs_at_t(
                    F.one_hot(x0_pred, num_classes=probs_x0.shape[-1]).float(), t_next
                )

            # Sample
            probs_flat = probs_xt_next.reshape(-1, probs_xt_next.shape[-1])
            if eta > 0:
                sampled_flat = torch.multinomial(probs_flat, 1)
            else:
                sampled_flat = probs_flat.argmax(dim=-1, keepdim=True)
            x = sampled_flat.view(batch_size, seq_len)

            # Show progress
            mask_ratio = (x == self.mask_token_id).float().mean().item()
            pbar.set_postfix({"t": f"{t:.3f}", "mask%": f"{mask_ratio * 100:.1f}%"})

        return x

    @torch.no_grad()
    def sample_l2t_with_loss_logging(
        self, latents, target_texts=None, seq_len=128, steps=1000, temperature=1.0
    ):
        """
        L2T Sampling with Loss Logging - Computes text loss at each step if target texts are provided.

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

        # Ensure latents are correct shape [batch, latent_dim]
        if latents.dim() == 1:
            latents = latents.unsqueeze(0)
        latents = latents.to(device)

        # Encode target texts if provided
        target_tokens = None
        if target_texts is not None:
            target_tokens = self.tokenizer(
                target_texts,
                padding=True,
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )["input_ids"].to(device)
            print(f"✓ Encoded {len(target_texts)} target texts for loss computation")

        # Start from fully masked tokens (t=1.0)
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )

        # Attention mask (all tokens valid)
        attention_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=device
        )

        # Latent conditioning (clean, t=0)
        latent_t = torch.zeros(batch_size, device=device)

        # Reverse time steps (from 1.0 to 0.0)
        eps = 1e-4
        timesteps = torch.linspace(1.0 - eps, eps, steps, device=device)

        # Loss tracking (similar to training loop)
        losses = []
        mask_ratios = []
        text_accuracies = []

        pbar = tqdm(timesteps, desc="Reverse diffusion (with loss logging)")
        for i, t in enumerate(pbar):
            t_vec = torch.full((batch_size,), float(t), device=device)

            # Model predicts p(x₀ | x_t, z)
            text_logits, _ = self.model(
                text_tokens=x,
                latents=latents.unsqueeze(1),
                text_timesteps=t_vec,
                latent_timesteps=latent_t,
                attention_mask=attention_mask,
            )

            # Apply temperature
            if temperature != 1.0:
                text_logits = text_logits / temperature

            # Compute text loss if target texts provided (similar to training)
            text_loss = None
            text_accuracy = 0.0
            if target_tokens is not None:
                # Compute cross-entropy loss on current predictions
                vocab_size = text_logits.shape[-1]
                # Only compute loss on masked positions
                current_mask = x == self.mask_token_id

                if current_mask.any():
                    text_loss_unmasked = F.cross_entropy(
                        text_logits.view(-1, vocab_size),
                        target_tokens.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(batch_size, seq_len)

                    mask_sum = current_mask.sum().float().clamp(min=1)
                    text_loss = (text_loss_unmasked * current_mask).sum() / mask_sum

                    # Compute accuracy on masked positions
                    pred_tokens = torch.argmax(text_logits, dim=-1)
                    text_accuracy = (
                        (pred_tokens[current_mask] == target_tokens[current_mask])
                        .float()
                        .mean()
                        .item()
                    )

                losses.append(text_loss.item() if text_loss is not None else 0.0)
                text_accuracies.append(text_accuracy)

            # Convert to probabilities for x₀
            probs_x0 = torch.softmax(text_logits, dim=-1)

            # Get transition probabilities for x_{t-Δt}
            if i < steps - 1:
                t_next = timesteps[i + 1]
                t_next_vec = torch.full((batch_size,), float(t_next), device=device)

                # Get logits for x_{t-Δt}
                logits_xt_next = self.text_noise_schedule.logits_at_t(
                    probs_x0, t_next_vec
                )
                probs_xt_next = torch.softmax(logits_xt_next, dim=-1)
            else:
                # Final step: sample from p(x₀)
                probs_xt_next = probs_x0

            # Sample new tokens
            probs_flat = probs_xt_next.reshape(-1, probs_xt_next.shape[-1])
            sampled_flat = torch.multinomial(probs_flat, 1)
            x = sampled_flat.view(batch_size, seq_len)

            # Track mask ratio
            mask_ratio = (x == self.mask_token_id).float().mean().item()
            mask_ratios.append(mask_ratio)

            # Log progress similar to training loop
            if i % max(1, steps // 10) == 0:
                pbar.set_postfix(
                    {
                        "t": f"{t:.3f}",
                        "mask%": f"{mask_ratio * 100:.1f}%",
                        "loss": f"{text_loss.item() if text_loss is not None else 0.0:.4f}",
                        "acc": f"{text_accuracy:.4f}",
                    }
                )

        return x, {
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
        DDIM-style sampling with loss logging.

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

        # Ensure latents are correct shape
        if latents.dim() == 1:
            latents = latents.unsqueeze(0)
        latents = latents.to(device)

        # Encode target texts if provided
        target_tokens = None
        if target_texts is not None:
            target_tokens = self.tokenizer(
                target_texts,
                padding=True,
                truncation=True,
                max_length=seq_len,
                return_tensors="pt",
            )["input_ids"].to(device)
            print(f"✓ Encoded {len(target_texts)} target texts for loss computation")

        # Start from fully masked
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )

        attention_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=device
        )
        latent_t = torch.zeros(batch_size, device=device)

        # DDIM timesteps (subsampled)
        eps = 1e-4
        all_timesteps = torch.linspace(1.0 - eps, eps, steps + 1, device=device)

        losses = []
        mask_ratios = []
        text_accuracies = []

        pbar = tqdm(range(steps), desc="DDIM sampling (with loss logging)")
        for i in pbar:
            t = all_timesteps[i]
            t_next = all_timesteps[i + 1]

            t_vec = torch.full((batch_size,), float(t), device=device)

            # Model predicts logits for x₀
            text_logits, _ = self.model(
                text_tokens=x,
                latents=latents.unsqueeze(1),
                text_timesteps=t_vec,
                latent_timesteps=latent_t,
                attention_mask=attention_mask,
            )

            if temperature != 1.0:
                text_logits = text_logits / temperature

            # Compute text loss if target texts provided
            text_loss = None
            text_accuracy = 0.0
            if target_tokens is not None:
                vocab_size = text_logits.shape[-1]
                current_mask = x == self.mask_token_id

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

            # Get probabilities for x₀
            probs_x0 = torch.softmax(text_logits, dim=-1)

            # Get probabilities for x_t
            probs_xt = self.text_noise_schedule.probs_at_t(probs_x0, t_vec)

            # Get probabilities for x_{t_next}
            if eta > 0:
                probs_xt_next = self.text_noise_schedule.probs_at_t(probs_x0, t_next)
            else:
                x0_pred = torch.argmax(probs_x0, dim=-1)
                probs_xt_next = self.text_noise_schedule.probs_at_t(
                    F.one_hot(x0_pred, num_classes=probs_x0.shape[-1]).float(), t_next
                )

            # Sample
            probs_flat = probs_xt_next.reshape(-1, probs_xt_next.shape[-1])
            if eta > 0:
                sampled_flat = torch.multinomial(probs_flat, 1)
            else:
                sampled_flat = probs_flat.argmax(dim=-1, keepdim=True)
            x = sampled_flat.view(batch_size, seq_len)

            mask_ratio = (x == self.mask_token_id).float().mean().item()
            mask_ratios.append(mask_ratio)

            if i % max(1, steps // 10) == 0:
                pbar.set_postfix(
                    {
                        "t": f"{t:.3f}",
                        "mask%": f"{mask_ratio * 100:.1f}%",
                        "loss": f"{text_loss.item() if text_loss is not None else 0.0:.4f}",
                        "acc": f"{text_accuracy:.4f}",
                    }
                )

        return x, {
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
        """Load .npy files"""
        import glob

        files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))

        if num_samples and len(files) > num_samples:
            import random

            files = random.sample(files, num_samples)
        elif num_samples:
            files = files[:num_samples]

        latents = []
        for f in tqdm(files, desc="Loading latents"):
            data = np.load(f)
            latent_dim = 32  # From your config
            if data.shape[0] >= latent_dim:
                data = data[:latent_dim]
            else:
                data = np.pad(data, (0, latent_dim - data.shape[0]))
            latents.append(torch.from_numpy(data).float())

        if latents:
            return torch.stack(latents, dim=0)

        print(f"No .npy files found, creating random latents")
        return torch.randn(num_samples or 3, 32)


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

    # Load target texts if provided
    target_texts = None
    if args.target_texts and os.path.exists(args.target_texts):
        with open(args.target_texts, "r") as f:
            target_texts = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(target_texts)} target texts from {args.target_texts}")

    # Create sampler
    sampler = ReverseDiffusionSampler(args.checkpoint, args.config)

    # Load latents
    latents = sampler.load_latents(args.npy_dir, args.num_samples)
    print(f"Loaded {latents.shape[0]} latents")

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
            if args.algorithm == "reverse":
                tokens = sampler.sample_l2t(
                    batch,
                    seq_len=args.seq_len,
                    steps=args.steps,
                    temperature=args.temperature,
                )
            else:  # ddim
                tokens = sampler.sample_l2t_ddim(
                    batch,
                    seq_len=args.seq_len,
                    steps=args.steps,
                    temperature=args.temperature,
                    eta=args.eta,
                )

        texts = sampler.decode(tokens)
        all_texts.extend(texts)

        for j, text in enumerate(texts):
            print(f"\nSample {i + j + 1}:")
            print(f"{text}")

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
