import hydra
import tqdm
import torch
import numpy as np
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from latentDLM_mmdit.utils import parse_dtype
from latentDLM_mmdit.checkpoints_mmdit import load_full_model
from latentDLM_mmdit.sampling import get_sampler


class L2TModelWrapper(torch.nn.Module):
    """Wraps MultimodalMMDiT to expose model(z_t, t) -> text_logits interface.

    For L2T (latent-to-text) mode: latent timesteps are 0 (clean latents),
    only text tokens are diffused.
    """
    def __init__(self, model, latents=None):
        super().__init__()
        self.model = model
        self.register_buffer("_latents", latents if latents is not None else torch.empty(0))
        # Forward config from wrapped model
        self.config = model.config

    @property
    def latents(self):
        return self._latents if self._latents.numel() > 0 else None

    def forward(self, z_t, t, **kwargs):
        batch_size = z_t.shape[0]
        device = z_t.device
        # For L2T: latent timesteps = 0 (clean/no noise)
        latent_t = torch.zeros(batch_size, device=device)
        latents = self.latents
        if latents is not None:
            if latents.shape[0] == 1:
                latents = latents.expand(batch_size, -1)
            elif latents.shape[0] < batch_size:
                latents = latents[:batch_size]
        result = self.model(z_t, latents, t, latent_t)
        text_logits = result[0]
        return text_logits


@hydra.main(config_path="../configs", config_name="generate", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    print(f"Generating {args.num_samples} samples from {args.path}")

    ckpt_path = hydra.utils.to_absolute_path(args.path)
    config = args
    torch.manual_seed(args.get("seed", 0))

    model, text_noise_schedule, latent_diffusion, tokenizer, old_config, _ = load_full_model(
        ckpt_path, device=device
    )
    model.eval()
    old_config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(old_config.training.dtype)

    # Load latents if provided
    latent_path = args.get("latent_path", None)
    if latent_path is not None:
        latent_path = hydra.utils.to_absolute_path(latent_path)
        print(f"Loading latents from {latent_path}")
        if latent_path.endswith(".npy"):
            all_latents = torch.from_numpy(np.load(latent_path)).float().to(device)
        else:
            all_latents = torch.load(latent_path, map_location=device, weights_only=True).float()
    else:
        all_latents = None

    # Wrap model for L2T sampling (model(z_t, t) interface)
    wrapped_model = L2TModelWrapper(model, latents=all_latents).to(device)

    sampler = get_sampler(
        old_config, wrapped_model, tokenizer, text_noise_schedule,
        min_p=args.min_p, new_config=config
    )
    wrapped_model.eval()

    samples = []
    with tqdm.tqdm(total=args.num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            for i in range(0, args.num_samples, args.batch_size):
                bs = min(args.batch_size, args.num_samples - i)
                z_t = sampler.generate(bs, args.num_denoising_steps, decode=False, show_progress=False)
                samples.append(z_t)
                pbar.update(bs)
    samples = torch.cat(samples, dim=0).cpu()

    torch.save(samples, hydra.utils.to_absolute_path(args.samples_path))


if __name__ == "__main__":
    main()
