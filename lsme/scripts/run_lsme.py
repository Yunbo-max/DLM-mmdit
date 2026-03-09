"""
Entry point: LSME editing.

Loads a trained MMDiT checkpoint, computes attribute centroids,
and runs LSME editing on input texts.

Usage:
    python -m lsme.scripts.run_lsme \
        --checkpoint_path checkpoints/mmdit_latent \
        --latent_dir data/latents \
        --metadata_file data/latents/metadata.json \
        --attribute sentiment --target_value positive \
        --mask_ratio 0.3 --steps 100 \
        --input_file data/test_negatives.txt \
        --output_file results/lsme_edited.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Import from existing codebase
from mmdit_latent.checkpoints import load_checkpoint
from mmdit_latent.sampling import get_sampler

# Import LSME modules
from lsme.sample_lsme import LSMESampler
from lsme.latent_utils.attribute_encoder import AttributeLatentEncoder


def main():
    parser = argparse.ArgumentParser(description="LSME: Latent-Steered Masked Editing")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained MMDiT checkpoint")
    parser.add_argument("--latent_dir", type=str, required=True,
                        help="Directory with .npy latent files")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="JSON file mapping latent filenames to attributes")
    parser.add_argument("--attribute", type=str, required=True,
                        help="Attribute name (e.g., 'sentiment', 'topic')")
    parser.add_argument("--target_value", type=str, required=True,
                        help="Target attribute value (e.g., 'positive')")
    parser.add_argument("--mask_ratio", type=float, default=0.3,
                        help="Edit strength: 0.0=no edit, 1.0=full regen")
    parser.add_argument("--steps", type=int, default=100,
                        help="Reverse diffusion steps")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--mask_mode", type=str, default="random",
                        choices=["random", "entropy", "suffix"],
                        help="Masking strategy")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Text file with one input per line")
    parser.add_argument("--input_text", type=str, default=None,
                        help="Single input text (alternative to --input_file)")
    parser.add_argument("--output_file", type=str, default="results/lsme_output.json",
                        help="Output JSON file")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    model, noise_schedule, tokenizer, config = load_checkpoint(
        args.checkpoint_path, device=device
    )
    model.eval()

    # Build LSME sampler
    lsme = LSMESampler(model, tokenizer, noise_schedule)
    lsme.to(device)

    # Load attribute encoder
    print(f"Loading latents from {args.latent_dir}...")
    encoder = AttributeLatentEncoder(args.latent_dir, args.metadata_file)
    encoder.compute_attribute_centroids(args.attribute)

    z_target = encoder.get_target_latent(
        args.attribute, args.target_value, device=device
    )
    print(f"Target latent for {args.attribute}={args.target_value}: shape {z_target.shape}")

    # Load input texts
    if args.input_file:
        with open(args.input_file) as f:
            input_texts = [line.strip() for line in f if line.strip()]
    elif args.input_text:
        input_texts = [args.input_text]
    else:
        print("Error: provide --input_file or --input_text")
        sys.exit(1)

    print(f"Editing {len(input_texts)} texts with mask_ratio={args.mask_ratio}, "
          f"steps={args.steps}, mode={args.mask_mode}...")

    # Run LSME editing in batches
    all_results = []
    for i in range(0, len(input_texts), args.batch_size):
        batch = input_texts[i:i + args.batch_size]
        z_batch = z_target.unsqueeze(0).expand(len(batch), -1)

        edited_texts, edit_masks = lsme.edit_from_text(
            batch, z_batch,
            mask_ratio=args.mask_ratio,
            steps=args.steps,
            temperature=args.temperature,
            mask_mode=args.mask_mode,
            max_length=args.max_length,
            decode=True,
        )

        for src, edt in zip(batch, edited_texts):
            all_results.append({
                "source": src,
                "edited": edt,
                "attribute": args.attribute,
                "target_value": args.target_value,
                "mask_ratio": args.mask_ratio,
            })

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Edited {len(all_results)} texts.")

    # Print a few examples
    for r in all_results[:3]:
        print(f"\n  Source:  {r['source'][:100]}...")
        print(f"  Edited:  {r['edited'][:100]}...")


if __name__ == "__main__":
    main()
