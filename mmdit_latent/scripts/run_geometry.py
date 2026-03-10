"""
Entry point: Latent Geometry Analysis.

Runs SSS, MTS, cluster separation, and interpolation visualization.

Usage:
    python -m lsme.scripts.run_geometry \
        --checkpoint_path checkpoints/mmdit_latent \
        --latent_dir data/latents \
        --metadata_file data/latents/metadata.json \
        --attribute sentiment \
        --output_dir results/geometry/ \
        --n_pairs 10 --n_points 10
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np

from mmdit_latent.checkpoints import load_checkpoint
from mmdit_latent.sampling import get_sampler

from mmdit_latent.latent_utils.attribute_encoder import AttributeLatentEncoder
from mmdit_latent.evaluation.metrics.latent_geometry import (
    semantic_smoothness_score,
    monotonic_transition_score,
    cluster_separation,
    latent_variance_ratio,
)


def main():
    parser = argparse.ArgumentParser(description="Latent Geometry Analysis")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--metadata_file", type=str, required=True)
    parser.add_argument("--attribute", type=str, default="sentiment")
    parser.add_argument("--source_value", type=str, default="negative")
    parser.add_argument("--target_value", type=str, default="positive")
    parser.add_argument("--n_pairs", type=int, default=10,
                        help="Number of latent pairs for SSS/MTS")
    parser.add_argument("--n_points", type=int, default=10,
                        help="Interpolation points per pair")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Samples per interpolation point")
    parser.add_argument("--output_dir", type=str, default="results/geometry/")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    print("Loading checkpoint...")
    model, noise_schedule, tokenizer, config = load_checkpoint(
        args.checkpoint_path, device=device
    )
    model.eval()

    sampler = get_sampler(config, model, tokenizer, noise_schedule, compile_step=False)

    # Load latent encoder
    encoder = AttributeLatentEncoder(args.latent_dir, args.metadata_file)
    centroids = encoder.compute_attribute_centroids(args.attribute)

    z_source = centroids[args.source_value].to(device)
    z_target = centroids[args.target_value].to(device)

    results = {}

    # SSS
    print(f"\nComputing SSS ({args.source_value} -> {args.target_value})...")
    sss, trajectory = semantic_smoothness_score(
        model, sampler, z_source, z_target,
        n_points=args.n_points, n_samples=args.n_samples, device=device
    )
    results["sss"] = sss
    print(f"  SSS = {sss:.4f}")

    # MTS
    print(f"\nComputing MTS ({args.source_value} -> {args.target_value})...")
    from transformers import pipeline
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device, truncation=True, max_length=512, top_k=None
    )
    mts, scores = monotonic_transition_score(
        model, sampler, z_source, z_target,
        classifier=classifier, target_label="POSITIVE",
        n_points=args.n_points, n_samples=args.n_samples, device=device
    )
    results["mts"] = mts
    results["mts_scores"] = scores
    print(f"  MTS = {mts:.4f}")

    # Cluster separation
    print("\nComputing cluster separation...")
    all_latents, all_labels = [], []
    for filename, attrs in encoder.metadata.items():
        if args.attribute in attrs and filename in encoder.latents:
            all_latents.append(encoder.latents[filename])
            all_labels.append(attrs[args.attribute])
    if len(all_latents) > 1:
        latent_tensor = torch.stack(all_latents)
        label_to_int = {v: i for i, v in enumerate(set(all_labels))}
        label_ints = [label_to_int[l] for l in all_labels]

        sil = cluster_separation(latent_tensor, np.array(label_ints))
        var_ratio = latent_variance_ratio(latent_tensor, np.array(label_ints))
        results["cluster_separation"] = sil
        results["variance_ratio"] = var_ratio
        print(f"  Silhouette: {sil:.4f}")
        print(f"  Variance ratio: {var_ratio:.4f}")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "geometry_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'geometry_results.json'}")


if __name__ == "__main__":
    main()
