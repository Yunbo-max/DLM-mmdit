# File: mmdit_latent/merge_shards.py
"""
Merge multiple preprocessed data parts (from multi-GPU preprocessing) into one.

Usage:
  python -m mmdit_latent.merge_shards \
    --inputs mmdit_latent/data_part0 mmdit_latent/data_part1 \
    --output mmdit_latent/data
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm


SHARD_SIZE = 100_000


def main():
    parser = argparse.ArgumentParser(description="Merge preprocessed data parts")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input directories to merge")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--val_size", type=int, default=1000,
                        help="Number of validation samples in merged output")
    args = parser.parse_args()

    output_dir = Path(args.output)
    shard_dir = output_dir / "latent_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Collect all latents and JSONL entries, re-index shards
    current_shard_latents = []
    current_shard_id = 0
    total_samples = 0

    train_f = open(output_dir / "train_data.jsonl", "w")
    val_f = open(output_dir / "validation_data.jsonl", "w")

    def flush_shard():
        nonlocal current_shard_latents, current_shard_id
        if not current_shard_latents:
            return
        arr = np.stack(current_shard_latents).astype(np.float32)
        np.save(shard_dir / f"shard_{current_shard_id:04d}.npy", arr)
        print(f"  Saved shard {current_shard_id}: {arr.shape}")
        current_shard_id += 1
        current_shard_latents = []

    latent_dim = None

    for input_dir in args.inputs:
        input_dir = Path(input_dir)
        print(f"Processing: {input_dir}")

        # Read all JSONL files (train + val from this part)
        for jsonl_name in ["train_data.jsonl", "validation_data.jsonl"]:
            jsonl_path = input_dir / jsonl_name
            if not jsonl_path.exists():
                continue

            # Load shards from this part
            part_shards = {}

            with open(jsonl_path, "r") as f:
                for line in tqdm(f, desc=f"  {jsonl_name}"):
                    item = json.loads(line)

                    # Load latent from source shard
                    src_shard_id = item["shard"]
                    if src_shard_id not in part_shards:
                        src_path = input_dir / "latent_shards" / f"shard_{src_shard_id:04d}.npy"
                        part_shards[src_shard_id] = np.load(src_path)

                    latent = part_shards[src_shard_id][item["idx"]]
                    if latent_dim is None:
                        latent_dim = latent.shape[-1]

                    # Write to merged output with new shard/idx
                    idx_in_shard = len(current_shard_latents)
                    entry = {
                        "text": item["text"],
                        "shard": current_shard_id,
                        "idx": idx_in_shard,
                    }
                    line_out = json.dumps(entry) + "\n"

                    if total_samples < args.val_size:
                        val_f.write(line_out)
                    else:
                        train_f.write(line_out)

                    current_shard_latents.append(latent)
                    total_samples += 1

                    if len(current_shard_latents) >= SHARD_SIZE:
                        flush_shard()

    flush_shard()
    train_f.close()
    val_f.close()

    # Save metadata
    metadata = {
        "latent_dim": int(latent_dim) if latent_dim else 32,
        "shard_size": SHARD_SIZE,
        "num_shards": current_shard_id,
        "total_samples": total_samples,
        "train_samples": total_samples - min(args.val_size, total_samples),
        "val_samples": min(args.val_size, total_samples),
        "merged_from": [str(p) for p in args.inputs],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Merged {len(args.inputs)} parts → {output_dir}")
    print(f"  Total samples: {total_samples}")
    print(f"  Shards: {current_shard_id}")
    print(f"  Train: {metadata['train_samples']}, Val: {metadata['val_samples']}")


if __name__ == "__main__":
    main()
