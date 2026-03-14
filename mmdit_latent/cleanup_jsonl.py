"""Remove JSONL entries that reference missing shards or out-of-bounds indices."""
import json
import sys
from pathlib import Path
import numpy as np

def cleanup(data_root):
    data_root = Path(data_root)
    shard_dir = data_root / "latent_shards"

    # Discover existing shards and their sizes
    shard_sizes = {}
    for shard_file in sorted(shard_dir.glob("shard_*.npy")):
        shard_id = int(shard_file.stem.split("_")[1])
        arr = np.load(shard_file, mmap_mode='r')
        shard_sizes[shard_id] = arr.shape[0]
        print(f"  shard_{shard_id:04d}: {arr.shape[0]} entries")

    print(f"\nFound {len(shard_sizes)} shards")

    for split in ["train_data.jsonl", "validation_data.jsonl"]:
        src = data_root / split
        if not src.exists():
            continue

        dst = data_root / f"{split}.clean"
        kept, removed = 0, 0

        with open(src, 'r') as fin, open(dst, 'w') as fout:
            for line in fin:
                entry = json.loads(line)
                shard_id = entry.get("shard")
                idx = entry.get("idx")

                if shard_id not in shard_sizes:
                    removed += 1
                    continue
                if idx >= shard_sizes[shard_id]:
                    removed += 1
                    continue

                fout.write(line)
                kept += 1

        # Replace original
        src.rename(data_root / f"{split}.bak")
        dst.rename(src)
        print(f"\n{split}: kept {kept}, removed {removed}")

if __name__ == "__main__":
    data_root = sys.argv[1] if len(sys.argv) > 1 else "mmdit_latent/data"
    cleanup(data_root)
