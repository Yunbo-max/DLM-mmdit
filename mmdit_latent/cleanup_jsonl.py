"""Remove JSONL entries that reference missing shards or out-of-bounds indices.
Also handles corrupted lines with concatenated JSON objects."""
import json
import re
import sys
from pathlib import Path
import numpy as np


def split_concatenated_json(line):
    """Split a line that may contain multiple concatenated JSON objects."""
    # Match individual JSON objects: {"..."}
    return re.findall(r'\{[^{}]*\}', line)


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
        # Try .bak first (from previous cleanup run), then original
        bak = data_root / f"{split}.bak"
        src = data_root / split
        if bak.exists():
            read_from = bak
        elif src.exists():
            read_from = src
        else:
            continue

        dst = data_root / f"{split}.clean"
        kept, removed = 0, 0

        with open(read_from, 'r') as fin, open(dst, 'w') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                # Try parsing as single JSON first
                try:
                    entry = json.loads(line)
                    entries = [entry]
                except json.JSONDecodeError:
                    # Line has concatenated JSON objects — split them
                    entries = []
                    for match in split_concatenated_json(line):
                        try:
                            entries.append(json.loads(match))
                        except json.JSONDecodeError:
                            removed += 1

                for entry in entries:
                    shard_id = entry.get("shard")
                    idx = entry.get("idx")

                    if shard_id not in shard_sizes:
                        removed += 1
                        continue
                    if idx >= shard_sizes[shard_id]:
                        removed += 1
                        continue

                    fout.write(json.dumps(entry) + "\n")
                    kept += 1

        # Replace original
        if src.exists() and src != read_from:
            src.unlink()
        dst.rename(src)
        print(f"\n{split}: kept {kept}, removed {removed}")

if __name__ == "__main__":
    data_root = sys.argv[1] if len(sys.argv) > 1 else "mmdit_latent/data"
    cleanup(data_root)
