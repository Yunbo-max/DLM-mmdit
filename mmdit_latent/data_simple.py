# File: mmdit_latent/data_simple.py
import json
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import random


class SimpleLatentDataset(Dataset):
    """
    Dataset that loads text and latent pairs from JSONL + sharded latent files.

    Two latent storage formats:
      1. Per-sample .npy files (small scale):
         {"text": "...", "latent_path": "latents/sample_0000.npy"}

      2. Sharded .npy files (large scale, 1B+):
         {"text": "...", "shard": 5, "idx": 42317}
         Latent loaded from: latent_shards/shard_0005.npy[42317]

    JSONL format: only byte offsets are stored in memory, not the text.
    JSON format: loaded fully into memory (for small datasets).
    """

    def __init__(self, data_path, tokenizer, max_length=512, max_samples=None,
                 data_root=None):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_root = Path(data_root) if data_root else self.data_path.parent

        # Shard cache: loaded shards stay in memory (LRU-style)
        self._shard_cache = {}
        self._max_cached_shards = 8  # keep at most 8 shards in memory (~8 * 100K * 32 * 4 = ~100MB)

        if str(self.data_path).endswith('.jsonl'):
            self._init_jsonl(max_samples)
        else:
            self._init_json(max_samples)

    def _init_json(self, max_samples):
        """Load small JSON dataset into memory."""
        self.mode = 'json'
        with open(self.data_path, 'r') as f:
            self.samples = json.load(f)
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} samples from {self.data_path} (JSON, in-memory)")

    def _init_jsonl(self, max_samples):
        """
        Index a JSONL file by byte offsets — only offsets are stored in RAM.
        For 1B samples this uses ~8GB RAM (8 bytes per offset) instead of ~hundreds of GB.
        """
        self.mode = 'jsonl'
        self.offsets = []
        print(f"Indexing {self.data_path} ...")
        with open(self.data_path, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                self.offsets.append(offset)
                if max_samples and len(self.offsets) >= max_samples:
                    break
        print(f"Indexed {len(self.offsets)} samples from {self.data_path} (JSONL, lazy)")

    def __len__(self):
        if self.mode == 'json':
            return len(self.samples)
        return len(self.offsets)

    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except Exception as e:
            print(f"WARNING: Error loading sample {idx}: {e}")
            return self._fallback_item()

    def _get_sample(self, idx):
        """Get the raw dict for sample idx."""
        if self.mode == 'json':
            return self.samples[idx]
        with open(self.data_path, 'r') as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            return json.loads(line)

    def _load_shard(self, shard_id):
        """Load a latent shard, with simple caching."""
        if shard_id in self._shard_cache:
            return self._shard_cache[shard_id]

        shard_path = self.data_root / "latent_shards" / f"shard_{shard_id:04d}.npy"
        shard_data = np.load(shard_path, mmap_mode='r')  # memory-mapped, not loaded into RAM

        # Evict oldest if cache full
        if len(self._shard_cache) >= self._max_cached_shards:
            oldest = next(iter(self._shard_cache))
            del self._shard_cache[oldest]

        self._shard_cache[shard_id] = shard_data
        return shard_data

    def _load_item(self, idx):
        item = self._get_sample(idx)
        text = item.get('text', '')

        # Tokenize text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        result = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
        }

        # Load latent — sharded format
        if 'shard' in item and 'idx' in item:
            shard_data = self._load_shard(item['shard'])
            latent = shard_data[item['idx']]  # (latent_dim,)
            latent_tensor = torch.from_numpy(latent.copy()).float()
            if latent_tensor.dim() == 1:
                latent_tensor = latent_tensor.unsqueeze(0)
            result['latent'] = latent_tensor

        # Load latent — per-file format (backward compat)
        elif 'latent_path' in item:
            latent_path = self.data_root / item['latent_path']
            if latent_path.exists():
                latent = np.load(latent_path)
                latent_tensor = torch.from_numpy(latent).float()
                if latent_tensor.dim() == 1:
                    latent_tensor = latent_tensor.unsqueeze(0)
                result['latent'] = latent_tensor

        return result

    def _fallback_item(self):
        """Return a valid dummy sample when loading fails."""
        result = {
            'input_ids': torch.full((self.max_length,), self.tokenizer.pad_token_id, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
        }
        return result


def collate_fn(batch):
    """Collate function that handles optional latents with variable length."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    # Handle latents
    if 'latent' in batch[0]:
        latents = [item['latent'] for item in batch if 'latent' in item]
        if len(latents) == len(batch):
            max_len = max(lat.shape[0] for lat in latents)
            latent_dim = latents[0].shape[-1]
            padded_latents = torch.zeros(len(latents), max_len, latent_dim)
            for i, lat in enumerate(latents):
                padded_latents[i, :lat.shape[0]] = lat
            result['latent'] = padded_latents

    return result


def get_simple_dataloaders(config, tokenizer):
    """
    Create train/val dataloaders with latent support.

    Supports:
      - .json  → in-memory (small datasets)
      - .jsonl → lazy byte-offset indexing (large datasets, up to ~1B)
    Latent formats:
      - Per-sample .npy files (latent_path)
      - Sharded .npy files (shard + idx) — recommended for 1B+
    """
    train_path = Path(config.data.data_files.train)
    val_path = (
        Path(config.data.data_files.validation)
        if hasattr(config.data, 'data_files') and hasattr(config.data.data_files, 'validation')
        else None
    )

    data_root = config.data.get('latent_data_root', None)

    train_ds = SimpleLatentDataset(
        train_path, tokenizer,
        max_length=config.model.max_seq_len,
        max_samples=config.data.get('max_samples', None),
        data_root=data_root,
    )

    if val_path and val_path.exists():
        test_ds = SimpleLatentDataset(
            val_path, tokenizer,
            max_length=config.model.max_seq_len,
            max_samples=config.data.get('max_val_samples', 1000),
            data_root=data_root,
        )
    else:
        print("No validation file found, splitting training data")
        total_size = len(train_ds)
        val_size = min(1000, total_size // 10)
        indices = list(range(total_size))
        random.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        from torch.utils.data import Subset
        test_ds = Subset(train_ds, val_indices)
        train_ds = Subset(train_ds, train_indices)

    # Distributed sampler
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        test_sampler = DistributedSampler(test_ds, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    num_workers = config.data.get('num_workers', 4)
    use_persistent = num_workers > 0

    train_dl = DataLoader(
        train_ds,
        batch_size=config.training.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        timeout=3600 if use_persistent else 0,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=config.training.eval_batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(num_workers, 2),
        pin_memory=True,
        drop_last=False,
    )

    return train_dl, test_dl
