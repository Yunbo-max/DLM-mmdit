# File: mmdit_latent/data_simple.py
import json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import random


class SimpleLatentDataset(Dataset):
    """Dataset that loads text and latent pairs from JSON."""

    def __init__(self, json_path, tokenizer, max_length=512, max_samples=None):
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load JSON data
        with open(json_path, 'r') as f:
            self.samples = json.load(f)

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except Exception as e:
            # Fallback: return a dummy sample on error to avoid crashing training
            print(f"WARNING: Error loading sample {idx}: {e}")
            return self._fallback_item()

    def _load_item(self, idx):
        item = self.samples[idx]
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

        # Load latent if available
        if 'latent_path' in item:
            latent_path = Path(self.json_path.parent) / item['latent_path']
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
            # All samples have latents — pad to same length
            max_len = max(lat.shape[0] for lat in latents)
            latent_dim = latents[0].shape[-1]
            padded_latents = torch.zeros(len(latents), max_len, latent_dim)
            for i, lat in enumerate(latents):
                padded_latents[i, :lat.shape[0]] = lat
            result['latent'] = padded_latents

    return result


def get_simple_dataloaders(config, tokenizer):
    """Create train/val dataloaders from JSON files with latent support."""

    # Create datasets
    train_json = Path(config.data.data_files.train)
    val_json = (
        Path(config.data.data_files.validation)
        if hasattr(config.data, 'data_files') and hasattr(config.data.data_files, 'validation')
        else None
    )

    train_ds = SimpleLatentDataset(
        train_json,
        tokenizer,
        max_length=config.model.max_seq_len,
        max_samples=config.data.get('max_samples', None)
    )

    if val_json and val_json.exists():
        test_ds = SimpleLatentDataset(
            val_json,
            tokenizer,
            max_length=config.model.max_seq_len,
            max_samples=config.data.get('max_val_samples', 1000)
        )
    else:
        # Create validation split from training
        print("No validation JSON found, splitting training data")
        total_size = len(train_ds)
        val_size = min(1000, total_size // 10)
        indices = list(range(total_size))
        random.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        from torch.utils.data import Subset
        test_ds = Subset(train_ds, val_indices)
        train_ds = Subset(train_ds, train_indices)

    # Create distributed sampler if needed
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        test_sampler = DistributedSampler(test_ds, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    num_workers = config.data.get('num_workers', 4)
    use_persistent = num_workers > 0

    # Create dataloaders with persistent workers and prefetch
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
