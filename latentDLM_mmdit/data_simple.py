
        
        

# File: latentDLM_mmdit/data_simple.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import json
import random
import os
from typing import Optional, Tuple

class DirectFileDataset(Dataset):
    """Dataset that reads directly from tokenized .npz and latent .npy files.

    Uses lazy loading: scans minimal files upfront, loads more on-demand during training.
    """

    def __init__(self,
                 token_dir: str,
                 latent_dir: str,
                 max_samples: Optional[int] = None,
                 initial_scan_limit: int = 10000,
                 verbose: bool = True,
                 max_seq_len: Optional[int] = None):
        """
        Args:
            token_dir: Directory containing tokenized .npz files
            latent_dir: Directory containing latent .npy files
            max_samples: Maximum number of samples to use (for dataset length)
            initial_scan_limit: Number of files to scan initially (default: 10000)
            verbose: Print debug information
            max_seq_len: Maximum sequence length (truncates longer sequences if set)
        """
        self.token_dir = Path(token_dir)
        self.latent_dir = Path(latent_dir)
        self.max_samples = max_samples
        self.verbose = verbose
        self.max_seq_len = max_seq_len

        if verbose:
            print(f"Initializing LAZY dataset from:")
            print(f"  Token dir: {self.token_dir}")
            print(f"  Latent dir: {self.latent_dir}")
            print(f"  Initial scan: {initial_scan_limit:,} files")
            print(f"  Max samples: {max_samples if max_samples else 'unlimited'}")

        # Quick initial scan - just enough to start training
        valid_files = []
        scanned_count = 0

        if verbose:
            print(f"Quick scan: loading first {initial_scan_limit:,} files...")

        # Fast initial scan
        with os.scandir(self.token_dir) as entries:
            for entry in entries:
                if not entry.name.endswith('.npz'):
                    continue

                scanned_count += 1

                # Get corresponding latent file name
                latent_name = entry.name.replace('_tokens.npz', '.npy')
                if latent_name == entry.name:
                    latent_name = entry.name.replace('.npz', '.npy')

                latent_file = self.latent_dir / latent_name

                if latent_file.exists():
                    valid_files.append((Path(entry.path), latent_file))

                # Stop after initial scan limit
                if len(valid_files) >= initial_scan_limit:
                    break

        self.file_pairs = valid_files
        self.initial_scan_complete = True

        # Set dataset length
        if max_samples:
            self._length = max_samples
        else:
            # Use initial scan size, will expand if needed
            self._length = len(self.file_pairs)

        if verbose:
            print(f"✓ Quick initialization complete:")
            print(f"  Loaded: {len(self.file_pairs):,} file pairs")
            print(f"  Dataset length: {self._length:,}")
            print(f"  Training can start immediately!")
            print(f"  (More files will be loaded on-demand during training)")

            # Show some examples
            if len(self.file_pairs) > 0:
                print("\nSample file mappings:")
                for i in range(min(3, len(self.file_pairs))):
                    token_f, latent_f = self.file_pairs[i]
                    print(f"  Token: {token_f.name}")
                    print(f"  Latent: {latent_f.name}")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Use modulo to cycle through available files if idx exceeds loaded files
        actual_idx = idx % len(self.file_pairs)
        token_file, latent_file = self.file_pairs[actual_idx]

        try:
            # Load tokenized data
            token_data = np.load(token_file)
            input_ids = token_data['input_ids'].astype(np.int32)
            attention_mask = token_data['attention_mask'].astype(bool)

            # Ensure correct shapes
            if input_ids.ndim == 0:
                input_ids = np.array([input_ids])
            if attention_mask.ndim == 0:
                attention_mask = np.array([attention_mask])

            # Truncate to max_seq_len if specified
            if self.max_seq_len is not None and len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                attention_mask = attention_mask[:self.max_seq_len]

            # Load latent
            latent = np.load(latent_file)

            # Convert to tensors
            input_ids_tensor = torch.from_numpy(input_ids).long()
            attention_mask_tensor = torch.from_numpy(attention_mask).float()
            latent_tensor = torch.from_numpy(latent).float()

            # Ensure latent has correct shape [1, latent_dim]
            if latent_tensor.dim() == 1:
                latent_tensor = latent_tensor.unsqueeze(0)  # [1, latent_dim]
            elif latent_tensor.dim() == 2:
                if latent_tensor.shape[0] != 1:
                    latent_tensor = latent_tensor[0:1]  # Take first if multiple

            return {
                'input_ids': input_ids_tensor,
                'attention_mask': attention_mask_tensor,
                'latent': latent_tensor,
                'token_file': str(token_file.name),
                'latent_file': str(latent_file.name)
            }

        except Exception as e:
            if self.verbose:
                print(f"Error loading files {token_file}, {latent_file}: {e}")
            # Return zeros as fallback
            return self._create_empty_sample()
    
    def _create_empty_sample(self):
        """Create an empty sample for error cases."""
        return {
            'input_ids': torch.zeros(512, dtype=torch.long),
            'attention_mask': torch.zeros(512, dtype=torch.float),
            'latent': torch.zeros(1, 1024, dtype=torch.float),
            'token_file': 'error',
            'latent_file': 'error'
        }


class PreTokenizedDataset(Dataset):
    """Dataset that reads pre-tokenized data from your specific directory structure."""
    
    def __init__(self, base_dir: str, split: str = "train", max_samples: Optional[int] = None):
        """
        Args:
            base_dir: Base directory (e.g., sonar-embeddings-1024d-interact)
            split: "train" or "val"
            max_samples: Maximum samples to load
        """
        self.base_dir = Path(base_dir)
        self.split = split
        
        # Directories
        self.token_dir = self.base_dir / "tokens" / split
        self.latent_dir = self.base_dir / "latents" / split
        self.text_dir = self.base_dir / "texts" / split
        
        print(f"Token dir: {self.token_dir}")
        print(f"Latent dir: {self.latent_dir}")
        
        # Get all token files
        token_files = sorted(list(self.token_dir.glob("*.npz")))
        
        if max_samples:
            token_files = token_files[:max_samples]
        
        # Create file pairs
        self.file_pairs = []
        for token_file in token_files:
            # Get corresponding latent file name
            latent_name = token_file.stem.replace('_tokens', '')
            latent_file = self.latent_dir / f"{latent_name}.npy"
            
            if latent_file.exists():
                self.file_pairs.append((token_file, latent_file))
        
        print(f"Loaded {len(self.file_pairs)} file pairs from {base_dir}")
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        token_file, latent_file = self.file_pairs[idx]
        
        # Load tokenized data
        token_data = np.load(token_file)
        
        # Extract arrays - handle both npz and direct numpy formats
        if 'input_ids' in token_data:
            input_ids = token_data['input_ids']
            attention_mask = token_data['attention_mask']
        else:
            # Assume it's a dictionary-like structure
            input_ids = token_data['input_ids']
            attention_mask = token_data['attention_mask']
        
        # Load latent
        latent = np.load(latent_file)
        
        # Convert to tensors
        input_ids = torch.from_numpy(input_ids.astype(np.int32)).long()
        attention_mask = torch.from_numpy(attention_mask.astype(bool)).float()
        latent = torch.from_numpy(latent).float()
        
        # Ensure latent has correct shape [1, latent_dim]
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'latent': latent
        }


def get_direct_dataloaders(config, tokenizer=None):
    """Get dataloaders that read directly from tokenized and latent files."""
    from torch.utils.data import DataLoader, random_split

    # Extract paths from config
    data_config = config.data

    # Option 1: Direct paths (for your current structure)
    if hasattr(data_config, 'token_dir') and hasattr(data_config, 'latent_dir'):
        token_dir = data_config.token_dir
        latent_dir = data_config.latent_dir

        print(f"Loading from token_dir: {token_dir}")
        print(f"Loading from latent_dir: {latent_dir}")

        # Get initial scan limit from config (default: 10000)
        initial_scan_limit = data_config.get('initial_scan_limit', 10000)

        # Get max_seq_len from config or model config (default: None)
        max_seq_len = data_config.get('max_seq_len', None)
        if max_seq_len is None and hasattr(config, 'model'):
            max_seq_len = getattr(config.model, 'max_seq_len', None)

        # Create dataset with lazy loading
        full_dataset = DirectFileDataset(
            token_dir=token_dir,
            latent_dir=latent_dir,
            max_samples=data_config.get('max_samples', None),
            initial_scan_limit=initial_scan_limit,
            verbose=True,
            max_seq_len=max_seq_len
        )

    # Option 2: Base directory with standard structure
    elif hasattr(data_config, 'data_dir'):
        base_dir = Path(data_config.data_dir)
        split = data_config.get('split', 'train')

        print(f"Loading from base_dir: {base_dir}, split: {split}")

        # Create dataset
        full_dataset = PreTokenizedDataset(
            base_dir=base_dir,
            split=split,
            max_samples=data_config.get('max_samples', None)
        )

    else:
        raise ValueError("Either token_dir/latent_dir or data_dir must be specified in config")

    print(f"Total samples: {len(full_dataset)}")

    # Split into train/val
    val_ratio = data_config.get('val_ratio', 0.05)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Collate function
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # Get latent dimension from config or first sample
        latent_dim = config.model.get('latent_dim', 1024)
        
        # Process latents
        latents = []
        for item in batch:
            latent = item['latent']
            # Ensure shape [1, latent_dim]
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            elif latent.dim() == 2 and latent.shape[0] != 1:
                latent = latent[0:1]
            latents.append(latent)
        
        latent_tensor = torch.cat(latents, dim=0)  # [batch_size, 1, latent_dim]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'latent': latent_tensor
        }
    
    # Create distributed samplers
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
            seed=config.training.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # DataLoader settings
    num_workers = data_config.get('num_workers', 16)
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        timeout=3600  # 1 hour timeout for slow data loading
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=min(8, num_workers),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        timeout=3600  # 1 hour timeout for slow data loading
    )
    
    return train_loader, val_loader


# For backward compatibility
def get_simple_dataloaders(config, tokenizer=None):
    """Main entry point - automatically chooses the right data loader."""
    # Check which type of data we're using
    use_preprocessed = config.data.get('use_preprocessed', True)
    
    if use_preprocessed:
        print("Using preprocessed file-based data loader")
        return get_direct_dataloaders(config, tokenizer)
    else:
        print("Using JSON-based data loader (needs tokenizer)")
        # This would be your old JSON-based loader
        # You can keep it for backward compatibility
        from latentDLM_mmdit.data_simple_legacy import get_json_dataloaders
        return get_json_dataloaders(config, tokenizer)


