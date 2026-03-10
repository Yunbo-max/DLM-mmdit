"""
Data loading for LSME experiments.

Adapted from MDLM (https://github.com/kuleshov-group/mdlm) dataloader.py.
Provides tokenizer loading, dataset loading, and DataLoader creation for
the three LSME benchmarks (Yelp, Amazon, GYAFC).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def get_tokenizer(config):
    """
    Load a HuggingFace tokenizer from config.

    Adapted from MDLM dataloader.get_tokenizer().

    Args:
        config: dict or OmegaConf with "tokenizer_name_or_path" key.
                Defaults to "bert-base-uncased" if not specified.

    Returns:
        tokenizer: PreTrainedTokenizer
    """
    from transformers import AutoTokenizer

    name = config.get("tokenizer_name_or_path", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(name)

    # Ensure special tokens exist (MDLM pattern)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.cls_token or "[CLS]"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token or "[SEP]"

    return tokenizer


def load_editing_dataset(dataset_name, split="test", max_samples=500,
                         filter_fn=None, text_key="text", label_key="label"):
    """
    Load a dataset for text editing experiments.

    Supports: yelp_polarity, amazon_polarity, gyafc.

    Args:
        dataset_name: str, HuggingFace dataset name
        split: str, "train" or "test"
        max_samples: int
        filter_fn: optional callable(item) -> bool to filter samples
        text_key: str, key for text field
        label_key: str, key for label field

    Returns:
        texts: list of str
        labels: list of int or str
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)

    texts, labels = [], []
    for item in ds:
        if filter_fn is not None and not filter_fn(item):
            continue
        if len(texts) >= max_samples:
            break
        texts.append(item[text_key])
        labels.append(item[label_key])

    return texts, labels


def load_yelp_negative(split="test", max_samples=500):
    """Load negative Yelp reviews for sentiment editing."""
    return load_editing_dataset(
        "yelp_polarity", split=split, max_samples=max_samples,
        filter_fn=lambda item: item["label"] == 0,
    )


def load_yelp_positive(split="train", max_samples=5000):
    """Load positive Yelp reviews for computing target centroid."""
    return load_editing_dataset(
        "yelp_polarity", split=split, max_samples=max_samples,
        filter_fn=lambda item: item["label"] == 1,
    )


def load_amazon_domain(domain_label=None, split="test", max_samples=500):
    """
    Load Amazon reviews, optionally filtering by domain.

    Args:
        domain_label: int or None. If None, loads all.
        split: str
        max_samples: int

    Returns:
        texts: list of str
        labels: list of int
    """
    filter_fn = None
    if domain_label is not None:
        filter_fn = lambda item: item["label"] == domain_label
    return load_editing_dataset(
        "amazon_polarity", split=split, max_samples=max_samples,
        filter_fn=filter_fn, text_key="content",
    )


def load_gyafc(split="test", max_samples=500):
    """
    Load GYAFC informal/formal sentence pairs.

    Note: GYAFC may require manual download. Falls back gracefully.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("gyafc", split=split)
        informal = [item["informal"] for item in ds][:max_samples]
        formal = [item["formal"] for item in ds][:max_samples]
        return informal, formal
    except Exception:
        return [], []


class TextEditingDataset(Dataset):
    """
    Dataset wrapper for text editing experiments.

    Tokenizes source texts and provides them as batches for LSME editing.

    Adapted from MDLM data pattern.

    Args:
        texts: list of str, source texts
        tokenizer: PreTrainedTokenizer
        max_length: int
        labels: optional list, parallel labels
    """

    def __init__(self, texts, tokenizer, max_length=512, labels=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

        # Pre-tokenize all texts
        self.encodings = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "text": self.texts[idx],
        }
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def get_dataloaders(texts, tokenizer, batch_size=32, max_length=512,
                    labels=None, shuffle=False, num_workers=0):
    """
    Create a DataLoader for text editing.

    Adapted from MDLM dataloader.get_dataloaders() — simplified for
    inference/editing (no distributed sampling).

    Args:
        texts: list of str
        tokenizer: PreTrainedTokenizer
        batch_size: int
        max_length: int
        labels: optional list
        shuffle: bool
        num_workers: int

    Returns:
        DataLoader
    """
    dataset = TextEditingDataset(texts, tokenizer, max_length, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
