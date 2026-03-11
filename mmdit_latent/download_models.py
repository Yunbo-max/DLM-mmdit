# File: mmdit_latent/download_models.py
"""
Download tokenizer and encoder models locally into mmdit_latent/data/models/.
Run from repo root:
    python -m mmdit_latent.download_models
"""

import os
from pathlib import Path


def main():
    # Resolve paths relative to repo root
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "data" / "models"

    tokenizer_dir = models_dir / "bert-base-uncased"
    encoder_dir = models_dir / "Qwen3-Embedding-8B"

    # 1. Download BERT tokenizer
    print(f"Downloading bert-base-uncased → {tokenizer_dir}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"  Done: {tokenizer_dir}")

    # 2. Download Qwen3-Embedding-8B
    print(f"\nDownloading Qwen/Qwen3-Embedding-8B → {encoder_dir}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        trust_remote_code=True,
        truncate_dim=32,
    )
    model.save(str(encoder_dir))
    print(f"  Done: {encoder_dir}")

    print(f"\nAll models saved to: {models_dir}")
    print(f"\nTo preprocess with local models:")
    print(f"  python -m mmdit_latent.preprocess_data \\")
    print(f"    --dataset Skylion007/openwebtext \\")
    print(f"    --tokenizer mmdit_latent/data/models/bert-base-uncased \\")
    print(f"    --encoder mmdit_latent/data/models/Qwen3-Embedding-8B")


if __name__ == "__main__":
    main()
