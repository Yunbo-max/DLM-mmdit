# File: mmdit_latent/preprocess_data.py
"""
Preprocess text datasets for mmdit_latent training.

Option C pipeline (chunk-based, recommended):
  1. Load documents from HuggingFace dataset or text file
  2. Tokenize each document with BERT tokenizer (same as training)
  3. Split into max_seq_len-sized chunks (like baseline MDLM)
  4. Decode each chunk back to text
  5. Encode each chunk with Qwen3-Embedding-8B → 32-dim latent
  6. Save latents as sharded .npy files + JSONL index

Output structure:
  output_dir/
    latent_shards/shard_0000.npy  (100K x 32, float32)
    train_data.jsonl              (one JSON per line)
    validation_data.jsonl
    metadata.json

Usage:
  # 4096 chunks (default, matches training config)
  python -m mmdit_latent.preprocess_data \
    --dataset Skylion007/openwebtext \
    --output_dir /path/to/data_root \
    --max_seq_len 4096

  # 512 chunks (for faster training experiments)
  python -m mmdit_latent.preprocess_data \
    --dataset Skylion007/openwebtext \
    --output_dir /path/to/data_root_512 \
    --max_seq_len 512

  # From a local text file
  python -m mmdit_latent.preprocess_data \
    --text_file /path/to/texts.txt \
    --output_dir /path/to/data_root

  # Limit samples for testing
  python -m mmdit_latent.preprocess_data \
    --dataset Skylion007/openwebtext \
    --output_dir /path/to/data_root \
    --max_docs 10000
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SHARD_SIZE = 100_000  # samples per shard file


def chunk_document(text, tokenizer, max_seq_len, min_chunk_tokens=32):
    """
    Tokenize a document and split into fixed-size chunks.
    Same logic as baseline MDLM (baseline/data.py tokenize_dataset).

    Returns list of (token_ids, decoded_text) tuples.
    """
    bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    # Tokenize full document without truncation
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Add BOS/EOS
    token_ids = [bos_id] + token_ids + [eos_id]

    # Split into max_seq_len chunks
    chunks = []
    for start in range(0, len(token_ids), max_seq_len):
        chunk_tokens = token_ids[start:start + max_seq_len]

        # Skip very short last chunks (mostly padding)
        if len(chunk_tokens) < min_chunk_tokens:
            continue

        # Pad to max_seq_len
        pad_len = max_seq_len - len(chunk_tokens)
        attention_len = len(chunk_tokens)
        if pad_len > 0:
            chunk_tokens = chunk_tokens + [tokenizer.pad_token_id] * pad_len

        # Decode back to text for Qwen3 encoding
        # Only decode the real tokens (not padding)
        decoded_text = tokenizer.decode(chunk_tokens[:attention_len],
                                         skip_special_tokens=True)

        if len(decoded_text.strip()) < 10:  # skip near-empty chunks
            continue

        chunks.append({
            "tokens": chunk_tokens,
            "text": decoded_text,
            "attention_len": attention_len,
        })

    return chunks


def stream_chunks_from_dataset(dataset_name, tokenizer, max_seq_len,
                                split="train", max_docs=None,
                                cache_dir="./data", text_key="text",
                                min_chunk_tokens=32):
    """Stream document chunks from a HuggingFace dataset."""
    from datasets import load_dataset

    print(f"Loading dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir,
                      trust_remote_code=True)

    doc_count = 0
    chunk_count = 0
    for i in tqdm(range(len(ds)), desc="Chunking documents"):
        if max_docs and doc_count >= max_docs:
            break

        row = ds[i]
        text = row.get(text_key, row.get("sentence", "")).strip()
        if len(text.split()) < 5:
            continue

        chunks = chunk_document(text, tokenizer, max_seq_len, min_chunk_tokens)
        for chunk in chunks:
            yield chunk
            chunk_count += 1

        doc_count += 1

    print(f"Processed {doc_count} documents → {chunk_count} chunks")


def stream_chunks_from_file(text_file, tokenizer, max_seq_len,
                             max_docs=None, min_chunk_tokens=32):
    """Stream chunks from a plain text file (one document per line)."""
    doc_count = 0
    chunk_count = 0
    with open(text_file, "r") as f:
        for line in f:
            if max_docs and doc_count >= max_docs:
                break
            line = line.strip()
            if len(line.split()) < 5:
                continue

            chunks = chunk_document(line, tokenizer, max_seq_len, min_chunk_tokens)
            for chunk in chunks:
                yield chunk
                chunk_count += 1

            doc_count += 1

    print(f"Processed {doc_count} documents → {chunk_count} chunks")


def process_and_save(chunk_stream, output_dir, encoder_name="Qwen/Qwen3-Embedding-8B",
                     latent_dim=32, batch_size=256, device="cuda",
                     max_text_tokens=512, val_size=1000):
    """
    Consume chunk stream, encode latents, save as sharded files + JSONL.

    Writes train/val JSONL incrementally — constant memory regardless of dataset size.
    """
    from sentence_transformers import SentenceTransformer

    output_dir = Path(output_dir)
    shard_dir = output_dir / "latent_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading encoder: {encoder_name} (output dim={latent_dim})")
    model = SentenceTransformer(
        encoder_name,
        trust_remote_code=True,
        truncate_dim=latent_dim,
    )
    model.max_seq_length = max_text_tokens

    train_path = output_dir / "train_data.jsonl"
    val_path = output_dir / "validation_data.jsonl"

    # Accumulate for batched encoding
    batch_chunks = []
    # Accumulate for current shard
    shard_latents = []
    current_shard = 0
    total_processed = 0

    train_f = open(train_path, "w")
    val_f = open(val_path, "w")

    def flush_shard():
        nonlocal shard_latents, current_shard
        if not shard_latents:
            return
        shard_array = np.stack(shard_latents).astype(np.float32)
        shard_file = shard_dir / f"shard_{current_shard:04d}.npy"
        np.save(shard_file, shard_array)
        print(f"  Saved shard {current_shard}: {shard_array.shape} -> {shard_file}")
        current_shard += 1
        shard_latents = []

    def encode_and_write(chunks_batch):
        nonlocal total_processed, shard_latents
        texts = [c["text"] for c in chunks_batch]

        embeddings = model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
            device=device,
        )

        for chunk, emb in zip(chunks_batch, embeddings):
            idx_in_shard = len(shard_latents)
            entry = {
                "text": chunk["text"],
                "shard": current_shard,
                "idx": idx_in_shard,
            }
            line = json.dumps(entry) + "\n"

            if total_processed < val_size:
                val_f.write(line)
            else:
                train_f.write(line)

            shard_latents.append(emb)
            total_processed += 1

            if len(shard_latents) >= SHARD_SIZE:
                flush_shard()

    pbar = tqdm(desc="Processing chunks")
    for chunk in chunk_stream:
        batch_chunks.append(chunk)

        if len(batch_chunks) >= batch_size:
            encode_and_write(batch_chunks)
            pbar.update(len(batch_chunks))
            pbar.set_postfix(samples=total_processed, shard=current_shard)
            batch_chunks = []

    # Process remaining
    if batch_chunks:
        encode_and_write(batch_chunks)
        pbar.update(len(batch_chunks))

    # Flush last partial shard
    flush_shard()
    pbar.close()

    train_f.close()
    val_f.close()

    # Save metadata
    metadata = {
        "latent_dim": latent_dim,
        "shard_size": SHARD_SIZE,
        "num_shards": current_shard,
        "total_samples": total_processed,
        "train_samples": total_processed - min(val_size, total_processed),
        "val_samples": min(val_size, total_processed),
        "encoder": encoder_name,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!")
    print(f"  Total chunks: {total_processed}")
    print(f"  Train: {metadata['train_samples']}")
    print(f"  Val: {metadata['val_samples']}")
    print(f"  Shards: {current_shard}")
    print(f"  Latent dim: {latent_dim}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess text data for mmdit_latent (Option C: chunk-based)"
    )
    # Input
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset name (e.g. Skylion007/openwebtext)")
    parser.add_argument("--text_file", type=str, default=None,
                        help="Plain text file (one doc per line)")
    parser.add_argument("--text_key", type=str, default="text",
                        help="Key for text field in HF dataset")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--cache_dir", type=str, default="./data",
                        help="HuggingFace cache directory")

    # Output
    parser.add_argument("--output_dir", type=str, default="mmdit_latent/data",
                        help="Output directory (default: mmdit_latent/data)")

    # Chunking
    parser.add_argument("--max_seq_len", type=int, default=4096,
                        help="Chunk size in BERT tokens (must match training config)")
    parser.add_argument("--min_chunk_tokens", type=int, default=32,
                        help="Discard chunks shorter than this")
    parser.add_argument("--tokenizer", type=str, default="mmdit_latent/data/models/bert-base-uncased",
                        help="BERT tokenizer for chunking (must match training)")

    # Encoder
    parser.add_argument("--encoder", type=str, default="mmdit_latent/data/models/Qwen3-Embedding-8B",
                        help="Sentence encoder model (local path or HuggingFace name)")
    parser.add_argument("--latent_dim", type=int, default=32,
                        help="Output latent dimension (Qwen3-Embedding supports 32-4096)")
    parser.add_argument("--max_encoder_tokens", type=int, default=512,
                        help="Max tokens for Qwen3 encoder (its own tokenizer)")

    # Processing
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Max number of source documents to process")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for Qwen3 encoding")
    parser.add_argument("--val_size", type=int, default=1000,
                        help="Number of validation samples")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for encoding")

    args = parser.parse_args()

    if args.dataset is None and args.text_file is None:
        parser.error("Provide either --dataset or --text_file")

    # Load BERT tokenizer for chunking
    from transformers import AutoTokenizer
    print(f"Loading chunking tokenizer: {args.tokenizer}")
    bert_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Ensure pad token exists
    if bert_tokenizer.pad_token_id is None:
        bert_tokenizer.pad_token = bert_tokenizer.eos_token

    # Create chunk stream
    if args.text_file:
        chunk_stream = stream_chunks_from_file(
            args.text_file, bert_tokenizer, args.max_seq_len,
            max_docs=args.max_docs,
            min_chunk_tokens=args.min_chunk_tokens,
        )
    else:
        chunk_stream = stream_chunks_from_dataset(
            args.dataset, bert_tokenizer, args.max_seq_len,
            split=args.split, max_docs=args.max_docs,
            cache_dir=args.cache_dir, text_key=args.text_key,
            min_chunk_tokens=args.min_chunk_tokens,
        )

    # Process and save
    process_and_save(
        chunk_stream,
        args.output_dir,
        encoder_name=args.encoder,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        device=args.device,
        max_text_tokens=args.max_encoder_tokens,
        val_size=args.val_size,
    )

    print(f"\nTo train with 4096:")
    print(f"  CONFIG_NAME=mdlm_mmdit_latent LATENT_DATA_ROOT={args.output_dir} bash train_mmdit_latent.sh")
    print(f"\nTo train with 512:")
    print(f"  CONFIG_NAME=mdlm_mmdit_latent_512 LATENT_DATA_ROOT={args.output_dir} bash train_mmdit_latent.sh")


if __name__ == "__main__":
    main()
