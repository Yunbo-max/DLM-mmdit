"""
Fluency metrics — Pillar 1 of DLM-Eval Suite.

Measures: Perplexity (GPT-2) and grammar error rate.

PPL computation adapted from discrete-diffusion-guidance
(https://github.com/naver-ai/discrete-diffusion-guidance) eval_utils.py
with chunked evaluation for long sequences.
"""

import torch
import numpy as np


def compute_perplexity(texts, model_name="gpt2", batch_size=8, device="cuda",
                       max_length=512, chunk_size=1024):
    """
    Compute perplexity of generated texts using a pretrained LM.

    Handles long sequences by chunking (adapted from
    discrete-diffusion-guidance eval_utils.compute_generative_ppl).

    Args:
        texts: list of str
        model_name: HuggingFace model name (default: gpt2)
        batch_size: int
        device: str
        max_length: int, max tokens per text
        chunk_size: int, context window for chunked PPL on long sequences

    Returns:
        dict with "ppl_mean", "ppl_std", "ppl_median", "ppl_per_sample"
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            input_ids = encoding["input_ids"]  # (B, L)
            attn_mask = encoding["attention_mask"]  # (B, L)
            seq_len = input_ids.shape[1]

            if seq_len <= chunk_size:
                # Standard PPL computation
                logits = model(input_ids, attention_mask=attn_mask).logits
                shift_logits = logits[:, :-1]  # (B, L-1, V)
                shift_labels = input_ids[:, 1:]  # (B, L-1)
                shift_mask = attn_mask[:, 1:]  # (B, L-1)

                loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
                losses = loss_fn(
                    shift_logits.transpose(1, 2), shift_labels
                )  # (B, L-1)
                losses = (losses * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)
                ppls.extend(losses.exp().cpu().tolist())
            else:
                # Chunked PPL for sequences longer than model context
                # Adapted from discrete-diffusion-guidance
                for b in range(input_ids.shape[0]):
                    total_loss = 0.0
                    total_tokens = 0
                    ids = input_ids[b][attn_mask[b].bool()]  # remove padding
                    for start in range(0, len(ids) - 1, chunk_size):
                        end = min(start + chunk_size, len(ids))
                        chunk = ids[start:end].unsqueeze(0)
                        logits = model(chunk).logits
                        shift_logits = logits[:, :-1]
                        shift_labels = chunk[:, 1:]
                        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
                        total_loss += loss_fn(
                            shift_logits.transpose(1, 2), shift_labels
                        ).item()
                        total_tokens += shift_labels.shape[1]
                    if total_tokens > 0:
                        ppls.append(np.exp(total_loss / total_tokens))

    del model
    torch.cuda.empty_cache()

    return {
        "ppl_mean": float(np.mean(ppls)),
        "ppl_std": float(np.std(ppls)),
        "ppl_median": float(np.median(ppls)),
        "ppl_per_sample": ppls,
    }


def compute_grammar_errors(texts):
    """
    Count grammar errors using language_tool_python (if available).

    Args:
        texts: list of str

    Returns:
        dict with "errors_mean", "errors_per_sample"
    """
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool("en-US")
    except ImportError:
        return {"errors_mean": None, "errors_per_sample": None,
                "note": "language_tool_python not installed"}

    errors = []
    for text in texts:
        matches = tool.check(text)
        errors.append(len(matches))

    tool.close()
    return {
        "errors_mean": float(np.mean(errors)),
        "errors_per_sample": errors,
    }
