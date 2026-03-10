"""
Yelp Sentiment Editing Benchmark.

Task: Edit negative reviews to positive using LSME.
Dataset: Yelp Review Polarity (binary: positive/negative).
Metrics: Sentiment accuracy, BLEU, PPL.
"""

import torch
import numpy as np
from pathlib import Path

from mmdit_latent.data.dataloader import load_yelp_negative, load_yelp_positive
from mmdit_latent.data.preprocessing import preprocess_texts


def get_negative_samples(split="test", max_samples=500):
    """Get preprocessed negative (label=0) samples for editing to positive."""
    texts, _ = load_yelp_negative(split=split, max_samples=max_samples)
    return preprocess_texts(texts)


def run_yelp_benchmark(lsme_sampler, attribute_encoder,
                       mask_ratios=(0.1, 0.3, 0.5, 0.7),
                       steps=100, max_samples=500, device="cuda"):
    """
    Run the full Yelp sentiment editing benchmark.

    Protocol:
    1. Take negative reviews from test set
    2. Get z_positive from attribute encoder
    3. Run LSME at multiple mask_ratios
    4. Evaluate with DLM-Eval Suite

    Args:
        lsme_sampler: LSMESampler instance
        attribute_encoder: AttributeLatentEncoder with Yelp latents loaded
        mask_ratios: tuple of float
        steps: int, diffusion steps
        max_samples: int
        device: str

    Returns:
        results: dict mapping mask_ratio -> evaluation results
    """
    from mmdit_latent.evaluation.eval_suite import DLMEvalSuite

    # Load negative test samples
    source_texts = get_negative_samples(max_samples=max_samples)

    # Get target latent for positive sentiment
    z_positive = attribute_encoder.get_target_latent(
        "sentiment", "positive", device=device
    )

    eval_suite = DLMEvalSuite({
        "device": device,
        "fluency_model": "gpt2",
        "classifier_name": "distilbert-base-uncased-finetuned-sst-2-english",
    })

    results = {}
    for mr in mask_ratios:
        print(f"\n--- mask_ratio={mr} ---")

        edited_texts, _ = lsme_sampler.edit_from_text(
            source_texts,
            z_positive.unsqueeze(0).expand(len(source_texts), -1),
            mask_ratio=mr, steps=steps, decode=True,
        )

        result = eval_suite.evaluate_editing(
            source_texts, edited_texts, target_attribute="POSITIVE"
        )
        result["mask_ratio"] = mr
        results[f"mask_ratio_{mr}"] = result

        # Quick summary
        acc = result["controllability"]["accuracy"]
        ppl = result["fluency"]["ppl_mean"]
        bleu = result["edit_quality"]["bleu_mean"]
        print(f"  Accuracy: {acc:.3f}, PPL: {ppl:.1f}, BLEU: {bleu:.3f}")

    return results
