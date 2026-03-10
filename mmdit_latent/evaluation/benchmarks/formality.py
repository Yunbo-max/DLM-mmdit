"""
GYAFC Formality Transfer Benchmark.

Task: Edit informal text to formal using LSME.
Dataset: GYAFC (Grammarly's Yahoo Answers Formality Corpus).
Metrics: Formality classifier accuracy, BERTScore (meaning preservation).
"""

import torch

from mmdit_latent.data.dataloader import load_gyafc
from mmdit_latent.data.preprocessing import preprocess_texts


def load_gyafc_data(split="test", max_samples=500):
    """
    Load GYAFC informal sentences.

    Note: GYAFC may require manual download. Falls back gracefully.

    Args:
        split: str
        max_samples: int

    Returns:
        informal_texts: list of str
        formal_references: list of str (if available)
    """
    informal, formal = load_gyafc(split=split, max_samples=max_samples)
    if informal:
        informal = preprocess_texts(informal)
        formal = formal[:len(informal)]
    return informal, formal


def run_formality_benchmark(lsme_sampler, attribute_encoder,
                            mask_ratios=(0.1, 0.3, 0.5, 0.7),
                            steps=100, max_samples=500, device="cuda"):
    """
    Run the GYAFC formality transfer benchmark.

    Protocol:
    1. Take informal sentences
    2. Get z_formal from attribute encoder
    3. Run LSME at multiple mask_ratios
    4. Evaluate formality accuracy and BERTScore vs references

    Returns:
        results: dict mapping mask_ratio -> evaluation results
    """
    from mmdit_latent.evaluation.eval_suite import DLMEvalSuite

    informal_texts, formal_references = load_gyafc_data(max_samples=max_samples)
    if not informal_texts:
        print("GYAFC data not available. Skipping formality benchmark.")
        return {}

    z_formal = attribute_encoder.get_target_latent(
        "formality", "formal", device=device
    )

    eval_suite = DLMEvalSuite({
        "device": device,
        "fluency_model": "gpt2",
    })

    results = {}
    for mr in mask_ratios:
        print(f"\n--- mask_ratio={mr} ---")

        edited_texts, _ = lsme_sampler.edit_from_text(
            informal_texts,
            z_formal.unsqueeze(0).expand(len(informal_texts), -1),
            mask_ratio=mr, steps=steps, decode=True,
        )

        result = eval_suite.evaluate_editing(
            informal_texts, edited_texts, target_attribute="formal"
        )
        result["mask_ratio"] = mr
        results[f"mask_ratio_{mr}"] = result

    return results
