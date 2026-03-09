"""
Amazon Topic Transfer Benchmark.

Task: Edit electronics reviews to books domain using LSME.
Dataset: Amazon Reviews (multi-domain).
Metrics: Topic classifier accuracy, content preservation.
"""

import torch

from lsme.data.dataloader import load_amazon_domain
from lsme.data.preprocessing import preprocess_texts


def load_amazon_data(domain="electronics", domain_label=None,
                     split="test", max_samples=500):
    """
    Load Amazon reviews for a specific domain.

    Args:
        domain: str, domain name for labeling
        domain_label: int or None, label to filter by
        split: str
        max_samples: int

    Returns:
        texts: list of str
        domains: list of str
    """
    texts, _ = load_amazon_domain(
        domain_label=domain_label, split=split, max_samples=max_samples
    )
    texts = preprocess_texts(texts)
    domains = [domain] * len(texts)
    return texts, domains


def run_amazon_benchmark(lsme_sampler, attribute_encoder,
                         source_domain="electronics", target_domain="books",
                         mask_ratios=(0.1, 0.3, 0.5, 0.7),
                         steps=100, max_samples=500, device="cuda"):
    """
    Run the Amazon topic transfer benchmark.

    Protocol:
    1. Take reviews from source domain
    2. Get z_target from target domain centroid
    3. Run LSME at multiple mask_ratios
    4. Evaluate topic accuracy and content preservation

    Returns:
        results: dict mapping mask_ratio -> evaluation results
    """
    from lsme.evaluation.eval_suite import DLMEvalSuite

    source_texts, _ = load_amazon_data(source_domain, max_samples=max_samples)

    z_target = attribute_encoder.get_target_latent(
        "topic", target_domain, device=device
    )

    eval_suite = DLMEvalSuite({
        "device": device,
        "fluency_model": "gpt2",
    })

    results = {}
    for mr in mask_ratios:
        print(f"\n--- mask_ratio={mr} ---")

        edited_texts, _ = lsme_sampler.edit_from_text(
            source_texts,
            z_target.unsqueeze(0).expand(len(source_texts), -1),
            mask_ratio=mr, steps=steps, decode=True,
        )

        result = eval_suite.evaluate_editing(
            source_texts, edited_texts, target_attribute=target_domain
        )
        result["mask_ratio"] = mr
        results[f"mask_ratio_{mr}"] = result

    return results
