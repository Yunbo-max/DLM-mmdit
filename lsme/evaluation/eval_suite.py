"""
DLM-Eval Suite — 6-pillar evaluation framework for discrete diffusion LMs.

Solves gap: Gap 7 — no comprehensive DLM evaluation suite.
Pillars: Fluency, Controllability, Edit Quality, Latent Geometry, Diversity, Efficiency.
"""

import json
from pathlib import Path

from .metrics.fluency import compute_perplexity, compute_grammar_errors
from .metrics.controllability import compute_attribute_accuracy
from .metrics.edit_quality import compute_edit_quality
from .metrics.latent_geometry import (
    semantic_smoothness_score, monotonic_transition_score,
    cluster_separation, latent_variance_ratio
)
from .metrics.diversity import compute_diversity_metrics
from .metrics.efficiency import EfficiencyTracker


class DLMEvalSuite:
    """
    Comprehensive 6-pillar evaluation framework for DLMs.

    Args:
        config: dict or OmegaConf with:
            fluency_model: str (GPT-2 model for PPL)
            classifier_name: str (sentiment/topic classifier)
            sentence_encoder: str (for BERTScore, SSS)
            device: str
    """

    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cuda")
        self.fluency_model = config.get("fluency_model", "gpt2")
        self.classifier_name = config.get("classifier_name",
            "distilbert-base-uncased-finetuned-sst-2-english")
        self.sentence_encoder_name = config.get("sentence_encoder",
            "all-MiniLM-L6-v2")
        self._classifier = None
        self._sentence_encoder = None

    def evaluate_generation(self, generated_texts):
        """
        Evaluate unconditional/conditional generation.

        Returns dict with fluency, diversity, and efficiency results.
        """
        results = {}
        results["fluency"] = compute_perplexity(
            generated_texts, model_name=self.fluency_model, device=self.device
        )
        results["diversity"] = compute_diversity_metrics(generated_texts)
        return results

    def evaluate_editing(self, source_texts, edited_texts, target_attribute):
        """
        Evaluate LSME editing quality — all 6 pillars except geometry.

        Args:
            source_texts: list of str, original texts
            edited_texts: list of str, edited texts
            target_attribute: str, target attribute label

        Returns dict with fluency, controllability, edit_quality, diversity.
        """
        results = {}

        results["fluency"] = compute_perplexity(
            edited_texts, model_name=self.fluency_model, device=self.device
        )
        results["controllability"] = compute_attribute_accuracy(
            edited_texts, target_label=target_attribute,
            classifier_name=self.classifier_name, device=self.device
        )
        results["edit_quality"] = compute_edit_quality(source_texts, edited_texts)
        results["diversity"] = compute_diversity_metrics(edited_texts)

        return results

    def evaluate_latent_geometry(self, model, sampler, latent_pairs, labels):
        """
        Evaluate latent space structure.

        Args:
            model: MultimodalMMDiT
            sampler: sampler with .generate()
            latent_pairs: list of (z_a, z_b) tuples for SSS/MTS
            labels: (N,) attribute labels for cluster separation

        Returns dict with sss, mts, cluster_separation.
        """
        results = {}

        # SSS: average over pairs
        sss_scores = []
        for z_a, z_b in latent_pairs:
            sss, _ = semantic_smoothness_score(
                model, sampler, z_a, z_b, device=self.device
            )
            sss_scores.append(sss)
        results["sss_mean"] = float(sum(sss_scores) / len(sss_scores)) if sss_scores else 0.0

        return results

    def full_evaluation(self, source_texts, edited_texts, target_attribute,
                        model=None, sampler=None, latent_pairs=None, labels=None):
        """
        Run all applicable pillars.

        Returns combined results dict.
        """
        results = self.evaluate_editing(source_texts, edited_texts, target_attribute)

        if model is not None and sampler is not None and latent_pairs is not None:
            results["latent_geometry"] = self.evaluate_latent_geometry(
                model, sampler, latent_pairs, labels
            )

        return results

    def save_results(self, results, output_path):
        """Save results as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable values
        def _convert(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return obj

        clean = json.loads(json.dumps(results, default=_convert))
        with open(output_path, "w") as f:
            json.dump(clean, f, indent=2)

    def print_summary(self, results):
        """Print a human-readable summary table."""
        print("\n" + "=" * 60)
        print("DLM-Eval Suite Results")
        print("=" * 60)

        for pillar, metrics in results.items():
            print(f"\n--- {pillar.upper()} ---")
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if "per_sample" not in k and not isinstance(v, list):
                        print(f"  {k}: {v}")
            else:
                print(f"  {metrics}")

        print("\n" + "=" * 60)
