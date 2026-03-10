"""
Entry point: Run baseline methods through the same DLM-Eval Suite.

Baselines:
  - MDLM (unconditional generation — no editing)
  - ReMDM (remasking — editing without attribute control)
  - LatentOps (latent-space style transfer — continuous)
  - DiffusER (edit-based diffusion)

Each baseline is expected to produce edited/generated texts,
which are then evaluated with the same 6-pillar suite.

Usage:
    python -m lsme.scripts.run_baselines \
        --baseline_results_dir results/baselines/ \
        --output_dir results/comparison/ \
        --device cuda
"""

import argparse
import json
from pathlib import Path

from mmdit_latent.evaluation.eval_suite import DLMEvalSuite


def evaluate_baseline_outputs(results_dir, eval_suite, target_attribute):
    """
    Evaluate pre-generated baseline outputs.

    Expects each baseline to have a JSON file with:
    [{"source": "...", "edited": "..."}, ...]

    Args:
        results_dir: Path with baseline JSON files
        eval_suite: DLMEvalSuite instance
        target_attribute: str

    Returns:
        comparison: dict mapping baseline_name -> eval results
    """
    results_dir = Path(results_dir)
    comparison = {}

    for json_file in sorted(results_dir.glob("*.json")):
        baseline_name = json_file.stem
        print(f"\nEvaluating baseline: {baseline_name}")

        with open(json_file) as f:
            data = json.load(f)

        source_texts = [r.get("source", "") for r in data]
        edited_texts = [r.get("edited", r.get("generated", "")) for r in data]

        if not edited_texts or not edited_texts[0]:
            print(f"  Skipping {baseline_name}: no texts found")
            continue

        eval_results = eval_suite.evaluate_editing(
            source_texts, edited_texts, target_attribute
        )
        comparison[baseline_name] = eval_results

        acc = eval_results.get("controllability", {}).get("accuracy", "N/A")
        ppl = eval_results.get("fluency", {}).get("ppl_mean", "N/A")
        print(f"  Accuracy: {acc}, PPL: {ppl}")

    return comparison


def print_comparison_table(comparison):
    """Print a unified comparison table."""
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'Attr-Acc':>10} {'PPL':>10} {'BLEU':>10} "
          f"{'Distinct-1':>12} {'Self-BLEU':>12}")
    print("-" * 80)

    for method, results in comparison.items():
        acc = results.get("controllability", {}).get("accuracy", 0)
        ppl = results.get("fluency", {}).get("ppl_mean", 0)
        bleu = results.get("edit_quality", {}).get("bleu_mean", 0)
        d1 = results.get("diversity", {}).get("distinct_1", 0)
        sb = results.get("diversity", {}).get("self_bleu", 0)
        print(f"{method:<20} {acc:>10.3f} {ppl:>10.1f} {bleu:>10.3f} "
              f"{d1:>12.3f} {sb:>12.3f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Baseline Comparison")
    parser.add_argument("--baseline_results_dir", type=str, required=True,
                        help="Directory containing baseline output JSON files")
    parser.add_argument("--output_dir", type=str, default="results/comparison/")
    parser.add_argument("--target_attribute", type=str, default="POSITIVE")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    eval_suite = DLMEvalSuite({
        "device": args.device,
        "fluency_model": "gpt2",
        "classifier_name": "distilbert-base-uncased-finetuned-sst-2-english",
    })

    comparison = evaluate_baseline_outputs(
        args.baseline_results_dir, eval_suite, args.target_attribute
    )

    print_comparison_table(comparison)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nComparison saved to {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
