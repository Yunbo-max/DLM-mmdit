"""
Entry point: Full DLM-Eval Suite evaluation.

Runs the 6-pillar evaluation on LSME editing results.

Usage:
    python -m lsme.scripts.run_eval \
        --results_file results/lsme_output.json \
        --output_dir results/eval/ \
        --device cuda
"""

import argparse
import json
from pathlib import Path

from lsme.evaluation.eval_suite import DLMEvalSuite


def main():
    parser = argparse.ArgumentParser(description="DLM-Eval Suite")
    parser.add_argument("--results_file", type=str, required=True,
                        help="JSON file from run_lsme.py")
    parser.add_argument("--output_dir", type=str, default="results/eval/",
                        help="Output directory for eval results")
    parser.add_argument("--fluency_model", type=str, default="gpt2")
    parser.add_argument("--classifier_name", type=str,
                        default="distilbert-base-uncased-finetuned-sst-2-english")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load LSME results
    with open(args.results_file) as f:
        results_data = json.load(f)

    source_texts = [r["source"] for r in results_data]
    edited_texts = [r["edited"] for r in results_data]
    target_attribute = results_data[0].get("target_value", "POSITIVE")

    print(f"Evaluating {len(edited_texts)} edited texts...")
    print(f"Target attribute: {target_attribute}")

    # Run evaluation
    eval_suite = DLMEvalSuite({
        "device": args.device,
        "fluency_model": args.fluency_model,
        "classifier_name": args.classifier_name,
    })

    eval_results = eval_suite.evaluate_editing(
        source_texts, edited_texts, target_attribute
    )

    # Save and print
    output_dir = Path(args.output_dir)
    eval_suite.save_results(eval_results, output_dir / "eval_results.json")
    eval_suite.print_summary(eval_results)

    print(f"\nResults saved to {output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
