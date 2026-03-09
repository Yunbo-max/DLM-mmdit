"""
Diversity metrics — Pillar 5 of DLM-Eval Suite.

Measures: Self-BLEU, Distinct-1/2/3.

Distinct-N adapted from ConGenBench
(https://github.com/princeton-nlp/ConGenBench) evaluation/all_metrics.py
with both corpus-level and per-sentence variants.
"""

import numpy as np
from collections import Counter


def compute_diversity_metrics(texts):
    """
    Compute diversity metrics for a set of generated texts.

    Args:
        texts: list of str

    Returns:
        dict with "self_bleu", "distinct_1/2/3" (corpus-level),
        and "distinct_1/2/3_per_sentence" (per-sentence, ConGenBench-style)
    """
    results = {}

    # Self-BLEU (lower = more diverse)
    results["self_bleu"] = _compute_self_bleu(texts)

    # Distinct-n corpus-level (higher = more diverse)
    for n in [1, 2, 3]:
        results[f"distinct_{n}"] = _compute_distinct_n(texts, n)

    # Distinct-n per-sentence (ConGenBench style: average per-sentence diversity)
    for n in [1, 2, 3]:
        results[f"distinct_{n}_per_sentence"] = _compute_distinct_n_per_sentence(
            texts, n
        )

    return results


def _compute_self_bleu(texts, max_pairs=1000):
    """
    Self-BLEU: average BLEU of each text against all others.
    Lower = more diverse.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return None

    if len(texts) < 2:
        return 0.0

    smoother = SmoothingFunction().method1
    tokenized = [t.split() for t in texts]

    scores = []
    n = len(tokenized)
    # Subsample pairs if too many
    if n * (n - 1) > max_pairs:
        indices = np.random.choice(n, size=min(n, int(np.sqrt(max_pairs * 2))),
                                   replace=False)
    else:
        indices = range(n)

    for i in indices:
        refs = [tokenized[j] for j in range(n) if j != i]
        if not tokenized[i]:
            continue
        score = sentence_bleu(refs, tokenized[i], smoothing_function=smoother)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def _compute_distinct_n(texts, n):
    """
    Distinct-n (corpus-level): ratio of unique n-grams to total n-grams
    across all texts. Higher = more diverse.
    """
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / total


def _compute_distinct_n_per_sentence(texts, n):
    """
    Distinct-n (per-sentence): compute distinct-n for each sentence
    then average. Adapted from ConGenBench evaluation/all_metrics.py.
    """
    scores = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            continue
        scores.append(len(set(ngrams)) / len(ngrams))
    return float(np.mean(scores)) if scores else 0.0
