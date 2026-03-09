"""
Edit quality metrics — Pillar 3 of DLM-Eval Suite.

Measures: BLEU, ROUGE-L, BERTScore, edit distance.
"""

import numpy as np


def compute_edit_quality(source_texts, edited_texts):
    """
    Compute edit quality metrics between source and edited texts.

    Args:
        source_texts: list of str, original texts
        edited_texts: list of str, edited texts

    Returns:
        dict with "bleu", "rouge_l", "bertscore_f1", "edit_distance",
              and per-sample breakdowns
    """
    results = {}

    # BLEU
    bleu_scores = _compute_bleu(source_texts, edited_texts)
    results["bleu_mean"] = float(np.mean(bleu_scores))
    results["bleu_per_sample"] = bleu_scores

    # ROUGE-L
    rouge_scores = _compute_rouge_l(source_texts, edited_texts)
    results["rouge_l_mean"] = float(np.mean(rouge_scores))
    results["rouge_l_per_sample"] = rouge_scores

    # BERTScore
    bert_scores = _compute_bertscore(source_texts, edited_texts)
    results["bertscore_f1_mean"] = float(np.mean(bert_scores))
    results["bertscore_f1_per_sample"] = bert_scores

    # Edit distance (normalized)
    edit_dists = _compute_edit_distance(source_texts, edited_texts)
    results["edit_distance_mean"] = float(np.mean(edit_dists))
    results["edit_distance_per_sample"] = edit_dists

    return results


def _compute_bleu(references, hypotheses):
    """Sentence-level BLEU between each (reference, hypothesis) pair."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return [0.0] * len(references)

    smoother = SmoothingFunction().method1
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        if not hyp_tokens:
            scores.append(0.0)
            continue
        score = sentence_bleu([ref_tokens], hyp_tokens,
                              smoothing_function=smoother)
        scores.append(score)
    return scores


def _compute_rouge_l(references, hypotheses):
    """ROUGE-L F1 between each (reference, hypothesis) pair."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue
        lcs_len = _lcs_length(ref_tokens, hyp_tokens)
        precision = lcs_len / len(hyp_tokens)
        recall = lcs_len / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            scores.append(f1)
    return scores


def _lcs_length(x, y):
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _compute_bertscore(references, hypotheses):
    """BERTScore F1 between each pair."""
    try:
        from evaluate import load
        bertscore = load("bertscore")
        results = bertscore.compute(
            predictions=hypotheses, references=references, lang="en"
        )
        return results["f1"]
    except ImportError:
        return [0.0] * len(references)


def _compute_edit_distance(source_texts, edited_texts):
    """Normalized Levenshtein edit distance (character-level)."""
    scores = []
    for src, edt in zip(source_texts, edited_texts):
        dist = _levenshtein(src, edt)
        max_len = max(len(src), len(edt), 1)
        scores.append(dist / max_len)
    return scores


def _levenshtein(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]
