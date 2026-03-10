"""
Controllability metrics — Pillar 2 of DLM-Eval Suite.

Measures: Classifier accuracy and attribute match rate.
"""

import numpy as np
import torch


def compute_attribute_accuracy(texts, target_label, classifier_name=None,
                               classifier=None, device="cuda"):
    """
    Measure fraction of texts classified as the target attribute.

    Args:
        texts: list of str, generated/edited texts
        target_label: str or int, the desired attribute label
        classifier_name: str, HuggingFace classifier model name
            (e.g. "distilbert-base-uncased-finetuned-sst-2-english")
        classifier: optional pre-loaded classifier pipeline
        device: str

    Returns:
        dict with "accuracy", "confidence_mean", "confidence_std",
              "predictions", "confidences"
    """
    if classifier is None:
        from transformers import pipeline
        classifier = pipeline("text-classification", model=classifier_name,
                              device=device, truncation=True, max_length=512)

    results = classifier(texts, batch_size=32)

    predictions = []
    confidences = []
    correct = 0

    for r in results:
        pred_label = r["label"]
        pred_score = r["score"]
        predictions.append(pred_label)
        confidences.append(pred_score)

        # Handle both string and int label matching
        if str(pred_label).lower() == str(target_label).lower():
            correct += 1

    accuracy = correct / len(texts) if texts else 0.0

    return {
        "accuracy": accuracy,
        "confidence_mean": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)),
        "predictions": predictions,
        "confidences": confidences,
    }


def compute_attribute_scores(texts, classifier=None, classifier_name=None,
                             target_label=None, device="cuda"):
    """
    Get raw classifier probability for the target attribute per text.

    Useful for MTS (Monotonic Transition Score) computation.

    Args:
        texts: list of str
        classifier: optional pre-loaded pipeline
        classifier_name: str, HuggingFace model name
        target_label: str, the label to get probability for
        device: str

    Returns:
        scores: list of float, P(target_label | text) for each text
    """
    if classifier is None:
        from transformers import pipeline
        classifier = pipeline("text-classification", model=classifier_name,
                              device=device, truncation=True, max_length=512,
                              top_k=None)

    results = classifier(texts, batch_size=32)

    scores = []
    for r in results:
        label_scores = {item["label"]: item["score"] for item in r}
        score = label_scores.get(target_label, 0.0)
        scores.append(score)

    return scores
