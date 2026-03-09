"""
Latent geometry metrics — Pillar 4 of DLM-Eval Suite (NOVEL).

Solves gap: Gap 6 — latent space geometry unexplored in DLMs.
Inspired by: Cosmos (smooth latent diffusion), LatentOps (ODE traversal), Optimus (text VAE)

Metrics:
  - Semantic Smoothness Score (SSS)
  - Monotonic Transition Score (MTS)
  - Cluster Separation (silhouette score)
  - Interpolation Fluency
  - Latent Variance Ratio
"""

import numpy as np
import torch

from lsme.latent_utils.interpolation import slerp, interpolation_path


def semantic_smoothness_score(model, sampler, z_a, z_b,
                              n_points=10, n_samples=5,
                              sentence_encoder=None, device="cuda"):
    """
    SSS: measure smoothness of semantic trajectory between two latents.

    SSS = (1/(N-1)) * sum_{i=1}^{N-1} sim(text_i, text_{i+1})

    SSS -> 1.0 = smooth (adjacent interp. points produce similar texts).
    SSS -> 0.0 = rough (texts change abruptly between adjacent points).

    Args:
        model: MultimodalMMDiT
        sampler: a sampler with .generate(latents=...) method
        z_a, z_b: (D,) latent vectors
        n_points: int, interpolation steps
        n_samples: int, samples per interpolation point (variance reduction)
        sentence_encoder: optional pre-loaded SentenceTransformer
        device: str

    Returns:
        sss: float in [0, 1]
        trajectory: list of (alpha, texts, embedding) tuples
    """
    if sentence_encoder is None:
        from sentence_transformers import SentenceTransformer
        sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    path, alphas = interpolation_path(z_a.to(device), z_b.to(device),
                                      n_points=n_points, method="slerp")
    trajectory = []
    embeddings = []

    for i, (alpha, z_interp) in enumerate(zip(alphas, path)):
        # Generate texts conditioned on this latent
        z_batch = z_interp.unsqueeze(0).expand(n_samples, -1)  # (n_samples, D)
        texts = sampler.generate(
            num_samples=n_samples, latents=z_batch,
            decode=True, show_progress=False
        )

        # Encode texts and average
        emb = sentence_encoder.encode(texts, convert_to_tensor=True)  # (n_samples, E)
        mean_emb = emb.mean(dim=0)  # (E,)
        embeddings.append(mean_emb)
        trajectory.append((alpha.item(), texts, mean_emb.cpu()))

    # Compute consecutive cosine similarities
    sims = []
    for i in range(len(embeddings) - 1):
        cos_sim = torch.nn.functional.cosine_similarity(
            embeddings[i].unsqueeze(0), embeddings[i + 1].unsqueeze(0)
        ).item()
        sims.append(cos_sim)

    sss = float(np.mean(sims)) if sims else 0.0
    return sss, trajectory


def monotonic_transition_score(model, sampler, z_source, z_target,
                               classifier, target_label,
                               n_points=10, n_samples=5, device="cuda"):
    """
    MTS: measure whether classifier confidence changes monotonically
    along an interpolation path.

    MTS = (1/(N-1)) * sum_{i=1}^{N-1} 1[c(text_{i+1}) >= c(text_i)]

    MTS = 1.0 = perfect monotonic increase (ideal latent structure).
    MTS = 0.5 = random walk (no structure).

    Args:
        model: MultimodalMMDiT
        sampler: a sampler with .generate(latents=...) method
        z_source, z_target: (D,) latent vectors (source attribute -> target attribute)
        classifier: callable or HF pipeline, returns P(target | text)
        target_label: str, the target attribute label
        n_points: int, interpolation steps
        n_samples: int, samples per point
        device: str

    Returns:
        mts: float in [0, 1]
        scores: list of classifier scores along interpolation
    """
    from lsme.evaluation.metrics.controllability import compute_attribute_scores

    path, alphas = interpolation_path(z_source.to(device), z_target.to(device),
                                      n_points=n_points, method="slerp")
    scores = []

    for z_interp in path:
        z_batch = z_interp.unsqueeze(0).expand(n_samples, -1)
        texts = sampler.generate(
            num_samples=n_samples, latents=z_batch,
            decode=True, show_progress=False
        )

        point_scores = compute_attribute_scores(
            texts, classifier=classifier, target_label=target_label
        )
        scores.append(float(np.mean(point_scores)))

    # Count monotonic increases
    monotonic = sum(
        1 for i in range(len(scores) - 1) if scores[i + 1] >= scores[i]
    )
    mts = monotonic / (len(scores) - 1) if len(scores) > 1 else 0.0

    return mts, scores


def cluster_separation(latents, labels):
    """
    Compute silhouette score measuring how well latents cluster by attribute.

    Args:
        latents: (N, D) tensor of latent vectors
        labels: (N,) integer labels

    Returns:
        silhouette: float in [-1, 1], higher = better separation
    """
    from sklearn.metrics import silhouette_score

    if isinstance(latents, torch.Tensor):
        latents = latents.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(labels):
        return 0.0

    return float(silhouette_score(latents, labels))


def latent_variance_ratio(latents, labels):
    """
    Compute ratio of between-class variance to within-class variance.

    Higher ratio = attributes explain more of latent variance.

    Args:
        latents: (N, D) tensor
        labels: (N,) integer or string labels

    Returns:
        ratio: float
    """
    if isinstance(latents, torch.Tensor):
        latents = latents.cpu().numpy()

    unique_labels = np.unique(labels)
    overall_mean = latents.mean(axis=0)

    between_var = 0.0
    within_var = 0.0

    for label in unique_labels:
        mask = np.array(labels) == label
        group = latents[mask]
        group_mean = group.mean(axis=0)
        n = len(group)

        between_var += n * np.sum((group_mean - overall_mean) ** 2)
        within_var += np.sum((group - group_mean) ** 2)

    if within_var < 1e-10:
        return float("inf")

    return float(between_var / within_var)


def interpolation_fluency(model, sampler, z_a, z_b, n_points=10,
                          ppl_model="gpt2", device="cuda"):
    """
    Mean PPL of texts generated along an interpolation path.

    Low fluency at intermediate points suggests latent space has "holes."

    Args:
        model: MultimodalMMDiT
        sampler: sampler with .generate()
        z_a, z_b: (D,) latent vectors
        n_points: int
        ppl_model: str, model name for PPL computation
        device: str

    Returns:
        mean_ppl: float
        ppls: list of float, PPL at each interpolation point
    """
    from lsme.evaluation.metrics.fluency import compute_perplexity

    path, alphas = interpolation_path(z_a.to(device), z_b.to(device),
                                      n_points=n_points, method="slerp")
    ppls = []

    for z_interp in path:
        texts = sampler.generate(
            num_samples=5, latents=z_interp.unsqueeze(0).expand(5, -1),
            decode=True, show_progress=False
        )
        result = compute_perplexity(texts, model_name=ppl_model, device=device)
        ppls.append(result["ppl_mean"])

    return float(np.mean(ppls)), ppls
