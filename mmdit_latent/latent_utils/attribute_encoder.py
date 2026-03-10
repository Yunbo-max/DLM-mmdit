"""
Attribute Latent Encoder — encode target attributes into latent vectors for LSME.

Uses the model's own training data to extract attribute-conditioned latent statistics.
No retraining needed: the latent space is assumed to already cluster by attributes
due to joint training with MMDiT.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .interpolation import slerp


class AttributeLatentEncoder:
    """
    Encode target attributes into latent vectors z_target for LSME conditioning.

    Loads pre-computed latent vectors and their metadata (attribute labels),
    computes per-attribute centroids, and provides target latent retrieval.

    Input:
        latent_dir:    str — directory containing .npy latent files
        metadata_file: str — JSON mapping filenames to attributes

    Output:
        z_target: (D,) float32 — target latent centroid
    """

    def __init__(self, latent_dir, metadata_file):
        """
        Load pre-computed latent vectors and their metadata.

        Args:
            latent_dir: Directory containing .npy latent files from training data
            metadata_file: JSON/CSV mapping filenames to attributes
                e.g., {"sample_001.npy": {"sentiment": "positive", "topic": "food"}}
        """
        self.latent_dir = Path(latent_dir)
        self.metadata = self._load_metadata(metadata_file)
        self.latents = {}  # filename -> tensor
        self.centroids = {}  # attribute_name -> {value -> tensor}

        self._load_latents()

    def _load_metadata(self, metadata_file):
        """Load attribute metadata from JSON."""
        with open(metadata_file) as f:
            return json.load(f)

    def _load_latents(self):
        """Load all .npy latent files referenced in metadata."""
        for filename in self.metadata:
            path = self.latent_dir / filename
            if path.exists():
                self.latents[filename] = torch.from_numpy(
                    np.load(str(path))
                ).float()

    def compute_attribute_centroids(self, attribute_name):
        """
        Compute the mean latent vector for each value of the given attribute.

        Args:
            attribute_name: str, e.g. "sentiment", "topic", "formality"

        Returns:
            centroids: dict[str, Tensor], e.g. {"positive": tensor(D,), "negative": tensor(D,)}
        """
        groups = {}  # value -> list of tensors

        for filename, attrs in self.metadata.items():
            if attribute_name not in attrs:
                continue
            if filename not in self.latents:
                continue
            value = attrs[attribute_name]
            if value not in groups:
                groups[value] = []
            groups[value].append(self.latents[filename])

        centroids = {}
        for value, tensors in groups.items():
            stacked = torch.stack(tensors)  # (N, D)
            centroids[value] = stacked.mean(dim=0)  # (D,)

        self.centroids[attribute_name] = centroids
        return centroids

    def get_target_latent(self, attribute_name, target_value, device=None):
        """
        Return the centroid latent for the target attribute value.

        Args:
            attribute_name: str, e.g. "sentiment"
            target_value: str, e.g. "positive"
            device: optional torch device

        Returns:
            z_target: Tensor (D,)
        """
        if attribute_name not in self.centroids:
            self.compute_attribute_centroids(attribute_name)

        z = self.centroids[attribute_name][target_value]
        if device is not None:
            z = z.to(device)
        return z

    def get_directional_target(self, attribute_name, source_value, target_value,
                               z_source, alpha=1.0, device=None):
        """
        Compute directional edit latent: z_source + alpha * (z_target - z_source_centroid).

        Preserves source content better than hard centroid replacement.

        Args:
            attribute_name: str
            source_value: str, source attribute value
            target_value: str, target attribute value
            z_source: (D,) source sample latent
            alpha: float, edit strength
            device: optional torch device

        Returns:
            z_target: (D,) edited latent
        """
        z_src_centroid = self.get_target_latent(attribute_name, source_value, device)
        z_tgt_centroid = self.get_target_latent(attribute_name, target_value, device)
        if device is not None:
            z_source = z_source.to(device)

        direction = z_tgt_centroid - z_src_centroid
        return z_source + alpha * direction

    def get_nearest_neighbor(self, attribute_name, target_value, z_query, device=None):
        """
        Return the nearest real training latent in the target class.

        Stays on the data manifold — more realistic than mean centroid.

        Args:
            attribute_name: str
            target_value: str
            z_query: (D,) query latent
            device: optional torch device

        Returns:
            z_nn: (D,) nearest neighbor latent
        """
        candidates = []
        for filename, attrs in self.metadata.items():
            if attrs.get(attribute_name) == target_value and filename in self.latents:
                candidates.append(self.latents[filename])

        if not candidates:
            return self.get_target_latent(attribute_name, target_value, device)

        candidates = torch.stack(candidates)  # (N, D)
        if device is not None:
            candidates = candidates.to(device)
            z_query = z_query.to(device)

        dists = torch.cdist(z_query.unsqueeze(0), candidates).squeeze(0)  # (N,)
        nn_idx = dists.argmin()
        return candidates[nn_idx]

    def interpolate(self, z_source, z_target, alpha):
        """
        Spherical linear interpolation between two latents.

        Args:
            z_source: (D,) source latent
            z_target: (D,) target latent
            alpha: float in [0, 1]

        Returns:
            z_interp: (D,) interpolated latent
        """
        return slerp(z_source, z_target, alpha)

    def list_attributes(self):
        """Return all attribute names found in metadata."""
        attrs = set()
        for meta in self.metadata.values():
            attrs.update(meta.keys())
        return sorted(attrs)

    def list_values(self, attribute_name):
        """Return all values for a given attribute."""
        values = set()
        for meta in self.metadata.values():
            if attribute_name in meta:
                values.add(meta[attribute_name])
        return sorted(values)
