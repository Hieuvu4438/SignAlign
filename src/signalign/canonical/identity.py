"""Robust signer identity utilities used by the canonical reconstruction."""

from __future__ import annotations

import numpy as np


def huber_location(
    values: np.ndarray,
    delta: float = 1.5,
    iterations: int = 10,
) -> np.ndarray:
    """Estimate a robust location independently for every shape coefficient."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"invalid identity samples: {values.shape}")
    estimate = np.median(values, axis=0)
    scale = np.median(np.abs(values - estimate), axis=0) * 1.4826 + 1e-6
    for _ in range(iterations):
        residual = (values - estimate) / scale
        weight = np.minimum(1.0, delta / (np.abs(residual) + 1e-8))
        estimate = (weight * values).sum(axis=0) / np.maximum(
            weight.sum(axis=0), 1e-8
        )
    return estimate.astype(np.float32)


def farthest_point_indices(features: np.ndarray, count: int) -> np.ndarray:
    """Select a deterministic pose-diverse subset after feature normalization."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError(f"invalid pose features: {features.shape}")
    count = min(int(count), len(features))
    if count <= 0:
        raise ValueError("count must be positive")
    scale = features.std(axis=0)
    normalized = (features - features.mean(axis=0)) / np.where(
        scale > 1e-8, scale, 1.0
    )
    selected = [int(np.argmax(np.square(normalized).sum(axis=1)))]
    nearest = np.square(normalized - normalized[selected[0]]).sum(axis=1)
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(
            nearest,
            np.square(normalized - normalized[index]).sum(axis=1),
        )
    return np.asarray(selected, dtype=np.int64)
