"""Robust shape initialization and pose-diverse signer calibration."""
from __future__ import annotations

import numpy as np
import torch

from .model import BOUNDARY, Decoder


def huber_location(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    estimate = np.median(values, axis=0)
    scale = np.median(np.abs(values - estimate), axis=0) * 1.4826 + 1e-6
    for _ in range(10):
        residual = (values - estimate) / scale
        weight = np.minimum(1.0, 1.5 / (np.abs(residual) + 1e-8))
        estimate = (weight * values).sum(axis=0) / np.maximum(weight.sum(axis=0), 1e-8)
    return estimate.astype(np.float32)


def calibration_indices(parameters: dict[str, np.ndarray], count: int = 200) -> np.ndarray:
    """Deterministic farthest-point sampling in standardized pose space."""
    features = np.concatenate((
        parameters["body_pose"].reshape(-1, 21, 3)[:, 15:21].reshape(-1, 18),
        parameters["left_hand_pose"], parameters["right_hand_pose"],
    ), axis=1).astype(np.float64)
    count = min(int(count), len(features))
    if count < 1:
        raise ValueError("calibration_frames must be positive")
    scale = features.std(axis=0)
    normalized = (features - features.mean(axis=0)) / np.where(scale > 1e-8, scale, 1.0)
    selected = [int(np.argmax(np.square(normalized).sum(axis=1)))]
    nearest = np.square(normalized - normalized[selected[0]]).sum(axis=1)
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.square(normalized - normalized[index]).sum(axis=1))
    return np.asarray(selected, dtype=np.int64)


def centered_hand_loss(vertices, reference, hand_vertex_ids):
    """Mean coordinate-wise MSE after independent hand-centroid subtraction."""
    terms = []
    for ids in hand_vertex_ids:
        prediction, target = vertices[:, ids], reference[:, ids]
        terms.append(torch.square(
            prediction - prediction.mean(dim=1, keepdim=True)
            - target + target.mean(dim=1, keepdim=True)
        ).mean())
    return 0.5 * (terms[0] + terms[1])


def fit_identity(
    decoder: Decoder, parameters: dict[str, np.ndarray],
    initializer_vertices: np.ndarray, hand_vertex_ids: tuple[torch.Tensor, torch.Tensor],
    calibration_frames: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit shared shape with temporary per-frame finger nuisance residuals."""
    selected = calibration_indices(parameters, calibration_frames)
    observed = {key: value[selected] for key, value in parameters.items()}
    tensors = decoder.parameters(observed)
    reference = decoder.tensor(initializer_vertices[selected] * BOUNDARY)
    initial = decoder.tensor(huber_location(observed["betas"])[None])
    beta = torch.nn.Parameter(initial.clone())
    left = torch.nn.Parameter(torch.zeros_like(tensors["left_hand_pose"]))
    right = torch.nn.Parameter(torch.zeros_like(tensors["right_hand_pose"]))
    optimizer = torch.optim.Adam((beta, left, right), lr=0.01)
    best_loss, best_beta = float("inf"), None
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        values = dict(tensors)
        values.update(betas=beta.expand(len(selected), -1),
                      left_hand_pose=tensors["left_hand_pose"] + left,
                      right_hand_pose=tensors["right_hand_pose"] + right)
        vertices = decoder.model(**values, return_verts=True).vertices
        loss = (centered_hand_loss(vertices, reference, hand_vertex_ids)
                + 0.02 * torch.square(vertices - reference).mean()
                + 1e-4 * (left.square().mean() + right.square().mean()))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite signer-shape objective")
        loss.backward()
        optimizer.step()
        current = float(loss.detach())
        if current < best_loss:
            best_loss, best_beta = current, beta.detach().clone()
    if best_beta is None:
        raise RuntimeError("No finite signer-shape solution")
    return best_beta.cpu().numpy().reshape(10), selected
