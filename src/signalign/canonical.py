"""Restricted canonical pose refitting under the shared signer shape."""
from __future__ import annotations

import numpy as np
import torch

from .identity import centered_hand_loss
from .model import BOUNDARY, Decoder


def canonicalize(
    decoder: Decoder, parameters: dict[str, np.ndarray], initializer_vertices: np.ndarray,
    shared_shape: np.ndarray, hand_vertex_ids: tuple[torch.Tensor, torch.Tensor],
    clip_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Refit shoulder/elbow/wrist and finger poses; freeze all other variables."""
    result = {key: value.copy() for key, value in parameters.items()}
    result["betas"][:] = shared_shape
    meshes = np.empty_like(initializer_vertices)
    mask = torch.zeros((1, 63), dtype=torch.float32, device=decoder.device)
    mask[:, 15 * 3:21 * 3] = 1.0
    beta = decoder.tensor(shared_shape.reshape(1, 10))
    for clip in sorted(set(clip_ids)):
        indices = np.flatnonzero(clip_ids == clip)
        for start in range(0, len(indices), 32):
            ids = indices[start:start + 32]
            values = decoder.parameters({key: value[ids] for key, value in parameters.items()})
            reference = decoder.tensor(initializer_vertices[ids] * BOUNDARY)
            body = torch.nn.Parameter(torch.zeros_like(values["body_pose"]))
            left = torch.nn.Parameter(torch.zeros_like(values["left_hand_pose"]))
            right = torch.nn.Parameter(torch.zeros_like(values["right_hand_pose"]))
            optimizer = torch.optim.Adam((body, left, right), lr=0.01)

            def forward():
                current = dict(values)
                current.update(betas=beta.expand(len(ids), -1),
                               body_pose=values["body_pose"] + body * mask,
                               left_hand_pose=values["left_hand_pose"] + left,
                               right_hand_pose=values["right_hand_pose"] + right)
                return decoder.model(**current, return_verts=True).vertices

            best_loss, best, stale = float("inf"), None, 0
            for _ in range(300):
                optimizer.zero_grad(set_to_none=True)
                vertices = forward()
                loss = (centered_hand_loss(vertices, reference, hand_vertex_ids)
                        + 0.02 * torch.square(vertices - reference).mean())
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite canonical pose objective")
                loss.backward()
                optimizer.step()
                current = float(loss.detach())
                if current < best_loss - 1e-10:
                    best_loss = current
                    best = tuple(x.detach().clone() for x in (body, left, right))
                    stale = 0
                else:
                    stale += 1
                if stale >= 15:
                    break
            if best is None:
                raise RuntimeError("No finite canonical pose solution")
            with torch.no_grad():
                for target, value in zip((body, left, right), best):
                    target.copy_(value)
                vertices = forward()
                for hand_ids in hand_vertex_ids:
                    prediction, target = vertices[:, hand_ids], reference[:, hand_ids]
                    residual = torch.linalg.vector_norm(
                        prediction - prediction.mean(1, keepdim=True)
                        - target + target.mean(1, keepdim=True), dim=-1,
                    ).mean() * 1000
                    if float(residual) > 8.0:
                        raise RuntimeError("Canonical hand residual exceeds 8 mm")
                meshes[ids] = (vertices * decoder.boundary).cpu().numpy()
                result["body_pose"][ids] = (values["body_pose"] + body * mask).cpu().numpy()
                result["left_hand_pose"][ids] = (values["left_hand_pose"] + left).cpu().numpy()
                result["right_hand_pose"][ids] = (values["right_hand_pose"] + right).cpu().numpy()
    return result, meshes
