"""SO(3) maps and the proper, scale-normalized palm coordinate frame."""
from __future__ import annotations

import torch


def _unit(vector: torch.Tensor):
    norm = torch.linalg.vector_norm(vector, dim=-1)
    finite = torch.isfinite(vector).all(dim=-1) & torch.isfinite(norm)
    safe = torch.nan_to_num(vector) / torch.nan_to_num(norm, nan=0.0).clamp_min(1e-6)[..., None]
    return safe, finite & (norm > 1e-6)


def palm_canonical(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize wrist-first thumb/index/middle/ring/little joints.

    Return canonical coordinates and per-hand geometry validity. Rows are
    points; columns of the proper palm basis are its local axes.
    """
    centered = joints - joints[..., :1, :]
    finite = torch.isfinite(joints).flatten(-2).all(dim=-1)
    transverse, transverse_valid = _unit(centered[..., 5, :] - centered[..., 17, :])
    longitudinal_raw = 0.5 * (centered[..., 5, :] + centered[..., 17, :])
    longitudinal, longitudinal_valid = _unit(
        longitudinal_raw - (longitudinal_raw * transverse).sum(-1, keepdim=True) * transverse
    )
    normal, normal_valid = _unit(torch.cross(transverse, longitudinal, dim=-1))
    longitudinal = torch.cross(normal, transverse, dim=-1)
    frame = torch.stack((transverse, longitudinal, normal), dim=-1)
    determinant = torch.det(frame)
    scale = torch.linalg.vector_norm(centered[..., 9, :], dim=-1)
    coordinates = torch.nan_to_num(centered) @ torch.nan_to_num(frame)
    coordinates = coordinates / torch.nan_to_num(scale, nan=0.0).clamp_min(1e-6)[..., None, None]
    valid = (finite & transverse_valid & longitudinal_valid & normal_valid
             & torch.isfinite(scale) & (scale > 1e-6) & torch.isfinite(determinant)
             & ((determinant - 1.0).abs() <= 1e-3))
    return torch.nan_to_num(coordinates), valid


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def so3_exp_map(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    K = skew(axis_angle)
    angle2 = angle.square()
    small = angle < 1e-4
    a = torch.where(
        small,
        1.0 - angle2 / 6.0 + angle2.square() / 120.0,
        torch.sin(angle) / angle.clamp_min(1e-12),
    )
    b = torch.where(
        small,
        0.5 - angle2 / 24.0 + angle2.square() / 720.0,
        (1.0 - torch.cos(angle)) / angle2.clamp_min(1e-12),
    )
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(axis_angle.shape[:-1] + (3, 3))
    return eye + a[..., None] * K + b[..., None] * (K @ K)


def so3_log_map(rotation: torch.Tensor) -> torch.Tensor:
    """Differentiable matrix-to-rotvec map for proper rotations away from pi."""
    vector = 0.5 * torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    cos_angle = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5)
    cos_angle = cos_angle.clamp(-1.0, 1.0).unsqueeze(-1)
    angle = torch.atan2(sin_angle, cos_angle)
    factor = torch.where(
        sin_angle < 1e-5,
        1.0 + angle.square() / 6.0,
        angle / sin_angle.clamp_min(1e-8),
    )
    result = vector * factor
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("SO(3) logarithm produced non-finite rotvec")
    return result
