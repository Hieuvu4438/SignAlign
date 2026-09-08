from __future__ import annotations

import torch


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


def apply_lie_residual(
    baseline_rotation: torch.Tensor,
    delta: torch.Tensor,
    radius_rad: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = torch.as_tensor(radius_rad, dtype=delta.dtype, device=delta.device)
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-8)
    bounded = delta * torch.clamp(radius / norm, max=1.0)
    return so3_exp_map(bounded) @ baseline_rotation, bounded
