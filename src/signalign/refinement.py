"""Wrist-locked palm-canonical refinement with a hard 12-degree finger bound."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import palm_canonical, so3_exp_map
from .model import Decoder, HAND_JOINTS


def _fit_side(decoder, values, rotations, baseline, expert, available, side):
    base = rotations[side]
    joint_ids = torch.as_tensor(HAND_JOINTS[side], device=decoder.device)
    target, target_valid = palm_canonical(expert)
    _, reference_valid = palm_canonical(baseline["joints"].index_select(1, joint_ids))
    usable = available & target_valid & reference_valid
    delta = torch.nn.Parameter(torch.zeros_like(base[..., 0]))
    best_energy = torch.full((len(base),), float("inf"), device=decoder.device)
    best_rotation = base.detach().clone()
    if bool(usable.any()):
        optimizer = torch.optim.Adam((delta,), lr=0.03)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            applied = delta * torch.clamp(math.radians(12.0) / norm.clamp_min(1e-12), max=1.0)
            rotation = so3_exp_map(applied) @ base
            current = (rotation, rotations[1]) if side == 0 else (rotations[0], rotation)
            decoded = decoder.decode(values, current)
            represented, valid = palm_canonical(decoded["joints"].index_select(1, joint_ids))
            # No residual-magnitude prior: the hard ball independently bounds
            # each applied local finger rotation.
            energy = F.smooth_l1_loss(
                represented[..., 1:, :], target[..., 1:, :], reduction="none", beta=1.0,
            ).sum(-1).mean(-1)
            finite = valid & torch.isfinite(energy)
            with torch.no_grad():
                improved = usable & finite & (energy < best_energy)
                best_energy[improved] = energy.detach()[improved]
                best_rotation[improved] = rotation.detach()[improved]
            safe_energy = torch.where(finite, energy, torch.zeros_like(energy))
            weights = (usable & finite).to(safe_energy.dtype)
            if not bool(weights.any()):
                break
            loss = (safe_energy * weights).sum() / weights.sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite palm-canonical objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((delta,), 1.0)
            optimizer.step()
            scheduler.step()
    accepted = usable & torch.isfinite(best_energy)
    return torch.where(accepted[:, None, None, None], best_rotation, base), accepted


def refine_hands(
    decoder: Decoder, parameters: dict[str, np.ndarray], vertices: np.ndarray,
    expert_joints: np.ndarray, available: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Refine valid expert hands while preserving exact fallback states."""
    result = {key: value.copy() for key, value in parameters.items()}
    meshes = vertices.copy()
    accepted = np.zeros_like(available, dtype=bool)
    for start in range(0, len(vertices), 8):
        end = min(start + 8, len(vertices))
        values = decoder.parameters({key: value[start:end] for key, value in parameters.items()})
        rotations = tuple(so3_exp_map(values[key].reshape(-1, 15, 3))
                          for key in ("left_hand_pose", "right_hand_pose"))
        baseline = decoder.decode(values, rotations)
        expert = decoder.tensor(expert_joints[start:end])
        flags = torch.as_tensor(available[start:end], dtype=torch.bool, device=decoder.device)
        left, left_ok = _fit_side(decoder, values, rotations, baseline, expert[:, 0], flags[:, 0], 0)
        right, right_ok = _fit_side(decoder, values, rotations, baseline, expert[:, 1], flags[:, 1], 1)
        with torch.no_grad():
            final = decoder.decode(values, (left, right))
        accepted[start:end] = torch.stack((left_ok, right_ok), dim=1).cpu().numpy()
        for index in range(end - start):
            if accepted[start + index].any():
                meshes[start + index] = final["vertices"][index].cpu().numpy()
                for side, key in enumerate(("left_hand_pose", "right_hand_pose")):
                    if accepted[start + index, side]:
                        result[key][start + index] = final[key][index].cpu().numpy()
    return result, meshes, accepted
