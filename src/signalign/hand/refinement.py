"""Palm-canonical, bounded, finger-only SMPL-X refinement."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from signalign.hand.model import STATE_KEYS, CanonicalBatch, FrozenSMPLX
from signalign.io.arrays import atomic_savez
from signalign.io.obj import load_obj, write_obj
from signalign.io_utils import atomic_write_json, sha256_file
from signalign.manifest import HandFrameRecord, read_hand_manifest
from signalign.model.kinematics import apply_lie_residual, so3_log_map

SIDES = ("left", "right")
SMPLX_HAND_JOINTS = {
    "left": (20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70),
    "right": (21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75),
}


def palm_canonical(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove wrist translation, palm orientation, and global hand scale."""
    centered = joints - joints[..., :1, :]
    transverse = F.normalize(centered[..., 5, :] - centered[..., 17, :], dim=-1)
    longitudinal_raw = 0.5 * (centered[..., 5, :] + centered[..., 17, :])
    longitudinal = F.normalize(
        longitudinal_raw
        - (longitudinal_raw * transverse).sum(-1, keepdim=True) * transverse,
        dim=-1,
    )
    normal = F.normalize(torch.cross(transverse, longitudinal, dim=-1), dim=-1)
    longitudinal = torch.cross(normal, transverse, dim=-1)
    frame = torch.stack((transverse, longitudinal, normal), dim=-1)
    scale = torch.linalg.vector_norm(centered[..., 9, :], dim=-1).clamp_min(1e-6)
    return centered @ frame / scale[..., None, None], torch.det(frame)


def _load_wilor(
    records: Sequence[HandFrameRecord],
    root: Path,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    availability: list[np.ndarray] = []
    joints: list[np.ndarray] = []
    for record in records:
        path = root / record.sign / f"{record.source_frame_id:06d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["rgb_sha256"]) != record.rgb_sha256:
                raise RuntimeError(f"WiLoR/RGB hash mismatch: {record.record_id}")
            availability.append(np.asarray(archive["available"], dtype=bool))
            joints.append(np.asarray(archive["joints3d"], dtype=np.float32))
    return (
        torch.as_tensor(np.stack(availability), dtype=torch.bool, device=device),
        torch.as_tensor(np.stack(joints), dtype=torch.float32, device=device),
    )


def _fit_side(
    model: FrozenSMPLX,
    expert_joints: torch.Tensor,
    available: torch.Tensor,
    side: str,
    *,
    radius_deg: float,
    steps: int,
    learning_rate: float,
    residual_prior: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_left = model.left_rotation
    base_right = model.right_rotation
    base = base_left if side == "left" else base_right
    delta = torch.nn.Parameter(torch.zeros(
        base.shape[0], 15, 3, dtype=base.dtype, device=base.device
    ))
    optimizer = torch.optim.Adam((delta,), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1)
    )
    joint_ids = torch.as_tensor(SMPLX_HAND_JOINTS[side], device=base.device)
    with torch.no_grad():
        baseline = model.decode(base_left, base_right)
        reference = baseline["joints"].index_select(1, joint_ids)
        effective_expert = torch.where(
            available[:, None, None], expert_joints, reference
        )
        target, target_det = palm_canonical(effective_expert)
        _, reference_det = palm_canonical(reference)
        if not bool(((target_det > 0.999) & (reference_det > 0.999)).all()):
            raise RuntimeError("degenerate or improper palm frame")
    best_energy = torch.full((base.shape[0],), float("inf"), device=base.device)
    best_rotation = base.detach().clone()
    radius_rad = np.deg2rad(radius_deg)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        rotation, bounded = apply_lie_residual(base, delta, radius_rad)
        decoded = model.decode(
            rotation if side == "left" else base_left,
            rotation if side == "right" else base_right,
        )
        predicted, determinant = palm_canonical(
            decoded["joints"].index_select(1, joint_ids)
        )
        if not bool((determinant > 0.999).all()):
            raise RuntimeError("predicted an improper palm frame")
        data = F.smooth_l1_loss(
            predicted[..., 1:, :], target[..., 1:, :], reduction="none"
        ).sum(-1).mean(-1)
        prior = delta.square().sum(-1).mean(-1)
        energy = data + residual_prior * prior
        with torch.no_grad():
            improved = available & torch.isfinite(energy) & (energy < best_energy)
            best_energy[improved] = energy.detach()[improved]
            best_rotation[improved] = rotation.detach()[improved]
        weight = available.to(energy.dtype)
        loss = (energy * weight).sum() / weight.sum().clamp_min(1)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite palm-canonical fitting objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((delta,), 1.0)
        optimizer.step()
        scheduler.step()
    relative = best_rotation @ base.transpose(-1, -2)
    projected, bounded = apply_lie_residual(
        base, so3_log_map(relative), radius_rad
    )
    trust = torch.rad2deg(torch.linalg.vector_norm(bounded, dim=-1))
    return projected, trust


def refine_hands(
    manifest: Path,
    output_root: Path,
    model_root: Path,
    wilor_root: Path,
    *,
    device: str = "cuda",
    batch_size: int = 8,
    radius_deg: float = 12.0,
    steps: int = 40,
    learning_rate: float = 0.03,
    residual_prior: float = 0.2,
    limit: int | None = None,
) -> dict[str, object]:
    """Run the final method; no target mesh or evaluator asset is accepted."""
    if radius_deg <= 0 or radius_deg > 30:
        raise ValueError(f"invalid finger trust radius: {radius_deg}")
    all_records = read_hand_manifest(manifest)
    records = all_records if limit is None else all_records[:limit]
    if (output_root / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed refinement: {output_root}")
    implementation_sha = sha256_file(Path(__file__))
    accepted_frames = 0
    accepted_hands = 0
    unavailable_frames = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        states = [Path(item.canonical_state_path) for item in batch]
        for record, state in zip(batch, states, strict=True):
            if sha256_file(state) != record.state_sha256:
                raise RuntimeError(f"canonical state changed: {record.record_id}")
            if sha256_file(Path(record.canonical_obj_path)) != record.obj_sha256:
                raise RuntimeError(f"canonical mesh changed: {record.record_id}")
        canonical = CanonicalBatch.from_npz(states, device)
        model = FrozenSMPLX(model_root, canonical)
        available, expert = _load_wilor(batch, wilor_root, device)
        left, left_trust = _fit_side(
            model, expert[:, 0], available[:, 0], "left",
            radius_deg=radius_deg, steps=steps, learning_rate=learning_rate,
            residual_prior=residual_prior,
        )
        right, right_trust = _fit_side(
            model, expert[:, 1], available[:, 1], "right",
            radius_deg=radius_deg, steps=steps, learning_rate=learning_rate,
            residual_prior=residual_prior,
        )
        final_left = torch.where(
            available[:, 0, None, None, None], left, model.left_rotation
        )
        final_right = torch.where(
            available[:, 1, None, None, None], right, model.right_rotation
        )
        with torch.no_grad():
            final = model.decode(final_left, final_right)
        for index, record in enumerate(batch):
            accepted_sides = [
                side for side, side_index in zip(SIDES, range(2))
                if bool(available[index, side_index])
            ]
            source_state = Path(record.canonical_state_path)
            source_obj = Path(record.canonical_obj_path)
            output_state = output_root / "states" / record.sign / f"{record.source_frame_id:06d}.npz"
            output_obj = output_root / "meshes" / record.sign / f"{record.source_frame_id:06d}.obj"
            decision_path = output_root / "decisions" / record.sign / f"{record.source_frame_id:06d}.json"
            if output_state.exists() or output_obj.exists() or decision_path.exists():
                raise FileExistsError(f"refusing to reuse existing output: {record.record_id}")
            if accepted_sides:
                _, faces = load_obj(source_obj)
                vertices = final["vertices"][index].cpu().numpy().astype(np.float32)
                write_obj(output_obj, vertices, faces)
                with np.load(source_state, allow_pickle=False) as archive:
                    state = {key: np.asarray(archive[key]).copy() for key in archive.files}
                if "left" in accepted_sides:
                    state["left_hand_pose"] = final["left_hand_pose"][
                        index:index + 1
                    ].cpu().numpy().astype(np.float32)
                if "right" in accepted_sides:
                    state["right_hand_pose"] = final["right_hand_pose"][
                        index:index + 1
                    ].cpu().numpy().astype(np.float32)
                state["vertices"] = vertices
                atomic_savez(output_state, **state)
                accepted_frames += 1
                accepted_hands += len(accepted_sides)
                reason = "PALM_CANONICAL_REFINEMENT"
            else:
                output_state.parent.mkdir(parents=True, exist_ok=True)
                output_obj.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_state, output_state)
                shutil.copy2(source_obj, output_obj)
                unavailable_frames += 1
                reason = "HAND_EXPERT_UNAVAILABLE"
            with np.load(source_state, allow_pickle=False) as before, np.load(
                output_state, allow_pickle=False
            ) as after:
                protected = [key for key in STATE_KEYS if key not in {
                    "left_hand_pose", "right_hand_pose"
                }]
                if any(not np.array_equal(before[key], after[key]) for key in protected):
                    raise RuntimeError(f"protected state changed: {record.record_id}")
                if "left" not in accepted_sides and not np.array_equal(
                    before["left_hand_pose"], after["left_hand_pose"]
                ):
                    raise RuntimeError(f"left fallback changed: {record.record_id}")
                if "right" not in accepted_sides and not np.array_equal(
                    before["right_hand_pose"], after["right_hand_pose"]
                ):
                    raise RuntimeError(f"right fallback changed: {record.record_id}")
            atomic_write_json(
                decision_path,
                {
                    "schema_version": "signalign.hand-refinement.v1",
                    "record_id": record.record_id,
                    "accepted_sides": accepted_sides,
                    "reason": reason,
                    "finger_radius_deg": radius_deg,
                    "left_max_update_deg": float(left_trust[index].max()),
                    "right_max_update_deg": float(right_trust[index].max()),
                    "wrist_locked": True,
                    "body_locked": True,
                    "shape_locked": True,
                    "camera_locked": True,
                    "objective_uses_ground_truth": False,
                    "implementation_sha256": implementation_sha,
                    "input_hashes": {
                        "rgb": record.rgb_sha256,
                        "state": record.state_sha256,
                        "mesh": record.obj_sha256,
                        "wilor": sha256_file(
                            wilor_root / record.sign / f"{record.source_frame_id:06d}.npz"
                        ),
                    },
                    "output_hashes": {
                        "state": sha256_file(output_state),
                        "mesh": sha256_file(output_obj),
                    },
                },
            )
    summary = {
        "schema_version": "signalign.hand-refinement-summary.v1",
        "status": "ok",
        "frames": len(records),
        "manifest_frames": len(all_records),
        "accepted_frames": accepted_frames,
        "accepted_hands": accepted_hands,
        "full_frame_fallbacks": unavailable_frames,
        "finger_radius_deg": radius_deg,
        "optimization_steps": steps,
        "learning_rate": learning_rate,
        "residual_prior": residual_prior,
        "wrist_locked": True,
        "explicit_bone_normalization": False,
        "confidence_filtering": False,
        "transformer": False,
        "objective_uses_ground_truth": False,
        "implementation_sha256": implementation_sha,
    }
    atomic_write_json(output_root / "summary.json", summary)
    return summary
