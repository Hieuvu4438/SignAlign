from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from signalign.canonical.identity import farthest_point_indices, huber_location
from signalign.io_utils import atomic_write_json, sha256_file
from signalign.manifest import FrameRecord, read_jsonl


PARAMETER_SHAPES = {
    "betas": (10,),
    "global_orient": (3,),
    "body_pose": (63,),
    "left_hand_pose": (45,),
    "right_hand_pose": (45,),
    "jaw_pose": (3,),
    "leye_pose": (3,),
    "reye_pose": (3,),
    "expression": (10,),
    "transl": (3,),
}


@dataclass(frozen=True)
class InitializerFrame:
    record: FrameRecord
    result_path: Path
    mesh_path: Path


def initializer_frame_paths(root: Path, record: FrameRecord) -> InitializerFrame:
    base = root / record.sign / "smplifyx"
    stem = f"low_{record.source_frame_id}"
    return InitializerFrame(record, base / "results" / f"{stem}.pkl", base / "meshes" / f"{stem}.obj")


def load_initializer_parameters(path: Path) -> dict[str, np.ndarray]:
    # Trusted local artifacts produced by the frozen monocular initializer.
    with path.open("rb") as handle:
        raw: Any = pickle.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected parameter mapping")
    result: dict[str, np.ndarray] = {}
    for key, shape in PARAMETER_SHAPES.items():
        if key not in raw:
            raise KeyError(f"{path}: missing {key}")
        value = np.asarray(raw[key], dtype=np.float32).reshape(-1)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{path}: invalid {key} {value.shape}")
        result[key] = value
    return result


def load_obj_vertices(path: Path, expected: int = 10475) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                if len(fields) != 4:
                    raise ValueError(f"{path}: malformed vertex")
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    result = np.asarray(vertices, dtype=np.float32)
    if result.shape != (expected, 3) or not np.isfinite(result).all():
        raise ValueError(f"{path}: vertices {result.shape}")
    return result


def load_mano_smplx_ids(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        mapping: Any = pickle.load(handle)
    if not isinstance(mapping, dict) or set(mapping) != {"left_hand", "right_hand"}:
        raise ValueError(f"{path}: unexpected MANO-SMPL-X mapping")
    left = np.asarray(mapping["left_hand"], dtype=np.int64)
    right = np.asarray(mapping["right_hand"], dtype=np.int64)
    for name, indices in (("left", left), ("right", right)):
        if indices.shape != (778,) or len(np.unique(indices)) != 778:
            raise ValueError(f"{path}: invalid {name} mapping {indices.shape}")
        if indices.min() < 0 or indices.max() >= 10475:
            raise IndexError(f"{path}: invalid {name} vertex range")
    return left, right


def _all_initializer_frames(
    manifest_root: Path, initializer_root: Path
) -> list[InitializerFrame]:
    frames = []
    for manifest_path in sorted(manifest_root.glob("*.jsonl")):
        frames.extend(
            initializer_frame_paths(initializer_root, record)
            for record in read_jsonl(manifest_path)
        )
    missing = [str(path) for frame in frames for path in (frame.result_path, frame.mesh_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} initializer artifacts; first: {missing[0]}"
        )
    return frames


def _pose_feature(parameters: dict[str, np.ndarray]) -> np.ndarray:
    # Upper-limb and hand rotations provide a deterministic pose-diversity proxy.
    body = parameters["body_pose"].reshape(21, 3)[15:21].reshape(-1)
    hands = np.concatenate((parameters["left_hand_pose"], parameters["right_hand_pose"]))
    return np.concatenate((body, hands)).astype(np.float32)


def estimate_signer_identity(
    initializer_root: Path,
    manifest_root: Path,
    output_npz: Path,
    calibration_frames: int = 50,
    huber_delta: float = 1.5,
    model_root: Path | None = None,
    mano_smplx_ids: Path | None = None,
    refine_steps: int = 0,
    learning_rate: float = 0.01,
    beta_anchor_weight: float = 0.001,
    whole_mesh_weight: float = 0.02,
    device: str = "cpu",
) -> dict[str, object]:
    observations = []
    for frame in _all_initializer_frames(manifest_root, initializer_root):
        parameters = load_initializer_parameters(frame.result_path)
        observations.append((frame, parameters["betas"], _pose_feature(parameters)))
    selected_indices = farthest_point_indices(
        np.stack([item[2] for item in observations]), calibration_frames
    )
    selected = [observations[index] for index in selected_indices]
    beta_values = np.stack([item[1] for item in selected])
    robust_beta = huber_location(beta_values, delta=huber_delta)
    shared_beta = robust_beta.copy()
    refinement: dict[str, object] | None = None
    if refine_steps:
        if model_root is None or mano_smplx_ids is None:
            raise ValueError("model_root and mano_smplx_ids are required for beta refinement")
        import smplx
        import torch

        left_np, right_np = load_mano_smplx_ids(mano_smplx_ids)
        left_ids = torch.as_tensor(left_np, dtype=torch.long, device=device)
        right_ids = torch.as_tensor(right_np, dtype=torch.long, device=device)
        frames = [item[0] for item in selected]
        parameters = _stack_parameters(frames)
        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=device)
        reference = tensor(np.stack([load_obj_vertices(frame.mesh_path) for frame in frames]))
        reference = reference * tensor(np.asarray([1.0, -1.0, -1.0], dtype=np.float32))
        model = smplx.create(
            str(model_root), model_type="smplx", gender="neutral", num_betas=10,
            use_pca=False, use_face_contour=True,
        ).to(device)
        model.eval()
        batch = len(frames)
        beta_initial = tensor(robust_beta.reshape(1, 10))
        beta = torch.nn.Parameter(beta_initial.clone())
        left_delta = torch.nn.Parameter(torch.zeros((batch, 45), dtype=torch.float32, device=device))
        right_delta = torch.nn.Parameter(torch.zeros((batch, 45), dtype=torch.float32, device=device))
        fixed = {
            key: tensor(parameters[key]) for key in (
                "global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                "expression", "transl",
            )
        }

        def forward_vertices():
            return model(
                left_hand_pose=tensor(parameters["left_hand_pose"]) + left_delta,
                right_hand_pose=tensor(parameters["right_hand_pose"]) + right_delta,
                betas=beta.expand(batch, -1), return_verts=True, **fixed,
            ).vertices

        def centered_mse(vertices):
            losses = []
            for indices in (left_ids, right_ids):
                prediction = vertices[:, indices]
                reference_region = reference[:, indices]
                losses.append(torch.square(
                    prediction - prediction.mean(dim=1, keepdim=True)
                    - reference_region + reference_region.mean(dim=1, keepdim=True)
                ).mean())
            return 0.5 * (losses[0] + losses[1])

        with torch.no_grad():
            initial_vertices = forward_vertices()
            initial_hand_mm = float(torch.sqrt(centered_mse(initial_vertices) * 3.0) * 1000)
        optimizer = torch.optim.Adam((beta, left_delta, right_delta), lr=learning_rate)
        best_loss = float("inf")
        best_state = None
        for _ in range(refine_steps):
            optimizer.zero_grad(set_to_none=True)
            vertices = forward_vertices()
            hand_loss = centered_mse(vertices)
            whole_loss = torch.square(vertices - reference).mean()
            anchor = torch.square(beta - beta_initial).mean()
            pose_anchor = 1e-4 * (
                torch.square(left_delta).mean() + torch.square(right_delta).mean()
            )
            loss = hand_loss + whole_mesh_weight * whole_loss + beta_anchor_weight * anchor + pose_anchor
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite shared-beta refinement loss")
            loss.backward()
            optimizer.step()
            current = float(loss.detach())
            if current < best_loss:
                best_loss = current
                best_state = (beta.detach().clone(), left_delta.detach().clone(), right_delta.detach().clone())
        if best_state is None:
            raise RuntimeError("shared-beta refinement produced no valid state")
        with torch.no_grad():
            beta.copy_(best_state[0])
            left_delta.copy_(best_state[1])
            right_delta.copy_(best_state[2])
            final_hand_mm = float(torch.sqrt(centered_mse(forward_vertices()) * 3.0) * 1000)
        shared_beta = beta.detach().cpu().numpy().reshape(10).astype(np.float32)
        refinement = {
            "steps": refine_steps,
            "learning_rate": learning_rate,
            "beta_anchor_weight": beta_anchor_weight,
            "whole_mesh_weight": whole_mesh_weight,
            "initial_hand_rms_mm": initial_hand_mm,
            "final_hand_rms_mm": final_hand_mm,
            "best_loss": best_loss,
        }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        beta=shared_beta,
        robust_beta=robust_beta,
        calibration_betas=beta_values,
        calibration_features=np.stack([item[2] for item in selected]),
        calibration_frame_ids=np.asarray([item[0].record.source_frame_id for item in selected], dtype=np.int64),
    )
    report = {
        "schema_version": "signalign.signer-identity.v1",
        "scope": "signer",
        "initializer": str(initializer_root.resolve()),
        "candidate_frames": len(observations),
        "calibration_frames": len(selected),
        "selection": "farthest_point_upper_limb_and_hand_pose_diversity",
        "estimator": "huber_location",
        "huber_delta": huber_delta,
        "beta": shared_beta.tolist(),
        "robust_beta": robust_beta.tolist(),
        "canonical_refinement": refinement,
        "selected": [
            {"sign": item[0].record.sign, "frame_id": item[0].record.source_frame_id}
            for item in selected
        ],
    }
    atomic_write_json(output_npz.with_suffix(".json"), report)
    return report


def _stack_parameters(frames: list[InitializerFrame]) -> dict[str, np.ndarray]:
    loaded = [load_initializer_parameters(frame.result_path) for frame in frames]
    return {key: np.stack([item[key] for item in loaded]) for key in PARAMETER_SHAPES}


def canonical_refit(
    initializer_root: Path,
    manifest_root: Path,
    identity_npz: Path,
    model_root: Path,
    mano_smplx_ids: Path,
    output_root: Path,
    *,
    device: str = "cpu",
    steps: int = 100,
    learning_rate: float = 0.01,
    chunk_size: int = 32,
    hand_weight: float = 1.0,
    whole_mesh_weight: float = 0.02,
    pose_anchor_weight: float = 1e-4,
    max_hand_residual_mm: float = 3.0,
    signs: set[str] | None = None,
) -> dict[str, object]:
    import smplx
    import torch

    if (output_root / "run_manifest.json").exists() and signs is None:
        raise FileExistsError(f"Refusing to overwrite completed canonical fit: {output_root}")
    with np.load(identity_npz, allow_pickle=False) as archive:
        shared_beta = np.asarray(archive["beta"], dtype=np.float32).reshape(1, 10)
    left_ids_np, right_ids_np = load_mano_smplx_ids(mano_smplx_ids)
    hand_ids_np = np.concatenate((left_ids_np, right_ids_np))
    left_ids = torch.as_tensor(left_ids_np, dtype=torch.long, device=device)
    right_ids = torch.as_tensor(right_ids_np, dtype=torch.long, device=device)
    hand_ids = torch.as_tensor(hand_ids_np, dtype=torch.long, device=device)
    model = smplx.create(
        str(model_root), model_type="smplx", gender="neutral", num_betas=10,
        use_pca=False, use_face_contour=True,
    ).to(device)
    model.eval()
    body_mask = torch.zeros((1, 63), dtype=torch.float32, device=device)
    body_mask[:, 15 * 3:21 * 3] = 1.0  # shoulders, elbows, wrists only
    boundary_x180 = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
    summaries = []
    manifest_paths = sorted(manifest_root.glob("*.jsonl"))
    if signs is not None:
        unknown = signs - {path.stem for path in manifest_paths}
        if unknown:
            raise ValueError(f"Unknown signs: {sorted(unknown)}")
        manifest_paths = [path for path in manifest_paths if path.stem in signs]
    for manifest_path in manifest_paths:
        records = read_jsonl(manifest_path)
        destination = output_root / "clips" / manifest_path.stem / "mesh_parametric_final.npz"
        sign_report_path = destination.with_suffix(".json")
        if destination.exists() or sign_report_path.exists():
            if not destination.is_file() or not sign_report_path.is_file():
                raise RuntimeError(f"Incomplete resumable output for {manifest_path.stem}")
            import json
            cached = json.loads(sign_report_path.read_text(encoding="utf-8"))
            if cached.get("sha256") != sha256_file(destination) or cached.get("frames") != len(records):
                raise RuntimeError(f"Invalid resumable output for {manifest_path.stem}")
            summaries.append(cached)
            continue
        frames = [initializer_frame_paths(initializer_root, record) for record in records]
        parameters = _stack_parameters(frames)
        sign_vertices = []
        fitted_body_pose = []
        fitted_left_hand_pose = []
        fitted_right_hand_pose = []
        left_residual_per_frame = []
        right_residual_per_frame = []
        chunk_reports = []
        for chunk_start in range(0, len(frames), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(frames))
            chunk = frames[chunk_start:chunk_end]
            batch = len(chunk)
            arrays = {key: value[chunk_start:chunk_end] for key, value in parameters.items()}
            def tensor(value):
                return torch.as_tensor(value, dtype=torch.float32, device=device)
            reference_np = np.stack(
                [load_obj_vertices(frame.mesh_path) for frame in chunk]
            ) * boundary_x180
            reference = tensor(reference_np)
            body_init = tensor(arrays["body_pose"])
            left_init = tensor(arrays["left_hand_pose"])
            right_init = tensor(arrays["right_hand_pose"])
            body_delta = torch.nn.Parameter(torch.zeros_like(body_init))
            left_delta = torch.nn.Parameter(torch.zeros_like(left_init))
            right_delta = torch.nn.Parameter(torch.zeros_like(right_init))
            optimizer = torch.optim.Adam((body_delta, left_delta, right_delta), lr=learning_rate)
            fixed = {
                "global_orient": tensor(arrays["global_orient"]),
                "jaw_pose": tensor(arrays["jaw_pose"]),
                "leye_pose": tensor(arrays["leye_pose"]),
                "reye_pose": tensor(arrays["reye_pose"]),
                "expression": tensor(arrays["expression"]),
                "transl": tensor(arrays["transl"]),
                "betas": tensor(shared_beta).expand(batch, -1),
            }

            def forward_vertices():
                return model(
                    body_pose=body_init + body_delta * body_mask,
                    left_hand_pose=left_init + left_delta,
                    right_hand_pose=right_init + right_delta,
                    return_verts=True,
                    **fixed,
                ).vertices

            with torch.no_grad():
                initial_vertices = forward_vertices()
                initial_hand_mm = float(
                    torch.linalg.vector_norm(
                        initial_vertices[:, hand_ids] - reference[:, hand_ids], dim=-1
                    ).mean() * 1000
                )
            best_loss = float("inf")
            best_deltas = None
            stale = 0
            completed_steps = 0
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                vertices = forward_vertices()
                fitted_left = vertices[:, left_ids]
                fitted_right = vertices[:, right_ids]
                target_left = reference[:, left_ids]
                target_right = reference[:, right_ids]
                centered_left = fitted_left - fitted_left.mean(dim=1, keepdim=True)
                centered_right = fitted_right - fitted_right.mean(dim=1, keepdim=True)
                centered_target_left = target_left - target_left.mean(dim=1, keepdim=True)
                centered_target_right = target_right - target_right.mean(dim=1, keepdim=True)
                hand_loss = 0.5 * (
                    torch.square(centered_left - centered_target_left).mean()
                    + torch.square(centered_right - centered_target_right).mean()
                )
                whole_loss = torch.square(vertices - reference).mean()
                anchor = (
                    torch.square(body_delta * body_mask).mean()
                    + torch.square(left_delta).mean()
                    + torch.square(right_delta).mean()
                )
                loss = hand_weight * hand_loss + whole_mesh_weight * whole_loss + pose_anchor_weight * anchor
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{manifest_path.stem}: non-finite refit loss")
                loss.backward()
                optimizer.step()
                completed_steps = step + 1
                current = float(loss.detach())
                if current < best_loss - 1e-10:
                    best_loss = current
                    best_deltas = (
                        body_delta.detach().clone(), left_delta.detach().clone(), right_delta.detach().clone()
                    )
                    stale = 0
                else:
                    stale += 1
                if stale >= 15:
                    break
            if best_deltas is None:
                raise RuntimeError(f"{manifest_path.stem}: optimizer produced no valid state")
            with torch.no_grad():
                body_delta.copy_(best_deltas[0])
                left_delta.copy_(best_deltas[1])
                right_delta.copy_(best_deltas[2])
                fitted = forward_vertices()
                final_body = body_init + body_delta * body_mask
                final_left = left_init + left_delta
                final_right = right_init + right_delta
                fitted_left = fitted[:, left_ids]
                fitted_right = fitted[:, right_ids]
                target_left = reference[:, left_ids]
                target_right = reference[:, right_ids]
                left_absolute_mm = float(torch.linalg.vector_norm(fitted_left - target_left, dim=-1).mean() * 1000)
                right_absolute_mm = float(torch.linalg.vector_norm(fitted_right - target_right, dim=-1).mean() * 1000)
                left_frame_mm = torch.linalg.vector_norm(
                    fitted_left - fitted_left.mean(dim=1, keepdim=True)
                    - target_left + target_left.mean(dim=1, keepdim=True), dim=-1
                ).mean(dim=1) * 1000
                right_frame_mm = torch.linalg.vector_norm(
                    fitted_right - fitted_right.mean(dim=1, keepdim=True)
                    - target_right + target_right.mean(dim=1, keepdim=True), dim=-1
                ).mean(dim=1) * 1000
                left_mm = float(left_frame_mm.mean())
                right_mm = float(right_frame_mm.mean())
                final_vertices = fitted.detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(final_vertices).all():
                raise FloatingPointError(f"{manifest_path.stem}: non-finite canonical vertices")
            if max(left_mm, right_mm) > max_hand_residual_mm:
                raise RuntimeError(
                    f"{manifest_path.stem}[{chunk_start}:{chunk_end}]: hand residual "
                    f"{left_mm:.3f}/{right_mm:.3f} mm exceeds {max_hand_residual_mm:.3f}"
                )
            # The canonical model uses camera convention; export boundary is exactly one x180.
            sign_vertices.append(final_vertices * boundary_x180)
            fitted_body_pose.append(final_body.detach().cpu().numpy().astype(np.float32))
            fitted_left_hand_pose.append(final_left.detach().cpu().numpy().astype(np.float32))
            fitted_right_hand_pose.append(final_right.detach().cpu().numpy().astype(np.float32))
            left_residual_per_frame.append(left_frame_mm.detach().cpu().numpy().astype(np.float32))
            right_residual_per_frame.append(right_frame_mm.detach().cpu().numpy().astype(np.float32))
            chunk_reports.append({
                "start": chunk_start,
                "end": chunk_end,
                "steps": completed_steps,
                "initial_hand_residual_mm": initial_hand_mm,
                "left_hand_residual_mm": left_mm,
                "right_hand_residual_mm": right_mm,
                "max_left_hand_residual_mm": float(left_frame_mm.max()),
                "max_right_hand_residual_mm": float(right_frame_mm.max()),
                "left_hand_absolute_residual_mm": left_absolute_mm,
                "right_hand_absolute_residual_mm": right_absolute_mm,
                "best_loss": best_loss,
            })
        vertices = np.concatenate(sign_vertices, axis=0)
        final_body_pose = np.concatenate(fitted_body_pose, axis=0)
        final_left_hand_pose = np.concatenate(fitted_left_hand_pose, axis=0)
        final_right_hand_pose = np.concatenate(fitted_right_hand_pose, axis=0)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            mesh_parametric=vertices,
            faces=np.asarray(model.faces, dtype=np.int64),
            frame_ids=np.asarray([record.source_frame_id for record in records], dtype=np.int64),
            shared_beta=shared_beta[0],
            betas=np.broadcast_to(shared_beta, (len(records), 10)).copy(),
            global_orient=parameters["global_orient"],
            body_pose=final_body_pose,
            left_hand_pose=final_left_hand_pose,
            right_hand_pose=final_right_hand_pose,
            jaw_pose=parameters["jaw_pose"],
            leye_pose=parameters["leye_pose"],
            reye_pose=parameters["reye_pose"],
            expression=parameters["expression"],
            transl=parameters["transl"],
            left_hand_initializer_residual_mm=np.concatenate(left_residual_per_frame),
            right_hand_initializer_residual_mm=np.concatenate(right_residual_per_frame),
        )
        sign_report = {
            "sign": manifest_path.stem,
            "frames": len(records),
            "sha256": sha256_file(destination),
            "chunks": chunk_reports,
        }
        atomic_write_json(sign_report_path, sign_report)
        summaries.append(sign_report)
    report = {
        "schema_version": "signalign.canonical-refit.v1",
        "initializer": str(initializer_root.resolve()),
        "identity": str(identity_npz.resolve()),
        "model": str((model_root / "smplx" / "SMPLX_NEUTRAL.npz").resolve()),
        "model_sha256": sha256_file(model_root / "smplx" / "SMPLX_NEUTRAL.npz"),
        "mano_smplx_correspondence": str(mano_smplx_ids.resolve()),
        "mano_smplx_correspondence_sha256": sha256_file(mano_smplx_ids),
        "objective_uses_ground_truth": False,
        "objective_uses_evaluator_upper_body_mask": False,
        "hand_objective": "translation_aligned_mano_smplx_correspondence",
        "optimized_parameters": ["left_hand_pose", "right_hand_pose", "shoulder_elbow_wrist_body_pose"],
        "temporal_pose_loss": False,
        "export_transform": "x180_exactly_once",
        "steps": steps,
        "learning_rate": learning_rate,
        "signs": len(summaries),
        "frames": sum(item["frames"] for item in summaries),
        "items": summaries,
    }
    if signs is None:
        atomic_write_json(output_root / "run_manifest.json", report)
    else:
        partition_id = hashlib.sha256(
            "\n".join(sorted(signs)).encode("utf-8")
        ).hexdigest()[:12]
        atomic_write_json(
            output_root / f"partial_run_manifest_{partition_id}.json", report
        )
    return report
