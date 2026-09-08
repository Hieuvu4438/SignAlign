"""End-to-end inference orchestration with no evaluation data in its API."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from signalign.canonical.materialize import materialize_canonical_states
from signalign.canonical.refinement import canonical_refit, estimate_signer_identity
from signalign.hand.parallel import refine_hands_parallel
from signalign.io_utils import atomic_write_json, load_config, sha256_file
from signalign.manifest import prepare_inference_manifests


FORBIDDEN_INFERENCE_PATHS = {
    "evaluator",
    "evaluation_root",
    "ground_truth_root",
    "gt_root",
    "target_mesh_root",
}


def _path(config: dict[str, Any], name: str) -> Path:
    try:
        return Path(config["paths"][name]).resolve()
    except KeyError as error:
        raise KeyError(f"missing paths.{name}") from error


def validate_inference_config(config: dict[str, Any]) -> None:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise TypeError("paths must be a mapping")
    forbidden = FORBIDDEN_INFERENCE_PATHS & set(paths)
    if forbidden:
        raise ValueError(f"evaluation data is forbidden in inference config: {sorted(forbidden)}")
    required = {
        "rgb_root",
        "signs_file",
        "segments_file",
        "initializer_root",
        "smplx_model_root",
        "mano_smplx_ids",
        "wilor_root",
        "output_root",
    }
    if missing := required - set(paths):
        raise KeyError(f"missing inference paths: {sorted(missing)}")
    for name in required - {"output_root"}:
        if not _path(config, name).exists():
            raise FileNotFoundError(f"paths.{name}: {_path(config, name)}")


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _completed_canonical_report(
    output_root: Path, expected_signs: int, expected_frames: int
) -> dict[str, Any] | None:
    path = output_root / "run_manifest.json"
    if not path.is_file():
        return None
    report = _json(path)
    if report.get("signs") != expected_signs or report.get("frames") != expected_frames:
        raise RuntimeError(f"canonical completion count mismatch: {path}")
    for item in report.get("items", []):
        sequence = output_root / "clips" / str(item["sign"]) / "mesh_parametric_final.npz"
        if not sequence.is_file() or sha256_file(sequence) != item.get("sha256"):
            raise RuntimeError(f"canonical completion hash mismatch: {sequence}")
    if len(report.get("items", [])) != expected_signs:
        raise RuntimeError(f"canonical completion item mismatch: {path}")
    return report


def _completed_materialization_report(
    output_root: Path,
    hand_manifest: Path,
    expected_signs: int,
    expected_frames: int,
) -> dict[str, Any] | None:
    path = output_root / "materialization.json"
    if not path.is_file() or not hand_manifest.is_file():
        return None
    report = _json(path)
    if report.get("signs") != expected_signs or report.get("frames") != expected_frames:
        raise RuntimeError(f"materialization completion count mismatch: {path}")
    if report.get("manifest_sha256") != sha256_file(hand_manifest):
        raise RuntimeError(f"materialization manifest hash mismatch: {hand_manifest}")
    return report


def run_inference(config_path: Path) -> dict[str, object]:
    """Run RGB manifests -> canonical SMPL-X -> bounded hand refinement."""
    config = load_config(config_path)
    validate_inference_config(config)
    set_reproducible(int(config.get("seed", 20260903)))
    output_root = _path(config, "output_root")
    completion = output_root / "inference_summary.json"
    if completion.exists():
        raise FileExistsError(f"refusing to overwrite completed inference: {output_root}")

    protocol = config["protocol"]
    manifests = output_root / "manifests"
    manifest_report = prepare_inference_manifests(
        _path(config, "rgb_root"),
        _path(config, "signs_file"),
        _path(config, "segments_file"),
        manifests,
        expected_signs=int(protocol["expected_signs"]),
        expected_frames=int(protocol["expected_frames"]),
    )

    identity_cfg = config["identity"]
    identity_path = output_root / "identity" / "signer.npz"
    identity_report_path = identity_path.with_suffix(".json")
    if identity_path.is_file() and identity_report_path.is_file():
        identity_report = _json(identity_report_path)
        if identity_report.get("candidate_frames") != int(protocol["expected_frames"]):
            raise RuntimeError(f"identity frame count mismatch: {identity_report_path}")
        if identity_report.get("calibration_frames") != int(
            identity_cfg["calibration_frames"]
        ):
            raise RuntimeError(f"identity calibration mismatch: {identity_report_path}")
    else:
        identity_report = estimate_signer_identity(
            _path(config, "initializer_root"),
            manifests,
            identity_path,
            calibration_frames=int(identity_cfg["calibration_frames"]),
            huber_delta=float(identity_cfg["huber_delta"]),
            model_root=_path(config, "smplx_model_root"),
            mano_smplx_ids=_path(config, "mano_smplx_ids"),
            refine_steps=int(identity_cfg["refine_steps"]),
            learning_rate=float(identity_cfg["learning_rate"]),
            beta_anchor_weight=float(identity_cfg["beta_anchor_weight"]),
            whole_mesh_weight=float(identity_cfg["whole_mesh_weight"]),
            device=str(config["runtime"]["device"]),
        )

    canonical_cfg = config["canonicalization"]
    canonical_root = output_root / "canonical_fit"
    canonical_report = _completed_canonical_report(
        canonical_root,
        int(protocol["expected_signs"]),
        int(protocol["expected_frames"]),
    )
    if canonical_report is None:
        canonical_report = canonical_refit(
            _path(config, "initializer_root"),
            manifests,
            identity_path,
            _path(config, "smplx_model_root"),
            _path(config, "mano_smplx_ids"),
            canonical_root,
            device=str(config["runtime"]["device"]),
            steps=int(canonical_cfg["steps"]),
            learning_rate=float(canonical_cfg["learning_rate"]),
            chunk_size=int(canonical_cfg["chunk_size"]),
            hand_weight=float(canonical_cfg["hand_weight"]),
            whole_mesh_weight=float(canonical_cfg["whole_mesh_weight"]),
            pose_anchor_weight=float(canonical_cfg["pose_anchor_weight"]),
            max_hand_residual_mm=float(canonical_cfg["max_hand_residual_mm"]),
        )

    canonical_frames = output_root / "canonical_frames"
    hand_manifest = output_root / "hand_manifest.jsonl"
    materialization_report = _completed_materialization_report(
        canonical_frames,
        hand_manifest,
        int(protocol["expected_signs"]),
        int(protocol["expected_frames"]),
    )
    if materialization_report is None:
        materialization_report = materialize_canonical_states(
            canonical_root, manifests, canonical_frames, hand_manifest
        )

    hand_cfg = config["hand_refinement"]
    hand_root = output_root / "predictions"
    hand_report = refine_hands_parallel(
        hand_manifest,
        hand_root,
        _path(config, "smplx_model_root"),
        _path(config, "wilor_root"),
        workers=int(config["runtime"].get("hand_workers", 1)),
        device=str(config["runtime"]["device"]),
        batch_size=int(config["runtime"]["batch_size"]),
        radius_deg=float(hand_cfg["radius_deg"]),
        steps=int(hand_cfg["steps"]),
        learning_rate=float(hand_cfg["learning_rate"]),
        residual_prior=float(hand_cfg["residual_prior"]),
        seed=int(config.get("seed", 20260903)),
    )

    result = {
        "schema_version": "signalign.inference.v1",
        "status": "ok",
        "method": "signer-consistent initialization with palm-canonical hand refinement",
        "frames": manifest_report["frames"],
        "signs": manifest_report["signs"],
        "uses_transformer": False,
        "uses_sequence_network": False,
        "uses_ground_truth_during_inference": False,
        "artifacts": {
            "manifest": str((manifests / "summary.json").resolve()),
            "identity": str(identity_path.resolve()),
            "identity_sha256": sha256_file(identity_path),
            "canonical_fit": str(canonical_root.resolve()),
            "hand_manifest": str(hand_manifest.resolve()),
            "hand_manifest_sha256": sha256_file(hand_manifest),
            "predictions": str(hand_root.resolve()),
        },
        "stages": {
            "identity": identity_report["schema_version"],
            "canonicalization": canonical_report["schema_version"],
            "materialization": materialization_report["schema_version"],
            "hand_refinement": hand_report["schema_version"],
        },
    }
    atomic_write_json(completion, result)
    return result


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
