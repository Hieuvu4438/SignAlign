"""Convert raw WiLoR predictions into the immutable SignAlign hand cache."""

from __future__ import annotations

import os
from pathlib import Path
import pickle

import numpy as np

from signalign.io.arrays import atomic_savez
from signalign.io_utils import atomic_write_json, sha256_file
from signalign.manifest import FrameRecord, read_jsonl


SIDES = ("left", "right")
REFLECT_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def _records(manifest_root: Path) -> list[FrameRecord]:
    records = []
    for path in sorted(manifest_root.glob("*.jsonl")):
        records.extend(read_jsonl(path))
    return records


def _record_id(record: FrameRecord) -> str:
    return f"{record.sign}/{record.source_frame_id}"


def canonical_right_to_left_rotation(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float32)
    return REFLECT_X @ value @ REFLECT_X


def build_wilor_frame_manifest(
    manifest_root: Path, output: Path
) -> dict[str, object]:
    """Freeze RGB paths, dimensions, and hashes before third-party extraction."""
    from PIL import Image

    records = _records(manifest_root)
    items = []
    hashes = {}
    for record in records:
        path = Path(record.source_path).resolve()
        with Image.open(path) as image:
            width, height = image.size
        digest = sha256_file(path)
        hashes[str(path)] = digest
        items.append(
            {
                "image_key": _record_id(record),
                "image_path": str(path),
                "expected_width": width,
                "expected_height": height,
            }
        )
    report = {
        "schema_version": "signalign.wilor-frame-manifest.v1",
        "frame_count": len(items),
        "manifest_root": str(manifest_root.resolve()),
        "image_sha256": hashes,
        "records": items,
    }
    atomic_write_json(output, report)
    return report


def merge_wilor_sidecars(sources: list[Path], output: Path) -> dict[str, object]:
    """Merge per-sign raw WiLoR sidecars without changing their observations."""
    paths = sorted(path.resolve() for path in sources)
    if not paths:
        raise ValueError("at least one WiLoR sidecar is required")
    common_fields = (
        "format",
        "wilor_repository_commit",
        "wilor_checkpoint_sha256",
        "detector_checkpoint_sha256",
        "model_config_sha256",
    )
    baseline: dict[str, object] | None = None
    images: dict[str, object] = {}
    source_records = []
    runtime = 0.0
    dropouts = 0
    for path in paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        meta = payload.get("meta", {})
        if meta.get("format") != "wilor_raw_v3":
            raise RuntimeError(f"unsupported WiLoR sidecar format: {path}")
        signature = {field: meta.get(field) for field in common_fields}
        if baseline is None:
            baseline = signature
        elif signature != baseline:
            raise RuntimeError(f"WiLoR provenance differs across sidecars: {path}")
        current = payload.get("images")
        if not isinstance(current, dict):
            raise TypeError(f"WiLoR images must be a mapping: {path}")
        duplicates = set(images) & set(current)
        if duplicates:
            raise RuntimeError(f"duplicate WiLoR image keys: {sorted(duplicates)[:3]}")
        images.update(current)
        runtime += float(meta.get("runtime_seconds", 0.0))
        dropouts += int(meta.get("detector_dropout_frames", 0))
        source_records.append({"path": str(path), "sha256": sha256_file(path)})
    assert baseline is not None
    artifact = {
        "meta": {
            **baseline,
            "frame_count": len(images),
            "runtime_seconds": runtime,
            "detector_dropout_frames": dropouts,
            "merged_sidecars": source_records,
        },
        "images": images,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return {
        "schema_version": "signalign.wilor-sidecar-merge.v1",
        "status": "ok",
        "sidecars": len(paths),
        "frames": len(images),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def import_wilor_sidecar(
    manifest_root: Path, source_pickle: Path, output_root: Path
) -> dict[str, object]:
    records = _records(manifest_root)
    with source_pickle.open("rb") as handle:
        source = pickle.load(handle)
    if source.get("meta", {}).get("format") != "wilor_raw_v3":
        raise RuntimeError("unsupported WiLoR sidecar format")
    images = source.get("images", {})
    expected = {_record_id(record) for record in records}
    if set(images) != expected:
        raise RuntimeError("WiLoR sidecar coverage differs from inference manifests")
    unavailable = {side: 0 for side in SIDES}
    for record in records:
        record_id = _record_id(record)
        hands = images[record_id]["hands"]
        chosen = {}
        for side in SIDES:
            is_right = float(side == "right")
            candidates = [hand for hand in hands if float(hand["is_right"]) == is_right]
            if candidates:
                chosen[side] = max(
                    candidates, key=lambda hand: float(hand["detector_confidence"])
                )
        available = np.zeros(2, dtype=bool)
        confidence = np.zeros(2, dtype=np.float32)
        joints = np.zeros((2, 21, 3), dtype=np.float32)
        rotations = np.tile(np.eye(3, dtype=np.float32), (2, 15, 1, 1))
        boxes = np.zeros((2, 4), dtype=np.float32)
        for side_index, side in enumerate(SIDES):
            hand = chosen.get(side)
            if hand is None:
                unavailable[side] += 1
                continue
            available[side_index] = True
            confidence[side_index] = float(hand["detector_confidence"])
            hand_joints = np.asarray(hand["pred_keypoints_3d"], dtype=np.float32)
            hand_rotations = np.asarray(hand["pred_mano_pose_rotmat"], dtype=np.float32)
            if hand_joints.shape != (21, 3) or hand_rotations.shape != (15, 3, 3):
                raise RuntimeError(f"WiLoR shape drift: {record_id}/{side}")
            if side == "left":
                hand_joints = hand_joints @ REFLECT_X
                hand_rotations = canonical_right_to_left_rotation(hand_rotations)
            if not np.allclose(np.linalg.det(hand_rotations), 1.0, atol=2e-4):
                raise RuntimeError(f"improper WiLoR rotation: {record_id}/{side}")
            if not np.allclose(
                hand_rotations.swapaxes(-1, -2) @ hand_rotations,
                np.eye(3),
                atol=2e-4,
            ):
                raise RuntimeError(f"non-orthogonal WiLoR rotation: {record_id}/{side}")
            joints[side_index] = hand_joints - hand_joints[:1]
            rotations[side_index] = hand_rotations
            boxes[side_index] = np.asarray(hand["detector_box_xyxy"], dtype=np.float32)
        rgb = Path(record.source_path)
        atomic_savez(
            output_root / record.sign / f"{record.source_frame_id:06d}.npz",
            side_names=np.asarray(SIDES),
            available=available,
            detector_confidence=confidence,
            joints3d=joints,
            pose_rotmat=rotations,
            bbox_xyxy=boxes,
            coord_frame=np.asarray("canonical_hand_root_centered"),
            unit=np.asarray("WiLoR_MANO_model_unit"),
            left_conversion=np.asarray("F_x_R_F_x_and_x_reflection"),
            rgb_sha256=np.asarray(sha256_file(rgb)),
            record_id=np.asarray(record_id),
            source_pickle_sha256=np.asarray(sha256_file(source_pickle)),
            checkpoint_sha256=np.asarray(source["meta"]["wilor_checkpoint_sha256"]),
            detector_sha256=np.asarray(source["meta"]["detector_checkpoint_sha256"]),
            source_commit=np.asarray(source["meta"]["wilor_repository_commit"]),
        )
    report = {
        "schema_version": "signalign.wilor-observation-summary.v1",
        "status": "ok",
        "frames": len(records),
        "written": len(records),
        "unavailable_by_side": unavailable,
        "source_pickle_sha256": sha256_file(source_pickle),
        "checkpoint_sha256": source["meta"]["wilor_checkpoint_sha256"],
        "detector_sha256": source["meta"]["detector_checkpoint_sha256"],
        "source_commit": source["meta"]["wilor_repository_commit"],
    }
    atomic_write_json(output_root / "summary.json", report)
    return report


def validate_wilor_cache(
    manifest_root: Path, root: Path, output: Path
) -> dict[str, object]:
    failures = []
    availability = np.zeros(2, dtype=np.int64)
    records = _records(manifest_root)
    for record in records:
        path = root / record.sign / f"{record.source_frame_id:06d}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                if tuple(archive["side_names"].tolist()) != SIDES:
                    raise RuntimeError("side order drift")
                if str(archive["rgb_sha256"]) != sha256_file(Path(record.source_path)):
                    raise RuntimeError("RGB hash drift")
                valid = np.asarray(archive["available"], dtype=bool)
                joints = np.asarray(archive["joints3d"], dtype=np.float64)
                rotations = np.asarray(archive["pose_rotmat"], dtype=np.float64)
                if valid.shape != (2,) or joints.shape != (2, 21, 3):
                    raise RuntimeError("WiLoR cache shape drift")
                if rotations.shape != (2, 15, 3, 3):
                    raise RuntimeError("WiLoR rotation cache shape drift")
                if not np.isfinite(joints).all() or not np.isfinite(rotations).all():
                    raise RuntimeError("non-finite WiLoR cache")
                for side in np.where(valid)[0]:
                    if not np.allclose(np.linalg.det(rotations[side]), 1.0, atol=2e-4):
                        raise RuntimeError("improper hand reflection")
                    if np.max(np.abs(joints[side, 0])) > 1e-6:
                        raise RuntimeError("hand is not wrist centered")
                availability += valid
        except Exception as error:
            failures.append({"record_id": _record_id(record), "error": str(error)})
    report = {
        "schema_version": "signalign.wilor-cache-validation.v1",
        "status": "ok" if not failures else "failed",
        "frames": len(records),
        "available_left": int(availability[0]),
        "available_right": int(availability[1]),
        "failures": failures,
    }
    atomic_write_json(output, report)
    if failures:
        raise RuntimeError(f"WiLoR cache validation failed: {output}")
    return report
