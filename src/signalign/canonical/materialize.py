"""Materialize canonical sequence archives into per-frame SMPL-X states."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from signalign.io.arrays import atomic_savez
from signalign.io.obj import write_obj
from signalign.io_utils import atomic_write_json, sha256_file
from signalign.manifest import HandFrameRecord, read_jsonl, write_jsonl

STATE_KEYS = (
    "betas",
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
    "transl",
)


def materialize_canonical_states(
    fit_root: Path,
    manifest_root: Path,
    output_root: Path,
    hand_manifest: Path,
) -> dict[str, object]:
    """Export one immutable parameter state and mesh per input frame."""
    records_out: list[HandFrameRecord] = []
    sign_summaries: list[dict[str, object]] = []
    for manifest_path in sorted(manifest_root.glob("*.jsonl")):
        records = read_jsonl(manifest_path)
        sequence_path = fit_root / "clips" / manifest_path.stem / "mesh_parametric_final.npz"
        with np.load(sequence_path, allow_pickle=False) as archive:
            sequence = {key: np.asarray(archive[key]).copy() for key in archive.files}
        if len(sequence["frame_ids"]) != len(records):
            raise RuntimeError(f"frame count mismatch: {manifest_path.stem}")
        faces = np.asarray(sequence["faces"], dtype=np.int64)
        for index, record in enumerate(records):
            if int(sequence["frame_ids"][index]) != record.source_frame_id:
                raise RuntimeError(f"frame ID mismatch: {record.sign}/{record.source_frame_id}")
            state_path = output_root / "states" / record.sign / f"{record.source_frame_id:06d}.npz"
            obj_path = output_root / "meshes" / record.sign / f"{record.source_frame_id:06d}.obj"
            vertices = np.asarray(sequence["mesh_parametric"][index], dtype=np.float32)
            arrays = {
                key: np.asarray(sequence[key][index:index + 1], dtype=np.float32)
                for key in STATE_KEYS
            }
            arrays.update(
                {
                    "vertices": vertices,
                    "coord_frame": np.asarray("evaluator_camera"),
                    "unit": np.asarray("meter"),
                    "source_sequence_sha256": np.asarray(sha256_file(sequence_path)),
                }
            )
            if state_path.exists() or obj_path.exists():
                if not state_path.is_file() or not obj_path.is_file():
                    raise RuntimeError(f"partial canonical frame: {record.sign}/{record.source_frame_id}")
            else:
                atomic_savez(state_path, **arrays)
                write_obj(obj_path, vertices, faces)
            rgb_path = Path(record.source_path)
            records_out.append(
                HandFrameRecord(
                    record_id=f"{record.sign}/{record.source_frame_id}",
                    sign=record.sign,
                    sign_class=record.sign_class,
                    frame_index=index,
                    source_frame_id=record.source_frame_id,
                    rgb_path=str(rgb_path.resolve()),
                    canonical_state_path=str(state_path.resolve()),
                    canonical_obj_path=str(obj_path.resolve()),
                    rgb_sha256=sha256_file(rgb_path),
                    state_sha256=sha256_file(state_path),
                    obj_sha256=sha256_file(obj_path),
                )
            )
        sign_summaries.append({"sign": manifest_path.stem, "frames": len(records)})
    write_jsonl(records_out, hand_manifest)
    report = {
        "schema_version": "signalign.canonical-materialization.v1",
        "status": "ok",
        "signs": len(sign_summaries),
        "frames": len(records_out),
        "manifest": str(hand_manifest.resolve()),
        "manifest_sha256": sha256_file(hand_manifest),
        "items": sign_summaries,
    }
    atomic_write_json(output_root / "materialization.json", report)
    return report
