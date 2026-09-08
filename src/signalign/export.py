"""Materialize frozen predictions in the official evaluator directory layout."""

from __future__ import annotations

import shutil
from pathlib import Path

from signalign.io.obj import load_obj, validate_mesh
from signalign.io_utils import atomic_write_json, sha256_file
from signalign.manifest import read_hand_manifest


def export_evaluation_layout(
    hand_manifest: Path, prediction_root: Path, output_root: Path
) -> dict[str, object]:
    records = read_hand_manifest(hand_manifest)
    items: dict[str, int] = {}
    for record in records:
        source = prediction_root / "meshes" / record.sign / f"{record.source_frame_id:06d}.obj"
        destination = output_root / record.sign / "smplifyx" / "meshes" / f"{record.frame_index:03d}.obj"
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise RuntimeError(f"different evaluation artifact exists: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        vertices, faces = load_obj(destination)
        validate_mesh(vertices, faces)
        items[record.sign] = items.get(record.sign, 0) + 1
    report = {
        "schema_version": "signalign.evaluation-layout.v1",
        "status": "ok",
        "signs": len(items),
        "frames": sum(items.values()),
        "items": [{"sign": sign, "frames": count} for sign, count in sorted(items.items())],
    }
    atomic_write_json(output_root / "materialization_summary.json", report)
    return report
