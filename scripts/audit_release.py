#!/usr/bin/env python3
"""Fail-closed audit of a completed public-code inference run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from signalign.io_utils import atomic_write_json, sha256_file, tree_sha256
from signalign.manifest import read_hand_manifest
from signalign.pipeline import load_config, validate_inference_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_inference_config(config)
    root = Path(config["paths"]["output_root"])
    expected_frames = int(config["protocol"]["expected_frames"])
    expected_signs = int(config["protocol"]["expected_signs"])
    completion_path = root / "inference_summary.json"
    if not completion_path.is_file():
        raise FileNotFoundError(completion_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "ok" or completion.get("frames") != expected_frames:
        raise RuntimeError("inference completion manifest is invalid")
    if completion.get("uses_transformer") is not False:
        raise RuntimeError("completion manifest does not exclude Transformer")
    if completion.get("uses_ground_truth_during_inference") is not False:
        raise RuntimeError("completion manifest does not exclude ground truth")
    hand_manifest = root / "hand_manifest.jsonl"
    records = read_hand_manifest(hand_manifest)
    if len(records) != expected_frames:
        raise RuntimeError("hand manifest frame count mismatch")
    signs = {record.sign for record in records}
    if len(signs) != expected_signs:
        raise RuntimeError("hand manifest sign count mismatch")
    reason_histogram: dict[str, int] = {}
    accepted_hands = 0
    for record in records:
        stem = f"{record.source_frame_id:06d}"
        decision_path = root / "predictions/decisions" / record.sign / f"{stem}.json"
        state_path = root / "predictions/states" / record.sign / f"{stem}.npz"
        mesh_path = root / "predictions/meshes" / record.sign / f"{stem}.obj"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("objective_uses_ground_truth") is not False:
            raise RuntimeError(f"ground-truth objective flag: {record.record_id}")
        for lock in ("wrist_locked", "body_locked", "shape_locked", "camera_locked"):
            if decision.get(lock) is not True:
                raise RuntimeError(f"missing {lock}: {record.record_id}")
        if decision["output_hashes"]["state"] != sha256_file(state_path):
            raise RuntimeError(f"state hash mismatch: {record.record_id}")
        if decision["output_hashes"]["mesh"] != sha256_file(mesh_path):
            raise RuntimeError(f"mesh hash mismatch: {record.record_id}")
        reason = str(decision["reason"])
        reason_histogram[reason] = reason_histogram.get(reason, 0) + 1
        accepted_hands += len(decision["accepted_sides"])
    canonical = list((root / "canonical_fit/clips").glob("*/mesh_parametric_final.npz"))
    report = {
        "schema_version": "signalign.release-audit.v1",
        "decision": "PASS",
        "signs": len(signs),
        "frames": len(records),
        "canonical_sequences": len(canonical),
        "states": len(list((root / "predictions/states").glob("*/*.npz"))),
        "meshes": len(list((root / "predictions/meshes").glob("*/*.obj"))),
        "decisions": len(list((root / "predictions/decisions").glob("*/*.json"))),
        "accepted_hands": accepted_hands,
        "reason_histogram": reason_histogram,
        "target_paths_in_inference_config": 0,
        "source_tree_sha256": tree_sha256(
            Path(__file__).resolve().parents[1] / "src/signalign"
        ),
        "completion_sha256": sha256_file(completion_path),
        "hand_manifest_sha256": sha256_file(hand_manifest),
    }
    counts = (
        report["canonical_sequences"],
        report["states"],
        report["meshes"],
        report["decisions"],
    )
    if counts != (expected_signs, expected_frames, expected_frames, expected_frames):
        raise RuntimeError(f"release artifact count mismatch: {counts}")
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
