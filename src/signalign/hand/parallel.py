"""Parallel execution that preserves the sequential batch partition exactly."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from signalign.hand.refinement import refine_hands
from signalign.io_utils import atomic_write_json, sha256_file
from signalign.manifest import HandFrameRecord, read_hand_manifest, write_jsonl


def _run_shard(arguments: dict[str, Any]) -> dict[str, object]:
    import torch

    torch.manual_seed(int(arguments.pop("seed")))
    torch.use_deterministic_algorithms(True, warn_only=True)
    return refine_hands(**arguments)


def _batch_preserving_partitions(
    records: list[HandFrameRecord], batch_size: int, workers: int
) -> list[list[HandFrameRecord]]:
    batches = [records[start : start + batch_size] for start in range(0, len(records), batch_size)]
    partitions = []
    for worker in range(workers):
        start = len(batches) * worker // workers
        end = len(batches) * (worker + 1) // workers
        partitions.append([record for batch in batches[start:end] for record in batch])
    return [partition for partition in partitions if partition]


def refine_hands_parallel(
    manifest: Path,
    output_root: Path,
    model_root: Path,
    wilor_root: Path,
    *,
    workers: int,
    device: str = "cuda",
    batch_size: int = 8,
    radius_deg: float = 12.0,
    steps: int = 40,
    learning_rate: float = 0.03,
    residual_prior: float = 0.2,
    seed: int = 20260903,
) -> dict[str, object]:
    """Run disjoint contiguous batch shards and merge their immutable artifacts."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return refine_hands(
            manifest,
            output_root,
            model_root,
            wilor_root,
            device=device,
            batch_size=batch_size,
            radius_deg=radius_deg,
            steps=steps,
            learning_rate=learning_rate,
            residual_prior=residual_prior,
        )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to merge into non-empty output: {output_root}")
    records = read_hand_manifest(manifest)
    partitions = _batch_preserving_partitions(records, batch_size, workers)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shard_root = Path(
        tempfile.mkdtemp(prefix=".signalign-hand-shards-", dir=output_root.parent)
    )
    arguments = []
    owner: dict[str, Path] = {}
    for index, partition in enumerate(partitions):
        root = shard_root / f"part_{index:02d}"
        shard_manifest = root / "manifest.jsonl"
        write_jsonl(partition, shard_manifest)
        for record in partition:
            owner[record.record_id] = root / "output"
        arguments.append(
            {
                "manifest": shard_manifest,
                "output_root": root / "output",
                "model_root": model_root,
                "wilor_root": wilor_root,
                "device": device,
                "batch_size": batch_size,
                "radius_deg": radius_deg,
                "steps": steps,
                "learning_rate": learning_rate,
                "residual_prior": residual_prior,
                "seed": seed + index,
            }
        )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(arguments), mp_context=context) as pool:
        summaries = list(pool.map(_run_shard, arguments))

    for record in records:
        source_root = owner[record.record_id]
        relative = Path(record.sign) / f"{record.source_frame_id:06d}"
        for folder, suffix in (("states", ".npz"), ("meshes", ".obj"), ("decisions", ".json")):
            source = source_root / folder / relative.with_suffix(suffix)
            destination = output_root / folder / relative.with_suffix(suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
        decision_path = output_root / "decisions" / relative.with_suffix(".json")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision["output_hashes"]["state"] != sha256_file(
            output_root / "states" / relative.with_suffix(".npz")
        ):
            raise RuntimeError(f"merged state hash mismatch: {record.record_id}")
        if decision["output_hashes"]["mesh"] != sha256_file(
            output_root / "meshes" / relative.with_suffix(".obj")
        ):
            raise RuntimeError(f"merged mesh hash mismatch: {record.record_id}")
    implementations = {item["implementation_sha256"] for item in summaries}
    if len(implementations) != 1:
        raise RuntimeError("hand implementation differed across workers")
    report = {
        "schema_version": "signalign.hand-refinement-summary.v1",
        "status": "ok",
        "frames": len(records),
        "manifest_frames": len(records),
        "accepted_frames": sum(int(item["accepted_frames"]) for item in summaries),
        "accepted_hands": sum(int(item["accepted_hands"]) for item in summaries),
        "full_frame_fallbacks": sum(int(item["full_frame_fallbacks"]) for item in summaries),
        "finger_radius_deg": radius_deg,
        "optimization_steps": steps,
        "learning_rate": learning_rate,
        "residual_prior": residual_prior,
        "wrist_locked": True,
        "explicit_bone_normalization": False,
        "confidence_filtering": False,
        "transformer": False,
        "objective_uses_ground_truth": False,
        "implementation_sha256": implementations.pop(),
        "parallel_workers": len(partitions),
        "batch_partition_preserved": True,
        "shards": summaries,
    }
    atomic_write_json(output_root / "summary.json", report)
    shutil.rmtree(shard_root)
    return report
