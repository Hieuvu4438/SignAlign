#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import cv2
import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IndexedHands(torch.utils.data.Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        record_key, confidence, box, dataset, hand_index = self.entries[index]
        item = dataset[hand_index]
        item["record_key"] = record_key
        item["detector_confidence"] = np.float32(confidence)
        item["detector_box_xyxy"] = np.asarray(box, dtype=np.float32)
        return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--hand-batch-size", type=int, default=16)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to

    manifest_path = args.frame_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    expected_hashes = manifest["image_sha256"]
    for record in records:
        path = Path(record["image_path"]).resolve()
        if sha256(path) != expected_hashes[str(path)]:
            raise RuntimeError(f"RGB hash mismatch: {path}")

    checkpoint = args.checkpoint.resolve()
    detector_path = args.detector.resolve()
    config_path = args.model_config.resolve()
    original_cwd = Path.cwd()
    # WiLoR resolves MANO assets relative to its repository root.
    import os
    os.chdir(repo)
    try:
        model, model_cfg = load_wilor(str(checkpoint), str(config_path))
        detector = YOLO(str(detector_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        detector = detector.to(device)
        images = {record["image_key"]: {"hands": []} for record in records}
        dropout = 0
        start_time = time.monotonic()
        for start in range(0, len(records), args.frame_batch_size):
            chunk = records[start:start + args.frame_batch_size]
            frames = []
            for record in chunk:
                frame = cv2.imread(record["image_path"])
                if frame is None:
                    raise RuntimeError(f"failed to load RGB: {record['image_path']}")
                if frame.shape[1] != record["expected_width"] or frame.shape[0] != record["expected_height"]:
                    raise RuntimeError(f"RGB size mismatch: {record['image_key']}")
                frames.append(frame)
            detections = detector(frames, conf=0.3, verbose=False)
            entries = []
            for record, frame, result in zip(chunk, frames, detections):
                data = result.boxes.data.detach().cpu().numpy()
                if data.size == 0:
                    dropout += 1
                    continue
                boxes = data[:, :4].astype(np.float32)
                confidence = data[:, 4].astype(np.float32)
                right = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
                hand_dataset = ViTDetDataset(
                    model_cfg, frame, boxes, right,
                    rescale_factor=args.rescale_factor, fp16=False,
                )
                for hand_index in range(len(hand_dataset)):
                    entries.append((
                        record["image_key"], confidence[hand_index], boxes[hand_index],
                        hand_dataset, hand_index,
                    ))
            loader = torch.utils.data.DataLoader(
                IndexedHands(entries), batch_size=args.hand_batch_size,
                shuffle=False, num_workers=0,
            )
            for batch in loader:
                record_keys = list(batch.pop("record_key"))
                batch_confidence = batch.pop("detector_confidence").numpy()
                batch_boxes = batch.pop("detector_box_xyxy").numpy()
                batch = recursive_to(batch, device)
                with torch.inference_mode():
                    output = model(batch)
                joints = output["pred_keypoints_3d"].detach().cpu().numpy()
                rotations = output["pred_mano_params"]["hand_pose"].detach().cpu().numpy()
                handedness = batch["right"].detach().cpu().numpy()
                for index, record_key in enumerate(record_keys):
                    images[record_key]["hands"].append({
                        "is_right": float(handedness[index]),
                        "detector_confidence": float(batch_confidence[index]),
                        "detector_box_xyxy": batch_boxes[index],
                        "pred_keypoints_3d": joints[index],
                        "pred_mano_pose_rotmat": rotations[index],
                    })
        runtime = time.monotonic() - start_time
    finally:
        os.chdir(original_cwd)

    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True,
    ).splitlines()
    artifact = {
        "meta": {
            "format": "wilor_raw_v3",
            "left_hand_convention": "canonical_right_requires_x_reflection",
            "rotation_representation": "SO3_matrix",
            "camera_translation_units": "WiLoR_full_image_camera_units",
            "image_size_order": "width_height",
            "detection_selection": "highest_detector_confidence_per_side",
            "frame_manifest": str(manifest_path),
            "frame_manifest_sha256": sha256(manifest_path),
            "frame_manifest_sources_verified": True,
            "exporter_sha256": sha256(Path(__file__).resolve()),
            "wilor_repository_commit": commit,
            "wilor_repository_dirty": bool(status),
            "wilor_repository_tracked_changes": status,
            "wilor_checkpoint_sha256": sha256(checkpoint),
            "detector_checkpoint_sha256": sha256(detector_path),
            "model_config_sha256": sha256(config_path),
            "base_focal_length": float(model_cfg.EXTRA.FOCAL_LENGTH),
            "rescale_factor": args.rescale_factor,
            "fast_mode": False,
            "frame_batch_size": args.frame_batch_size,
            "hand_batch_size": args.hand_batch_size,
            "frame_count": len(records),
            "detector_dropout_frames": dropout,
            "runtime_seconds": runtime,
            "frames_per_second": len(records) / max(runtime, 1e-9),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        },
        "images": images,
    }
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.out.resolve().open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({
        "frames": len(records), "dropout": dropout, "runtime_seconds": runtime,
        "fps": len(records) / max(runtime, 1e-9), "out": str(args.out.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
