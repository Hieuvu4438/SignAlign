#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def rotmat_to_axis_angle_batch(rotmats: torch.Tensor) -> torch.Tensor:
    # rotmats: [N, 15, 3, 3] -> [N, 15, 3]
    n, j = rotmats.shape[:2]
    aa = []
    for i in range(n):
        aa_i = []
        for k in range(j):
            r = rotmats[i, k].detach().cpu().numpy().astype(np.float32)
            vec, _ = cv2.Rodrigues(r)
            aa_i.append(vec.reshape(3))
        aa.append(np.stack(aa_i, axis=0))
    return torch.from_numpy(np.stack(aa, axis=0)).float()


def collect_images(img_folder: str, exts):
    paths = []
    root = Path(img_folder)
    for ext in exts:
        paths.extend(root.glob(ext))
    return sorted(paths)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_record_key(image_name: str, image_root: str | None) -> str:
    """Map a DexAvatar filename to SignAlign's ``sign/source_frame_id`` key."""
    if image_root is None:
        return image_name
    match = re.search(r"\d+", Path(image_name).stem)
    if match is None:
        raise ValueError(f"image filename has no frame identifier: {image_name}")
    return f"{Path(image_root).name}/{int(match.group())}"


def validate_frame_manifest(manifest: dict) -> None:
    records = manifest.get("records", [])
    if len(records) != int(manifest.get("frame_count", -1)):
        raise ValueError("Frame-manifest count does not match its records")
    image_keys = [str(record.get("image_key", "")) for record in records]
    if any(not key for key in image_keys) or len(set(image_keys)) != len(image_keys):
        raise ValueError("Frame-manifest image keys must be nonempty and unique")

    source_maps = {
        "image_path": manifest.get("image_sha256", {}),
        "video_path": manifest.get("video_sha256", {}),
    }
    for source_field, expected_hashes in source_maps.items():
        referenced = {
            str(Path(record[source_field]).resolve())
            for record in records
            if source_field in record
        }
        if referenced != set(expected_hashes):
            raise ValueError(
                f"Frame-manifest {source_field} set does not match its hash map"
            )
        for source in sorted(referenced):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256(path) != str(expected_hashes[source]):
                raise ValueError(f"Source hash mismatch: {path}")


def git_provenance(repository: Path) -> tuple[str, list[str]]:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changes = subprocess.run(
        ["git", "-C", str(repository), "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return revision, changes


def iter_input_images(
    img_folder: str | None, frame_manifest: str | None, exts,
    *, verify_manifest: bool = True,
):
    if frame_manifest is None:
        for img_path in collect_images(img_folder, exts):
            yield img_path.name, cv2.imread(str(img_path))
        return

    manifest_path = Path(frame_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {
        "cusp_sl_wilor_frame_manifest_v1",
        "cusp_sl_wilor_frame_manifest_v2",
        "signalign.wilor-frame-manifest.v1",
    }:
        raise ValueError(f"Unsupported frame manifest schema: {manifest.get('schema_version')}")
    if verify_manifest:
        validate_frame_manifest(manifest)
    current_video = None
    capture = None
    try:
        for record in manifest["records"]:
            if "image_path" in record:
                image_path = Path(record["image_path"]).resolve()
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Could not decode image: {image_path}")
                height, width = image.shape[:2]
                expected = (
                    int(record["expected_width"]),
                    int(record["expected_height"]),
                )
                if (width, height) != expected:
                    raise ValueError(
                        f"Decoded size {(width, height)} != manifest {expected}: "
                        f"{image_path}"
                    )
                yield str(record["image_key"]), image
                continue
            video = str(Path(record["video_path"]).resolve())
            if video != current_video:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(video)
                if not capture.isOpened():
                    raise RuntimeError(f"Could not open video: {video}")
                current_video = video
            frame_number = int(record["frame_number"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Could not decode {video} frame {frame_number}")
            height, width = image.shape[:2]
            expected = (int(record["expected_width"]), int(record["expected_height"]))
            if (width, height) != expected:
                raise ValueError(
                    f"Decoded size {(width, height)} != manifest {expected}: "
                    f"{video} frame {frame_number}"
                )
            yield str(record["image_key"]), image
    finally:
        if capture is not None:
            capture.release()


def chunked(iterable, size):
    """Yield bounded lists while preserving the manifest's exact frame order."""
    if size <= 0:
        raise ValueError("Batch size must be positive")
    group = []
    for item in iterable:
        group.append(item)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


def main():
    parser = argparse.ArgumentParser(description="WiLoR -> HaMeR compatibility exporter")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--img_folder", type=str)
    inputs.add_argument("--frame_manifest", type=str)
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Clean checkout of the official WiLoR repository",
    )
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--file_type", nargs='+', default=['*.jpg', '*.png', '*.jpeg'])
    parser.add_argument("--fast", action='store_true', default=False)
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=1,
        help="Number of source frames jointly sent through detector/hand batching",
    )
    parser.add_argument(
        "--hand_batch_size",
        type=int,
        default=16,
        help="Maximum detected hand crops per WiLoR forward pass",
    )
    args = parser.parse_args()
    if args.frame_batch_size <= 0 or args.hand_batch_size <= 0:
        parser.error("--frame_batch_size and --hand_batch_size must be positive")

    img_folder = str(Path(args.img_folder).resolve()) if args.img_folder else None
    frame_manifest = (
        str(Path(args.frame_manifest).resolve()) if args.frame_manifest else None
    )
    manifest_payload = None
    if frame_manifest is not None:
        manifest_payload = json.loads(
            Path(frame_manifest).read_text(encoding="utf-8")
        )
        validate_frame_manifest(manifest_payload)
    out_folder = Path(args.out_folder).resolve()
    repository = args.repo.resolve()
    if not (repository / "wilor").is_dir():
        raise FileNotFoundError(f"not a WiLoR checkout: {repository}")
    sys.path.insert(0, str(repository))
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    repository_commit, repository_changes = git_provenance(repository)
    started = time.perf_counter()
    checkpoint_path = (args.checkpoint or repository / 'pretrained_models' / 'wilor_final.ckpt').resolve()
    detector_path = (args.detector or repository / 'pretrained_models' / 'detector.pt').resolve()
    config_path = (args.model_config or repository / 'pretrained_models' / 'model_config.yaml').resolve()
    for asset in (checkpoint_path, detector_path, config_path):
        if not asset.is_file():
            raise FileNotFoundError(asset)
    os.chdir(repository)

    hamer_out = out_folder / "hamer"
    wilor_out = out_folder / "wilor"
    if (
        (hamer_out.exists() or wilor_out.exists())
        and os.environ.get("WILOR_ALLOW_EXISTING") != "1"
    ):
        raise FileExistsError(f"WiLoR output already exists under: {out_folder}")
    hamer_out.mkdir(parents=True, exist_ok=True)
    wilor_out.mkdir(parents=True, exist_ok=True)

    model, model_cfg = load_wilor(
        checkpoint_path=str(checkpoint_path),
        cfg_path=str(config_path),
    )
    if args.fast:
        torch.set_float32_matmul_precision('high')
        model = model.half()
        model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True

    detector = YOLO(str(detector_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    results_hamer = {}
    results_wilor = {
        "meta": {
            "format": "wilor_raw_v3",
            "left_hand_convention": "canonical_right_requires_x_reflection",
            "rotation_representation": "SO3_matrix",
            "camera_translation_units": "WiLoR_full_image_camera_units",
            "image_size_order": "width_height",
            "detection_selection": "highest_detector_confidence_per_side",
            "frame_manifest": frame_manifest,
            "frame_manifest_sha256": (
                sha256(Path(frame_manifest)) if frame_manifest is not None else None
            ),
            "frame_manifest_sources_verified": frame_manifest is not None,
            "exporter_sha256": sha256(Path(__file__).resolve()),
            "wilor_repository_commit": repository_commit,
            "wilor_repository_dirty": bool(repository_changes),
            "wilor_repository_tracked_changes": repository_changes,
            "wilor_checkpoint_sha256": sha256(checkpoint_path),
            "detector_checkpoint_sha256": sha256(detector_path),
            "model_config_sha256": sha256(config_path),
            "base_focal_length": float(model_cfg.EXTRA.FOCAL_LENGTH),
            "rescale_factor": args.rescale_factor,
            "fast_mode": args.fast,
            "frame_batch_size": args.frame_batch_size,
            "hand_batch_size": args.hand_batch_size,
        },
        "images": {},
    }

    expected_progress = (
        int(manifest_payload["frame_count"])
        if manifest_payload is not None
        else None
    )
    processed = 0

    def log_progress() -> None:
        if processed % 25 == 0 or processed == expected_progress:
            elapsed = time.perf_counter() - started
            total = expected_progress if expected_progress is not None else "?"
            print(
                f"[wilor] {processed}/{total} frames "
                f"({processed / max(elapsed, 1e-9):.2f} fps)",
                flush=True,
            )

    image_iterator = iter_input_images(
        img_folder, frame_manifest, args.file_type, verify_manifest=False
    )
    for frame_group in chunked(image_iterator, args.frame_batch_size):
        if any(image is None for _, image in frame_group):
            bad = [key for key, image in frame_group if image is None]
            raise RuntimeError(f"Could not decode source frames: {bad[:3]}")

        # Ultralytics accepts a list of numpy images and returns one Results
        # object per frame.  This preserves per-frame NMS while amortizing the
        # detector launch over the group.
        detection_groups = detector(
            [image for _, image in frame_group], conf=0.3, verbose=False
        )
        if len(detection_groups) != len(frame_group):
            raise RuntimeError("Detector result count differs from source batch")

        states = []
        hand_samples = []
        for frame_index, ((img_key, img_cv2), detections) in enumerate(
            zip(frame_group, detection_groups)
        ):
            bboxes, is_right, detector_confidence = [], [], []
            for det in detections:
                box = det.boxes.data.cpu().detach().squeeze().numpy()
                cls = det.boxes.cls.cpu().detach().squeeze().item()
                bboxes.append(box[:4].tolist())
                is_right.append(cls)
                detector_confidence.append(float(box[4]))

            state = {
                "img_key": img_key,
                "raw_key": public_record_key(img_key, img_folder),
                "boxes": None,
                "detector_confidence": detector_confidence,
                "pred_kp2d": [],
                "pred_kp3d": [],
                "box_center": [],
                "box_size": [],
                "right": [],
                "cam_t": [],
                "global_orient": [],
                "pose_rotmat": [],
                "betas": [],
                "raw_hands": [],
            }
            states.append(state)
            if not bboxes:
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            state["boxes"] = boxes
            dataset = ViTDetDataset(
                model_cfg,
                img_cv2,
                boxes,
                right,
                rescale_factor=args.rescale_factor,
                fp16=args.fast,
            )
            for person_index in range(len(dataset)):
                sample = dataset[person_index]
                sample["_frame_index"] = np.int64(frame_index)
                hand_samples.append(sample)

        if hand_samples:
            dataloader = torch.utils.data.DataLoader(
                hand_samples,
                batch_size=args.hand_batch_size,
                shuffle=False,
                num_workers=0,
            )
            for batch in dataloader:
                frame_indices = batch.pop("_frame_index")
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                multiplier = 2 * batch['right'] - 1
                pred_cam = out['pred_cam']
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                box_center = batch['box_center'].float()
                box_size = batch['box_size'].float()
                img_size = batch['img_size'].float()
                scaled_focal = (
                    float(model_cfg.EXTRA.FOCAL_LENGTH)
                    / model_cfg.MODEL.IMAGE_SIZE
                    * img_size.max(dim=1).values
                )
                pred_cam_t_full = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal
                ).detach().cpu()

                pred_kp2d = out['pred_keypoints_2d'].detach().cpu()
                pred_kp3d = out['pred_keypoints_3d'].detach().cpu()
                pred_global_orient = (
                    out['pred_mano_params']['global_orient']
                    .detach()
                    .cpu()
                    .reshape(-1, 1, 3, 3)
                )
                pred_pose_rotmat = (
                    out['pred_mano_params']['hand_pose']
                    .detach()
                    .cpu()
                    .reshape(-1, 15, 3, 3)
                )
                pred_pose_aa = rotmat_to_axis_angle_batch(pred_pose_rotmat)
                pred_betas = (
                    out['pred_mano_params']['betas']
                    .detach()
                    .cpu()
                    .reshape(-1, 10)
                )

                for index in range(pred_kp2d.shape[0]):
                    state = states[int(frame_indices[index])]
                    person_id = int(batch['personid'][index].detach().cpu())
                    state["pred_kp2d"].append(pred_kp2d[index])
                    state["pred_kp3d"].append(pred_kp3d[index])
                    state["box_center"].append(box_center[index].detach().cpu())
                    state["box_size"].append(box_size[index].detach().cpu())
                    state["right"].append(batch['right'][index].detach().cpu())
                    state["cam_t"].append(pred_cam_t_full[index])
                    state["global_orient"].append(pred_global_orient[index])
                    state["pose_rotmat"].append(pred_pose_rotmat[index])
                    state["betas"].append(pred_betas[index])
                    state["raw_hands"].append({
                        'pred_keypoints_2d': pred_kp2d[index].numpy(),
                        'pred_keypoints_3d': pred_kp3d[index].numpy(),
                        # Preserve the complete MANO pose contract needed by
                        # downstream wrist/forearm alignment. Left-hand values
                        # remain in WiLoR's canonical-right convention.
                        'pred_mano_global_orient_rotmat': pred_global_orient[index].numpy(),
                        'pred_mano_pose_rotmat': pred_pose_rotmat[index].numpy(),
                        'pred_mano_pose_axis_angle': pred_pose_aa[index].numpy(),
                        'pred_mano_betas': pred_betas[index].numpy(),
                        'box_center': box_center[index].detach().cpu().numpy(),
                        'box_size': float(box_size[index].detach().cpu()),
                        'detector_box_xyxy': state["boxes"][person_id].astype(np.float32),
                        'detector_confidence': state["detector_confidence"][person_id],
                        'is_right': float(batch['right'][index].detach().cpu()),
                        'cam_t': pred_cam_t_full[index].numpy(),
                        'focal_length_px': float(scaled_focal[index].detach().cpu()),
                        'image_size_wh': img_size[index].detach().cpu().numpy(),
                    })

        for state in states:
            processed += 1
            img_key = state["img_key"]
            if not state["pred_kp2d"]:
                # Keep detector dropout explicit so consumers can select the
                # target-independent base-pose fallback.
                results_wilor['images'][state["raw_key"]] = {'hands': []}
                log_progress()
                continue

            pred_kp2d_t = torch.stack(state["pred_kp2d"], dim=0).float()
            pred_kp3d_t = torch.stack(state["pred_kp3d"], dim=0).float()
            box_center_t = torch.stack(state["box_center"], dim=0).float()
            box_size_t = torch.stack(state["box_size"], dim=0).float()
            right_t = torch.stack(state["right"], dim=0).float()
            cam_t_t = torch.stack(state["cam_t"], dim=0).float()
            global_orient_t = torch.stack(state["global_orient"], dim=0).float()
            pose_rotmat_t = torch.stack(state["pose_rotmat"], dim=0).float()
            betas_t = torch.stack(state["betas"], dim=0).float()
            pred_dict = {
                'pred_keypoints_2d': pred_kp2d_t,
                'pred_keypoints_3d': pred_kp3d_t,
                'pred_mano_params': {
                    'global_orient': global_orient_t,
                    'hand_pose': pose_rotmat_t,
                    'betas': betas_t,
                }
            }
            results_hamer[img_key] = [
                pred_dict,
                box_center_t,
                box_size_t,
                right_t,
                cam_t_t.numpy(),
            ]
            results_wilor['images'][state["raw_key"]] = {'hands': state["raw_hands"]}
            log_progress()

    if frame_manifest is not None:
        manifest = json.loads(Path(frame_manifest).read_text(encoding="utf-8"))
        expected_frames = int(manifest["frame_count"])
        if len(results_wilor["images"]) != expected_frames:
            raise ValueError(
                f"WiLoR output coverage {len(results_wilor['images'])} "
                f"!= manifest {expected_frames}"
            )
    results_wilor["meta"]["frame_count"] = len(results_wilor["images"])
    results_wilor["meta"]["detector_dropout_frames"] = sum(
        not record["hands"] for record in results_wilor["images"].values()
    )
    runtime_seconds = time.perf_counter() - started
    results_wilor["meta"]["runtime_seconds"] = runtime_seconds
    results_wilor["meta"]["frames_per_second"] = (
        len(results_wilor["images"]) / runtime_seconds
    )
    results_wilor["meta"]["device"] = str(device)
    results_wilor["meta"]["cuda_device_name"] = (
        torch.cuda.get_device_name(device) if device.type == 'cuda' else None
    )
    results_wilor["meta"]["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == 'cuda'
        else None
    )

    with open(hamer_out / 'hamer.pkl', 'wb') as f:
        pickle.dump(results_hamer, f)
    with open(wilor_out / 'wilor.pkl', 'wb') as f:
        pickle.dump(results_wilor, f)

    print(f"Saved: {hamer_out / 'hamer.pkl'}")
    print(f"Saved: {wilor_out / 'wilor.pkl'}")


if __name__ == '__main__':
    main()
