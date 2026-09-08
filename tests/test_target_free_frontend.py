from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from signalign.frontend.initializer import build_initializer_view
from signalign.frontend.wilor import (
    build_wilor_frame_manifest,
    import_wilor_sidecar,
    merge_wilor_sidecars,
    validate_wilor_cache,
)
from signalign.manifest import prepare_inference_manifests


def test_synthetic_target_free_frontend_contract(tmp_path: Path) -> None:
    rgb_root = tmp_path / "rgb"
    rgb = rgb_root / "sign_001" / "frame_0001.png"
    rgb.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), color=(10, 20, 30)).save(rgb)
    signs = tmp_path / "signs.txt"
    signs.write_text("sign_001 class_001\n", encoding="utf-8")
    segments = tmp_path / "segments.json"
    segments.write_text(json.dumps({"sign_001": [1, 1]}), encoding="utf-8")

    manifests = tmp_path / "manifests"
    summary = prepare_inference_manifests(
        rgb_root,
        signs,
        segments,
        manifests,
        expected_signs=1,
        expected_frames=1,
    )
    assert summary["target_free"] is True

    frame_manifest = tmp_path / "wilor_frames.json"
    frozen = build_wilor_frame_manifest(manifests, frame_manifest)
    assert frozen["frame_count"] == 1
    assert frozen["records"][0]["image_key"] == "sign_001/1"

    identity_rotations = np.tile(np.eye(3, dtype=np.float32), (15, 1, 1))
    raw = {
        "meta": {
            "format": "wilor_raw_v3",
            "wilor_checkpoint_sha256": "checkpoint-digest",
            "detector_checkpoint_sha256": "detector-digest",
            "wilor_repository_commit": "a" * 40,
        },
        "images": {
            "sign_001/1": {
                "hands": [
                    {
                        "is_right": 1.0,
                        "detector_confidence": 0.9,
                        "detector_box_xyxy": np.asarray([1, 2, 20, 22], np.float32),
                        "pred_keypoints_3d": np.zeros((21, 3), np.float32),
                        "pred_mano_pose_rotmat": identity_rotations,
                    }
                ]
            }
        },
    }
    sidecar = tmp_path / "wilor_raw.pkl"
    with sidecar.open("wb") as handle:
        pickle.dump(raw, handle)
    merged_sidecar = tmp_path / "wilor_merged.pkl"
    merge = merge_wilor_sidecars([sidecar], merged_sidecar)
    assert merge["frames"] == 1
    cache = tmp_path / "wilor_cache"
    imported = import_wilor_sidecar(manifests, merged_sidecar, cache)
    assert imported["frames"] == 1
    assert imported["unavailable_by_side"] == {"left": 1, "right": 0}
    validation = validate_wilor_cache(manifests, cache, tmp_path / "validation.json")
    assert validation["status"] == "ok"
    assert validation["available_right"] == 1

    primary = tmp_path / "primary"
    result = primary / "sign_001" / "smplifyx/results/frame_0001.pkl"
    mesh = primary / "sign_001" / "smplifyx/meshes/frame_0001.obj"
    result.parent.mkdir(parents=True)
    mesh.parent.mkdir(parents=True)
    result.write_bytes(b"parameters")
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    view = tmp_path / "initializer_view"
    selection = build_initializer_view(manifests, primary, primary, view)
    assert selection["primary_frames"] == 1
    assert (view / "sign_001/smplifyx/results/frame_0001.pkl").is_symlink()
