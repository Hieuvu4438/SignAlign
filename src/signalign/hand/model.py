"""Frozen SMPL-X state used by bounded finger-only refinement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from signalign.model.kinematics import so3_exp_map, so3_log_map

BOUNDARY_X180 = torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32)
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


@dataclass
class CanonicalBatch:
    arrays: dict[str, torch.Tensor]
    cached_vertices: torch.Tensor

    @classmethod
    def from_npz(cls, paths: Sequence[Path], device: str) -> CanonicalBatch:
        loaded: list[dict[str, np.ndarray]] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["coord_frame"]) != "evaluator_camera":
                    raise RuntimeError(f"coordinate contract mismatch: {path}")
                if str(archive["unit"]) != "meter":
                    raise RuntimeError(f"unit contract mismatch: {path}")
                loaded.append(
                    {
                        key: np.asarray(archive[key], dtype=np.float32)
                        for key in (*STATE_KEYS, "vertices")
                    }
                )
        arrays = {
            key: torch.as_tensor(
                np.concatenate([item[key] for item in loaded], axis=0),
                dtype=torch.float32,
                device=device,
            )
            for key in STATE_KEYS
        }
        vertices = torch.as_tensor(
            np.stack([item["vertices"] for item in loaded]),
            dtype=torch.float32,
            device=device,
        )
        return cls(arrays=arrays, cached_vertices=vertices)


class FrozenSMPLX:
    """Decode finger rotations while every non-finger parameter stays fixed."""

    def __init__(self, model_root: Path, state: CanonicalBatch) -> None:
        import smplx

        self.state = state
        self.model = smplx.create(
            str(model_root),
            model_type="smplx",
            gender="neutral",
            num_betas=10,
            use_pca=False,
            use_face_contour=True,
        ).to(state.cached_vertices.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.boundary = BOUNDARY_X180.to(state.cached_vertices.device)
        self.left_rotation = so3_exp_map(
            state.arrays["left_hand_pose"].reshape(-1, 15, 3)
        )
        self.right_rotation = so3_exp_map(
            state.arrays["right_hand_pose"].reshape(-1, 15, 3)
        )

    def decode(
        self,
        left_rotation: torch.Tensor,
        right_rotation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        values = self.state.arrays
        left_pose = so3_log_map(left_rotation).reshape(-1, 45)
        right_pose = so3_log_map(right_rotation).reshape(-1, 45)
        output = self.model(
            betas=values["betas"],
            global_orient=values["global_orient"],
            body_pose=values["body_pose"],
            left_hand_pose=left_pose,
            right_hand_pose=right_pose,
            jaw_pose=values["jaw_pose"],
            leye_pose=values["leye_pose"],
            reye_pose=values["reye_pose"],
            expression=values["expression"],
            transl=values["transl"],
            return_verts=True,
        )
        return {
            "vertices": output.vertices * self.boundary,
            "joints": output.joints * self.boundary,
            "left_hand_pose": left_pose,
            "right_hand_pose": right_pose,
        }
