"""The common SMPL-X decoder and its input conventions."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .geometry import so3_log_map

PARAMETER_SIZES = {
    "betas": 10, "global_orient": 3, "body_pose": 63,
    "left_hand_pose": 45, "right_hand_pose": 45, "jaw_pose": 3,
    "leye_pose": 3, "reye_pose": 3, "expression": 10, "transl": 3,
}
BOUNDARY = np.array([1.0, -1.0, -1.0], dtype=np.float32)
HAND_JOINTS = (
    (20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70),
    (21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75),
)


def copy_parameters(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy finite, batched axis-angle parameters; never infer a hand mean."""
    result = {}
    count = None
    for key, width in PARAMETER_SIZES.items():
        value = np.asarray(values[key], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != width or not np.isfinite(value).all():
            raise ValueError(f"{key} must be a finite [frames, {width}] array")
        if count is None:
            count = len(value)
        if not count or len(value) != count:
            raise ValueError("All parameters must contain the same nonzero number of frames")
        result[key] = value.copy()
    return result


class Decoder:
    """Neutral SMPL-X with residual hand poses around its learned hand mean.

    Input translations are in meters. Vertices/joints returned by ``decode``
    use the exported camera frame, with [1, -1, -1] applied exactly once.
    Initializer reference vertices must already use that exported frame.
    """

    def __init__(self, model_root: str | Path, device: str = "cuda") -> None:
        import smplx

        self.device = torch.device(device)
        self.model = smplx.create(
            str(model_root), model_type="smplx", gender="neutral", num_betas=10,
            use_pca=False, use_face_contour=True, flat_hand_mean=False,
        ).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.boundary = self.tensor(BOUNDARY)

    def tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def parameters(self, values: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {key: self.tensor(values[key]) for key in PARAMETER_SIZES}

    def decode(
        self, parameters: Mapping[str, torch.Tensor],
        rotations: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        values = dict(parameters)
        if rotations is not None:
            values["left_hand_pose"] = so3_log_map(rotations[0]).reshape(-1, 45)
            values["right_hand_pose"] = so3_log_map(rotations[1]).reshape(-1, 45)
        output = self.model(**values, return_verts=True)
        return {
            "vertices": output.vertices * self.boundary,
            "joints": output.joints * self.boundary,
            "left_hand_pose": values["left_hand_pose"],
            "right_hand_pose": values["right_hand_pose"],
        }
