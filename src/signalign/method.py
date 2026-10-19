"""Signer-consistent canonicalization followed by bounded finger retargeting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .canonical import canonicalize
from .identity import fit_identity
from .model import Decoder, copy_parameters
from .refinement import refine_hands


@dataclass(frozen=True)
class Reconstruction:
    parameters: dict[str, np.ndarray]
    vertices: np.ndarray
    shared_shape: np.ndarray
    calibration_indices: np.ndarray
    refined_hands: np.ndarray


def prepare_wilor_joints(joints: np.ndarray) -> np.ndarray:
    """Convert raw right-hand-model WiLoR joints into anatomical hands.

    Shape is [frames, 2, 21, 3], sides [left, right], wrist-first, then
    thumb/index/middle/ring/little. Call once on raw WiLoR joints; do not
    reflect observations that are already anatomically converted.
    """
    value = np.asarray(joints, dtype=np.float32).copy()
    if value.ndim != 4 or value.shape[1:] != (2, 21, 3):
        raise ValueError("Expected [frames, 2, 21, 3] WiLoR joints")
    value[:, 0, :, 0] *= -1
    return value - value[:, :, :1, :]


class SignAlign:
    """Reconstruct one signer from frozen initializer and hand observations.

    Supply batched SMPL-X parameters and actual initializer mesh vertices in
    meters in the exported camera frame. Hand parameters follow the decoder's
    learned-hand-mean residual convention. Keep the initializer reference mesh;
    do not silently replace it with a differently decoded reconstruction.

    ``hand_vertex_ids`` contains left/right 778-vertex MANO-to-SMPL-X mappings.
    ``model_root`` is the licensed SMPL-X model directory. No dataset
    annotations or ground-truth geometry are inputs to this API.
    """

    def __init__(self, model_root: str | Path, hand_vertex_ids: Sequence[np.ndarray],
                 device: str = "cuda") -> None:
        if len(hand_vertex_ids) != 2:
            raise ValueError("Provide left and right hand vertex correspondences")
        ids = [np.asarray(value, dtype=np.int64) for value in hand_vertex_ids]
        for value in ids:
            if (value.shape != (778,) or len(np.unique(value)) != 778
                    or value.min() < 0 or value.max() >= 10475):
                raise ValueError("Each hand needs 778 unique SMPL-X vertex indices")
        self.decoder = Decoder(model_root, device)
        self.hand_vertex_ids = tuple(torch.as_tensor(value, device=self.decoder.device)
                                     for value in ids)

    def reconstruct(
        self, parameters: Mapping[str, np.ndarray], initializer_vertices: np.ndarray,
        expert_joints: np.ndarray, available: np.ndarray,
        clip_ids: Sequence[str] | None = None, *, calibration_frames: int = 200,
    ) -> Reconstruction:
        """Fit one shared signer shape and refine the supplied frames.

        Frames within each clip must be in temporal order. Expert joints must
        be anatomical left/right, wrist-first; camera and scale matching are
        unnecessary. Invalid/unavailable hands retain their canonical state.
        Outputs follow input order and inputs are never modified.
        """
        source = copy_parameters(parameters)
        count = len(source["betas"])
        reference = np.asarray(initializer_vertices, dtype=np.float32)
        expert = np.asarray(expert_joints, dtype=np.float32)
        flags = np.asarray(available, dtype=bool)
        clips = np.asarray([""] * count if clip_ids is None else clip_ids, dtype=str)
        if reference.shape != (count, 10475, 3) or not np.isfinite(reference).all():
            raise ValueError("Initializer vertices must be finite [frames, 10475, 3]")
        if expert.shape != (count, 2, 21, 3) or flags.shape != (count, 2):
            raise ValueError("Expected joints [frames,2,21,3] and availability [frames,2]")
        if clips.shape != (count,):
            raise ValueError("Provide one clip ID per frame")
        order = np.argsort(clips, kind="stable")
        undo = np.argsort(order)
        source = {key: value[order] for key, value in source.items()}
        reference = reference[order]
        with torch.enable_grad():
            shared, selected = fit_identity(
                self.decoder, source, reference, self.hand_vertex_ids, calibration_frames,
            )
            canonical, meshes = canonicalize(
                self.decoder, source, reference, shared, self.hand_vertex_ids, clips[order],
            )
            final, meshes, accepted = refine_hands(
                self.decoder, canonical, meshes, expert[order], flags[order],
            )
        return Reconstruction(
            parameters={key: value[undo] for key, value in final.items()},
            vertices=meshes[undo], shared_shape=shared.copy(),
            calibration_indices=order[selected], refined_hands=accepted[undo],
        )
