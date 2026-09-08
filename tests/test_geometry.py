import numpy as np
import torch

from signalign.canonical.identity import farthest_point_indices, huber_location
from signalign.hand.refinement import SMPLX_HAND_JOINTS, _fit_side, palm_canonical
from signalign.model.kinematics import apply_lie_residual


def synthetic_hand() -> torch.Tensor:
    joints = torch.zeros(1, 21, 3)
    joints[0, 5] = torch.tensor([1.0, 2.0, 0.1])
    joints[0, 9] = torch.tensor([0.0, 2.5, 0.2])
    joints[0, 17] = torch.tensor([-1.0, 2.0, -0.1])
    for index in range(1, 21):
        if not bool(joints[0, index].any()):
            joints[0, index] = torch.tensor([index % 5 - 2.0, index / 8.0, index / 30.0])
    return joints


def test_palm_canonical_removes_similarity_transform() -> None:
    hand = synthetic_hand()
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0],
         [torch.sin(angle), torch.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    transformed = 2.3 * (hand @ rotation.T) + torch.tensor([4.0, -3.0, 2.0])
    first, first_det = palm_canonical(hand)
    second, second_det = palm_canonical(transformed)
    assert torch.allclose(first, second, atol=2e-6)
    assert torch.all(first_det > 0.999)
    assert torch.all(second_det > 0.999)


def test_lie_residual_is_bounded() -> None:
    identity = torch.eye(3).reshape(1, 1, 3, 3)
    delta = torch.tensor([[[1.0, -2.0, 3.0]]])
    radius = np.deg2rad(12.0)
    _, bounded = apply_lie_residual(identity, delta, radius)
    assert float(torch.linalg.vector_norm(bounded, dim=-1).max()) <= radius + 1e-7


def test_huber_identity_resists_outlier_and_pose_sampling_is_deterministic() -> None:
    shapes = np.asarray([[1.0, -2.0], [1.1, -1.9], [0.9, -2.1], [100.0, 80.0]])
    estimate = huber_location(shapes)
    assert np.allclose(estimate, [1.0, -2.0], atol=0.2)

    features = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    first = farthest_point_indices(features, 3)
    second = farthest_point_indices(features, 3)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 3


class FrozenHandStub:
    def __init__(self) -> None:
        identity = torch.eye(3).reshape(1, 1, 3, 3)
        self.left_rotation = identity.repeat(1, 15, 1, 1)
        self.right_rotation = identity.repeat(1, 15, 1, 1)
        self.joints = torch.zeros(1, 76, 3)
        self.joints[:, list(SMPLX_HAND_JOINTS["left"])] = synthetic_hand()

    def decode(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        dependency = 0.0 * (left.sum() + right.sum())
        return {"joints": self.joints + dependency}


def test_unavailable_hand_keeps_canonical_rotations() -> None:
    model = FrozenHandStub()
    rotations, trust = _fit_side(
        model,
        synthetic_hand(),
        torch.tensor([False]),
        "left",
        radius_deg=12.0,
        steps=2,
        learning_rate=0.03,
        residual_prior=0.2,
    )
    assert torch.equal(rotations, model.left_rotation)
    assert torch.equal(trust, torch.zeros_like(trust))
