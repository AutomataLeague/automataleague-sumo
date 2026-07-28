import math

import pytest
import torch

from automataleague_sumo.envs.sumo.spatial import (
    projected_gravity,
    quat_rotate_inverse,
    tilt_angle,
    yaw_from_quat,
)

_IDENTITY = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def _yaw_q(yaw: float) -> torch.Tensor:
    return torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])


def test_identity_rotation_is_a_no_op():
    v = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(quat_rotate_inverse(_IDENTITY, v), v, atol=1e-6)


def test_inverse_rotation_undoes_a_yaw():
    """A world +x vector seen from a body yawed by +90 deg points along body -y."""
    out = quat_rotate_inverse(_yaw_q(math.pi / 2), torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(out, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6)


def test_projected_gravity_is_down_when_upright():
    assert torch.allclose(
        projected_gravity(_IDENTITY), torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6)


def test_projected_gravity_is_unit_length_under_any_rotation():
    q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    assert torch.linalg.norm(projected_gravity(q), dim=-1).item() == pytest.approx(1.0)


def test_tilt_angle_is_zero_when_upright_and_grows_when_tipped():
    assert tilt_angle(_IDENTITY).item() == pytest.approx(0.0, abs=1e-6)
    # 90 degrees about x: body up now points along world -y, so tilt is pi/2.
    q = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])
    assert tilt_angle(q).item() == pytest.approx(math.pi / 2, abs=1e-6)


def test_tilt_angle_is_invariant_to_yaw():
    for yaw in (0.0, 1.0, -2.5, math.pi):
        assert tilt_angle(_yaw_q(yaw)).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.3, math.pi / 2])
def test_yaw_from_quat_round_trips(yaw):
    assert yaw_from_quat(_yaw_q(yaw)).item() == pytest.approx(yaw, abs=1e-6)


def test_everything_is_batched():
    q = torch.cat([_IDENTITY, _yaw_q(1.0), _yaw_q(-2.0)], dim=0)
    assert projected_gravity(q).shape == (3, 3)
    assert tilt_angle(q).shape == (3,)
    assert yaw_from_quat(q).shape == (3,)
