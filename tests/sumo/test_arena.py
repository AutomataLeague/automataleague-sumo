import math

import pytest
import torch

from automataleague_sumo.envs.sumo.arena import (
    SpawnPose,
    radial_distance,
    spawn_poses,
    yaw_quat,
)
from automataleague_sumo.envs.sumo.config import SumoConfig


def test_spawns_are_diametrically_opposite():
    a, b = spawn_poses(SumoConfig())
    assert a.x == pytest.approx(-b.x)
    assert a.y == pytest.approx(b.y)
    assert a.y == pytest.approx(0.0)


def test_spawns_sit_at_the_configured_radius():
    cfg = SumoConfig()
    a, b = spawn_poses(cfg)
    for p in (a, b):
        assert math.hypot(p.x, p.y) == pytest.approx(cfg.spawn_radius)


def test_spawns_face_each_other():
    a, b = spawn_poses(SumoConfig())
    # A sits at -x facing +x; B sits at +x facing -x.
    assert a.yaw == pytest.approx(0.0)
    assert abs(b.yaw) == pytest.approx(math.pi)
    # Each robot's heading points at the other.
    for p, q in ((a, b), (b, a)):
        heading = (math.cos(p.yaw), math.sin(p.yaw))
        toward = (q.x - p.x, q.y - p.y)
        dot = heading[0] * toward[0] + heading[1] * toward[1]
        assert dot > 0, "robot is facing away from its opponent"


def test_spawn_is_a_180_degree_rotation_of_the_opponent():
    """The load-bearing symmetry: rotating the arena by pi maps side A onto side B.

    Every later mirror-symmetry claim rests on this.
    """
    a, b = spawn_poses(SumoConfig())
    theta = math.pi
    c, s = math.cos(theta), math.sin(theta)
    # Rotate A's pose about the ring centre by pi; it must land exactly on B.
    assert c * a.x - s * a.y == pytest.approx(b.x)
    assert s * a.x + c * a.y == pytest.approx(b.y)
    assert math.isclose(abs(b.yaw - (a.yaw + theta)), 0.0, abs_tol=1e-6)


def test_yaw_quat_is_a_unit_quaternion_in_mujoco_order():
    q = yaw_quat(0.0)
    assert q == [1.0, 0.0, 0.0, 0.0]
    q = yaw_quat(math.pi / 2)
    assert len(q) == 4
    assert sum(c * c for c in q) == pytest.approx(1.0)
    # Rotation about z only: the x and y components stay zero.
    assert q[1] == pytest.approx(0.0)
    assert q[2] == pytest.approx(0.0)
    assert q[3] == pytest.approx(math.sin(math.pi / 4))


def test_radial_distance_is_batched():
    xy = torch.tensor([[0.0, 0.0], [3.0, 4.0], [-1.0, 0.0]])
    assert torch.allclose(radial_distance(xy), torch.tensor([0.0, 5.0, 1.0]))


def test_spawn_pose_is_frozen():
    a, _ = spawn_poses(SumoConfig())
    assert isinstance(a, SpawnPose)
    with pytest.raises(Exception):
        a.x = 1.0
