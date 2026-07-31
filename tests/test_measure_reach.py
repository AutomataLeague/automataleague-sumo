"""The action-scale envelope measurement.

Light, because this is a measurement tool rather than task logic, but not absent:
a tool that silently ignored `action_scale` would report a flat table and the
conclusion drawn from it would be exactly backwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.measure_reach import Envelope


@pytest.fixture(scope="module")
def envelope():
    return Envelope("g1")


def test_the_home_pose_measures_a_plausible_robot(envelope):
    """Anchors the geometry against numbers known from elsewhere: the G1 stands
    0.784 m at the pelvis, and the home stance has the feet together."""
    home = envelope.home_geometry
    assert home["crouch"] == pytest.approx(0.784, abs=0.02), "leg length is wrong"
    assert home["stride"] == pytest.approx(0.0, abs=0.02), "home is not feet-square"
    assert 0.1 < home["stance"] < 0.4, f"implausible hip width {home['stance']}"


def test_the_envelope_grows_with_action_scale(envelope):
    """The whole point of the tool. A version that ignored `scale` would report
    identical rows and make every conclusion drawn from the table wrong."""
    rng = np.random.default_rng(0)
    small = envelope.measure(0.2, 400, rng)
    large = envelope.measure(0.8, 400, rng)
    for metric in ("stance", "stride", "reach"):
        assert large[metric] > small[metric] + 0.05, metric
    # crouch is a leg LENGTH, so a bigger scale reaches a smaller one.
    assert large["crouch"] < small["crouch"] - 0.05


def test_a_zero_scale_reaches_only_the_home_pose(envelope):
    """The control: at scale 0 every action maps to the same commanded pose, so
    the envelope must collapse onto the home geometry no matter what is sampled."""
    rng = np.random.default_rng(0)
    at_zero = envelope.measure(0.0, 200, rng)
    for metric in ("stance", "stride", "crouch", "reach"):
        assert at_zero[metric] == pytest.approx(
            envelope.home_geometry[metric], abs=1e-6), metric


def test_joint_limits_clip_a_large_scale_and_not_a_small_one(envelope):
    """`clipped` is what would tell you more scale has stopped buying reach. It
    has to actually respond to the mechanical limits to mean that."""
    rng = np.random.default_rng(0)
    assert envelope.measure(0.1, 400, rng)["clipped"] == pytest.approx(0.0, abs=1e-3)
    assert envelope.measure(2.0, 400, rng)["clipped"] > 0.2


def test_the_envelope_uses_the_robots_per_joint_scale(envelope):
    """A scalar here measures a different robot from the one being trained.

    Verified against the real defect: before this, introducing a 2.5x arm
    multiplier left the reported arm reach unchanged at 0.42 m, because the tool
    applied one number to every joint. With it, the same measurement reports
    0.59 m. Asserted by clearing the multipliers and watching `reach` fall, since
    `reach` depends only on the arm joints.
    """
    rng = np.random.default_rng(0)
    with_arms = envelope.measure(0.5, 600, rng)["reach"]

    original = envelope.robot.joint_scale
    try:
        envelope.robot.joint_scale = {}
        without = envelope.measure(0.5, 600, rng)["reach"]
    finally:
        envelope.robot.joint_scale = original

    assert with_arms > without + 0.05, (
        f"the arm multiplier changed nothing: {without:.3f} -> {with_arms:.3f} m, "
        f"so the tool is applying a scalar")
