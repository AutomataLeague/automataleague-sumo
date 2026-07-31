"""Per-joint action scale: the legs and the arms want different amounts of range.

The G1's home pose is a relaxed carry with the elbows bent 1.28 rad. At a uniform
action_scale of 0.5 the elbow can only reach 0.78 rad, so the arm can never get
within 45 degrees of straight and simply hangs — not a policy choice, a hard
kinematic cap. Raising the scale globally is blocked: 0.5 is the largest uniform
value that cannot crouch into an instant loss.
"""

from __future__ import annotations

import numpy as np
import pytest

from automataleague_sumo.robots import RobotSpec, get_robot


def _spec(**kwargs):
    base = dict(
        name="toy", mjcf_path="/dev/null", base_body="pelvis", nominal_height=1.0,
        joint_names=["left_knee_joint", "left_elbow_joint"],
        actuator_names=["left_knee_joint", "left_elbow_joint"],
        home_joint_qpos=np.zeros(2), action_scale=0.5)
    base.update(kwargs)
    return RobotSpec(**base)


def test_an_empty_joint_scale_is_the_scalar_everywhere():
    """The default must not change any existing robot's action space."""
    assert np.allclose(_spec().scale_vector(), [0.5, 0.5])


def test_a_multiplier_applies_only_to_matching_joints():
    """Matching by substring must not leak onto the joints it does not name —
    widening the leg window is exactly what the crouch measurement forbids."""
    v = _spec(joint_scale={"elbow": 2.5}).scale_vector()
    assert v[0] == pytest.approx(0.5), "the knee window was widened"
    assert v[1] == pytest.approx(1.25)


def test_the_config_override_rescales_without_losing_the_proportions():
    """SumoConfig.action_scale retunes a run; the per-joint multipliers describe
    the robot's proportions and must survive that."""
    spec = _spec(joint_scale={"elbow": 2.5})
    assert np.allclose(spec.scale_vector(0.2), [0.2, 0.5])


def test_a_multiplier_matching_nothing_is_an_error():
    """It would otherwise silently do nothing, and the arms would go on hanging
    while the config claimed to have fixed them."""
    with pytest.raises(ValueError, match="matches no joint"):
        _spec(joint_scale={"tentacle": 2.0})


def test_a_non_positive_multiplier_is_an_error():
    with pytest.raises(ValueError, match="must be > 0"):
        _spec(joint_scale={"elbow": 0.0})


# ------------------------------------------------------------------- the G1

def test_the_g1_can_now_straighten_its_arm():
    """The measured defect, pinned. A straight arm is elbow 0; the reachable
    minimum must be near it rather than 45 degrees short."""
    g1 = get_robot("g1")
    i = g1.joint_names.index("left_elbow_joint")
    home, scale = float(g1.home_joint_qpos[i]), float(g1.scale_vector()[i])
    assert home - scale < 0.10, (
        f"the elbow bottoms out at {home - scale:.2f} rad "
        f"({np.degrees(home - scale):.0f} deg of bend) and still cannot push")


def test_the_g1_leg_window_is_untouched():
    """0.5 is the largest window that cannot crouch into an instant loss
    (tools/measure_reach.py). Widening it would trade a hanging arm for a policy
    that can fold itself into a defeat."""
    g1 = get_robot("g1")
    v = g1.scale_vector()
    for i, name in enumerate(g1.joint_names):
        if any(k in name for k in ("hip", "knee", "ankle", "waist")):
            assert v[i] == pytest.approx(0.5), name
