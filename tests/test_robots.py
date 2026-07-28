import numpy as np
import pytest

from automataleague_sumo.robots import ROBOTS, get_robot


def test_g1_is_registered():
    assert "g1" in ROBOTS


def test_g1_spec_shape():
    g1 = get_robot("g1")
    assert g1.name == "g1"
    assert g1.n_joints == 29
    assert g1.action_dim == 29
    assert g1.proprio_dim == 9 + 3 * 29
    assert g1.base_body == "pelvis"
    assert g1.nominal_height == pytest.approx(0.79)
    assert len(g1.home_joint_qpos) == 29


def test_g1_uses_the_mjx_model():
    # Primitive colliders. The mesh-collider g1.xml is unusable with two humanoids
    # per world under MuJoCo-Warp.
    assert get_robot("g1").mjcf_path.endswith("g1_mjx.xml")


def test_g1_joint_and_actuator_names_match_the_compiled_model():
    import mujoco

    g1 = get_robot("g1")
    model = g1.load_spec().compile()
    for name in g1.joint_names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0, name
    for name in g1.actuator_names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0, name
    assert model.nu == 29


def test_g1_home_qpos_is_a_stable_standing_pose():
    """Dropping the robot at nominal height in its home stance must not collapse."""
    import mujoco

    g1 = get_robot("g1")
    spec = g1.load_spec()
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[2] = g1.nominal_height
    data.qpos[7:] = g1.home_joint_qpos
    data.ctrl[:] = g1.home_joint_qpos
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert data.qpos[2] > 0.6, f"robot collapsed to z={data.qpos[2]:.3f}"


def test_foot_geoms_exist():
    import mujoco

    g1 = get_robot("g1")
    model = g1.load_spec().compile()
    assert len(g1.foot_geoms) == 8
    for name in g1.foot_geoms:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, name


def test_unknown_robot_raises():
    with pytest.raises(ValueError, match="Unknown robot"):
        get_robot("nope")


def test_home_qpos_length_mismatch_raises():
    from automataleague_sumo.robots.base import RobotSpec

    with pytest.raises(ValueError, match="home_joint_qpos"):
        RobotSpec(
            name="bad", mjcf_path="x.xml", base_body="b", nominal_height=1.0,
            joint_names=["j1", "j2"], actuator_names=["j1", "j2"],
            home_joint_qpos=np.zeros(3),
        )
