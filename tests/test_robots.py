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
    assert g1.nominal_height == pytest.approx(0.784)
    assert len(g1.home_joint_qpos) == 29


def test_g1_home_stance_has_bent_knees():
    """Straight legs are a locked-knee inverted pendulum with no ankle leverage.
    The MJX model's own `home` keyframe bends them, and we follow it."""
    g1 = get_robot("g1")
    knee_idx = g1.joint_names.index("left_knee_joint")
    hip_idx = g1.joint_names.index("left_hip_pitch_joint")
    assert g1.home_joint_qpos[knee_idx] == pytest.approx(0.3)
    assert g1.home_joint_qpos[hip_idx] == pytest.approx(-0.1)


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


def test_load_spec_enables_collision_on_the_collision_geoms():
    """Upstream's MJX model ships with all contacts disabled and re-enables
    foot-floor pairs in a scene file we do not vendor. Without this override the
    robot falls straight through the floor."""
    import mujoco

    model = get_robot("g1").load_spec().compile()
    colliding = [
        g for g in range(model.ngeom)
        if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
    ]
    assert colliding, "every geom is non-colliding; load_spec did not override"
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_box_collision")
    assert model.geom_contype[foot] == 1
    assert model.geom_conaffinity[foot] == 1


def test_load_spec_leaves_visual_geoms_non_colliding():
    import mujoco

    model = get_robot("g1").load_spec().compile()
    visual = [g for g in range(model.ngeom) if model.geom_group[g] == 2]
    assert visual, "no visual geoms found; the group convention changed"
    for g in visual:
        assert model.geom_contype[g] == 0, mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, g)


def test_g1_settles_on_its_feet_without_sinking_or_exploding():
    """A physical sanity check on the vendored model, its home stance, and the
    collision override — not a balance test.

    A humanoid holding a stance under a plain position controller is an
    inverted pendulum and will eventually topple; learning to balance is what
    the standing bootstrap is for. What must hold here is that the robot lands on
    the ground rather than falling through it, that it does not sink into the
    floor, and that the simulation stays finite.
    """
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
    for _ in range(200):
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all(), "simulation diverged"
    assert data.qpos[2] > 0.2, f"robot sank or fell through the floor: z={data.qpos[2]:.3f}"
    assert data.ncon > 0, "no contacts at all; the robot is not touching the ground"


def test_g1_stance_is_not_immediately_unstable():
    """The stance should survive a short settling window. This is the honest,
    achievable version of "can it stand": it catches a badly wrong home pose or
    nominal height without pretending a passive humanoid balances indefinitely.
    """
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
    for _ in range(100):    # 0.4 s at the model's 0.004 s timestep
        mujoco.mj_step(model, data)
    assert data.qpos[2] > 0.6, f"stance collapsed immediately: z={data.qpos[2]:.3f}"


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
