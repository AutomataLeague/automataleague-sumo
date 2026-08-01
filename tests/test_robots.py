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


# ------------------------------------------------- grafted collision primitives

def test_extra_colliders_do_not_change_the_robots_mass():
    """They exist to make contact happen where the robot visibly is, not to change
    what it weighs.

    Honest about its own limits: every G1 body carries an explicit <inertial>, and
    MuJoCo ignores geom mass when one is present, so for THIS robot the assertion
    cannot fail — verified by setting the collider mass to 5 kg and watching the
    total stay at 66.6823. It is kept as insurance for a future robot without
    explicit inertials, where a grafted collider really would add weight. The
    assertion that does bite for the G1 is
    `test_the_grafted_colliders_actually_collide`.
    """

    from automataleague_sumo.envs.sumo.scene import build_sumo_model

    plain = get_robot("g1")
    plain.extra_colliders = []
    bare, _ = build_sumo_model(plain)
    full, _ = build_sumo_model(get_robot("g1"))

    assert full.ngeom > bare.ngeom, "no colliders were added at all"
    assert float(full.body_mass.sum()) == pytest.approx(float(bare.body_mass.sum()))
    assert np.allclose(full.body_inertia.sum(0), bare.body_inertia.sum(0), atol=1e-9)


def test_the_grafted_colliders_actually_collide():
    """Group and contype have to be set or the geoms are decorative. Checked by
    driving two robots into each other and requiring the new bodies to appear in
    the contact list, since a bare `add_geom` defaults to a non-colliding group.
    """
    import mujoco

    from automataleague_sumo.envs.sumo.scene import build_sumo_model

    model, scene = build_sumo_model("g1")
    data = mujoco.MjData(model)
    data.qpos[:] = scene.home_qpos
    # Drop side B onto side A so every part is in contact with something.
    qa, qb = scene.a.base_qposadr, scene.b.base_qposadr
    data.qpos[qb:qb + 3] = data.qpos[qa:qa + 3]
    mujoco.mj_forward(model, data)

    a, b = set(scene.a.geom_ids.tolist()), set(scene.b.geom_ids.tolist())
    touched = set()
    for c in data.contact[:data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in a and g2 in b) or (g1 in b and g2 in a):
            for g in (g1, g2):
                touched.add(mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))[2:])

    for body in ("left_shoulder_roll_link", "right_shoulder_roll_link"):
        assert body in touched, (
            f"{body} carries a grafted collider but never appears in a contact, "
            f"so it is not colliding. Touched: {sorted(touched)}")


def test_a_collider_on_a_missing_body_is_an_error():
    """It would otherwise silently add nothing while the spec claimed the part
    was covered — which is the whole failure being fixed here."""
    from automataleague_sumo.robots.base import ExtraCollider

    spec = get_robot("g1")
    spec.extra_colliders = [ExtraCollider(body="tail_link", size=0.05,
                                          pos=(0.0, 0.0, 0.0))]
    with pytest.raises(ValueError, match="does not exist"):
        spec.load_spec()


def test_a_collider_must_be_either_a_sphere_or_a_capsule():
    from automataleague_sumo.robots.base import ExtraCollider

    with pytest.raises(ValueError, match="exactly one of"):
        ExtraCollider(body="torso_link", size=0.05)
    with pytest.raises(ValueError, match="exactly one of"):
        ExtraCollider(body="torso_link", size=0.05, pos=(0, 0, 0),
                      fromto=(0, 0, 0, 0, 0, 1))
